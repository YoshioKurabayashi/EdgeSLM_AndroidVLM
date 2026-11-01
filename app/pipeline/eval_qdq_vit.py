# -*- coding: utf-8 -*-
"""Phase 2-2: 量子化した画像エンコーダ（final_qdq）を PC の ORT CPU で動かし、精度と答えの品質の材料を作る（2026-09-28）。

    python -m app.pipeline.eval_qdq_vit acc                          # 評価用 16 枚: HF との比較（SRQ あり・なし）
    python -m app.pipeline.eval_qdq_vit feats --variant srq          # 全画像の特徴 → work/feats/vit_b70_21x30_srq.npz
    python -m app.pipeline.eval_qdq_vit feats --variant nosrq
    （その後、Phase 1 の venv で ref_vlm run --prep fixed --budgets 70 --vision npz:work/feats/vit_b70_21x30_srq.npz）

**EdgeSLM_Android の venv で動かす**（onnxruntime がある）。ORT は最適化を切って（ORT_DISABLE_ALL）、QDQ をそのまま
「整数に丸めて戻す」計算として動かす＝量子化だけを真似た値（Gemma4 の device_small の「PC の QDQ」と同じ考え方）。

feats の前処理は HF のプロセッサを使わず `hf_vlm.pixels_fixed_np`（numpy）で作る。この venv には torchvision が無く、
Gemma4Processor が import できないため（2026-09-28 に踏んだ。venv にパッケージは足さない）。HF と同じ値になることは、
実行のたびに最初に capture の評価用 16 枚（HF のプロセッサで作った pixel_values）と比べて確かめ、食い違えば止まる。

## 何を見るか（基本設計 Phase 2 の Q8）

| 比較 | 見るもの | 目安 |
|---|---|---|
| **SRQ なしの版 ↔ HF（SRQ なし）** | 量子化（16bit の活性・8bit の重み・C-8 の 8bit の右辺）の算術の誤差だけ（揺れが無い） | cos ≥ 0.99 |
| SRQ なしの版 ↔ 書き出した float の ONNX | 同上（HF との差は 1e-5 なので、ほぼ同じ値になるはず） | 同上 |
| SRQ ありの版 ↔ HF（SRQ あり） | 記録だけ。揺れが入る（自前の float でも cos 0.84 まで） | — |
| 答えの品質（feats → ref_vlm） | SRQ ありの版・なしの版の特徴で 126 問 | 基準（42/123）と同等、入れ替わりに偏りなし |

結果は `results/P2_vit_quant/{tag}_acc.json`、特徴は `work/feats/vit_{tag}_{variant}.npz`（images: 画像の相対パス、feats: [K,M,1536]）。
2026-09-28 の実行の記録: `results/P2_20260928_vit_quant.md` の 3 節（SRQ なしの版 ↔ HF（SRQ なし）は cos 最小 0.9972・中央値 0.9993 で合格）
と 6 節（SRQ ありの版の特徴で 44/123）。StudyMemo 7.9 節。
入力: quantize_vit の `*_final_qdq.onnx`（と export_vit の float の ONNX）、capture_vit が保存した評価用 16 枚の HF の値。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from app.pipeline import vlm_common as C

RES = C.RESULTS / "P2_vit_quant"
FEATS = C.WORK / "feats"
PIX_TOL = 1.2e-7                                  # 前処理の照合の許容（float32 の 1 目盛り＝1.0 付近の丸めの差まで）


# 版の名前（"float" / "nosrq" / "srq"）から ONNX のパスを返す。float は export_vit が書いた量子化前のもの。
def model_path(tag: str, variant: str) -> Path:
    d = C.OPS / f"vit_{tag}"
    return d / (f"vit_{tag}.onnx" if variant == "float" else f"vit_{tag}_{variant}_final_qdq.onnx")


def session(path: Path):
    """ORT の CPU のセッションを、グラフの最適化を全部切って開く。

    最適化を切るのは、Q/DQ の組を ORT が畳んで整数の演算に置き換えないようにし、
    「float の計算の途中で整数に丸めて戻す」量子化の真似をそのまま流すため（モジュールの説明のとおり）。
    """
    import onnxruntime as ort
    if not path.exists():
        raise SystemExit(f"{path} が無い（float は export_vit export、srq/nosrq は quantize_vit で作る）")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def rel(a, b) -> dict:
    """トークンごとの相対 L2（b が正解）と cos（画像全体）。"""
    # a・b は [M,1536]（または [1,M,1536]）。最後の軸を残して 2 次元に並べ直し、float64 で計算する（丸めの差を混ぜないため）
    a = np.asarray(a, np.float64).reshape(-1, a.shape[-1])
    b = np.asarray(b, np.float64).reshape(-1, b.shape[-1])
    r = np.linalg.norm(a - b, axis=-1) / np.maximum(np.linalg.norm(b, axis=-1), 1e-12)
    cos = float((a * b).sum() / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))
    return {"rel_mean": float(r.mean()), "rel_max": float(r.max()), "cos": cos}


# 画像ごとの比べの結果（rel の戻り値の並び）を、相対 L2 の平均・cos の最小・cos の中央値にまとめる。
def summarize(rows: list[dict]) -> dict:
    cs = sorted(r["cos"] for r in rows)
    return {"rel_mean": float(np.mean([r["rel_mean"] for r in rows])), "cos_min": cs[0], "cos_median": cs[len(cs) // 2]}


def acc(a) -> dict:
    """評価用 16 枚で、float・SRQ なし・SRQ ありの 3 つの ONNX の出力を HF の値と比べる（基本設計 Phase 2 の Q8）。

    引数: a = コマンドの引数（budget・portrait）。戻り値: 版ごとの時間と比べの結果の辞書（main が JSON に書く）。
    HF の値（feat_nosrq・feat_srq）と入力（pixel_values）は capture_vit が保存したもの（export_vit.load_capture で読む）。
    """
    from app.pipeline.export_vit import load_capture, tag_for
    tag, grid = tag_for(a.budget, a.portrait)
    cap = load_capture(tag)
    E = len(cap["ids"])
    out = {"tag": tag, "n_eval": E, "variants": {}}
    feats = {}
    # ① 3 つの版を順に開き、16 枚を 1 枚ずつ流す（ORT の CPU で 1 枚の時間の中央値も残す）
    for v in ("float", "nosrq", "srq"):
        p = model_path(tag, v)
        if not p.exists():
            print(f"（{p.name} が無いので飛ばす）")
            continue
        sess = session(p)
        ys, ts = [], []
        for j in range(E):
            t0 = time.time()
            ys.append(sess.run(None, {"pixel_values": cap["pixel_values"][j][None].astype(np.float32)})[0][0])
            ts.append(time.time() - t0)
        feats[v] = np.stack(ys)
        out["variants"][v] = {"ort_cpu_sec_median": round(float(np.median(ts)), 2)}
        print(f"  {v}: {E} 枚（1 枚 {np.median(ts):.1f}s）", flush=True)

    # ② 版 v の出力を基準 ref と 1 枚ずつ比べ、まとめを out に入れて表示する
    def cmp(v, ref_name, ref):
        rows = [rel(feats[v][j], ref[j]) for j in range(E)]
        out["variants"][v][f"vs_{ref_name}"] = summarize(rows) | {"per_image": rows}
        s = out["variants"][v][f"vs_{ref_name}"]
        print(f"  {v:6s} ↔ {ref_name:10s}: 相対L2 平均 {s['rel_mean']:.4f}  cos 最小 {s['cos_min']:.4f}  中央値 {s['cos_median']:.4f}")

    print("\n比較（評価用 16 枚）:")
    if "float" in feats:
        cmp("float", "hf_nosrq", cap["feat_nosrq"])
    if "nosrq" in feats:
        cmp("nosrq", "hf_nosrq", cap["feat_nosrq"])
        if "float" in feats:
            cmp("nosrq", "float", feats["float"])
    if "srq" in feats:
        cmp("srq", "hf_srq", cap["feat_srq"])
        cmp("srq", "hf_nosrq", cap["feat_nosrq"])
    # ③ 合否は「SRQ なしの版 ↔ HF（SRQ なし）」の cos 最小が 0.99 以上か（揺れの無い比べで算術の誤差だけを見る）
    if "nosrq" in feats:
        c = out["variants"]["nosrq"]["vs_hf_nosrq"]["cos_min"]
        out["pass_nosrq"] = bool(c >= 0.99)
        print(f"\n[Q8 算術の誤差] SRQ なしの版 ↔ HF（SRQ なし）の cos 最小 {c:.4f} → "
              f"{'合格' if c >= 0.99 else '★目安（0.99）に届かない'}")
    print("  （SRQ ありの版は答えの品質で判定する: feats → ref_vlm --vision npz:…）")
    return out


def check_pixels(tag: str, budget: int, grid: tuple[int, int], man: dict) -> dict:
    """numpy の前処理（pixels_fixed_np）が HF のプロセッサと同じ値を作るかを、capture の評価用 16 枚で確かめる。"""
    from PIL import Image
    from app.pipeline import hf_vlm as H
    from app.pipeline.export_vit import load_capture
    cap = load_capture(tag)
    by_id = {it["id"]: it["image"] for it in man["items"]}
    # 同じ画像を PIL でレターボックスにし、numpy でパッチに並べた値と、HF のプロセッサの値の差の最大を取る
    worst, n_diff = 0.0, 0
    for j, iid in enumerate(cap["ids"]):
        img_in = H.prepare_image(Image.open(C.EVAL_DIR / by_id[str(iid)]), "fixed", budget, grid=grid)
        d = np.abs(H.pixels_fixed_np(img_in, grid) - cap["pixel_values"][j])
        worst, n_diff = max(worst, float(d.max())), n_diff + int((d > 0).sum())
    ok = worst <= PIX_TOL
    print(f"  前処理の照合（HF のプロセッサ ↔ numpy、{len(cap['ids'])} 枚）: 差の最大 {worst:.3g}・違う値の数 {n_diff} → "
          f"{'一致' if ok else '★食い違う'}", flush=True)
    if not ok:
        raise SystemExit("numpy の前処理が HF と違う値を作っている。hf_vlm.pixels_fixed_np を見直すこと（特徴は作らない）")
    return {"n_images": len(cap["ids"]), "max_abs": worst, "n_diff": n_diff}


def feats(a) -> dict:
    """manifest の全画像（重複を除く）について、固定の格子の前処理 → ONNX → 特徴 [M,1536] を保存する。

    引数: a = コマンドの引数（budget・portrait・variant）。戻り値: 保存先・枚数・時間・前処理の照合の結果。
    保存した npz は Phase 1 の venv の ref_vlm に `--vision npz:…` で渡し、文章側は HF のまま答えの品質を測る。
    """
    from PIL import Image
    from app.pipeline import hf_vlm as H
    from app.pipeline.export_vit import tag_for
    tag, grid = tag_for(a.budget, a.portrait)
    man = json.loads((C.EVAL_DIR / "manifest.json").read_text(encoding="utf-8"))
    pix_check = check_pixels(tag, a.budget, grid, man)            # ★先に前処理を確かめる（違えば止まる）
    sess = session(model_path(tag, a.variant))
    # 同じ画像に複数の問いがあるので、画像の並びは重複を除く（順番は manifest の初出の順のまま）。M は特徴の数（70）
    images = list(dict.fromkeys(it["image"] for it in man["items"]))
    M = (grid[0] // C.POOL) * (grid[1] // C.POOL)
    out = np.zeros((len(images), M, C.TEXT_HIDDEN), np.float32)
    t0 = time.time()
    for k, rp in enumerate(images):
        img_in = H.prepare_image(Image.open(C.EVAL_DIR / rp), "fixed", a.budget, grid=grid)
        out[k] = sess.run(None, {"pixel_values": H.pixels_fixed_np(img_in, grid)[None]})[0][0]
        if (k + 1) % 10 == 0:
            print(f"  {k + 1}/{len(images)}（{time.time() - t0:.0f}s）", flush=True)
    FEATS.mkdir(parents=True, exist_ok=True)
    path = FEATS / f"vit_{tag}_{a.variant}.npz"
    np.savez(path, images=np.array(images), feats=out, variant=a.variant, tag=tag)
    print(f"-> {path}（{len(images)} 枚、{time.time() - t0:.0f}s）")
    return {"tag": tag, "variant": a.variant, "path": str(path), "n_images": len(images),
            "sec": round(time.time() - t0, 1), "pixel_check": pix_check}


# acc か feats を動かし、結果を results/P2_vit_quant/{tag}_acc.json（feats は {tag}_feats_{版}.json）に書く。
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["acc", "feats"])
    ap.add_argument("--budget", type=int, default=70, choices=sorted(C.FIXED_GRIDS))
    ap.add_argument("--portrait", action="store_true")
    ap.add_argument("--variant", default="srq", choices=["srq", "nosrq", "float"], help="feats で使う版")
    a = ap.parse_args()
    C.utf8_stdout()
    out = {"acc": acc, "feats": feats}[a.cmd](a)
    RES.mkdir(parents=True, exist_ok=True)
    name = f"{out['tag']}_{a.cmd}" + (f"_{a.variant}" if a.cmd == "feats" else "")
    C.write_json_atomic(RES / f"{name}.json", out)
    print(f"-> {RES / (name + '.json')}")


if __name__ == "__main__":
    main()
