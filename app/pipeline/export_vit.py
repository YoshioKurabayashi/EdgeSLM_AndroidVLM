# -*- coding: utf-8 -*-
"""Phase 2-1: 自前の配線（vit_graph.py）を HF と突き合わせ、ONNX に書き出す（2026-09-28）。

    python -m app.pipeline.export_vit check                  # 自前の配線 ↔ HF（capture_vit の正解）
    python -m app.pipeline.export_vit export                 # ONNX・spec・較正データ
    python -m app.pipeline.export_vit ortcheck               # 書き出した ONNX を ORT CPU で動かし、自前の配線・HF と比べる
    （共通の引数: --budget 70 [--portrait]）

**EdgeSLM_Android の venv（torch CPU・onnx・onnxruntime・onnxscript）で動かす**（EdgeSLM_AndroidGemma4 と同じ。
Phase 1 の venv には onnx が無い）。先に capture_vit.py（Phase 1 の venv）で正解を取っておくこと。

## 何を確かめるか

| 比較 | 合格の目安 | 分かること |
|---|---|---|
| 自前（SRQ なし）↔ HF（SRQ なし） | 出力の相対 L2 が 1e-4 以下（理想は 1e-5 以下） | 配線が正しい（位置の埋め込み・2 次元 RoPE・q/k/v の正規化・プーリングの並び・射影層） |
| 同上、途中の値（2 枚） | 層ごとに滑らかに増える（1e-6 の桁） | ずれたとき、どの層からずれたか。配線の誤りならある層で急に 1e-2 以上に跳ぶ |

合格の目安を 1e-4 にしたのは、HF の attention（SDPA）と自前の行列積で足し算の順序が違い、16 層・70 トークンの
「最大」を取ると 1e-5 前後になりうるため（Phase 1 の文章側でも順序だけで 2.6e-5）。配線の誤りは 1e-2 以上になるので区別できる。
| 自前（SRQ あり）↔ HF（SRQ あり） | 記録だけ | 計算の順序の違いが 8bit の境目で転ぶ分（Phase 1 と同じ現象）。大きすぎないか |
| ×√768 を省いた形 ↔ 省かない形 | 1e-6 以下 | fold_pool_scale（eps を 1/768 にする書き換え）が同じ計算であること |
| ONNX（ORT CPU）↔ 自前 | cos ≥ 0.999999 かつ 相対 L2 の最大 ≤ 1e-3 | 書き出しで計算が変わっていない（ORT の CPU は tanh などを近似で計算するので、torch より少しずれる。2026-09-28 の初回は最大 1.1e-4・cos 0.9999999999） |
| 自前（SRQ あり）↔ HF（SRQ あり）の途中の値 | 記録だけ | 層 0 で既に大きければ SRQ の掛け方の誤り、層を追って増えるなら 8bit の境目で転ぶ揺れ |

## 書き出しの約束（EdgeSLM_AndroidGemma4 の export_chunk と同じ）

- `torch.onnx.export(..., opset_version=21, dynamo=True, dynamic_axes=None)`。形はすべて固定
- 入力 `pixel_values` [1,N,768] float32（0〜1）、出力 `image_embeds` [1,M,1536] float32
- 位置の埋め込み・cos/sin は初期化子（定数）になる
- 出力先 `work/ops/vit_{tag}/`: `vit_{tag}.onnx`、`vit_{tag}.spec.json`（Phase 2-2 の量子化の設定）、`vit_{tag}_calib.npz`（較正の入力）
- 使っている演算の一覧を出し、v68 で問題になりうるもの（Expand・Gather・Where・Cast など）が無いかを確かめる

結果は `results/P2_vit_export/{tag}_{cmd}.json`（数値）。

## 実行の記録（2026-09-28、`results/P2_20260928_vit_export.md`。StudyMemo 7.6〜7.8 節）

- check: 自前（SRQ なし）↔ HF（SRQ なし）の出力の相対 L2 最大 1.1e-5・平均 1.2e-6 → 合格。途中の値は層を追って 1e-7 → 1e-5 と滑らかに増える
- ortcheck: ONNX（ORT CPU）↔ 自前は相対 L2 最大 1.1e-4・cos 0.99999999993。初めの目安 1e-4 をわずかに超えたので、
  目安を「cos ≥ 0.999999 かつ 最大 ≤ 1e-3」に改めた（最後の層の大きな値で相対的に目立ったもの）
- 自前（SRQ あり）↔ HF（SRQ あり）は記録だけ（SRQ の 8bit の境目で転ぶ揺れ。StudyMemo 7.8 節）
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter

import numpy as np

from app.pipeline import vlm_common as C

RES = C.RESULTS / "P2_vit_export"
# v68 で動いている演算（EdgeSLM_AndroidGemma4 の P0D の 14 種 ＋ Sub・Softmax）。これ以外が出たら要確認
EXPECTED_OPS = {"Add", "Sub", "Mul", "Div", "MatMul", "Reshape", "Transpose", "Slice", "Concat", "Neg", "Pow",
                "ReduceMean", "Sqrt", "Reciprocal", "Softmax", "Tanh", "Constant"}
# RISKY_OPS は v68 で問題になりうる演算と、その理由（C-1・C-10 などの禁止事項の番号は EdgeSLM_Android の StudyMemo の決まり）
PASS_TOL = 1e-4                    # 合格の目安（上の表を参照）
RISKY_OPS = {"Expand": "C-10（端末で 1002）", "Gather": "グラフ入力に直接つながると C-1", "Where": "要確認",
             "Cast": "float 以外の型が混ざる", "Range": "動的な形", "Shape": "動的な形", "If": "制御", "Loop": "制御"}


# 予算と向きから、ファイル名の札（例: "b70_21x30"）と格子を返す。capture_vit と同じ札。
def tag_for(budget: int, portrait: bool) -> tuple[str, tuple[int, int]]:
    grid = (C.PORTRAIT_GRIDS if portrait else C.FIXED_GRIDS)[budget]
    return f"b{budget}_{grid[0]}x{grid[1]}", grid


# capture_vit が保存した評価用の npz（入力・HF の出力・途中の値）を辞書で読む。eval_qdq_vit・probe_vit も使う。
def load_capture(tag: str) -> dict:
    p = C.CAPTURE / f"vit_{tag}.npz"
    if not p.exists():
        raise SystemExit(f"{p} が無い。先に Phase 1 の venv で python -m app.pipeline.capture_vit を動かすこと")
    z = np.load(p)
    return {k: z[k] for k in z.files}


def rel(a, b) -> dict:
    """トークン（行）ごとの相対 L2（‖a−b‖/‖b‖、b が正解）の最大・平均と、全体の cos。"""
    a = np.asarray(a, np.float64).reshape(-1, np.asarray(a).shape[-1])
    b = np.asarray(b, np.float64).reshape(-1, np.asarray(b).shape[-1])
    r = np.linalg.norm(a - b, axis=-1) / np.maximum(np.linalg.norm(b, axis=-1), 1e-12)
    cos = float((a * b).sum() / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-30))
    return {"rel_max": float(r.max()), "rel_mean": float(r.mean()), "cos": cos, "max_abs": float(np.abs(a - b).max())}


# 自前の配線（VitGraph）を組んで返す。fold = ×√768 を eps に吸収した形にするか。
def build(grid, fold=True):
    from app.pipeline.vit_graph import QatFile, VitGraph
    t0 = time.time()
    m = VitGraph(QatFile(), grid, fold_pool_scale=fold).eval()
    print(f"自前の配線を組んだ（{time.time() - t0:.1f}s、格子 {grid[0]}×{grid[1]}、N={m.N} M={m.M}）", flush=True)
    return m


def check(a) -> dict:
    """自前の配線を HF の正解（capture_vit）と突き合わせる（モジュールの説明の表の 1・2・4・6 行目と ×√768 の書き換え）。

    引数: a = コマンドの引数（budget・portrait）。戻り値: 比べの結果の辞書（pass = SRQ なしの出力の相対 L2 最大が 1e-4 以下か）。
    vit_graph.SRQ をこの中で切り替える（SRQ ありは丸めが 8bit の境目で転ぶので、記録だけ）。
    """
    import torch
    from app.pipeline import vit_graph as V
    tag, grid = tag_for(a.budget, a.portrait)
    cap = load_capture(tag)
    assert (cap["pos"] == V.grid_positions(*grid).numpy()).all(), "正解を取ったときの格子と違う"
    m = build(grid)
    E = len(cap["ids"])
    out = {"tag": tag, "grid": list(grid), "n_eval": E}

    # 1 枚の pixel_values [N,768] を [1,N,768] にして流す（keep で途中の値も返す）
    def run(pix, keep=False):
        with torch.no_grad():
            return m.run(torch.from_numpy(pix[None]), keep=keep)

    # 1) SRQ なし: 出力
    V.SRQ = False
    res = [rel(run(cap["pixel_values"][j])["image_embeds"][0].numpy(), cap["feat_nosrq"][j]) for j in range(E)]
    out["nosrq_output"] = {"rel_max": max(r["rel_max"] for r in res), "rel_mean": float(np.mean([r["rel_mean"] for r in res])),
                           "cos_min": min(r["cos"] for r in res), "per_image": res}
    # 2) SRQ なし: 途中の値（先頭の数枚）
    D = cap["dbg_nosrq_layers"].shape[0]
    steps = []
    for j in range(D):
        o = run(cap["pixel_values"][j], keep=True)
        st = {"patch_embed": rel(o["patch_embed"][0].numpy(), cap["dbg_nosrq_patch_embed"][j])["rel_max"]}
        for li in range(C.VIT_LAYERS):
            st[f"layer_{li}"] = rel(o[f"layer_{li}"][0].numpy(), cap["dbg_nosrq_layers"][j, li])["rel_max"]
        st["pooled"] = rel(o["pooled"][0].numpy(), cap["dbg_nosrq_pooled"][j])["rel_max"]
        steps.append(st)
    out["nosrq_steps"] = steps
    # 3) ×√768 を省いた形と省かない形
    m2 = build(grid, fold=False)
    with torch.no_grad():
        x = torch.from_numpy(cap["pixel_values"][0][None])
        out["fold_vs_nofold"] = rel(m(x)[0].numpy(), m2(x)[0].numpy())["rel_max"]
    del m2
    # 4) SRQ あり（記録だけ）
    V.SRQ = True
    res_s = [rel(run(cap["pixel_values"][j])["image_embeds"][0].numpy(), cap["feat_srq"][j]) for j in range(E)]
    # 途中の値も比べる（層 0 で既に大きい → SRQ の掛け方の誤り、層を追って増える → 境目で転ぶ揺れ）
    steps_s = []
    for j in range(D):
        o = run(cap["pixel_values"][j], keep=True)
        st = {"patch_embed": rel(o["patch_embed"][0].numpy(), cap["dbg_srq_patch_embed"][j])["rel_mean"]}
        for li in range(C.VIT_LAYERS):
            st[f"layer_{li}"] = rel(o[f"layer_{li}"][0].numpy(), cap["dbg_srq_layers"][j, li])["rel_mean"]
        st["pooled"] = rel(o["pooled"][0].numpy(), cap["dbg_srq_pooled"][j])["rel_mean"]
        steps_s.append(st)
    out["srq_steps_mean"] = steps_s
    V.SRQ = False
    out["srq_output"] = {"rel_max": max(r["rel_max"] for r in res_s), "rel_mean": float(np.mean([r["rel_mean"] for r in res_s])),
                         "cos_min": min(r["cos"] for r in res_s), "per_image": res_s}
    # SRQ あり・なしの HF どうしの差（SRQ そのものの大きさの目安）
    out["hf_srq_vs_nosrq"] = rel(cap["feat_srq"], cap["feat_nosrq"])

    # 5) 合否と表示
    ok = out["nosrq_output"]["rel_max"] <= PASS_TOL
    print(f"\n[SRQ なし] 出力の相対L2 最大 {out['nosrq_output']['rel_max']:.2e}  平均 {out['nosrq_output']['rel_mean']:.2e}"
          f"  cos 最小 {out['nosrq_output']['cos_min']:.8f}  → {'合格' if ok else '★不合格'}（目安 {PASS_TOL:g}、理想 1e-5）")
    for j, st in enumerate(steps):
        worst = max(st, key=st.get)
        print(f"  途中の値（{cap['ids'][j]}）: パッチの埋め込み {st['patch_embed']:.1e}  層0 {st['layer_0']:.1e}  "
              f"層15 {st['layer_15']:.1e}  プーリング {st['pooled']:.1e}  （最大 {worst} {st[worst]:.1e}）")
    print(f"[書き換え] ×√768 を省いた形 ↔ 省かない形 {out['fold_vs_nofold']:.2e}")
    print(f"[SRQ あり] 出力の相対L2 最大 {out['srq_output']['rel_max']:.2e}  平均 {out['srq_output']['rel_mean']:.2e}"
          f"  cos 最小 {out['srq_output']['cos_min']:.6f}（記録だけ。HF どうしの SRQ あり↔なしは "
          f"平均 {out['hf_srq_vs_nosrq']['rel_mean']:.2e}）")
    for j, st in enumerate(steps_s):
        print(f"  SRQ ありの途中の値（{cap['ids'][j]}、相対L2 の平均）: 層0 {st['layer_0']:.1e}  層1 {st['layer_1']:.1e}  "
              f"層3 {st['layer_3']:.1e}  層7 {st['layer_7']:.1e}  層11 {st['layer_11']:.1e}  層15 {st['layer_15']:.1e}")
    out["pass"] = bool(ok)
    return out


def export(a) -> dict:
    """自前の配線を ONNX に書き出し、演算・入出力・行列積の数を確かめ、量子化の設定（spec）と較正データを置く。

    戻り値: 書き出しの記録（problems が空なら pass）。出力は work/ops/vit_{tag}/ の下（モジュールの説明「書き出しの約束」）。
    """
    import onnx
    import torch
    from app.pipeline import vit_graph as V
    assert not V.SRQ, "SRQ=True のまま書き出すと丸め（Round/Clip）がグラフに入る"
    tag, grid = tag_for(a.budget, a.portrait)
    cap = load_capture(tag)
    cal_src = C.CAPTURE / f"vit_{tag}_calib.npz"
    if not cal_src.exists():                                     # 長い書き出しの前に確かめる
        raise SystemExit(f"{cal_src} が無い。capture_vit を動かし直すこと")
    m = build(grid)
    d = C.OPS / f"vit_{tag}"
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"vit_{tag}.onnx"
    # 前回の生成物を消してから書く（書き出しが途中で落ちたとき、古い ONNX を ortcheck が黙って確かめないように）
    for old in (path, d / f"vit_{tag}.onnx.data"):
        old.unlink(missing_ok=True)
    x = torch.from_numpy(cap["pixel_values"][0:1].astype(np.float32))
    t0 = time.time()
    torch.onnx.export(m, (x,), str(path), input_names=m.INPUT_NAMES, output_names=m.OUTPUT_NAMES,
                      dynamic_axes=None, opset_version=21, do_constant_folding=True, dynamo=True)
    # 書き出した ONNX を読み直して見張る: 想定外の演算・入出力の形・入力を最初に使う演算・行列積の数
    mo = onnx.load(str(path), load_external_data=True)
    ops = Counter(n.op_type for n in mo.graph.node)
    extra = sorted(set(ops) - EXPECTED_OPS)
    risky = {o: RISKY_OPS[o] for o in extra if o in RISKY_OPS}
    ins = [(v.name, [dd.dim_value for dd in v.type.tensor_type.shape.dim]) for v in mo.graph.input]
    outs = [(v.name, [dd.dim_value for dd in v.type.tensor_type.shape.dim]) for v in mo.graph.output]
    size = path.stat().st_size + sum(p.stat().st_size for p in d.glob(f"vit_{tag}.onnx.data"))
    print(f"書き出し {time.time() - t0:.0f}s  ノード {len(mo.graph.node)}  大きさ {size / 2**20:.0f}MiB")
    print("  演算:", dict(sorted(ops.items())))
    print("  入力:", ins, " 出力:", outs)
    if extra:
        print(f"  ★想定外の演算: {extra}" + (f"  要注意: {risky}" if risky else ""))
    # 入力 pixel_values を最初に使う演算（C-1: Gather/Reshape/Transpose/Slice に直接入ると ORT が量子化しない）
    first = [n.op_type for n in mo.graph.node if "pixel_values" in n.input]
    print(f"  pixel_values を最初に使う演算: {first}")
    # 行列積の数: 右辺が重み（初期化子）のもの = 16 層 × 7 本 ＋ パッチの埋め込み ＋ 射影層 = 114、
    #             活性×活性（q·k^T と 確率·v）= 16 層 × 2 = 32。重みの Transpose が畳まれていないと数が合わない
    inits = {i.name for i in mo.graph.initializer}
    mm = [n for n in mo.graph.node if n.op_type == "MatMul"]
    mm_w = sum(1 for n in mm if n.input[1] in inits)
    mm_aa = len(mm) - mm_w
    exp_w, exp_aa = C.VIT_LAYERS * 7 + 2, C.VIT_LAYERS * 2
    print(f"  行列積: 右辺が重み {mm_w}（想定 {exp_w}）・活性×活性 {mm_aa}（想定 {exp_aa}）")
    # 入口は「2·(p − 0.5)」の Sub のはず（pixel_values が Reshape などに直接入ると ORT が量子化しない。C-1）
    N, M = grid[0] * grid[1], (grid[0] // C.POOL) * (grid[1] // C.POOL)
    io_ok = ins == [("pixel_values", [1, N, C.PATCH_DIM])] and outs == [("image_embeds", [1, M, C.TEXT_HIDDEN])]
    problems = []
    if not io_ok:
        problems.append(f"入出力の形が想定と違う（動的な次元は 0 と表示される）: {ins} {outs}")
    if risky:
        problems.append(f"要注意の演算: {risky}")
    if (mm_w, mm_aa) != (exp_w, exp_aa):
        problems.append(f"行列積の数が想定と違う: 重み {mm_w}/{exp_w}、活性×活性 {mm_aa}/{exp_aa}")
    if first != ["Sub"]:
        problems.append(f"pixel_values を最初に使う演算が Sub でない: {first}（C-1 の確認が要る）")
    for p_ in problems:
        print(f"  ★{p_}")
    # 較正の入力（キー＝グラフの入力名。p2b_ops が 1 サンプルずつ [1,N,768] にして食わせる）
    cal = d / f"vit_{tag}_calib.npz"
    shutil.copyfile(cal_src, cal)
    n_cal = np.load(cal)["pixel_values"].shape[0]
    # ★weight_bits は 4 のまま（Gemma4 と同じ）。ORT に全部の重みを「INT4・出力チャネルごと」の DQ として出させ、
    #   Phase 2-2 の qat_inject がそれを見つけて書き換える（QAT の 112 本は INT8 の整数とスケールに、QAT でない
    #   パッチの埋め込み・射影層は INT8 の出力チャネルごとに作り直す）。Gemma4 の qat_inject は INT4 の DQ しか
    #   処理しないため。最終的に INT4 は残らない（残っていたら qat_inject が止まる）
    spec = {"model": str(path).replace("\\", "/"),
            "note": f"Gemma 4 E2B の画像エンコーダ（ViT 16 層）＋射影層、格子 {grid[0]}×{grid[1]}。"
                    "weight_bits=4 は qat_inject が書き換えるための仮の型（最終的には全部 INT8）",
            "per_channel_weights": True, "weight_bits": 4,
            "calib_npz": str(cal).replace("\\", "/")}
    (d / f"vit_{tag}.spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  較正データ {n_cal} 枚 -> {cal}\n-> {path}\n-> {d / f'vit_{tag}.spec.json'}")
    return {"tag": tag, "path": str(path), "nodes": len(mo.graph.node), "size_mib": round(size / 2**20, 1),
            "ops": dict(ops), "unexpected_ops": extra, "risky_ops": risky, "inputs": ins, "outputs": outs,
            "first_consumer_of_input": first, "matmul_weight": mm_w, "matmul_actact": mm_aa,
            "n_calib": int(n_cal), "problems": problems, "pass": not problems, "sec": round(time.time() - t0, 1)}


def ortcheck(a) -> dict:
    """書き出した ONNX を ORT の CPU（最適化なし）で動かし、自前の配線・HF（SRQ なし）と評価用 16 枚で比べる。"""
    import onnxruntime as ort
    import torch
    from app.pipeline import vit_graph as V
    tag, grid = tag_for(a.budget, a.portrait)
    cap = load_capture(tag)
    path = C.OPS / f"vit_{tag}" / f"vit_{tag}.onnx"
    if not path.exists():
        raise SystemExit(f"{path} が無い。先に export を動かすこと")
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL   # 書き出したままの計算を比べる
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    m = build(grid)
    V.SRQ = False
    E = len(cap["ids"])
    r_own, r_hf, times = [], [], []
    for j in range(E):
        x = cap["pixel_values"][j][None].astype(np.float32)
        t0 = time.time()
        y = sess.run(None, {"pixel_values": x})[0][0]
        times.append(time.time() - t0)
        with torch.no_grad():
            yo = m(torch.from_numpy(x))[0].numpy()
        r_own.append(rel(y, yo))
        r_hf.append(rel(y, cap["feat_nosrq"][j]))
    out = {"tag": tag, "onnx_vs_own": {"rel_max": max(r["rel_max"] for r in r_own),
                                        "rel_mean": float(np.mean([r["rel_mean"] for r in r_own])),
                                        "cos_min": min(r["cos"] for r in r_own)},
           "onnx_vs_hf_nosrq": {"rel_max": max(r["rel_max"] for r in r_hf), "cos_min": min(r["cos"] for r in r_hf)},
           "ort_cpu_sec_median": round(float(np.median(times)), 3)}
    # ORT の CPU は tanh などを近似で計算するので、torch とは 1e-4 程度ずれることがある（書き出しの誤りなら cos が大きく落ちる）
    ok = out["onnx_vs_own"]["cos_min"] >= 0.999999 and out["onnx_vs_own"]["rel_max"] <= 1e-3
    print(f"ONNX（ORT CPU）↔ 自前の配線: 相対L2 最大 {out['onnx_vs_own']['rel_max']:.2e}  平均 {out['onnx_vs_own']['rel_mean']:.2e}"
          f"  cos 最小 {out['onnx_vs_own']['cos_min']:.10f} → {'合格' if ok else '★不合格'}（目安 cos ≥ 0.999999・最大 ≤ 1e-3）")
    print(f"ONNX ↔ HF（SRQ なし）:        相対L2 最大 {out['onnx_vs_hf_nosrq']['rel_max']:.2e}  cos 最小 {out['onnx_vs_hf_nosrq']['cos_min']:.8f}")
    print(f"ORT CPU の時間（参考）: 1 枚 {out['ort_cpu_sec_median']}s")
    out["pass"] = bool(ok)
    return out


# check・export・ortcheck のどれかを動かし、results/P2_vit_export/{tag}_{cmd}.json に書く。
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "export", "ortcheck"])
    ap.add_argument("--budget", type=int, default=70, choices=sorted(C.FIXED_GRIDS))
    ap.add_argument("--portrait", action="store_true")
    a = ap.parse_args()
    C.utf8_stdout()
    out = {"check": check, "export": export, "ortcheck": ortcheck}[a.cmd](a)
    RES.mkdir(parents=True, exist_ok=True)
    C.write_json_atomic(RES / f"{out['tag']}_{a.cmd}.json", out)
    print(f"-> {RES / (out['tag'] + '_' + a.cmd + '.json')}")


if __name__ == "__main__":
    main()
