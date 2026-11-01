# -*- coding: utf-8 -*-
"""16 トークン版を「NPU のずれ」に合わせて書き直し、変換し直す（2026-10-02）。

    python -m app.pipeline.build_t16fix                    # 書き直し（5 分前後）→ 変換（QAIRT、4 時間前後。夜に回す）
    python -m app.pipeline.build_t16fix --steps rewrite    # 書き直しと、PC の CPU での同値の確かめだけ
    python -m app.pipeline.build_t16fix --steps convert    # 変換だけ
    python -m app.pipeline.build_t16fix --redo             # 済んでいてもやり直す
    python -m app.pipeline.build_t16fix --variant t        # ★PLE の tanh だけ直した版（g4t16t_W256）。2026-10-02 夜に追加

**EdgeSLM_Android の venv（$PY2）で動かす**。元の 16 トークン版（`work/ops/g4t16_W256/`）は変えない。
書き直した版は `work/ops/g4t16f_W256/`（f = fix）。端末では `device_vlm --text t16f`・`probe_rows --model t16f` で使う。

## 何を直すか（切り分けの 4 段目・`probe_fix` の結果、2026-10-01）

| 原因 | 書き直し | 小さいグラフでの効き目（層 0 の出力の差、画像の行） |
|---|---|---|
| ① （2026-10-01 の見立て。★下の訂正）QAIRT が正規化を HTP の RmsNorm にまとめると値がずれる | **正規化の「平均（ReduceMean）」を「1/N を並べた重み [N,1] との行列の掛け算（MatMul）」に**。RmsNorm の並びにならないので、まとめられない | まとめた版 150% → 24% |
| ② 形 [16 行 × 256] の tanh が 3 行目以降で壊れる（PLE の gate の GELU、35 層） | **tanh の前後で形を [1, 1, 16×256] に変え、1 行の形で計算する** | ①と合わせて 20% |

★2026-10-18 訂正: 旧「① QAIRT が正規化を HTP の RmsNorm にまとめると値がずれる」（原因と言い切っていた）。2026-10-03 の 7 段目で、
ずれていたのは基準にした PC の CPU の方（正規化の途中の値 x²・平均・1/√ の目盛りを文章だけで決めたので、画像の行で切れる）で、
NPU の RmsNorm は全部の行で正確（重みのない V の正規化の出力の二乗平均の平方根が 1.000）と分かった。① の書き直し（f）は
NPU を CPU の誤った値に合わせるもので、画像が見えなくなった（「画像が無い」12 問）。② も、tanh の値が NPU と CPU で違うのは事実だが、
直しても答えも logits も変わらなかった（t の版）。
決定（2026-10-03、利用者）: 文章側は元の 16 トークン版（t16）で確定し、f・t の版は採用しない（StudyMemo 10.7 節の 6・7 段目、
`results/P5_20261003_t16_accuracy.md`）。このファイルは、その切り分けで使った道具として残している。

- どちらも**計算の中身は同じ**（PC の CPU で、書き直す前と後の差 0.000%。小さいグラフで確認済み）。ここでも全体で確かめる
- 平均の重みは int8 で 127、目盛り 1/(127N)（＝ちょうど 1/N）。正規化 1 つに小さな行列の掛け算が 1 つ増えるが、重みは N 個だけ
- 「x² を x×x に」は効かなかった（RmsNorm にまとめられたまま）。「tanh を 2·sigmoid(2x)−1 に」は効いたが、新しい目盛りで CPU の計算が 0.5% 変わるので、形を変える方を使う

## 段

| 段 | 中身 |
|---|---|
| rewrite | 元の final_qdq を読み、ReduceMean と PLE の tanh を書き直して `g4t16f_W256_final_qdq.onnx` に。書き直しの数（正規化の数・35 層の tanh）を確かめ、PC の CPU で元と書き直した版を同じ入力（syn_nav_S の最初の 2 回ぶん）に通して差を見る |
| convert | build_t16 と同じ（build_context.sh、`QAIRT_INT4_PATCH=1` → 検査 → EPContext で包む）。変換後のグラフに RmsNorm が残っていないか、DDR の読み書きの見積もりも出す |

結果: `results/P5_t16/build_t16fix.json`（`--variant t` は `build_t16fix_t.json`）

## 2 つの版（`--variant`、2026-10-02 夜に追加）

| 版 | 書き直し | 作る場所 | 端末での名前 |
|---|---|---|---|
| f（既定） | 正規化の平均（①）＋ PLE の tanh（②） | `work/ops/g4t16f_W256/` | `--text t16f` |
| t | PLE の tanh（②）だけ。正規化は元のまま（HTP の RmsNorm にまとまる＝ 1 トークン版と同じ計算） | `work/ops/g4t16t_W256/` | `--text t16t` |

**なぜ t を作るか**: f を 126 問で評価すると、16 行まとめたときだけのずれ（②）は直ったが、点は 45 → 37 に下がり、
「画像が提供されていません」と答える誤りが 0 → 12 問に増えた。1 トークンあたりの時間も 87 → 101ms に伸びた（行列の掛け算 242 個ぶん）。
1 トークン版（正規化は RmsNorm にまとまったまま）がいちばん HF に近い（97/126）ので、16 行のときだけ起きる不具合（②）だけを直した版を試す。
t は tanh の前後で形を変えるだけなので、PC の CPU では元とビット単位で同じになるはず（確かめで 0.0000% を見る）。

## 実行の記録（StudyMemo 10.7 節）

- f: 変換 4.0 時間、context binary 1,467.5MiB（元と同じ）、検査は全部合格、変換後の RmsNorm 0 個。DDR の読みの見積もり 1.570GB（元 1.549GB）。
  書き直す前と後の PC の CPU どうしの差は logits 18%（書き直しの誤りではなく、平均の 1 つの値が 1 目盛りずれたのが育ったもの。diag_t16fix で確かめた）
- t: 変換 3.9 時間、1,467.1MiB、RmsNorm 242 個（元と同じ）、PC の CPU では元とビット一致（0.0000%）
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np

from app.pipeline import device_vit as D
from app.pipeline import vlm_common as C

# 書き直しの元（build_t16 の繋いだ final_qdq と、その入出力の記録 io.json）。元は読むだけで変えない
SRC = C.OPS / "g4t16_W256" / "g4t16_W256_final_qdq.onnx"
SRC_IO = C.OPS / "g4t16_W256" / "io.json"
# 版 → （グラフの名前、書き直しの種類）
VARIANTS = {"f": (D.T16F_NAME, ("normmm", "tanhflat")), "t": (D.T16T_NAME, ("tanhflat",))}


def use(variant: str):
    """作る版を選ぶ（名前・置き場所・書き直しの種類を切り替える）。既定は f（diag_t16fix がこの DST を読む）"""
    global VARIANT, NAME, FIXES, DIR, DST, EPCTX
    VARIANT = variant
    NAME, FIXES = VARIANTS[variant]
    DIR = C.OPS / NAME
    DST = DIR / f"{NAME}_final_qdq.onnx"
    EPCTX = DIR / f"{NAME}_a16w8_rhs8_epctx.onnx"


use("f")
# 既定の版は f（モジュールを読み込んだ時点で use("f") を済ませ、DST などの名前を決めておく）
RES = C.RESULTS / "P5_t16"
GIT_BASH = os.environ.get("GIT_BASH", r"C:\Program Files\Git\bin\bash.exe")


def rewrite() -> dict:
    """元の final_qdq を読み、ReduceMean と PLE の tanh を書き直して DST に保存する（段 rewrite）。

    戻り値: 書き直した数・全体の数・時間・PC の CPU での同値の確かめ（equiv_check）の結果。
    書き直しの部品（fix_normmm・fix_tanhflat）は probe_fix.py の小さいグラフで試したものをそのまま使う。
    """
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper
    from app.pipeline import probe_fix as PF
    t0 = time.time()
    m = onnx.load(str(SRC))
    print(f"  読み込み {SRC.name}（{time.time() - t0:.0f}s、演算 {len(m.graph.node)}）", flush=True)
    # ① probe_fix の Graph（ノードの出どころ・使い先を引ける包み）で包み、書き直す前の ReduceMean・Tanh の数を数える
    G = PF.Graph(m, batch=True)
    G.refresh()
    n_mean0 = sum(1 for n in G.g.node if n.op_type == "ReduceMean")
    n_tanh0 = sum(1 for n in G.g.node if n.op_type == "Tanh")
    # ---- 書き直しに要る形（ReduceMean と Tanh の入力）を ORT で 1 回回して取る ----
    need = sorted({n.input[0] for n in G.g.node if n.op_type in ("ReduceMean", "Tanh")})
    # 書き直しの部品は入力の形を知る必要がある。ONNX に形が書かれていないので、入力の値を出力に足して
    # 全部 0 の入力で 1 回だけ流し、形を PF.SHAPES に入れる。そのあと出力を元に戻す
    orig_out = [o for o in m.graph.output]
    keep_out = [helper.make_tensor_value_info(o.name, o.type.tensor_type.elem_type,
                                              [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in orig_out]
    del m.graph.output[:]
    m.graph.output.extend([helper.make_tensor_value_info(t, TensorProto.FLOAT, None) for t in need])
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    t1 = time.time()
    s = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
    feed = {v.name: np.zeros([d.dim_value for d in v.type.tensor_type.shape.dim], np.float32) for v in m.graph.input}
    PF.SHAPES = {t: list(r.shape) for t, r in zip(need, s.run(need, feed))}
    del s
    print(f"  形を取った（{len(need)} 本、{time.time() - t1:.0f}s）", flush=True)
    del m.graph.output[:]
    m.graph.output.extend(keep_out)
    # ---- 書き直し ----
    c_mean = PF.fix_normmm(G) if "normmm" in FIXES else 0
    c_tanh = PF.fix_tanhflat(G) if "tanhflat" in FIXES else 0
    print(f"  書き直し（版 {VARIANT}）: 正規化の平均 {c_mean}/{n_mean0} 個 → 行列の掛け算"
          + ("" if "normmm" in FIXES else "（この版では触らない）")
          + f"、PLE の tanh {c_tanh} 個を 1 行の形に"
          f"（tanh は全部で {n_tanh0} 個。残りは MLP の GELU の tanh で、壊れていないので触らない）", flush=True)
    if "normmm" in FIXES and c_mean != n_mean0:
        print(f"  ★平均の書き直しが {c_mean}/{n_mean0}（axes=[-1]・keepdims=1 でないものは残した。RmsNorm にまとめられる可能性）", flush=True)
    # PLE の門の tanh は 35 層に 1 個ずつ。ちょうど 35 個でなければ、別の tanh を書き換えた・見落としたおそれがあるので止める
    if c_tanh != 35:
        raise SystemExit(f"PLE の tanh は 35 層ぶんのはずが {c_tanh} 個")
    DIR.mkdir(parents=True, exist_ok=True)
    t1 = time.time()
    onnx.save(m, str(DST))
    print(f"  保存 {DST.name}（{DST.stat().st_size / 2**20:.0f}MiB、{time.time() - t1:.0f}s）", flush=True)
    del m, G
    return {"mean_rewritten": c_mean, "mean_total": n_mean0, "tanh_rewritten": c_tanh, "tanh_total": n_tanh0,
            "sec": round(time.time() - t0, 1), "equiv": equiv_check()}


def equiv_check() -> dict:
    """元と書き直した版を PC の CPU で同じ入力（syn_nav_S の最初の 2 回ぶん: system 文の 16 行、文字 8＋画像 8 行）に通して比べる"""
    # 入力は diag_t16・probe_t16 の道具（端末と同じ入れ方を Python で作るもの）で作る。2 回目の past は元のグラフの 1 回目の出力にそろえる
    # （書き直した版の出力を使うと、2 回目の差に 1 回目の差が混ざるため）。戻り値: 出力ごとの相対 L2 の最大と上位 5 本
    import gc
    from app.pipeline.diag_t16 import IMG, TABLES, Tables, load_items, old_prompt
    from app.pipeline.probe_t16 import Cpu, Feeder
    tabs = Tables(TABLES)
    it = load_items()["syn_nav_S"]
    ids = old_prompt(it["tail"])
    feats = np.fromfile(it["feat"], np.float32).reshape(-1, tabs.H)
    rows, k = [], 0
    for tid in ids[:32]:
        if tid == IMG:
            rows.append((tid, feats[k]))
            k += 1
        else:
            rows.append((tid, None))
    outs = {}
    feeds = None
    for tag, path in (("orig", SRC), ("fix", DST)):
        cpu = Cpu(path)
        if feeds is None:                                   # 入力は元のグラフで作る（2 回目の past は元の 1 回目の出力）
            fd = Feeder(cpu.ins, tabs)
            fd.reset()
            f0 = fd.feed(rows[:16])
            o0 = cpu.run(f0)
            fd.advance(o0, 16)
            f1 = fd.feed(rows[16:32])
            feeds = [f0, f1]
            outs[tag] = [o0, cpu.run(f1)]
        else:
            outs[tag] = [cpu.run(f) for f in feeds]
        del cpu
        gc.collect()
    # 出力ごと（logits と present の各本）に、2 回の中で大きい方の相対 L2 を取る
    worst = {}
    for b in range(2):
        for name, ref in outs["orig"][b].items():
            a = np.asarray(outs["fix"][b][name], np.float64)
            r = np.asarray(ref, np.float64)
            v = float(np.linalg.norm(a - r) / max(np.linalg.norm(r), 1e-12))
            worst[name] = max(worst.get(name, 0.0), v)
    top = sorted(worst.items(), key=lambda kv: -kv[1])[:5]
    print(f"  PC の CPU で元と書き直した版の差（最大）: logits {worst.get('logits', 0) * 100:.4f}%、"
          f"いちばん大きい出力 {top[0][0]} {top[0][1] * 100:.4f}%", flush=True)
    return {"logits_rel_max": worst.get("logits"), "top5": [[k, round(v, 6)] for k, v in top]}


def convert() -> dict:
    """書き直した版を QAIRT で変換し、検査・包む（build_t16.convert と同じ手順）。入出力が元と同じか・RmsNorm の数・DDR の見積もりも記録する。"""
    from app.pipeline import verify_artifacts, wrap_epctx
    from app.pipeline.build_vit_ctx import run_streaming
    if not DST.exists():
        raise SystemExit(f"{DST} が無い。先に --steps rewrite")
    if not Path(GIT_BASH).exists():
        raise SystemExit(f"Git Bash が無い: {GIT_BASH}")
    env = {**os.environ, "QAIRT_INT4_PATCH": "1"}       # 文章グラフの重みは 4bit（build_t16 と同じ）
    t0 = time.time()
    print("  build_context.sh（S5〜S7）…　4 時間前後かかる（元の 16 トークン版は 3.9 時間）", flush=True)
    rc = run_streaming([GIT_BASH, "scripts/build_context.sh", str(DST), str(DIR / "build"), NAME],
                       C.WORK / "logs" / f"t16{VARIANT}_convert.log", env)
    bins = sorted((DIR / "build").glob("ctx_*/*.bin"))
    out = {"sec": round(time.time() - t0, 1), "rc": rc}
    if rc != 0 or not bins:
        raise SystemExit(f"変換に失敗（終了コード {rc}）。{DIR / 'build'} の *_convert.log / *_gen.log を見ること")
    meta = DIR / "build" / f"{NAME}_meta.json"
    cmdlog = DIR / "build" / f"{NAME}_commands.txt"
    cmds = cmdlog.read_text(encoding="utf-8", errors="replace").splitlines() if cmdlog.exists() else []
    ver = verify_artifacts.verify_all(DST, meta, cmds)
    out["verify"] = [{"check": x["check"], "pass": bool(x.get("pass")), "detail": str(x.get("detail", ""))} for x in ver["checks"]]
    for x in out["verify"]:
        print(f"  {'合格' if x['pass'] else '★不合格'}  {x['check']}: {x['detail']}", flush=True)
    info = wrap_epctx.run(bins[0], DST, EPCTX, NAME, embed=False)
    # 入出力が元と同じか（端末の C++ がそのまま使えるか）
    src_io = json.loads(SRC_IO.read_text(encoding="utf-8"))
    got_in = [[i["name"], i["shape"]] for i in info["inputs"]]
    got_out = [[o["name"], o["shape"]] for o in info["outputs"]]
    same_io = got_in == [list(x) for x in src_io["inputs"]] and got_out == [list(x) for x in src_io["outputs"]]
    # 変換後のグラフ（DLC）の演算・DDR の見積もり
    # DLC の中身をバイト列のまま数える（演算の名前の文字列の出てくる回数）。正確な数ではなく、RmsNorm が残ったかの目安
    dlc = DIR / "build" / f"{NAME}.dlc"
    b = dlc.read_bytes() if dlc.exists() else b""
    ops = {k: b.count(k.encode()) for k in ("RmsNorm", "Tanh", "FullyConnected", "ReduceMean")}
    del b
    gen = DIR / "build" / f"{NAME}_gen.log"
    # 変換のログの「spill_bytes=…」などの行から、DDR へ書き出して読み戻す量の見積もりを読む
    ddr = {}
    if gen.exists():
        for k in ("spill_bytes", "fill_bytes", "write_total_bytes", "read_total_bytes"):
            mm = re.search(rf"{k}=(\d+)", gen.read_text(encoding="utf-8", errors="replace"))
            if mm:
                ddr[k] = int(mm.group(1))
    out.update({"ctx_bin": str(bins[0]), "ctx_mib": round(bins[0].stat().st_size / 2**20, 1), "epctx": str(EPCTX),
                "same_io_as_t16": same_io, "dlc_ops": ops, "ddr": ddr, "all_pass": bool(ver["all_pass"])})
    print(f"  context binary {out['ctx_mib']}MiB（元 1,467MiB、NPU の上限 約 1,710MiB）／入出力 {'元と同じ' if same_io else '★元と違う'}"
          f"／変換後の RmsNorm {ops.get('RmsNorm')} 個（{'0 のはず' if 'normmm' in FIXES else 'この版では元と同じだけ残る'}）"
          f"／{out['sec'] / 3600:.1f} 時間", flush=True)
    if ddr:
        print(f"  DDR の見積もり: 読み {ddr.get('read_total_bytes', 0) / 1e9:.3f}GB（元 1.549GB）・書き {ddr.get('write_total_bytes', 0) / 1e6:.1f}MB"
              f"（元 5.0MB）", flush=True)
    return out


# --steps の段を流し、results/P5_t16/build_t16fix.json（t の版は build_t16fix_t.json）に書く。済んだ段は飛ばす。
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="rewrite,convert")
    ap.add_argument("--redo", action="store_true")
    ap.add_argument("--variant", default="f", choices=list(VARIANTS),
                    help="f = 正規化の平均＋PLE の tanh を書き直す（g4t16f_W256）、t = PLE の tanh だけ（g4t16t_W256）")
    a = ap.parse_args()
    C.utf8_stdout()
    use(a.variant)
    steps = a.steps.split(",")
    RES.mkdir(parents=True, exist_ok=True)
    out_path = RES / ("build_t16fix.json" if a.variant == "f" else f"build_t16fix_{a.variant}.json")
    rec = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    t0 = time.time()
    if "rewrite" in steps:
        if DST.exists() and DST.stat().st_mtime > SRC.stat().st_mtime and not a.redo and rec.get("rewrite"):
            print(f"  書き直しは済んでいる（{DST.name}）", flush=True)
        else:
            print("\n[書き直し]", flush=True)
            rec["rewrite"] = rewrite()
            C.write_json_atomic(out_path, rec)
    if "convert" in steps:
        if EPCTX.exists() and EPCTX.stat().st_mtime > DST.stat().st_mtime and not a.redo:
            print(f"  変換は済んでいる（{EPCTX.name}）", flush=True)
        else:
            print("\n[変換]", flush=True)
            rec["convert"] = convert()
            C.write_json_atomic(out_path, rec)
    print(f"\nおわり（{(time.time() - t0) / 60:.0f} 分）-> {out_path}")


if __name__ == "__main__":
    main()
