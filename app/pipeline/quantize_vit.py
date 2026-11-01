# -*- coding: utf-8 -*-
"""Phase 2-2: 書き出した画像エンコーダの ONNX を量子化し、v68 に渡せる形（final_qdq）にする（2026-09-28）。

    python -m app.pipeline.quantize_vit                      # 70 トークン、SRQ あり・なしの 2 版
    python -m app.pipeline.quantize_vit --variants nosrq     # 片方だけ

**EdgeSLM_Android の venv で動かす**（export_vit と同じ）。先に `export_vit export` を済ませておくこと。

## 流れ（EdgeSLM_AndroidGemma4 の bench_prefill.build と同じ順。道具はそこから複製したもの）

| 段 | 道具 | 出力（work/ops/vit_{tag}/） |
|---|---|---|
| S2〜S4 前処理・較正・量子化 | `p2b_ops.prepare`（scheme `a16w8_rhs8`） | `vit_{tag}_prepped.onnx`、`vit_{tag}_a16w8_rhs8_qdq_ort.onnx` |
|   〃 の中で C-13 | `fix_norm_gamma`（RMSNorm の gamma を int16 の対称に） | （同上に含む） |
| QAT の値と SRQ | `qat_inject_vit`（SRQ あり／なし） | `vit_{tag}_{srq,nosrq}_qdq.onnx` |
| C-8 | `downgrade_actact_rhs`（活性×活性の行列積の右辺を 8bit に） | `vit_{tag}_{srq,nosrq}_final_qdq.onnx` |
| 検査 | `verify_artifacts`（C-1 float の区間・C-5 BatchNorm） | 結果の JSON |

- 活性は 16bit（A16、テンソル全体で 1 つの目盛り、MinMax）。較正は評価用と別の 64 枚（export_vit が用意した calib npz）
- 重みは INT8・出力チャネルごと（QAT の 112 本は整数とスケールをそのまま、パッチの埋め込み・射影層は作り直し）
- 結果は `results/P2_vit_quant/{tag}_quantize.json`
- 2026-09-28 の実行の記録: `results/P2_20260928_vit_quant.md`（249 秒。2 版とも C-1・C-5 合格、C-8 の右辺 32/32、大きさ 149.9MiB）
- 入力: `work/ops/vit_{tag}/vit_{tag}.onnx`（export_vit が書いた float の ONNX）・同じ名前の `.spec.json`・`_calib.npz`
- 出力: 上の表の `*_final_qdq.onnx`（これを build_vit_ctx が v68 の context binary にする）
- StudyMemo 7.9 節（量子化）・8.4〜8.5 節（C-14 の見張り）
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from app.pipeline import vlm_common as C

RES = C.RESULTS / "P2_vit_quant"
SCHEME = "a16w8_rhs8"


# 予算（画像トークン数）と向きから、ファイル名に使う札（例: "b70_21x30"）を作る。
# 格子（パッチの縦×横）は vlm_common の表から引く。縦長の写真用は PORTRAIT_GRIDS（後から足した表）。
def tag_for(budget: int, portrait: bool) -> str:
    grid = (C.PORTRAIT_GRIDS if portrait else C.FIXED_GRIDS)[budget]
    return f"b{budget}_{grid[0]}x{grid[1]}"


# 1 つの札について、量子化の各段の入出力のパスをまとめた辞書を返す。
# "qdq" と "final" は版（"srq" / "nosrq"）を引数に取る関数。2 版で同じ前処理の結果を使い回すため、
# 前処理・較正の出力（prepped・qdq_ort）は版に依らない名前にしてある。
def paths(tag: str) -> dict:
    name = f"vit_{tag}"
    d = C.OPS / name
    return {"name": name, "dir": d, "src": d / f"{name}.onnx", "prepped": d / f"{name}_prepped.onnx",
            "qdq_raw": d / f"{name}_{SCHEME}_qdq.onnx", "qdq_ort": d / f"{name}_{SCHEME}_qdq_ort.onnx",
            "qdq": lambda v: d / f"{name}_{v}_qdq.onnx", "final": lambda v: d / f"{name}_{v}_final_qdq.onnx"}


def main():
    """量子化の 4 段（前処理・較正・量子化 → QAT の値 → C-8 → 検査）を版ごとに流し、結果を JSON に書く。

    引数: --budget（画像トークン数。既定 70）・--portrait（縦長の格子）・--variants（"srq,nosrq"）・--redo-prepare。
    戻り値は無く、`results/P2_vit_quant/{tag}_quantize.json` に段ごとの結果を書く。
    前処理・較正（[1/4]）は時間がかかる（2026-09-28 の実行で全体 249 秒）ので、済んでいて入力より新しければ飛ばす。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=70, choices=sorted(C.FIXED_GRIDS))
    ap.add_argument("--portrait", action="store_true")
    ap.add_argument("--variants", default="srq,nosrq", help="srq（QAT の 8bit の丸めあり）・nosrq（なし）")
    ap.add_argument("--redo-prepare", action="store_true", help="較正・量子化（時間がかかる段）からやり直す")
    a = ap.parse_args()
    C.utf8_stdout()
    # 量子化の道具は EdgeSLM_Android の venv（onnx・onnxruntime）にしか無いので、関数の中で読み込む
    from app.pipeline import downgrade_actact_rhs, p2b_ops, verify_artifacts
    from app.pipeline import qat_inject_vit as Q

    # ① 入力の float の ONNX があるかを確かめる（export_vit export の出力）
    tag = tag_for(a.budget, a.portrait)
    P = paths(tag)
    if not P["src"].exists():
        raise SystemExit(f"{P['src']} が無い。先に export_vit export を動かすこと")
    out = {"tag": tag, "scheme": SCHEME, "variants": {}}
    t0 = time.time()

    # ---- S2〜S4: 前処理・較正・量子化（ORT）。済んでいて、元の ONNX・spec・較正データより新しければ飛ばす ----
    inputs_ = [P["src"], P["dir"] / f"{P['name']}.spec.json", P["dir"] / f"{P['name']}_calib.npz"]
    stale = P["qdq_ort"].exists() and any(p.exists() and p.stat().st_mtime > P["qdq_ort"].stat().st_mtime
                                          for p in inputs_)
    if a.redo_prepare or stale or not P["qdq_ort"].exists():
        if stale:
            print("  （元の ONNX・spec・較正データが量子化の結果より新しいので、やり直す）", flush=True)
        for old in (P["prepped"], P["qdq_raw"], P["qdq_ort"]):          # 古い生成物を使い回さない
            old.unlink(missing_ok=True)
        print(f"[1/4] 前処理・較正・量子化（{SCHEME}）…", flush=True)
        st = p2b_ops.prepare(C.OPS, P["name"], SCHEME, n_random=16)     # n_random は較正データが無いときだけ使われる
        ok = st.get("verdict") == "ready_for_convert"
        if ok:
            P["qdq_raw"].replace(P["qdq_ort"])                           # JSON を書く前に（書けなくても結果は残す）
        C.write_json_atomic(P["dir"] / f"status_prepare_{SCHEME}.json", st)
        out["prepare"] = {"verdict": st.get("verdict"), "calibration": st.get("calibration"),
                          "per_channel_weights": st.get("per_channel_weights"), "weight_bits": st.get("weight_bits")}
        print(f"  verdict={st.get('verdict')}（{time.time() - t0:.0f}s）", flush=True)
        if not ok:
            RES.mkdir(parents=True, exist_ok=True)
            C.write_json_atomic(RES / f"{tag}_quantize.json", out)
            raise SystemExit(f"prepare が通らなかった。{P['dir'] / f'status_prepare_{SCHEME}.json'} を見ること")
    else:
        print(f"[1/4] 前処理・較正・量子化は済んでいる（{P['qdq_ort'].name}）。やり直すなら --redo-prepare", flush=True)

    # ② 版ごとに: QAT の値を書く → C-8 → 検査。lin は QAT の線形層の一覧（どの重みに QAT の整数を書くか）
    lin = Q.vit_linears()
    for v in a.variants.split(","):
        srq = {"srq": True, "nosrq": False}[v]
        print(f"\n== 版: {v}（SRQ {'あり' if srq else 'なし'}）", flush=True)
        print("[2/4] QAT の値を書き込む", flush=True)
        inj = Q.inject(P["qdq_ort"], P["prepped"], lin, P["qdq"](v), srq=srq)
        print("[3/4] 活性×活性の行列積の右辺を 8bit に（C-8）", flush=True)
        dg = downgrade_actact_rhs.run(P["qdq"](v), P["prepped"], P["final"](v))
        # 想定の右辺の数: 1 層の注意に活性×活性の行列積が 2 つ（Q×Kᵀ と 確率×V）あり、16 層で 32
        exp_rhs = C.VIT_LAYERS * 2
        print(f"  右辺の数 {dg['n_rhs']}（想定 {exp_rhs}）・8bit に落とした {len(dg['done'])}・"
              f"対象外（既に 8bit など） {len(dg['skipped'])}", flush=True)
        if dg["n_rhs"] != exp_rhs:
            print(f"  ★活性×活性の行列積の右辺の数が想定と違う（{dg['n_rhs']} / {exp_rhs}）。prepped の形を確かめること", flush=True)
        print("[4/4] 検査（C-1・C-5・C-14）", flush=True)
        c1 = verify_artifacts.check_c1_no_float_segments(P["final"](v))
        c5 = verify_artifacts.check_c5_no_batchnorm(P["final"](v))
        # C-14（2026-09-29 に端末で発見）: 重みのスケールが同じ行列の中で極端に小さいチャネルは、端末の NPU で値が壊れる
        c14 = Q.scale_ratio_report(P["final"](v))
        print(f"  C-14: {'合格' if c14['ok'] else '★注意'}  重みのスケールの最小/最大 {c14['min_ratio']:.4f}"
              f"（{c14['min_ratio_tensor']}、下限 {Q.SCALE_FLOOR_RATIO:.4f}）", flush=True)
        print(f"  C-1: {'合格' if c1.ok else '★不合格'}  {c1.get('detail', '')}", flush=True)
        print(f"  C-5: {'合格' if c5.ok else '★不合格'}", flush=True)
        # ③ 版ごとの結果をまとめる。大きさは 2026-09-28 の実行で 149.9MiB（float の 587MiB の約 1/4。results/P2_20260928_vit_quant.md）
        size = P["final"](v).stat().st_size
        out["variants"][v] = {"final_qdq": str(P["final"](v)), "size_mib": round(size / 2**20, 1),
                              "inject": {k: inj[k] for k in ("written", "retyped_int8", "srq_inputs", "srq_outputs",
                                                             "approx_from_float", "remaining_int4", "scale_floored")},
                              "downgrade": {k: dg[k] for k in dg if k != "done"} | {"n_done": len(dg.get("done", []))},
                              "C1": {"ok": bool(c1.ok), "detail": str(c1.get("detail", ""))}, "C5": {"ok": bool(c5.ok)},
                              "C14": c14}
        print(f"-> {P['final'](v)}（{size / 2**20:.0f}MiB）", flush=True)
    out["sec"] = round(time.time() - t0, 1)
    RES.mkdir(parents=True, exist_ok=True)
    C.write_json_atomic(RES / f"{tag}_quantize.json", out)
    print(f"\n-> {RES / (tag + '_quantize.json')}（{out['sec']:.0f}s）")


if __name__ == "__main__":
    main()
