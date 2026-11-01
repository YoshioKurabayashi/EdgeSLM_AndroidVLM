# -*- coding: utf-8 -*-
"""演習 3: 端末の文章グラフの「入口の目盛り」を ONNX から手で読む（StudyMemo 4.2 節・6.5 節）。

    python -B -m app.study.ex03_entry_range        # numpy だけ。10〜30 秒（186MB の ONNX を読む）

## 何を確かめるか

端末の文章グラフは、入口の `inputs_embeds` を uint16 に量子化してから計算する。目盛りは**文章だけで較正**して決まっているので、
表せる範囲は `lo = (0 − zero_point) × scale`、`hi = (65535 − zero_point) × scale` に限られる。画像の特徴がこの範囲を超えると切り詰められる。

1. 本物の道具 `app/pipeline/onnx_scales.py` の `read_quant_points`（onnx パッケージを使わず protobuf を直接たどる）を、
   **ソースから取り出してそのまま**動かし、EdgeSLM_AndroidGemma4 の量子化済みグラフ（読むだけ）から入口の 4 点を読む
2. lo・hi を scale と zero point から自分で計算し直す
3. Phase 1 の実測（`results/P1_check_wiring/wiring_b70_srq-off.json`）: 画像の特徴がこの範囲の外に出た割合と、配線の一致（A↔B・B↔C）
"""
from __future__ import annotations

import json
import struct

from app.study._src import ROOT, load


def main():
    import app.pipeline.vlm_common as C
    f = load("app/pipeline/onnx_scales.py", ["DTYPES", "ENTRY_NAMES", "_varint", "_fields", "read_quant_points"],
             {"struct": struct, "C": C})
    path = C.DEVICE_CHUNK_A_QDQ
    print("== 1. 量子化済みグラフから入口の目盛りを読む（onnx_scales.read_quant_points）")
    if not path.exists():
        print(f"  {path} が無い（EdgeSLM_AndroidGemma4 の生成物）。3. だけ見る")
        got = None
    else:
        print(f"  {path.relative_to(ROOT.parent)}（{path.stat().st_size / 2**20:.0f}MiB）を読む…", flush=True)
        got = f["read_quant_points"](path, f["ENTRY_NAMES"])
        for k, v in got.items():
            print(f"  {k:42s} scale={v['scale']:.6g} zp={v['zero_point']:>5}  範囲 [{v['lo']}, {v['hi']}]")

    print("\n== 2. 範囲を自分で計算し直す（uint16）")
    rec = json.loads((ROOT / "results/P1_check_wiring/wiring_b70_srq-off.json").read_text(encoding="utf-8"))
    e = (got or rec["entry_ranges"])["inputs_embeds_QuantizeLinear"]
    lo, hi = (0 - e["zero_point"]) * e["scale"], (65535 - e["zero_point"]) * e["scale"]
    print(f"  lo = (0 − {e['zero_point']}) × {e['scale']:.6g} = {lo:.5f}")
    print(f"  hi = (65535 − {e['zero_point']}) × {e['scale']:.6g} = {hi:.5f}")
    print("  → 4.2 節の [−7.03, 3.52]。0 の位置が範囲の 2/3 のところにある（較正に使った文章の埋め込みが負の側に広かったため。実測でも文章の埋め込みは −2.60〜1.30）")

    print("\n== 3. Phase 1 の実測（SRQ なし、70 トークン、4 問）")
    print("  問題           | A↔B 相対L2 | B↔C 相対L2 最大 | 画像の特徴が inputs_embeds の範囲外 | 切り詰めても答えが同じ")
    for r in rec["runs"]:
        out = (r.get("values") or {}).get("outside_rate", {}).get("inputs_embeds_image")
        same = (r.get("clip_effect") or {}).get("answer_same")
        print(f"  {r['id']:14s} | {r['A_vs_B']['rel_l2_max']:10.1e} | {r['B_vs_C']['rel_l2_max']:14.1e} | "
              f"{out if out is not None else '-':>33} | {same}")
    print("  → A↔B が 0（ビット一致）＝ 端末と同じ入れ方（画像の位置に特徴、PLE は pad の行）は HF の本来の経路と同じ（6.5 節）")
    print("    B↔C（一度に ↔ 1 トークンずつ）は足し算の順序の差だけ。画像の特徴はほぼ範囲に収まる")


if __name__ == "__main__":
    main()
