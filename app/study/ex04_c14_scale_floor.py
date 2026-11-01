# -*- coding: utf-8 -*-
"""演習 4: C-14 ― ほぼ 0 の重みのチャネルの「小さすぎるスケール」と、その下限（StudyMemo 8 章）。

    python -B -m app.study.ex04_c14_scale_floor      # numpy だけ。1 秒

## 何が起きたか（8.1・8.4 節）

画像エンコーダのパッチの埋め込み（768→768）には、重みがほぼ 0 のチャネルが 222 本ある。重みを INT8・出力チャネルごとに量子化するとき、
`max|w| / 127` をそのままスケールにすると 4e-6 未満になり、**そのチャネルだけ端末の NPU で値が壊れた**（範囲の端に張り付き、符号もでたらめ）。
PC の QDQ（ORT）では壊れない。NPU は「入力のスケール × 重みのスケール ÷ 出力のスケール」を固定小数点の倍率で持つので、小さすぎると表せないとみられる。

## 何を確かめるか

1. 本物の `_from_float`（`app/pipeline/qat_inject_vit.py`）を**ソースから取り出して**、ほぼ 0 のチャネルを含む重みに使う
2. 下限 `SCALE_FLOOR_RATIO`（最大のチャネルの 1/64）で、何本のスケールが持ち上がるか・値がどれだけ変わるか
3. 見張り（`scale_ratio_report` と同じ「最小 ÷ 最大」）の考え方
"""
from __future__ import annotations

import numpy as np

from app.study._src import load


def main():
    f = load("app/pipeline/qat_inject_vit.py", ["SCALE_FLOOR_RATIO", "_from_float"])
    SCALE_FLOOR_RATIO, _from_float = f["SCALE_FLOOR_RATIO"], f["_from_float"]
    rng = np.random.default_rng(0)
    out_ch, in_ch = 768, 768
    w = rng.normal(0, 0.01, (out_ch, in_ch)).astype(np.float32)
    dead = rng.choice(out_ch, 222, replace=False)                      # ほぼ 0 のチャネル（本物は 222 本、max|w| 2e-8〜5e-4）
    w[dead] = rng.normal(0, 1, (222, in_ch)) * (10 ** rng.uniform(-8, -3.5, 222))[:, None]

    raw = np.maximum(np.abs(w).max(axis=1), 1e-12) / 127.0
    print("== 1. 下限なしのスケール（max|w| / 127）")
    print(f"  最大 {raw.max():.2e}、最小 {raw.min():.2e}（最小 ÷ 最大 = 1/{raw.max() / raw.min():,.0f}）")
    print(f"  端末で壊れた境目 s ≈ 4.0e-6（最大の 1/659）より小さいチャネル: {int((raw < 4.0e-6).sum())} 本")

    d = _from_float(w)
    s = d["scale"]
    print(f"\n== 2. 本物の _from_float（下限 = 最大 × SCALE_FLOOR_RATIO = 最大 × 1/{1 / SCALE_FLOOR_RATIO:.0f}）")
    print(f"  下限に持ち上げたチャネル: {d['n_floored']} 本。最小 ÷ 最大 = {s.min() / s.max():.4f}（= 1/64、境目 1/659 に対して約 10 倍の余裕）")
    deq = d["codes"].astype(np.float32) * s[:, None]
    fl = np.where(raw < s.max() * SCALE_FLOOR_RATIO)[0]                # 下限に持ち上げたチャネル
    keep = np.setdiff1d(np.arange(out_ch), fl)
    mc = np.abs(d["codes"][fl]).max(axis=1)
    print(f"  持ち上げたチャネルの整数の最大値: 中央値 {int(np.median(mc))}、最大 {int(mc.max())}（下限に近いチャネルだけ 127 近くまで使い、"
          f"ほとんど 0 のチャネルは 0〜数になる）")
    print(f"  持ち上げたチャネルの、戻した値のずれ最大 {np.abs(deq[fl] - w[fl]).max():.1e}（重みそのものが小さいので、絶対値ではわずか）")
    print(f"  ほかのチャネルは変わらない（スケールは max|w|/127 のまま。ずれ最大 {np.abs(deq[keep] - w[keep]).max():.1e} ＝ 丸めの半目盛り）")
    print("  → 小さい重みが少し粗く表されるだけで、出力への影響はほとんど無い。代わりに NPU の倍率が表せる範囲に入る")

    print("\n== 3. 見張り（scale_ratio_report）")
    print("  同じ行列の中では入力・出力のスケールは 1 つなので、重みのスケールの「最小 ÷ 最大」が極端に小さいものを探せば足りる。")
    print(f"  合格の条件: 最小 ÷ 最大 ≥ SCALE_FLOOR_RATIO（{SCALE_FLOOR_RATIO:.4f}）。quantize_vit.py の [4/4] の検査で毎回出す")
    print("  ★PC の QDQ は丸めは真似るが、NPU の計算の限界は真似ない。途中の値を出力に足した版（probe_vit.py）で、"
          "同じグラフを PC と端末で比べて見つけた（8.2 節）")


if __name__ == "__main__":
    main()
