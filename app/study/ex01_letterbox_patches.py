# -*- coding: utf-8 -*-
"""演習 1: 写真が 630 パッチ（70 トークン）になるまで — レターボックスとパッチの並び（StudyMemo 2 章・3.1 節・11.3 節）。

    python -B -m app.study.ex01_letterbox_patches       # numpy と PIL だけ。数秒（評価セットの写真を 1 枚読む）

## 何を確かめるか

1. 固定の格子（`vlm_common.FIXED_GRIDS`）: 70 トークン → 21×30 パッチ ＝ 336×480 ピクセル。縦×横 ＝ トークン数×9
2. レターボックス（`hf_vlm.prepare_image` の fixed）: 縦横比を保って縮め、余白を黒で埋める。**縦長の写真を横長の格子に入れると、
   使える画素が約半分になる**（PC の評価で 280 トークンの品質が落ちた。`vlm_common.py` の `PORTRAIT_GRIDS` の注記）
3. パッチの並び（`hf_vlm.pixels_fixed_np`）: (縦, py, 横, px, c) → (縦, 横, py, px, c) → [N, 768]。パッチ n ＝ y·横の数 ＋ x
4. 3×3 のプーリングで 630 → 70（出力 m ＝ ky·(横/3) ＋ kx）

`prepare_image`・`pixels_fixed_np`・`FIXED_GRIDS` は本物のソースから取り出して使う（`app/study/_src.py`）。
`pixels_fixed_np` は HF の `processor_config.json`（EdgeSLM_AndroidGemma4 が取得したもの、読むだけ）を読む。
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from app.study._src import ROOT, load


def main():
    import app.pipeline.vlm_common as C                      # 置き場所と定数だけ（import しても何も起きない）
    f = load("app/pipeline/hf_vlm.py", ["prepare_image", "pixels_fixed_np"], {"C": C, "Image": Image})
    prepare_image, pixels_fixed_np = f["prepare_image"], f["pixels_fixed_np"]

    print("== 1. 固定の格子（vlm_common.FIXED_GRIDS）")
    for k, (gh, gw) in C.FIXED_GRIDS.items():
        print(f"  {k:>3} トークン: {gh}×{gw} パッチ = {gh * C.PATCH}×{gw * C.PATCH} ピクセル、{gh * gw} パッチ = {k} × 9")

    print("\n== 2. レターボックス（prepare_image の fixed、70 トークン）")
    for name, size in (("横長の写真 4032×3024", (4032, 3024)), ("縦長の写真 3024×4032", (3024, 4032))):
        img = Image.new("RGB", size, (200, 120, 60))
        out = prepare_image(img, "fixed", 70)
        a = np.asarray(out)
        used = float((a.sum(-1) > 0).mean())
        print(f"  {name} → {out.size[0]}×{out.size[1]}、写真が占める割合 {used:.0%}（残りは黒の余白）")
        if size[1] > size[0]:
            pg = C.PORTRAIT_GRIDS[70]
            out2 = prepare_image(img, "fixed", 70, grid=pg)
            used2 = float((np.asarray(out2).sum(-1) > 0).mean())
            print(f"  （同じ縦長の写真を縦長の格子 PORTRAIT_GRIDS[70] = {pg} に入れると {out2.size[0]}×{out2.size[1]}、占める割合 {used2:.0%}）")
    print("  → 縦長の写真を横長の格子に入れると、写真は 252×336 に縮み、使える画素が約半分になる")

    print("\n== 3. パッチの並び（pixels_fixed_np）")
    item = next(p for p in sorted((C.EVAL_DIR / "images" / "car").glob("*.jpg")))
    img = prepare_image(Image.open(item), "fixed", 70)
    pv = pixels_fixed_np(img, C.FIXED_GRIDS[70])
    x = np.asarray(img, dtype=np.float32) / 255.0
    gh, gw = C.FIXED_GRIDS[70]
    y0, x0 = 7, 12                                         # パッチ (縦 7, 横 12) を手で切り出して比べる
    manual = x[y0 * 16:(y0 + 1) * 16, x0 * 16:(x0 + 1) * 16, :].reshape(-1)
    n = y0 * gw + x0
    print(f"  写真 {item.relative_to(ROOT)} → pixel_values {pv.shape}（{gh}×{gw} パッチ × 768 = 16×16×3）")
    print(f"  パッチ (縦 {y0}, 横 {x0}) は n = {y0}·{gw} + {x0} = {n} 番目。手で切り出した 768 個との最大差 {np.abs(pv[n] - manual).max():.1e}")
    print("  （端末の C++ も同じ並びで作る: android_app/.../cpp/runner/vlm_prep.cpp の vlm_pixels）")

    print("\n== 4. 3×3 のプーリングで 630 → 70")
    feat = np.random.default_rng(0).normal(size=(gh * gw, 8))
    t = feat.reshape(gh, gw // 3, 3, 8).mean(axis=2)                 # vit_graph.VitGraph.pool と同じ 2 段の平均
    t = t.reshape(gh // 3, 3, gw // 3, 8).mean(axis=1).reshape(-1, 8)
    ky, kx = 2, 4
    m = ky * (gw // 3) + kx
    block = [(ky * 3 + dy) * gw + (kx * 3 + dx) for dy in range(3) for dx in range(3)]
    print(f"  プーリングの後 {t.shape[0]} 個。出力 m = {ky}·{gw // 3} + {kx} = {m} は、パッチ {block} の平均")
    print(f"  手で 9 個を平均したものとの最大差 {np.abs(t[m] - feat[block].mean(0)).max():.1e}")


if __name__ == "__main__":
    main()
