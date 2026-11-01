# -*- coding: utf-8 -*-
"""演習 2: 画像エンコーダの書き直し（NPU 向けの「数学的に同じ形」）を numpy で確かめる（StudyMemo 7.2〜7.4 節）。

    python -B -m app.study.ex02_vit_rewrites       # numpy だけ。1 秒

## 何を確かめるか（`app/pipeline/vit_graph.py` の表「NPU 向けに変えたところ」）

1. **×√768 を消して eps に吸収**: `RMSNorm(c·x; ε) = RMSNorm(x; ε/c²)`（プーリングの後の ×√768 を掛けず、射影層の RMSNorm の eps を eps/768 にする）
2. **プーリング**: HF は one_hot の行列（630×70）× h で平均する。ここでは reshape → 平均を 2 回。同じ値で、定数行列を持たない
3. **2 次元 RoPE**: head_dim 64 の前半 32 を x、後半 32 を y に使う。回転した q・k の内積は**パッチの相対位置 (dx, dy) だけ**で決まる
   （位置が固定の格子なので、cos/sin の表は定数にできる）

`VIT_EPS`・`VIT_ROPE_THETA`・`VISION_HIDDEN` は本物の値（`app/pipeline/vlm_common.py`）。
"""
from __future__ import annotations

import numpy as np


def rmsnorm(x, eps):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps)


def rope2d_tables(pos, head_dim=64, theta=100.0):
    """vit_graph.rope2d_tables と同じ式を numpy で（前半 32 = x、後半 32 = y。各半分は cat(freqs, freqs)）"""
    spatial = head_dim // 2
    inv = 1.0 / (theta ** (np.arange(0, spatial, 2) / spatial))      # [16]
    cs, ss = [], []
    for i in range(2):
        fr = pos[:, i:i + 1] * inv[None, :]                          # [N,16]
        emb = np.concatenate([fr, fr], -1)                           # [N,32]
        cs.append(np.cos(emb))
        ss.append(np.sin(emb))
    return np.concatenate(cs, -1), np.concatenate(ss, -1)            # [N,64]


def rotate_half_2d(x):
    """vit_graph.rotate_half_2d と同じ（32 ずつ 2 つに分けて、それぞれの中で rotate_half）"""
    q = x.shape[-1] // 4
    return np.concatenate([-x[..., q:2 * q], x[..., :q], -x[..., 3 * q:], x[..., 2 * q:3 * q]], -1)


def main():
    import app.pipeline.vlm_common as C
    rng = np.random.default_rng(0)

    print("== 1. ×√768 を eps に吸収する（7.4 節）")
    x = rng.normal(0, 0.01, (70, C.VISION_HIDDEN))                  # プーリングの後の値（小さい値ほど eps が効く）
    c = C.VISION_HIDDEN ** 0.5
    a = rmsnorm(c * x, C.VIT_EPS)                                    # HF: ×√768 してから RMSNorm(eps)
    b = rmsnorm(x, C.VIT_EPS / C.VISION_HIDDEN)                      # ここ: 掛けずに RMSNorm(eps/768)
    print(f"  c = √{C.VISION_HIDDEN} = {c:.3f}、eps = {C.VIT_EPS:g} → eps/c² = {C.VIT_EPS / C.VISION_HIDDEN:.3g}")
    print(f"  最大差 {np.abs(a - b).max():.1e}（式の上では完全に同じ: c·x / sqrt(c²·mean(x²) + ε) = x / sqrt(mean(x²) + ε/c²)）")
    print("  → 掛け算 1 つと、その出力の量子化（Q/DQ）1 組が減る")

    print("\n== 2. プーリング: one_hot の行列 ↔ reshape と平均（7.3 節）")
    gh, gw = C.FIXED_GRIDS[70]
    h = rng.normal(size=(gh * gw, 16))
    onehot = np.zeros((gh * gw, (gh // 3) * (gw // 3)))
    for y in range(gh):
        for x_ in range(gw):
            onehot[y * gw + x_, (y // 3) * (gw // 3) + x_ // 3] = 1.0 / 9
    hf = onehot.T @ h                                                 # HF の形（定数行列 630×70 を持つ）
    t = h.reshape(gh, gw // 3, 3, -1).mean(2).reshape(gh // 3, 3, gw // 3, -1).mean(1).reshape(-1, h.shape[1])
    print(f"  定数行列 {onehot.shape}（{onehot.size:,} 個）を持つ形 と reshape→平均×2 の差 {np.abs(hf - t).max():.1e}")

    print("\n== 3. 2 次元 RoPE: 内積は相対位置だけで決まる（7.2 節）")
    q, k = rng.normal(size=64), rng.normal(size=64)
    def rot(v, p):
        cs, sn = rope2d_tables(np.array([p], dtype=float), theta=C.VIT_ROPE_THETA)
        return v * cs[0] + rotate_half_2d(v) * sn[0]
    pairs = [((3, 5), (7, 6)), ((10, 15), (14, 16)), ((0, 0), (4, 1))]           # どれも (dx, dy) = (4, 1)
    for pq, pk in pairs:
        print(f"  q の位置 {pq}、k の位置 {pk}（相対 (4, 1)）: 内積 {rot(q, pq) @ rot(k, pk):.6f}")
    print(f"  相対 (5, 1) にすると: {rot(q, (3, 5)) @ rot(k, (8, 6)):.6f}")
    print("  → 格子が固定なら位置も固定なので、cos/sin の表 [N,64] を定数にしてグラフに持てる（毎回計算しない）")


if __name__ == "__main__":
    main()
