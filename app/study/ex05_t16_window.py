# -*- coding: utf-8 -*-
"""演習 5: まとめ読み（16 トークン版）のグラフの約束 ― 窓・マスク・空の行・行の選択（StudyMemo 10.4・10.5 節）。

    python -B -m app.study.ex05_t16_window        # numpy だけ。1 秒

## 何を確かめるか

`app/pipeline/g4t_graph.py` の T トークン版と、端末の C++（`job.cpp` の複数トークンの経路）の約束を、2 層の小さな注意で再現する。

- 入力は T 行。本物は n ≤ T 行（プロンプトは最大 T、答えを作るときは 1）、残りは**空の行**（埋め込み 0、出力は使わない）
- 注意の窓 = cat(past P 行, この回の T 行)（past は右詰め。中身のある右端 n_valid 行だけマスクを開ける）
- 行 r が見てよいのは、past の中身のある所と、この回の 0〜r 行目（因果）。空の行のマスクは 0 行目と同じ
- 出力の K/V は「この回の T 行」。端末は**本物の n 行だけ**を past の右端に足す（空の行の K/V は足さない）
- `logit_sel` [T,1] で 1 行だけ選ぶ（`hidden^T @ logit_sel`。右辺が 0/1 の選択なので、8bit に落としても正確）

確かめ: プロンプトを T 行ずつ読んでから 1 行ずつ答えを作った結果が、全部を 1 行ずつ読んだ結果と（浮動小数の誤差の範囲で）一致する。
わざとマスクを間違えると、どこがずれるかも見る。
"""
from __future__ import annotations

import numpy as np

MASK_NEG = -60.0


def softmax(x):
    x = x - x.max(-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(-1, keepdims=True)


def make_mask(T, P, n, n_valid, mode="ok"):
    """g4t_graph / job.cpp の約束どおりのマスク。mode で、わざと間違えた版も作る"""
    m = np.full((T, P + T), MASK_NEG)
    for r in range(T):
        rr = r if r < n else 0                         # 空の行は 0 行目と同じ
        m[r, P - n_valid:P] = 0.0                      # past の中身のある右端
        if mode == "no_causal":
            m[r, P:P + n] = 0.0                        # ★この回の後ろの本物の行まで見える
        elif mode == "see_empty":
            m[r, P:P + T] = 0.0                        # ★空の行（埋め込み 0）まで見える
        else:
            m[r, P:P + rr + 1] = 0.0                   # この回の 0〜r 行目
    return m


class Model:
    def __init__(self, D, n_layers, rng):
        self.W = [[rng.normal(0, 0.4, (D, D)) for _ in range(3)] for _ in range(n_layers)]
        self.D = D
        self.head = None                                                       # 出力層（語彙ぶんの行列）。generate で作る

    def generate(self, emb, prompt, n_new, T, P, mode="ok", select_by_matmul=True):
        L, D = len(self.W), self.D
        past = [[np.zeros((P, D)), np.zeros((P, D))] for _ in range(L)]
        n_valid, outs, hs = 0, [], []

        def feed(ids):
            nonlocal n_valid
            n = len(ids)
            h = np.zeros((T, D))
            h[:n] = emb[ids]                                                   # 残りは空の行（0）
            mask = make_mask(T, P, n, n_valid, mode)
            for li, (Wq, Wk, Wv) in enumerate(self.W):
                q, k, v = h @ Wq, h @ Wk, h @ Wv
                pk, pv = past[li]
                a = softmax(q @ np.concatenate([pk, k]).T / np.sqrt(D) + mask) @ np.concatenate([pv, v])
                past[li] = [np.concatenate([pk[n:], k[:n]]), np.concatenate([pv[n:], v[:n]])]   # 本物の n 行だけ足す
                h = h + a                                                      # 残差
            n_valid = min(n_valid + n, P)
            if select_by_matmul:
                sel = np.zeros((T, 1))
                sel[n - 1, 0] = 1.0                                            # logit_sel: 最後の本物の行
                return (h.T @ sel).T[0]                                        # hidden^T @ logit_sel
            return h[n - 1]

        for i in range(0, len(prompt), T):                                     # プロンプトを T 個ずつ
            last = feed(prompt[i:i + T])
        for _ in range(n_new):                                                 # 答えは 1 個ずつ（0 行目だけ本物）
            hs.append(last)
            nxt = int(np.argmax(last @ self.head))
            outs.append(nxt)
            last = feed([nxt])
        return outs, np.stack(hs)


def main():
    rng = np.random.default_rng(1)
    V, D, P = 64, 16, 32
    emb = rng.normal(0, 1, (V, D))
    model = Model(D, 2, rng)
    model.head = rng.normal(0, 1, (D, V))
    prompt = list(rng.integers(0, V, 13))

    a, ha = model.generate(emb, prompt, 8, T=1, P=P)                           # 1 トークン版と同じ読み方
    print("== 1 トークンずつ ↔ T 行ずつ（2 層、プロンプト 13 語、答え 8 語、past 32 行）")
    print(f"  T = 1 : 答え {a}")
    for T in (4, 16):
        b, hb = model.generate(emb, prompt, 8, T=T, P=P)
        print(f"  T = {T:<2}: 答え {b}   出力のベクトルの最大差 {np.abs(ha - hb).max():.1e}")
    c, hc = model.generate(emb, prompt, 8, T=16, P=P, select_by_matmul=False)
    _, hd = model.generate(emb, prompt, 8, T=16, P=P)
    print(f"\n  logit_sel の行列積で選ぶ ↔ 行を直接取り出す: 最大差 {np.abs(hc - hd).max():.1e}")

    print("\n== わざとマスクを間違えると（T = 16）")
    for name, mode in (("この回の行どうしの因果を外す", "no_causal"), ("空の行を見えるようにする", "see_empty")):
        e, he = model.generate(emb, prompt, 8, T=16, P=P, mode=mode)
        print(f"  {name:16s}: 出力のベクトルの最大差 {np.abs(ha - he).max():.2f}、答え {e}"
              f"{'（答えも変わった）' if e != a else '（答えは同じでも中の値はずれている）'}")
    print("  → まとめて読むときは、この回の行どうしの因果と、空の行を塞ぐマスクの両方が要る（10.4 節）。")
    print("    1 層だけなら最後の行は全部の本物の行を見てよいので因果の誤りは表に出ない。2 層目の K/V を通して効いてくる")


if __name__ == "__main__":
    main()
