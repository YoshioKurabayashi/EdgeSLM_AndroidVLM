# -*- coding: utf-8 -*-
"""Phase 5: Gemma 4 E2B の文章グラフを「1 回で T トークン（既定 16）読む」形で組む（2026-09-29）。

配線そのもの（層の中の計算・重みの読み方）は `gemma4_graph.py`（EdgeSLM_AndroidGemma4 から**そのまま複製**、変更なし）を使う。
ここで変えるのは **入出力の約束だけ**:

| | 1 トークン版（gemma4_graph.DecodeChunk） | この T トークン版（DecodeChunkT） |
|---|---|---|
| 1 回に入れる行 | 1 | T（本物は n ≤ T 行。残りは空。空の行の出力は使わない） |
| past（入力） | [1,1,L,hd]（L=256） | [1,1,**P**,hd]（P = W − T = 240） |
| 注意の窓 | cat(past, 新) の後ろ L 個（= 256） | cat(past, 新)（= P + T = W = 256。**同じ長さ**） |
| present（出力） | 窓全体 [1,1,L,hd]（端末はそのまま次の past にする） | **新しい T 行だけ** [1,1,T,hd]（端末が本物の n 行だけを past に足す） |
| 層 13・14（後ろの層が使い回す K/V） | 同上 | **窓全体** [1,1,W,hd]（後ろのかたまりへ渡す必要があるため。端末は「後ろから T 行目」から読む） |
| logits | [1,1,V] | [1,**1**,V]（入力 `logit_sel` [1,T,1] で選んだ 1 行だけ。選ぶのは**最後の層のかたまり E の出口**。出力層（H）は 1 トークン版のものをそのまま使う） |

## なぜ新しい T 行だけを出すのか

16 行のうち本物が 1 行だけ（答えを作るとき）でも、窓全体を出す形だと空の 15 行が窓に入ってしまい、端末はそのまま使えない。
どのみち端末が「本物の n 行だけを past の右端に足す」ので、出すのは新しい行だけで足りる（出力 9.4MB → 0.6MB）。
端末は present の形を見て、新しい行が「present の後ろから T 行目」から始まると読む（窓全体の形でも同じ式で動く）。

## 出力層の前の「行の選択」（かたまり E の出口）

`h = hidden^T @ logit_sel`（[1,H,T] @ [1,T,1] = [1,H,1] → [1,1,H]）。`logit_sel` は 1 行だけ 1、ほかは 0 なので、選んだ行がそのまま出る。
これを**かたまり E（層 29〜34）の出口**で行い、E の hidden_out を [1,1,H] にする。すると出力層のかたまり（H: 最後の正規化＋出力層）は
**1 トークン版とまったく同じ形**になるので、Gemma4 で作って実機で動いている `g4head_L256_final_qdq.onnx` をそのまま使える（複製）。
- 出力層のかたまりの中で選ばないのは、グラフ入力（hidden_in）がそのまま Transpose に入ると ORT が量子化しない（C-1。1 トークン版の
  `shared_*` と同じ落とし穴）ため。E の中なら Transpose の入力は計算の途中の値なので量子化される
- 行列積の**右辺を logit_sel にする**のが要点。C-8（v68 の活性×活性の行列積は右辺を 8bit に落とす。downgrade_actact_rhs）で
  8bit になるのは logit_sel（0 と 1 だけなので 8bit でも正確）。逆（sel @ hidden）にすると hidden が 8bit になり、出力層の入力が荒れる
- 足し算・平均（ReduceSum / ReduceMean）で選ぶ形は、ORT の量子化がその演算を扱うか確かでないので使わない
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from app.pipeline import gemma4_graph as M

T_DEFAULT = 16
W_DEFAULT = 256          # 注意の窓（1 トークン版の L と同じ）


def layer_forward_t(ly: M.DecoderLayer, h, ple_i, cos, sin, attn_mask, past_k=None, past_v=None,
                    shared_kv=None):
    """`gemma4_graph.DecoderLayer.forward` と同じ計算で、窓を「cat(past, 新)」（切らない）にしたもの。

    戻り値: (h, new_k, new_v, win_k, win_v)。KV を使い回す層は new/win とも None。
    - new_k/new_v: この回の T 行 [1,1,T,hd]
    - win_k/win_v: 注意に使った窓 [1,1,P+T,hd]（後ろの層が使い回す）
    """
    T = h.shape[1]
    res = h
    x = ly.input_layernorm(h)
    q = ly.q_proj(x).view(1, T, ly.nh, ly.hd)
    q = M.rope(ly.q_norm(q), cos, sin)                                   # [1,T,8,hd]
    if shared_kv is None:
        k = M.rope(ly.k_norm(ly.k_proj(x).view(1, T, 1, ly.hd)), cos, sin)   # [1,T,1,hd]
        v = ly.v_norm(ly.v_proj(x).view(1, T, 1, ly.hd))
        new_k, new_v = k.transpose(1, 2), v.transpose(1, 2)             # [1,1,T,hd]
        # ★past を Concat の最初に入れる順は 1 トークン版と同じ（Qwen の注記: Slice に直接入る float 入力を ORT が量子化しない）。
        #   ここでは Slice しない（窓 = past P 個 ＋ 新しい T 個 = W 個）
        pk = torch.cat((past_k, new_k), dim=2)                           # [1,1,W,hd]
        pv = torch.cat((past_v, new_v), dim=2)
    else:
        pk, pv = shared_kv
        new_k = new_v = None
    s = torch.matmul(q, pk.transpose(2, 3)) + attn_mask                 # [1,T,8,W]
    Lw = pv.shape[2]
    p = torch.softmax(s, dim=-1).reshape(1, T * ly.nh, Lw)
    # ★確率×V の右辺は形を変えた V（1 トークン版と同じ。C-8 の downgrade が右辺だけを 8bit にできるように）
    ctx = torch.matmul(p, pv.reshape(1, Lw, ly.hd))                     # [1,T*8,hd]
    h = res + ly.post_attention_layernorm(ly.o_proj(ctx.reshape(1, T, ly.nh * ly.hd)))
    res = h
    x = ly.pre_feedforward_layernorm(h)
    h = res + ly.post_feedforward_layernorm(ly.down_proj(M.gelu_tanh(ly.gate_proj(x)) * ly.up_proj(x)))
    res = h
    g = M.gelu_tanh(ly.per_layer_input_gate(h)) * ple_i
    h = res + ly.post_per_layer_input_norm(ly.per_layer_projection(g))
    h = h * ly.layer_scalar
    if shared_kv is None:
        return h, new_k, new_v, pk, pv
    return h, None, None, None, None


class DecodeChunkT(M.DecodeChunk):
    """T トークン版のかたまり。重み・層の組み方は `gemma4_graph.DecodeChunk` をそのまま使い、forward と入出力だけ変える。

    入力: 1 トークン版と同じ名前（inputs_embeds/ple_tok または hidden_in/ple_all、cos/sin、attn_mask、past_*、shared_*）。
          `select_row=True`（最後の層のかたまり E）は末尾に **logit_sel** [1,T,1] が付き、hidden_out は選んだ 1 行 [1,1,H]。
    出力: hidden_out（または logits）、ple_all（入口のときだけ）、present_*。
          present_* は「新しい T 行」。ただし**後ろのかたまりが使い回す層（13・14）は窓全体**（merge_chunks が
          present_k_13 → shared_k_13 と繋ぐ。1 トークン版と同じ繋ぎ方のまま使えるように）
    """

    WINDOW_LAYERS = (13, 14)            # 後ろの層（15〜34）が使い回す K/V の出どころ（局所 13・全体 14）

    def __init__(self, qf, layers, with_ple_proj: bool = False, select_row: bool = False):
        super().__init__(qf, layers, with_ple_proj=with_ple_proj, with_head=False)
        if not self.layers:
            raise ValueError("T トークン版の出力層は 1 トークン版のもの（g4head_L256）をそのまま使う")
        self.select_row = select_row

    def input_names(self) -> list[str]:
        return super().input_names() + (["logit_sel"] if self.select_row else [])

    def forward(self, x0, ple_in=None, cos_s=None, sin_s=None, cos_g=None, sin_g=None, attn_mask=None, *caches):
        if self.select_row:                                             # 最後の入力が logit_sel
            caches, sel = caches[:-1], caches[-1]
        if self.with_ple_proj:
            proj = F.linear(x0, self.ple_proj) * self.ple_in_scale
            proj = proj.reshape(1, x0.shape[1], self.n_layers, self.ple_dim)
            ple_all = (self.ple_norm(proj) + ple_in) * self.ple_combine
        else:
            ple_all = ple_in
        h = x0
        kv, presents, ci = {}, [], 0
        n_own = 2 * len(self.kv_layers)
        for k, i in enumerate(self.ext_kv):
            kv[i] = (caches[n_own + 2 * k], caches[n_own + 2 * k + 1])
        for ly in self.layers:
            info = ly.info
            cos, sin = (cos_g, sin_g) if info["full"] else (cos_s, sin_s)
            ple_i = ple_all[:, :, info["i"], :]
            if info["shared"]:
                h, *_ = layer_forward_t(ly, h, ple_i, cos, sin, attn_mask, shared_kv=kv[info["kv_src"]])
            else:
                h, nk, nv, wk, wv = layer_forward_t(ly, h, ple_i, cos, sin, attn_mask, caches[ci], caches[ci + 1])
                ci += 2
                kv[info["i"]] = (wk, wv)
                if info["i"] in self.WINDOW_LAYERS:
                    presents += [wk, wv]
                else:
                    presents += [nk, nv]
        if self.select_row:
            h = torch.matmul(h.transpose(1, 2), sel).transpose(1, 2)    # [1,H,T]@[1,T,1] → [1,1,H]（選んだ行）
        return (h, ple_all, *presents) if self.with_ple_proj else (h, *presents)


def build_chunk(layers: list[int], head_ple: bool = False, select_row: bool = False) -> DecodeChunkT:
    qf = M.QatFile()
    return DecodeChunkT(qf, layers, with_ple_proj=head_ple, select_row=select_row).eval()
