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
どのみち端末が「本物の n 行だけを past の右端に足す」ので、出すのは新しい行だけで足りる（出力 約 9.4MB → 約 2.1MB。
新しい行だけなら 0.6MB 程度だが、層 13・14 は下の表のとおり窓全体を出すので、その 2 層が大半を占める。
★2026-10-18 訂正: 旧「9.4MB → 0.6MB」は、層 13・14 も新しい行だけを出すとした場合の値。StudyMemo 10.4 節）。
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

## 使う段・入力と出力・動かし方

- Phase 5（基本設計_Phase5.md、StudyMemo 10.4 節「グラフの約束」・13 章）。このファイル単体では動かさない
- build_t16.py（と書き直し版の build_t16fix.py）が build_chunk でかたまりを作り、ONNX に書き出して量子化・変換する
- 入力・出力の名前と形は DecodeChunkT の説明のとおり。マスク（行ごとの因果マスク、−60 で「見ない」）は端末の job.cpp が作る
- 窓を 256 のままにしたのは、softmax の長さ・マスクの形を 1 トークン版と同じにして量子化の振る舞いを近づけるため。
  その代わり覚えていられるのは直近 240 トークン（VLM の 1 枚目は約 107＋答え 96 以下で足りる。StudyMemo 10.4 節）
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from app.pipeline import gemma4_graph as M

# 1 回に読む行の数（T）と注意の窓の長さ（W）。past の長さは P = W − T = 240
T_DEFAULT = 16
W_DEFAULT = 256          # 注意の窓（1 トークン版の L と同じ）


def layer_forward_t(ly: M.DecoderLayer, h, ple_i, cos, sin, attn_mask, past_k=None, past_v=None,
                    shared_kv=None):
    """`gemma4_graph.DecoderLayer.forward` と同じ計算で、窓を「cat(past, 新)」（切らない）にしたもの。

    戻り値: (h, new_k, new_v, win_k, win_v)。KV を使い回す層は new/win とも None。
    - new_k/new_v: この回の T 行 [1,1,T,hd]
    - win_k/win_v: 注意に使った窓 [1,1,P+T,hd]（後ろの層が使い回す）

    引数: ly = gemma4_graph の 1 層、h = [1,T,H]（H=1536）、ple_i = この層の PLE [1,T,256]（256 は vlm_common.PLE_DIM）、
    cos/sin = この回の T 行ぶんの RoPE の表、attn_mask = [1,T,1,W] の加算マスク（見る 0・見ない −60。端末が行ごとに作る）、
    past_k/past_v = [1,1,P,hd]、shared_kv = 層 13・14 の窓（後ろの 20 層が自分の K/V を持たずに使い回す。KV sharing）。
    1 トークン版と違って窓の後ろを切らない（Slice しない）ので、窓の長さは P + T = W のまま。
    """
    # ① 注意の前の正規化と Q。q_norm の後に RoPE（回転の位置の埋め込み）をかける
    T = h.shape[1]
    res = h
    x = ly.input_layernorm(h)
    q = ly.q_proj(x).view(1, T, ly.nh, ly.hd)
    q = M.rope(ly.q_norm(q), cos, sin)                                   # [1,T,8,hd]
    # ② K/V: 自分で持つ層はこの回の T 行を作って past の右に足す。使い回す層は層 13・14 の窓をそのまま使う
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
    # ③ 注意の重み（softmax）と確率×V。ヘッド 8 個ぶんを T*8 行に並べ直して 1 回の行列積にする（K/V のヘッドは 1 個を共有）
    Lw = pv.shape[2]
    p = torch.softmax(s, dim=-1).reshape(1, T * ly.nh, Lw)
    # ★確率×V の右辺は形を変えた V（1 トークン版と同じ。C-8 の downgrade が右辺だけを 8bit にできるように）
    ctx = torch.matmul(p, pv.reshape(1, Lw, ly.hd))                     # [1,T*8,hd]
    # ④ 出力の射影と残差。Gemma 4 は注意・MLP のそれぞれの後ろにも正規化がある（post_*_layernorm）
    h = res + ly.post_attention_layernorm(ly.o_proj(ctx.reshape(1, T, ly.nh * ly.hd)))
    res = h
    # ⑤ MLP（gate と up の積を down で戻す。活性化は tanh 近似の GELU）
    x = ly.pre_feedforward_layernorm(h)
    h = res + ly.post_feedforward_layernorm(ly.down_proj(M.gelu_tanh(ly.gate_proj(x)) * ly.up_proj(x)))
    res = h
    # ⑥ 層ごとの埋め込み（PLE）を門を通して足し、層ごとの倍率（layer_scalar）をかける
    g = M.gelu_tanh(ly.per_layer_input_gate(h)) * ple_i
    h = res + ly.post_per_layer_input_norm(ly.per_layer_projection(g))
    h = h * ly.layer_scalar
    # ⑦ 戻り値: 自分で K/V を持つ層は (新しい T 行, 窓全体) の両方を返す。どちらを present に出すかは呼び手が決める
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

    # 窓全体を出す層の番号（ほかの層は新しい T 行だけ。出力が約 9.4MB → 約 2.1MB になり、その大半がこの 2 層。StudyMemo 10.4 節）
    WINDOW_LAYERS = (13, 14)            # 後ろの層（15〜34）が使い回す K/V の出どころ（局所 13・全体 14）

    # qf = QAT の重みの読み手、layers = このかたまりの層の番号、with_ple_proj = 入口のかたまり（PLE の射影を中で作る）、
    # select_row = 最後の層のかたまり E（出口で 1 行を選ぶ）。出力層は持たない（1 トークン版の g4head_L256 を複製して使うため）
    def __init__(self, qf, layers, with_ple_proj: bool = False, select_row: bool = False):
        super().__init__(qf, layers, with_ple_proj=with_ple_proj, with_head=False)
        if not self.layers:
            raise ValueError("T トークン版の出力層は 1 トークン版のもの（g4head_L256）をそのまま使う")
        self.select_row = select_row

    # 入力の名前は 1 トークン版と同じ並びで、E だけ最後に logit_sel が付く（forward はこの並びで受け取る）
    def input_names(self) -> list[str]:
        return super().input_names() + (["logit_sel"] if self.select_row else [])

    def forward(self, x0, ple_in=None, cos_s=None, sin_s=None, cos_g=None, sin_g=None, attn_mask=None, *caches):
        """T 行ぶんを層の順に通し、(hidden_out, [ple_all,] present_*) を返す。

        x0 = 入口なら inputs_embeds [1,T,1536]、途中のかたまりなら hidden_in。ple_in = 入口なら ple_tok、それ以外は ple_all。
        cos_s/sin_s は局所の注意の層、cos_g/sin_g は全体の注意の層の RoPE の表。caches は
        「自分の層の past_k, past_v を層の順に」→「使い回す外の層の shared_k, shared_v」→（E だけ）logit_sel の順に並ぶ。
        """
        if self.select_row:                                             # 最後の入力が logit_sel
            caches, sel = caches[:-1], caches[-1]
        # ① 入口のかたまりだけ: PLE の射影（inputs_embeds × 1536^-0.5 を線形層で 35 層ぶんに広げる）と表の行を足して ple_all を作る
        #    （35 層ぶん = n_layers、1 層 256 次元。gemma4_graph の DecodeChunk と同じ式）
        if self.with_ple_proj:
            proj = F.linear(x0, self.ple_proj) * self.ple_in_scale
            proj = proj.reshape(1, x0.shape[1], self.n_layers, self.ple_dim)
            ple_all = (self.ple_norm(proj) + ple_in) * self.ple_combine
        else:
            ple_all = ple_in
        # ② 外から受け取る K/V（使い回す層 13・14 の窓）を、自分の層の past の後ろの位置から取り出して kv に入れておく
        h = x0
        kv, presents, ci = {}, [], 0
        n_own = 2 * len(self.kv_layers)
        for k, i in enumerate(self.ext_kv):
            kv[i] = (caches[n_own + 2 * k], caches[n_own + 2 * k + 1])
        # ③ 層の順に通す。全体の注意の層（info["full"]）と局所の注意の層で RoPE の表を使い分ける
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
    # layers の層だけのかたまりを、QAT の重みで組んで返す（head_ple = 入口、select_row = 出口で 1 行を選ぶ E）。build_t16 が使う
    qf = M.QatFile()
    return DecodeChunkT(qf, layers, with_ple_proj=head_ple, select_row=select_row).eval()
