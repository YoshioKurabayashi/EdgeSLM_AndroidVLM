# -*- coding: utf-8 -*-
"""Phase 2: Gemma 4 E2B の画像エンコーダ（ViT 16 層）＋射影層を、NPU 向けに組み直す（2026-09-28）。

## 何をするか

HF の `Gemma4VisionModel`（パッチの埋め込み → 16 層 → 3×3 プーリング）と `Gemma4MultimodalEmbedder`（射影層）と
同じ計算を、**格子を固定して**素の torch 演算で書く。書き出した ONNX を量子化して v68 の NPU へ載せる（Phase 2-2 以降）。

- 重みは QAT モバイル版の safetensors から**直接**読む（transformers は通さない）。HF の実装とは独立に書いた配線を、
  HF の出力（capture_vit.py が保存）と突き合わせて確かめる（EdgeSLM_AndroidGemma4 の gemma4_graph.py と同じ考え方）
- QAT の Linear は、活性の目盛り（SRQ）と名前を `Lin.qat` に持つ。整数とスケールは Phase 2-2 の qat_inject が
  safetensors から読み直して QDQ に書き戻す（Gemma4 の qat_inject と同じく、ONNX の初期化子とは**値で**対応づける）

## 配線（transformers 5.14.1 の modeling_gemma4.py で確認。N＝パッチ数、M＝画像トークン数＝N/9）

    入力 pixel_values [1,N,768]（0〜1。1 パッチの並びは (py, px, c)、パッチの並びは左上から右へ・行ごとに下へ）
    x  = 2·(p − 0.5)
    h  = x @ Wp^T + POS                    Wp: patch_embedder.input_proj（768→768、QAT ではない float）
                                           POS: 位置の表の x 行と y 行を足した定数 [N,768]（格子が固定なので先に計算）
    16 層（HF の Gemma4VisionEncoderLayer と同じ順）:
        a = input_layernorm(h)
        q = RoPE2d(q_norm(q_proj(a) → [1,N,12,64]))、k も同じ、v = v_norm(v_proj(a))（v_norm は重みなし）
        s = q @ k^T                        スケール 1.0（q・k を正規化しているため）。マスクなし（双方向・パディングなし）
        h = h + post_attention_layernorm(o_proj(softmax(s) @ v))
        h = h + post_feedforward_layernorm(down(gelu_tanh(gate(b)) · up(b)))、b = pre_feedforward_layernorm(h)
    プーリング: 3×3 のパッチの平均 → [1,M,768]
    射影: RMSNorm（重みなし）→ @ We^T（768→1536、QAT ではない float）→ 出力 image_embeds [1,M,1536]

## NPU 向けに変えたところ（どれも数学的には同じ計算。check で HF と一致を確かめる）

| HF | ここ | 理由 |
|---|---|---|
| 位置の表を引く（Embedding） | 定数 POS を足す | 格子が固定なので位置も固定。Gather をグラフに入れない |
| 2 次元 RoPE の cos/sin を毎回計算 | 定数の表 [N,64] を掛ける | 同上。表は HF と同じ式（float32）で作る |
| one_hot の行列 × h で平均（float32） | reshape → ReduceMean を 2 回 | 重み（定数行列 630×70）を持たない。テンソルは 4 次元まで |
| プーリングの後に ×√768 | **掛けない**。射影層の RMSNorm の eps を eps/768 にする | RMSNorm(c·x; ε) = RMSNorm(x; ε/c²)。演算と量子化（Q/DQ）が 1 組減る。fp16 で計算される区間があっても溢れにくい（fold_pool_scale）。※テンソル全体で 1 つの目盛りなら、c 倍しても相対的な細かさは変わらないので、精度そのものの得ではない |
| RMSNorm は x·pow(mean(x²)+ε, −0.5)·w | x·rsqrt(mean(x²)+ε)·w | Gemma4 と同じ形。書き出すと Pow→ReduceMean→Add→Sqrt→Reciprocal→Mul になり、QAIRT が RmsNorm に融合する |
| gelu_pytorch_tanh | Pow を使わずに書く | RMSNorm の Pow と紛れないように（Gemma4 と同じ） |
| q/k/v を [1,12,N,64] にして attention | 同じ（4 次元の行列積） | k^T と v は行列積の右辺専用のテンソルになる（C-8 で右辺だけ 8bit に落とすため） |

## SRQ（QAT の活性の丸め）

`SRQ = True` にすると、各 Linear の入出力を QAT の目盛りで 8bit に丸める（HF の QuantizedLinear と同じ）。
**ONNX に書き出すときは False のまま**（量子化の工程で u8 の Q/DQ として入れる）。確認用。
"""
from __future__ import annotations

import json

import torch
import torch.nn.functional as F
from safetensors import safe_open

from app.pipeline import vlm_common as C

PFX = "model.vision_tower."
GELU_C = 0.7978845608028654        # sqrt(2/pi)
SRQ = False                        # ★確認用。ONNX に書き出すときは False


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    # ★Pow を使わない（RMSNorm の見分けが「Pow(x,2) → ReduceMean」を探すため、紛れないように）
    return 0.5 * x * (1.0 + torch.tanh(GELU_C * (x + 0.044715 * x * x * x)))


def apply_srq(x: torch.Tensor, scale: float) -> torch.Tensor:
    """QAT の活性の丸め（transformers の gemma_quant.apply_srq と同じ。scale 0 は何もしない）。"""
    if not scale:
        return x
    return torch.clamp(torch.round(x / scale), -128.0, 127.0) * scale


class QatFile:
    """QAT モバイル版のファイルから、名前を指定して必要なテンソルだけを読む（画像エンコーダ用）。"""

    def __init__(self):
        self.f = safe_open(str(C.HF_DIR / "model.safetensors"), framework="pt")
        self.cfg = json.loads((C.HF_DIR / "config.json").read_text(encoding="utf-8"))["vision_config"]
        self.eps = float(self.cfg["rms_norm_eps"])
        # 前提の確認（違えば配線を見直す必要がある）
        assert self.cfg["hidden_size"] == C.VISION_HIDDEN and self.cfg["num_hidden_layers"] == C.VIT_LAYERS
        assert self.cfg["num_attention_heads"] == C.VIT_HEADS and self.cfg["num_key_value_heads"] == C.VIT_HEADS
        assert self.cfg["head_dim"] == C.VIT_HEAD_DIM and self.cfg["intermediate_size"] == C.VIT_INTER
        assert self.cfg["patch_size"] == C.PATCH and self.cfg["pooling_kernel_size"] == C.POOL
        assert not self.cfg.get("standardize") and not self.cfg.get("use_clipped_linears")
        assert self.cfg["hidden_activation"] == "gelu_pytorch_tanh"
        assert float(self.cfg["rope_parameters"]["rope_theta"]) == C.VIT_ROPE_THETA

    def t(self, name: str) -> torch.Tensor:
        return self.f.get_tensor(name)

    def vec(self, name: str) -> torch.Tensor:
        return self.t(name).float()

    def linear(self, base: str) -> dict:
        """QAT の Linear（int8・出力チャネルごと）。整数・スケール・活性の目盛りと、float32 に戻した重み [out, in]。"""
        w = self.t(base + ".weight")
        assert w.dtype == torch.int8, f"{base}: int8 のはず（{w.dtype}）"
        s = self.t(base + ".weight_scale").float().reshape(-1)
        return {"name": base, "codes": w, "scale": s, "bits": 8, "w": w.float() * s[:, None],
                "in_act": float(self.t(base + ".input_activation_scale")),
                "out_act": float(self.t(base + ".output_activation_scale"))}


class RMSNorm(torch.nn.Module):
    """x * rsqrt(mean(x²)+eps) * w（Gemma4 の gemma4_graph.RMSNorm と同じ形）。重みなしは w=1。"""

    def __init__(self, w: torch.Tensor, eps: float):
        super().__init__()
        self.weight = torch.nn.Parameter(w.clone(), requires_grad=False)
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class Lin(torch.nn.Module):
    """バイアスなしの Linear。書き出すと MatMul と [in,out] の初期化子になる。qat に QAT の情報を持つ（Phase 2-2 で使う）。"""

    def __init__(self, d: dict):
        super().__init__()
        self.weight = torch.nn.Parameter(d["w"].clone(), requires_grad=False)
        self.qat = {k: d[k] for k in ("name", "bits", "in_act", "out_act")} if "in_act" in d else None

    def forward(self, x):
        if SRQ and self.qat:
            y = F.linear(apply_srq(x, self.qat["in_act"]), self.weight)
            return apply_srq(y, self.qat["out_act"])
        return F.linear(x, self.weight)


def grid_positions(gh: int, gw: int) -> torch.Tensor:
    """パッチ n の (x, y)。HF の image_processing_gemma4 と同じ（meshgrid(indexing='xy') → n = y·gw + x）。[N,2] int64。"""
    ys, xs = torch.meshgrid(torch.arange(gh), torch.arange(gw), indexing="ij")
    return torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)


def rope2d_tables(pos: torch.Tensor, head_dim: int = C.VIT_HEAD_DIM, theta: float = C.VIT_ROPE_THETA):
    """2 次元 RoPE の cos/sin の表 [N, head_dim]。HF の Gemma4VisionRotaryEmbedding と同じ式（float32）。

    head_dim の前半（32）を x、後半（32）を y に使う。各半分の中で周波数は 16 個、cat(freqs, freqs) で 32 にする。
    """
    spatial = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, spatial, 2, dtype=torch.int64).float() / spatial))   # [16]
    coss, sins = [], []
    for i in range(2):                                         # 0 = x、1 = y
        p = pos[:, i].float()[None, :]                         # [1,N]
        freqs = (inv_freq[:, None] @ p).transpose(0, 1)        # [N,16]（HF と同じく行列積で作る）
        emb = torch.cat((freqs, freqs), dim=-1)                # [N,32]
        coss.append(emb.cos())
        sins.append(emb.sin())
    return torch.cat(coss, dim=-1), torch.cat(sins, dim=-1)   # [N,64] ×2


def rotate_half_2d(x: torch.Tensor) -> torch.Tensor:
    """head_dim を 32 ずつ 2 つに分け、それぞれの中で rotate_half（HF の apply_multidimensional_rope と同じ）。"""
    q = C.VIT_HEAD_DIM // 4                                    # 16
    return torch.cat((-x[..., q:2 * q], x[..., :q], -x[..., 3 * q:], x[..., 2 * q:3 * q]), dim=-1)


class VitLayer(torch.nn.Module):
    def __init__(self, qf: QatFile, i: int):
        super().__init__()
        p = f"{PFX}encoder.layers.{i}."
        eps = qf.eps
        self.nh, self.hd = C.VIT_HEADS, C.VIT_HEAD_DIM
        self.input_layernorm = RMSNorm(qf.vec(p + "input_layernorm.weight"), eps)
        self.post_attention_layernorm = RMSNorm(qf.vec(p + "post_attention_layernorm.weight"), eps)
        self.pre_feedforward_layernorm = RMSNorm(qf.vec(p + "pre_feedforward_layernorm.weight"), eps)
        self.post_feedforward_layernorm = RMSNorm(qf.vec(p + "post_feedforward_layernorm.weight"), eps)
        self.q_proj = Lin(qf.linear(p + "self_attn.q_proj.linear"))
        self.k_proj = Lin(qf.linear(p + "self_attn.k_proj.linear"))
        self.v_proj = Lin(qf.linear(p + "self_attn.v_proj.linear"))
        self.o_proj = Lin(qf.linear(p + "self_attn.o_proj.linear"))
        self.q_norm = RMSNorm(qf.vec(p + "self_attn.q_norm.weight"), eps)
        self.k_norm = RMSNorm(qf.vec(p + "self_attn.k_norm.weight"), eps)
        self.v_norm = RMSNorm(torch.ones(self.hd), eps)                     # 重みなし（with_scale=False）
        self.gate_proj = Lin(qf.linear(p + "mlp.gate_proj.linear"))
        self.up_proj = Lin(qf.linear(p + "mlp.up_proj.linear"))
        self.down_proj = Lin(qf.linear(p + "mlp.down_proj.linear"))
        # cos/sin は全層で同じ表なので層には持たせず、VitGraph のバッファを forward の引数で受け取る（書き出しで重複させない）

    def forward(self, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        N = h.shape[1]
        res = h
        a = self.input_layernorm(h)
        q = self.q_norm(self.q_proj(a).view(1, N, self.nh, self.hd))
        k = self.k_norm(self.k_proj(a).view(1, N, self.nh, self.hd))
        v = self.v_norm(self.v_proj(a).view(1, N, self.nh, self.hd))
        q = q * cos + rotate_half_2d(q) * sin                               # cos/sin [1,N,1,64]
        k = k * cos + rotate_half_2d(k) * sin
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)   # [1,12,N,64]
        s = torch.matmul(q, k.transpose(2, 3))                              # [1,12,N,N]。スケール 1.0、マスクなし
        ctx = torch.matmul(torch.softmax(s, dim=-1), v)                      # [1,12,N,64]
        ctx = ctx.transpose(1, 2).reshape(1, N, self.nh * self.hd)
        h = res + self.post_attention_layernorm(self.o_proj(ctx))
        res = h
        b = self.pre_feedforward_layernorm(h)
        h = res + self.post_feedforward_layernorm(self.down_proj(gelu_tanh(self.gate_proj(b)) * self.up_proj(b)))
        return h


class VitGraph(torch.nn.Module):
    """画像エンコーダ＋射影層（格子を固定）。入力 pixel_values [1,N,768] → 出力 image_embeds [1,M,1536]。

    grid=(gh, gw)＝パッチの縦×横（例: 70 トークン → (21, 30)）。gh・gw は 3 の倍数。
    fold_pool_scale=True: プーリングの後の ×√768 を掛けず、射影層の RMSNorm の eps を eps/768 にする（同じ計算）。
    """

    INPUT_NAMES = ["pixel_values"]
    OUTPUT_NAMES = ["image_embeds"]

    def __init__(self, qf: QatFile, grid: tuple[int, int], fold_pool_scale: bool = True):
        super().__init__()
        gh, gw = grid
        assert gh % C.POOL == 0 and gw % C.POOL == 0, grid
        self.gh, self.gw = gh, gw
        self.N, self.M = gh * gw, (gh // C.POOL) * (gw // C.POOL)
        self.fold_pool_scale = fold_pool_scale
        pos = grid_positions(gh, gw)
        # パッチの埋め込み（QAT ではない float。bf16 を float32 に戻すだけで値は変わらない）
        self.patch_proj = Lin({"w": qf.vec(PFX + "patch_embedder.input_proj.weight")})
        table = qf.vec(PFX + "patch_embedder.position_embedding_table")          # [2, 10240, 768]
        self.register_buffer("pos_emb", (table[0][pos[:, 0]] + table[1][pos[:, 1]])[None])   # [1,N,768]
        cos, sin = rope2d_tables(pos)
        self.register_buffer("cos", cos[None, :, None, :])                     # [1,N,1,64]
        self.register_buffer("sin", sin[None, :, None, :])
        self.layers = torch.nn.ModuleList([VitLayer(qf, i) for i in range(C.VIT_LAYERS)])
        # 射影層（Gemma4MultimodalEmbedder）: RMSNorm（重みなし、eps は vision の rms_norm_eps）→ Linear（float）
        eps = qf.eps / C.VISION_HIDDEN if fold_pool_scale else qf.eps
        self.proj_norm = RMSNorm(torch.ones(C.VISION_HIDDEN), eps)
        self.proj = Lin({"w": qf.vec("model.embed_vision.embedding_projection.weight")})
        self.pool_scale = C.VISION_HIDDEN ** 0.5

    def pool(self, h: torch.Tensor) -> torch.Tensor:
        """3×3 のパッチを平均して [1,M,768] にする。パッチ n = y·gw + x、出力 m = ky·(gw/3) + kx（HF と同じ並び）。"""
        gh, gw, P, D = self.gh, self.gw, C.POOL, C.VISION_HIDDEN
        t = h.reshape(gh, gw // P, P, D).mean(dim=2)                            # 横の 3 つを平均 → [gh, gw/3, D]
        t = t.reshape(gh // P, P, gw // P, D).mean(dim=1)                       # 縦の 3 つを平均 → [gh/3, gw/3, D]
        return t.reshape(1, self.M, D)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.run(pixel_values)["image_embeds"]

    def run(self, pixel_values: torch.Tensor, keep: bool = False) -> dict:
        """keep=True なら途中の値（パッチの埋め込み・各層の出力・プーリングの後）も返す（HF との突き合わせ用）。"""
        out = {}
        x = 2.0 * (pixel_values - 0.5)
        h = self.patch_proj(x) + self.pos_emb
        if keep:
            out["patch_embed"] = h
        for i, ly in enumerate(self.layers):
            h = ly(h, self.cos, self.sin)
            if keep:
                out[f"layer_{i}"] = h
        p = self.pool(h)
        if not self.fold_pool_scale:
            p = p * self.pool_scale
        if keep:
            out["pooled"] = p if not self.fold_pool_scale else p * self.pool_scale   # HF の pooler の出力と同じ尺度で返す
        out["image_embeds"] = self.proj(self.proj_norm(p))
        return out

    def qat_linears(self) -> list[dict]:
        """QAT の Linear の一覧（Phase 2-2 で QDQ に書き戻すときの対応表の元）。16 層 × 7 本 = 112 本。"""
        return [m.qat for m in self.modules() if isinstance(m, Lin) and m.qat]
