# -*- coding: utf-8 -*-
"""Gemma 4 E2B のテキスト部分の decode（1トークン）を、NPU 向けに組み直す（Phase 0-D〜）。

## 重み

QAT モバイル版のファイルから**必要な層だけ**を直接読み、整数×スケールの float32 に戻す（transformers は通さない。
PC のメモリを食わず、HF の実装とは独立に書いた配線を HF と突き合わせて確かめられる）。
整数とスケールも持っておき、量子化後の QDQ へ**そのまま書き戻す**（qat_inject.py）。

## 配線（HF の Gemma4TextDecoderLayer と同じ順。transformers 5.14.1 で確認）

    注意:   h = h + post_attention_layernorm( o_proj( attn( input_layernorm(h) ) ) )
    MLP:    h = h + post_feedforward_layernorm( down( gelu_tanh(gate(x)) * up(x) ) ),  x = pre_feedforward_layernorm(h)
    PLE:    h = h + post_per_layer_input_norm( per_layer_projection( gelu_tanh(per_layer_input_gate(h)) * ple_i ) )
    最後:   h = h * layer_scalar

注意の中: q = RoPE(q_norm(q_proj(x)))、k = RoPE(k_norm(k_proj(x)))、v = v_norm(v_proj(x))（v_norm は重みなし）、
スケールは 1.0。KV を使い回す層（15〜34）は k/v を作らず、同じ種類の最後の層（局所は13、全体は14）の K/V を使う。

## I/O の約束（Qwen の export_qwen_decode.py と同じ考え方）

- KV キャッシュは長さ L 固定、毎ステップ1つ左へずらして末尾に書く: `present = cat(past, new)[:, :, 1:, :]`
  （past を Concat に入れる順にしている理由は Qwen の注記: Slice に直接入る float 入力を ORT が量子化しない）
- マスク `attn_mask` [1,1,1,L]（0 または MASK_NEG。C-9: −inf は量子化できない）
- cos/sin は PC で作って入力で渡す。局所の層 [1,1,1,256]、全体の層 [1,1,1,512]（p-RoPE。回さない次元は cos=1・sin=0）
- **MQA**: KV は1本なので、Q の8ヘッドを8行に並べて1回の MatMul にする（[1,1,8,hd] @ [1,1,hd,L]）。KV を8本へ複製しない
- L ≤ 512（局所の窓）なら、局所と全体の層でマスクを分けなくてよい

## 大きさ（config.json の text_config。2026-10-18 追記）

hidden 1536・層 35（局所 sliding と全体 full が 4:1 で並ぶ。層 4・9・14… が全体）・Q のヘッド 8・KV のヘッド 1・
head_dim 256（全体の層は 512）・MLP 6144（KV を使い回す層 15〜34 は倍の 12288）・PLE 256・語彙 262144・局所の窓 512。

## 使い方（2026-10-18 追記）

直接は動かさない。export_small（Phase 0-D、DecodeLayers）・export_chunk（Phase 1〜2、DecodeChunk）・qat_inject（重み）・
gen_pc（rope_tables）などが import する。解説は StudyMemo（EdgeSLM_AndroidGemma4）01「Gemma4E2B のつくり」・
02「QAT の丸めを端末で再現する」・03「35 層をかたまりで作って繋ぐ」。
"""
from __future__ import annotations

import json

import torch
import torch.nn.functional as F
from safetensors import safe_open

from app.pipeline import gemma4_common as G

MASK_NEG = -60.0                   # 空きスロットを塞ぐ値（Qwen と同じ。exp(−60) ≒ 0。C-9: −inf は量子化できない）
# safetensors の中のテキスト部分の重みの名前の頭（画像・音声の部品は別の頭）
PFX = "model.language_model."
GELU_C = 0.7978845608028654        # sqrt(2/pi)


def gelu_tanh(x: torch.Tensor) -> torch.Tensor:
    """GELU の tanh 近似: 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))。Gemma の MLP と PLE の門で使う。"""
    # ★Pow を使わない（前処理の RMSNorm の見分けが「Pow(x,2) → ReduceMean」を探すため、紛れないように）
    return 0.5 * x * (1.0 + torch.tanh(GELU_C * (x + 0.044715 * x * x * x)))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """最後の軸を前半・後半に分け、[−後半, 前半] に並べ替える（HF の rotate_half と同じ）。"""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """RoPE（回転位置埋め込み）: x·cos ＋ rotate_half(x)·sin。cos / sin は x の形に合わせて広がる（ヘッドの軸は 1）。"""
    return x * cos + rotate_half(x) * sin


def unpack(packed: torch.Tensor, bits: int, in_features: int) -> torch.Tensor:
    """transformers の gemma_quant.py と同じ規則（符号なしで詰め、4bit は −8、2bit は −2 ずらす）。[out, in] int8。

    packed [out, in×bits/8] の uint8 を、1 バイトの下の bit から順に取り出す（4bit は 2 個、2bit は 4 個）。
    値の範囲: 4bit は −8〜7、2bit は −2〜1。in_features で切るのは、詰めるときの端数の分を捨てるため。
    """
    p = packed.to(torch.uint8)
    if bits == 4:
        v = [(p & 0x0F).to(torch.int8) - 8, (p >> 4).to(torch.int8) - 8]
    elif bits == 2:
        v = [((p >> s) & 0x03).to(torch.int8) - 2 for s in (0, 2, 4, 6)]
    else:
        raise ValueError(bits)
    return torch.stack(v, dim=-1).reshape(p.shape[0], -1)[:, :in_features].contiguous()


class QatFile:
    """QAT モバイル版のファイルから、名前を指定して必要なテンソルだけを読む。"""

    def __init__(self):
        """model.safetensors を開く（中身はまだ読まない。get_tensor で名前ごとに読む）。cfg は config.json の text_config。"""
        self.f = safe_open(str(G.HF_DIR / "model.safetensors"), framework="pt")
        self.cfg = json.loads((G.HF_DIR / "config.json").read_text(encoding="utf-8"))["text_config"]
        self.eps = self.cfg["rms_norm_eps"]

    def t(self, name: str) -> torch.Tensor:
        """名前を指定してテンソルを 1 つ読む（読んだ型のまま）。"""
        return self.f.get_tensor(name)

    def linear(self, base: str, in_features: int) -> dict:
        """整数・スケール・bit と、float32 に戻した重み [out, in]。

        bit は詰めた幅から決める: int8 ならそのまま 8bit、幅が入力の 1/4 なら 2bit（出力層）、1/2 なら 4bit。
        戻り値: name・codes（整数 [out, in]）・scale（出力チャネルごと [out]）・bits・w（codes×scale）・
                in_act / out_act（QAT の活性の丸めのスケール。0 は「較正していない」印）
        """
        w = self.t(base + ".weight")
        s = self.t(base + ".weight_scale").float().reshape(-1)
        if w.dtype == torch.int8:
            q, bits = w, 8
        elif w.shape[1] * 4 == in_features:
            q, bits = unpack(w, 2, in_features), 2
        elif w.shape[1] * 2 == in_features:
            q, bits = unpack(w, 4, in_features), 4
        else:
            raise ValueError(f"{base}: bit を決められない（形 {list(w.shape)}、入力 {in_features}）")
        return {"name": base, "codes": q, "scale": s, "bits": bits, "w": q.float() * s[:, None],
                "in_act": float(self.t(base + ".input_activation_scale")),
                "out_act": float(self.t(base + ".output_activation_scale"))}

    def vec(self, name: str) -> torch.Tensor:
        """量子化されていないテンソル（正規化の重みなど）を float32 で読む。"""
        return self.t(name).float()

    def layer_info(self, i: int) -> dict:
        """層 i の性質を config から決める。

        戻り値: type（sliding_attention / full_attention）・full・shared（KV を自分で作らず使い回す層か。層 15〜34）・
                kv_src（使い回す K/V の出どころ。15 より前の、同じ種類の最後の層＝局所 13・全体 14）・
                hd（局所 256・全体 512）・inter（MLP の中間の幅。使い回す層は use_double_wide_mlp で倍）
        """
        c = self.cfg
        typ = c["layer_types"][i]
        full = typ == "full_attention"
        first_shared = c["num_hidden_layers"] - c["num_kv_shared_layers"]
        shared = i >= first_shared
        prev = c["layer_types"][:first_shared]
        src = (len(prev) - 1 - prev[::-1].index(typ)) if shared else i      # 使い回す K/V の出どころ
        hd = c["global_head_dim"] if full else c["head_dim"]
        inter = c["intermediate_size"] * (2 if (shared and c.get("use_double_wide_mlp")) else 1)
        return {"i": i, "type": typ, "full": full, "shared": shared, "kv_src": src, "hd": hd, "inter": inter}


class RMSNorm(torch.nn.Module):
    """x * rsqrt(mean(x²)+eps) * w。**書き出すと Pow(x,2) → ReduceMean → Add → Sqrt → Reciprocal → Mul の形になる**
    （前処理 normalize_rmsnorm_input が C-11 の書き換えをかけ、QAIRT が RmsNorm に融合する形。Qwen と同じ）。"""

    def __init__(self, w: torch.Tensor, eps: float):
        """w = 正規化の重み [次元]（Gemma の値をそのまま掛ける）、eps = 0 割りを防ぐ小さな値（config の rms_norm_eps）。"""
        super().__init__()
        self.weight = torch.nn.Parameter(w.clone(), requires_grad=False)
        self.eps = eps

    def forward(self, x):
        """最後の軸で 2 乗の平均を取り、その平方根で割って重みを掛ける（形は変わらない）。"""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


SRQ = False          # ★確認用。True にすると QAT の活性の 8bit の丸めを真似る（ONNX へ書き出すときは False のまま）


def apply_srq(x: torch.Tensor, scale: float) -> torch.Tensor:
    """QAT の活性の丸め（transformers の `apply_srq` と同じ。scale 0 は「較正していない」＝何もしない）。"""
    if not scale:
        return x
    return torch.clamp(torch.round(x / scale), -128.0, 127.0) * scale


class Lin(torch.nn.Module):
    """バイアスの無い線形層（y = x·Wᵀ）。重みは QatFile.linear の w（float32 に戻した値）。
    qat に名前・bit・活性のスケールを持たせておき、qat_inject が量子化後の ONNX に書き戻すときの目印にする。
    SRQ=True のときだけ、入力と出力に QAT の 8bit の丸めを掛ける（PC での確認用）。"""

    def __init__(self, d: dict):
        """d は QatFile.linear の戻り値。重み w [out, in] を学習しない Parameter として持つ。"""
        super().__init__()
        self.weight = torch.nn.Parameter(d["w"].clone(), requires_grad=False)
        self.qat = {k: d[k] for k in ("name", "bits", "in_act", "out_act")}

    def forward(self, x):
        """x [.., in] → [.., out]。SRQ=True なら入力と出力に QAT の 8bit の丸めを掛ける（スケール 0 の本は丸めない）。"""
        if SRQ:
            y = F.linear(apply_srq(x, self.qat["in_act"]), self.weight)
            return apply_srq(y, self.qat["out_act"])
        return F.linear(x, self.weight)


class DecoderLayer(torch.nn.Module):
    """Gemma 4 の 1 層（冒頭の「配線」の図のとおり）。重みは QatFile から層 i の分だけ読む。
    KV を使い回す層（shared）は k_proj・v_proj・k_norm を持たない。"""

    def __init__(self, qf: QatFile, i: int):
        """層 i の部品を作る: 正規化 5 つ（注意の前後・MLP の前後・PLE の後）、q/o の射影と q_norm、
        KV を自分で作る層なら k/v の射影と k_norm・v_norm（重みなし）、MLP の 3 本、PLE の門と射影、層ごとの倍率 layer_scalar。
        各 Lin の第 2 引数は入力の幅（o_proj は 8×hd、down_proj は MLP の中間の幅、PLE の射影は 256）。
        """
        super().__init__()
        info = qf.layer_info(i)
        self.info = info
        c, eps, H = qf.cfg, qf.eps, qf.cfg["hidden_size"]
        nh, hd = c["num_attention_heads"], info["hd"]
        p = f"{PFX}layers.{i}."
        self.nh, self.hd = nh, hd
        self.input_layernorm = RMSNorm(qf.vec(p + "input_layernorm.weight"), eps)
        self.post_attention_layernorm = RMSNorm(qf.vec(p + "post_attention_layernorm.weight"), eps)
        self.pre_feedforward_layernorm = RMSNorm(qf.vec(p + "pre_feedforward_layernorm.weight"), eps)
        self.post_feedforward_layernorm = RMSNorm(qf.vec(p + "post_feedforward_layernorm.weight"), eps)
        self.post_per_layer_input_norm = RMSNorm(qf.vec(p + "post_per_layer_input_norm.weight"), eps)
        self.q_proj = Lin(qf.linear(p + "self_attn.q_proj", H))
        self.q_norm = RMSNorm(qf.vec(p + "self_attn.q_norm.weight"), eps)
        if not info["shared"]:
            self.k_proj = Lin(qf.linear(p + "self_attn.k_proj", H))
            self.v_proj = Lin(qf.linear(p + "self_attn.v_proj", H))
            self.k_norm = RMSNorm(qf.vec(p + "self_attn.k_norm.weight"), eps)
            self.v_norm = RMSNorm(torch.ones(hd), eps)          # 重みなし（with_scale=False）
        self.o_proj = Lin(qf.linear(p + "self_attn.o_proj", nh * hd))
        self.gate_proj = Lin(qf.linear(p + "mlp.gate_proj", H))
        self.up_proj = Lin(qf.linear(p + "mlp.up_proj", H))
        self.down_proj = Lin(qf.linear(p + "mlp.down_proj", info["inter"]))
        self.per_layer_input_gate = Lin(qf.linear(p + "per_layer_input_gate", H))
        self.per_layer_projection = Lin(qf.linear(p + "per_layer_projection", c["hidden_size_per_layer_input"]))
        self.register_buffer("layer_scalar", qf.vec(p + "layer_scalar").reshape(1))

    def forward(self, h, ple_i, cos, sin, attn_mask, past_k=None, past_v=None, shared_kv=None):
        """h [1,T,H]・ple_i [1,T,256]・cos/sin [1,T,1,hd]・attn_mask [1,T,1,L]。
        戻り値: (h, present_k, present_v)。KV を使い回す層は shared_kv=(K, V) を受け取り、present は None。

        T は「1回で食べるトークンの数」。生成は T=1、入力の文を食べる（プレフィル）ときは T>1 にする。
        """
        # ① 注意の前半: 正規化 → q（q_norm → RoPE）。KV を自分で作る層は k・v も作り、キャッシュを左へずらして末尾に書く
        T = h.shape[1]
        res = h
        x = self.input_layernorm(h)
        q = self.q_proj(x).view(1, T, self.nh, self.hd)          # 8ヘッドを8行に並べる（MQA）
        q = rope(self.q_norm(q), cos, sin)                      # [1,T,8,hd]
        if shared_kv is None:
            k = rope(self.k_norm(self.k_proj(x).view(1, T, 1, self.hd)), cos, sin)    # [1,T,1,hd]
            v = self.v_norm(self.v_proj(x).view(1, T, 1, self.hd))
            # キャッシュは [1,1,L,hd]（頭の軸が1）なので、T の軸と入れ替えてから末尾に足す
            pk = torch.cat((past_k, k.transpose(1, 2)), dim=2)[:, :, T:, :]     # [1,1,L,hd]
            pv = torch.cat((past_v, v.transpose(1, 2)), dim=2)[:, :, T:, :]
        else:
            pk, pv = shared_kv
        # ② スコア = q·Kᵀ ＋ マスク。Gemma 4 は 1/√hd を掛けない（スケール 1.0。q_norm・k_norm で大きさをそろえているためと推定）
        s = torch.matmul(q, pk.transpose(2, 3)) + attn_mask     # [1,1,8,L]。スケール 1.0
        # ★確率×V は**形を変えた V**（[1,L,hd]）に掛ける。V のキャッシュ（グラフの出力）と行列積の右辺を別のテンソルにし、
        #   右辺だけを 8bit に落とせるようにする（C-8: v68 の act×act MatMul は u16×u16 が通らない）。
        #   同じテンソルのままだと、後処理（downgrade_actact_rhs）が見つけられず、見つけても **V のキャッシュ自体が 8bit** になる。
        #   K は転置を挟むので最初から別のテンソルになっている（2026-09-20）
        L = pv.shape[2]
        p = torch.softmax(s, dim=-1).reshape(1, T * self.nh, L)   # [1,T*8,L]
        ctx = torch.matmul(p, pv.reshape(1, L, self.hd))         # [1,T*8,hd]
        # ③ o_proj → 注意の後の正規化 → 残差に足す（Gemma は足す前にも正規化を挟む＝サンドイッチ型）
        h = res + self.post_attention_layernorm(self.o_proj(ctx.reshape(1, T, self.nh * self.hd)))
        res = h
        # ④ MLP: 正規化 → gelu_tanh(gate)·up → down → 正規化 → 残差に足す
        x = self.pre_feedforward_layernorm(h)
        h = res + self.post_feedforward_layernorm(self.down_proj(gelu_tanh(self.gate_proj(x)) * self.up_proj(x)))
        res = h
        # ⑤ PLE（層ごとの埋め込み）: h から門を作り、層 i の PLE の値 ple_i [1,T,256] と掛けて H に戻して足す。最後に層ごとの倍率
        g = gelu_tanh(self.per_layer_input_gate(h)) * ple_i
        h = res + self.post_per_layer_input_norm(self.per_layer_projection(g))
        h = h * self.layer_scalar
        if shared_kv is None:
            return h, pk, pv
        return h, None, None


class DecodeLayers(torch.nn.Module):
    """指定した連続する層だけの decode（Phase 0-D の小さなグラフ用）。

    入力:  hidden_in [1,1,H]、ple_{i} [1,1,256]（層ごとの PLE の値）、cos_s/sin_s [1,1,1,256]、cos_g/sin_g [1,1,1,512]、
           attn_mask [1,1,1,L]、past_k_{i}/past_v_{i}（KV を自分で作る層だけ）
    出力:  hidden_out [1,1,H]、present_k_{i}/present_v_{i}

    ★PLE の値は**層ごとに別の入力**にする。1つの入力 [1,1,n,256] から層ごとに取り出すと、グラフの入力が Gather に
      直接入り、ORT が量子化を挟まず float のまま流す（C-1 違反。2026-09-20 に踏んだ。Qwen の Slice と同じ落とし穴）
    """

    def __init__(self, qf: QatFile, layers: list[int]):
        """layers の層を並べる。使い回す層が読む K/V の出どころ（kv_src）がこのグラフの中に無ければ止める
        （DecodeLayers は外から K/V を受け取る入力を持たないため。それができるのは DecodeChunk）。
        """
        super().__init__()
        self.layer_ids = list(layers)
        self.layers = torch.nn.ModuleList([DecoderLayer(qf, i) for i in layers])
        self.kv_layers = [ly.info["i"] for ly in self.layers if not ly.info["shared"]]
        for ly in self.layers:
            if ly.info["shared"] and ly.info["kv_src"] not in self.kv_layers:
                raise ValueError(f"層{ly.info['i']} が使う K/V（層{ly.info['kv_src']}）がこのグラフに無い")

    def input_names(self) -> list[str]:
        """ONNX の入力の名前（forward の引数の順と同じ）。KV は自分で作る層の分だけ。"""
        names = ["hidden_in"] + [f"ple_{i}" for i in self.layer_ids] + ["cos_s", "sin_s", "cos_g", "sin_g", "attn_mask"]
        for i in self.kv_layers:
            names += [f"past_k_{i}", f"past_v_{i}"]
        return names

    def output_names(self) -> list[str]:
        """ONNX の出力の名前（forward の戻り値の順と同じ）。"""
        names = ["hidden_out"]
        for i in self.kv_layers:
            names += [f"present_k_{i}", f"present_v_{i}"]
        return names

    def forward(self, hidden_in, *rest):
        """rest を input_names の順で切り分け（PLE → cos/sin 4 本とマスク → キャッシュ）、層を順に通す。
        局所の層は cos_s/sin_s、全体の層は cos_g/sin_g を使う。作った K/V は kv に置き、使い回す層がそこから読む。
        """
        n_ple = len(self.layer_ids)
        ples = rest[:n_ple]
        cos_s, sin_s, cos_g, sin_g, attn_mask = rest[n_ple:n_ple + 5]
        caches = rest[n_ple + 5:]
        h = hidden_in
        kv, presents, ci = {}, [], 0
        for n, ly in enumerate(self.layers):
            info = ly.info
            cos, sin = (cos_g, sin_g) if info["full"] else (cos_s, sin_s)
            ple_i = ples[n]
            if info["shared"]:
                h, _, _ = ly(h, ple_i, cos, sin, attn_mask, shared_kv=kv[info["kv_src"]])
            else:
                h, pk, pv = ly(h, ple_i, cos, sin, attn_mask, caches[ci], caches[ci + 1])
                ci += 2
                kv[info["i"]] = (pk, pv)
                presents += [pk, pv]
        return (h, *presents)


class DecodeChunk(torch.nn.Module):
    """入口（埋め込みの入力と PLE の射影）＋指定した層（Phase 1〜2）。

    入力（`with_ple_proj=True` のとき）:
        inputs_embeds [1,1,H]（√H 倍済み）・ple_tok [1,1,35,256]（表を引いて √256 倍したもの）
        cos_s/sin_s [1,1,1,256]・cos_g/sin_g [1,1,1,512]・attn_mask [1,1,1,L]・past_k_{i}/past_v_{i}
    入力（`with_ple_proj=False`）: hidden_in [1,1,H]・ple_all [1,1,35,256]（前のかたまりが作ったもの）・以下同じ
    出力: hidden_out [1,1,H]・**ple_all**（入口のときだけ。次のかたまりへ渡す）・present_k_{i}/present_v_{i}

    PLE の値の作り方（HF の `project_per_layer_inputs` と同じ）:
        ple_all = ( RMSNorm( per_layer_model_projection(inputs_embeds) * H^-0.5 ) + ple_tok ) * 2^-0.5
    層ごとの取り出しは**グラフの中のテンソル**からなので、C-1（グラフ入力が Gather に直接入る）には当たらない。
    """

    def __init__(self, qf: QatFile, layers: list[int], with_ple_proj: bool = False, with_head: bool = False):
        """layers = このかたまりの層の番号、with_ple_proj = 入口（PLE の射影）を含める、with_head = 最後の正規化と出力層を含める
        （layers が空なら出力層だけのかたまり）。
        """
        super().__init__()
        self.layer_ids = list(layers)
        self.with_ple_proj = with_ple_proj
        self.with_head = with_head                                  # 最後の正規化＋出力層（logits）まで含める
        self.layers = torch.nn.ModuleList([DecoderLayer(qf, i) for i in layers])
        self.kv_layers = [ly.info["i"] for ly in self.layers if not ly.info["shared"]]
        self.n_layers = qf.cfg["num_hidden_layers"]
        # ★このかたまりの中に無い K/V（層13＝局所・層14＝全体）は**入力で受け取る**。
        #   35層を分けて作るとき、後ろのかたまり（層15以降）は前のかたまりが作った K/V を使う
        self.ext_kv = sorted({ly.info["kv_src"] for ly in self.layers
                              if ly.info["shared"] and ly.info["kv_src"] not in self.kv_layers})
        if with_ple_proj:
            w = qf.vec(f"{PFX}per_layer_model_projection.weight")       # [35*256, H]。QAT では量子化されていない
            self.ple_proj = torch.nn.Parameter(w.clone(), requires_grad=False)
            self.ple_norm = RMSNorm(qf.vec(f"{PFX}per_layer_projection_norm.weight"), qf.eps)
            self.ple_in_scale = float(qf.cfg["hidden_size"]) ** -0.5
            self.ple_combine = 2.0 ** -0.5
            self.ple_dim = qf.cfg["hidden_size_per_layer_input"]
        if with_head:
            self.final_norm = RMSNorm(qf.vec(f"{PFX}norm.weight"), qf.eps)
            self.lm_head = Lin(qf.linear("lm_head", qf.cfg["hidden_size"]))
            # 出力の tanh ソフトキャップ（30）は**掛けない**。単調なので argmax は変わらない（貪欲な生成では同じ）

    def input_names(self) -> list[str]:
        """ONNX の入力の名前（forward の引数の順と同じ）: 入口なら inputs_embeds・ple_tok、そうでなければ hidden_in・ple_all、
        続けて cos/sin 4 本・attn_mask・自分の KV（past_*）・前のかたまりの KV（shared_*）。
        """
        if not self.layers:                                          # 出力層だけのかたまり（最後の正規化＋lm_head）
            return ["hidden_in"]
        names = (["inputs_embeds", "ple_tok"] if self.with_ple_proj else ["hidden_in", "ple_all"]) + \
                ["cos_s", "sin_s", "cos_g", "sin_g", "attn_mask"]
        for i in self.kv_layers:
            names += [f"past_k_{i}", f"past_v_{i}"]
        for i in self.ext_kv:
            names += [f"shared_k_{i}", f"shared_v_{i}"]
        return names

    def output_names(self) -> list[str]:
        """ONNX の出力の名前（forward の戻り値の順と同じ）。"""
        names = (["logits"] if self.with_head else ["hidden_out"]) + (["ple_all"] if self.with_ple_proj else [])
        for i in self.kv_layers:
            names += [f"present_k_{i}", f"present_v_{i}"]
        return names

    def forward(self, x0, ple_in=None, cos_s=None, sin_s=None, cos_g=None, sin_g=None, attn_mask=None, *caches):
        """x0 は inputs_embeds（入口）か hidden_in。ple_in は ple_tok（入口）か ple_all。残りは input_names の順。
        出力層だけのかたまりは x0 だけを使う。戻り値は output_names の順。
        """
        if not self.layers:                                          # 出力層だけのかたまり
            return (self.lm_head(self.final_norm(x0)),)
        # ① 入口なら、35 層ぶんの PLE の値 ple_all [1,T,35,256] をグラフの中で作る（クラスの説明の式）
        if self.with_ple_proj:
            proj = F.linear(x0, self.ple_proj) * self.ple_in_scale
            proj = proj.reshape(1, x0.shape[1], self.n_layers, self.ple_dim)
            ple_all = (self.ple_norm(proj) + ple_in) * self.ple_combine
        else:
            ple_all = ple_in
        h = x0
        kv, presents, ci = {}, [], 0
        n_own = 2 * len(self.kv_layers)
        # ② 入力のキャッシュの並び: 自分の past_k/past_v（2×自分の KV の層の数）の後に、前のかたまりの shared_k/shared_v
        for k, i in enumerate(self.ext_kv):                          # 前のかたまりが作った K/V
            kv[i] = (caches[n_own + 2 * k], caches[n_own + 2 * k + 1])
        # ③ 層を順に通す（DecodeLayers.forward と同じ。PLE は ple_all から層の番号で取り出す）
        for ly in self.layers:
            info = ly.info
            cos, sin = (cos_g, sin_g) if info["full"] else (cos_s, sin_s)
            ple_i = ple_all[:, :, info["i"], :]
            if info["shared"]:
                h, _, _ = ly(h, ple_i, cos, sin, attn_mask, shared_kv=kv[info["kv_src"]])
            else:
                h, pk, pv = ly(h, ple_i, cos, sin, attn_mask, caches[ci], caches[ci + 1])
                ci += 2
                kv[info["i"]] = (pk, pv)
                presents += [pk, pv]
        # ④ 出力層まで含むかたまりなら、最後の正規化と lm_head で logits にする
        if self.with_head:
            h = self.lm_head(self.final_norm(h))                    # [1,1,V]
        return (h, ple_all, *presents) if self.with_ple_proj else (h, *presents)


def rope_tables(n_pos: int) -> dict:
    """HF の Gemma4TextRotaryEmbedding から、局所・全体の cos/sin 表 [n_pos, hd] を作る（重みは読まない）。

    戻り値: {"cos_s", "sin_s"}（局所、[n_pos, 256]）と {"cos_g", "sin_g"}（全体、[n_pos, 512]。p-RoPE で回さない次元は cos=1・sin=0）。
    式を自分で書かずに HF の部品を呼ぶのは、写し間違いで静かにずれるのを避けるため（Qwen の rope_cos_sin と同じ考え）。
    x は rotary が dtype を見るためだけの入れ物。
    """
    from transformers import AutoConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding
    cfg = AutoConfig.from_pretrained(str(G.HF_DIR)).text_config
    rot = Gemma4TextRotaryEmbedding(cfg)
    pos = torch.arange(n_pos).reshape(1, -1)
    x = torch.zeros(1, dtype=torch.float32)
    out = {}
    for typ, key in (("sliding_attention", "s"), ("full_attention", "g")):
        c, s = rot(x, pos, layer_type=typ)
        out[f"cos_{key}"], out[f"sin_{key}"] = c[0].float(), s[0].float()
    return out
