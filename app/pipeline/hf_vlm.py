# -*- coding: utf-8 -*-
"""PC（transformers）で Gemma 4 E2B の QAT 版を「画像つき」で動かすための共通部品（2026-09-27）。

ref_vlm.py（品質の評価）と check_wiring.py（配線の検算）の両方から使う。

## ここにあるもの

| 関数 | 役割 |
|---|---|
| `load()` | モデルとプロセッサを読む。SRQ（活性の 8bit の丸め）の有無を選べる |
| `prepare_image()` | 画像の前処理の方式を選ぶ。`native`＝HF のまま（縦横比を保って縮小し、余りはパディング）、`fixed`＝端末と同じ固定の格子に収める |
| `build_inputs()` | chat template で文を作り、画像トークン数（70/140/280）を指定してプロセッサに通す |
| `generate_hf()` | HF の本来の経路（pixel_values を渡す）で貪欲に生成する。**品質の正解** |
| `image_features()` | 画像エンコーダ＋射影層だけを通して、画像トークンの埋め込み N×1536 を得る |
| `device_style_embeds()` | 端末と同じ入れ方（inputs_embeds＋ple_tok）を作る |
| `device_style_step()` | 文章側だけを 1 トークンずつ進める（端末の decode グラフと同じ動き） |
| `generate_device_style()` | 端末と同じ入れ方で生成する。プロンプトを 1 トークンずつ（端末と同じ）か一度に食わせる |
| `generate_step()` | 画像の特徴を取り出して `generate_device_style(prefill="step")` を呼ぶ（ref_vlm の `--feed step`） |

## 「QAT 版を PC で動かす」とは

`quant_method: gemma` の重みは 2/4/8bit の整数＋スケールで入っていて、transformers が層ごとにその場で浮動小数へ戻して計算する。
各 Linear の入力と出力は、QAT で学習した目盛りで 8bit に丸める（SRQ）。端末も同じ目盛りを使うので、
**SRQ ありの PC の結果が「端末で目指す答え」**になる（EdgeSLM_AndroidGemma4 P0B と同じ考え方）。
"""
from __future__ import annotations

import time
from typing import Any

from PIL import Image

from app.pipeline import vlm_common as C

LM_HEAD_CHUNK = 32768    # 出力層を行ごとに分けて計算する幅（一度に戻すと 262144×1536×4B = 1.6GB 要る）


# ---------------------------------------------------------------------------
# 読み込み
# ---------------------------------------------------------------------------
def load(device: str = "cuda", srq: bool = True):
    """モデルとプロセッサを読む。

    - dtype は float32（端末の検算の基準にするので、bf16 の丸めを混ぜない）
    - srq=False のときは、活性のスケールを 0 にする（apply_srq はスケール 0 を「較正していない」とみなして何もしない）
    - 出力層（lm_head、2bit）は行を分けて計算する。中身は QuantizedLinear.forward と同じ（SRQ → 整数×スケール → SRQ）。
      この部分は EdgeSLM_AndroidGemma4/app/pipeline/ref_qat.py の load() を複製したもの（DoC/引き継ぎ一覧.md）
    """
    C.ensure_hf_env()
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor
    from transformers.integrations.gemma_quant import _unpack_int2, _unpack_int4, apply_srq

    # GPU の行列積で TF32（精度を落として速くする機能）を使わない。検算で差が出る原因になるため
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    t0 = time.time()
    proc = AutoProcessor.from_pretrained(str(C.HF_DIR), local_files_only=True)
    model = AutoModelForMultimodalLM.from_pretrained(str(C.HF_DIR), dtype=torch.float32, local_files_only=True)
    model.eval()

    n_zero = disable_srq(model) if not srq else 0

    head = model.lm_head
    bits = getattr(head, "num_bits", None)

    def head_forward(x):
        x = apply_srq(x, head.input_activation_scale)
        outs = []
        for s in range(0, head.out_features, LM_HEAD_CHUNK):
            e = min(s + LM_HEAD_CHUNK, head.out_features)
            packed = head.weight[s:e]
            q = _unpack_int2(packed, head.in_features) if bits == 2 else \
                _unpack_int4(packed, head.in_features) if bits == 4 else packed
            w = q.to(x.dtype) * head.weight_scale[s:e].to(x.dtype)
            outs.append(torch.nn.functional.linear(x, w))
        return apply_srq(torch.cat(outs, dim=-1), head.output_activation_scale)

    if bits in (2, 4, 8):
        head.forward = head_forward

    model.to(device)
    print(f"ロード {time.time() - t0:.0f}s  device={device}  SRQ={'あり' if srq else 'なし'}（0 にしたスケール {n_zero}本）"
          f"  lm_head={bits}bit", flush=True)
    return model, proc


def disable_srq(model) -> int:
    """SRQ（活性の 8bit の丸め）を外す。活性のスケールを 0 にすると apply_srq は何もしない（「較正していない」扱い）。

    スケールが Parameter でもバッファでも 0 にする。0 にした数を返し、**1 本も見つからなければ止める**
    （名前が変わっていると黙って SRQ ありのまま測ってしまうため）。元に戻せないので、読み直さない限り SRQ なしのまま。
    """
    import torch
    n = 0
    with torch.no_grad():
        for m in model.modules():
            for nm in ("input_activation_scale", "output_activation_scale"):
                p = getattr(m, nm, None)
                if isinstance(p, torch.Tensor):
                    p.zero_()
                    n += 1
    assert n > 0, "活性のスケールが見つからない（transformers の gemma_quant の名前が変わった？）"
    return n


# ---------------------------------------------------------------------------
# 画像の前処理
# ---------------------------------------------------------------------------
def prepare_image(img: Image.Image, prep: str, budget: int, grid: tuple[int, int] | None = None) -> Image.Image:
    """画像の前処理の方式を選ぶ。

    native: そのまま返す。HF のプロセッサが「縦横比を保ち、パッチ数が budget×9 以下になる最大の大きさ」に拡大・縮小し、
            足りない分をパディング（位置 -1）で埋める。画像ごとにパッチの並びが変わる＝**形が固定されない**
    fixed:  端末と同じ固定の格子（vlm_common.FIXED_GRIDS、例: 70 → 21×30 パッチ = 336×480px）に収める。
            縦横比を保って拡大・縮小し、余白は黒で埋める（レターボックス）。プロセッサから見ると「ちょうどの大きさ」なので
            大きさの変更もパディングも起きず、パッチ数は常に budget×9 になる

    grid=(gh, gw) を渡すと、fixed のときその格子に収める（既定は FIXED_GRIDS[budget]。縦長の格子 PORTRAIT_GRIDS 用）。

    どちらも、先に EXIF の向きを反映する（スマホの縦写真は、画素は横のまま「回転して表示せよ」という印だけ付いていることがある。
    HF はファイルから読むときだけこれを反映し、PIL の画像を渡したときは反映しない）。透明部分は白で埋める。
    """
    from PIL import ImageOps
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[-1])
        img = bg
    img = img.convert("RGB")
    if prep == "native":
        return img
    if prep != "fixed":
        raise ValueError(prep)
    gh, gw = grid or C.FIXED_GRIDS[budget]
    H, W = gh * C.PATCH, gw * C.PATCH
    s = min(W / img.width, H / img.height)
    nw, nh = max(1, round(img.width * s)), max(1, round(img.height * s))
    canvas = Image.new("RGB", (W, H), (0, 0, 0))
    canvas.paste(img.resize((nw, nh), Image.BICUBIC), ((W - nw) // 2, (H - nh) // 2))
    return canvas


def pixels_fixed_np(img_fixed: Image.Image, grid: tuple[int, int]):
    """固定の格子に収めた画像（prepare_image の fixed の出力）→ pixel_values [N,768] float32 を、**HF のプロセッサを通さずに** numpy で作る。

    なぜ要るか（2026-09-28）: HF の Gemma4Processor は torchvision を import するが、EdgeSLM_Android の venv（ONNX の評価に使う）には
    torchvision が無い（venv にパッケージは足さない約束）。画像がちょうど格子の大きさなら、プロセッサがやることは次の 2 つだけなので、
    同じことを numpy で書く（transformers 5.14.1 の image_processing_gemma4.py・image_processing_backends.py を読んで確かめた）:

    1. 大きさの変更: しない（`aspect_ratio_preserving_resize` は目標の大きさと同じなら画像をそのまま返す）
    2. `rescale`: `uint8 の画素 × rescale_factor`（1/255。torch では float32 で掛け算）。正規化は無し（do_normalize=false、平均 0・標準偏差 1）
    3. `convert_image_to_patches`: (C,H,W) → (縦のパッチ, 横のパッチ, パッチ内の y, パッチ内の x, C) → [N, 768]。パッチの番号は y·横の数 + x

    端末のアプリも同じ前処理を書くことになるので、その手本にもなる。**HF と同じ値になることは、使う側（eval_qdq_vit feats）が
    capture_vit で HF から取った pixel_values と比べて毎回確かめる**（食い違えば止まる）。
    """
    import json
    import numpy as np
    cfg = json.loads((C.HF_DIR / "processor_config.json").read_text(encoding="utf-8"))
    cfg = cfg.get("image_processor", cfg)
    # この関数が前提にしている設定から外れていたら止める（モデルの版が変わったときに黙って違う値を作らない）
    assert cfg.get("do_rescale", True) and not cfg.get("do_normalize", False), f"前処理の設定が想定と違う: {cfg}"
    assert cfg["patch_size"] == C.PATCH, cfg
    gh, gw = grid
    x = np.asarray(img_fixed.convert("RGB"), dtype=np.uint8)            # [H,W,3]
    assert x.shape == (gh * C.PATCH, gw * C.PATCH, 3), f"格子の大きさでない: {x.shape}（prepare_image の fixed を通すこと）"
    x = x.astype(np.float32) * np.float32(cfg["rescale_factor"])        # torch の「uint8 × Python の float」と同じく float32 で掛ける
    x = x.reshape(gh, C.PATCH, gw, C.PATCH, 3).transpose(0, 2, 1, 3, 4)  # (縦, py, 横, px, c) → (縦, 横, py, px, c)
    return np.ascontiguousarray(x.reshape(gh * gw, C.PATCH * C.PATCH * 3))


_KEEP = ("input_ids", "attention_mask", "pixel_values", "image_position_ids", "mm_token_type_ids")
_dropped_shown = False


def build_inputs(proc, img: Image.Image, question: str, budget: int, system: str | None, device: str,
                 expect_exact: bool = False):
    """chat template で文を作り、画像トークン数を指定してプロセッサに通す。

    文の形（Gemma 4 の chat template）:
        <bos><|turn>system\\n{system}<turn|>\\n<|turn>user\\n<|image|>{質問}<turn|>\\n<|turn>model\\n
    プロセッサが <|image|> を <|image>（始まり）＋ 画像トークン×N ＋ <image|>（終わり）に置き換える。

    ★ chat template が <bos> を既に書くので、トークン化では特別なトークンを足さない（add_special_tokens=False）。
      足すと <bos><bos> で始まる、端末では使わない形のプロンプトになる。先頭が <bos> 1 つであることを確かめる。
    expect_exact=True（fixed の前処理）のときは、画像トークン数がちょうど budget であることも確かめる。
    """
    global _dropped_shown
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})        # 文字列で渡す（リストだと末尾に空白が付く）
    msgs.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]})
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[[img]], return_tensors="pt", add_special_tokens=False,
                  images_kwargs={"max_soft_tokens": budget})
    ids = inputs["input_ids"][0]
    bos = proc.tokenizer.bos_token_id
    assert int(ids[0]) == bos and int(ids[1]) != bos, f"先頭の <bos> が 1 つではない: {ids[:3].tolist()}"
    n_img = int((ids == C.IMAGE_TOKEN_ID).sum())
    # 指定した上限を超えていないか（kwargs の渡し方を間違えると黙って既定の 280 になるので、ここで必ず確かめる）
    assert 0 < n_img <= budget, f"画像トークン数 {n_img} が上限 {budget} と合わない"
    if expect_exact:
        assert n_img == budget, f"固定の格子なのに画像トークン数が {n_img}（期待 {budget}）。プロセッサが大きさを変えた？"
    dropped = [k for k in inputs.keys() if k not in _KEEP]
    if dropped and not _dropped_shown:
        print(f"  （プロセッサの出力のうち使わないもの: {dropped}）", flush=True)
        _dropped_shown = True
    keep = {k: v for k, v in inputs.items() if k in _KEEP}
    return {k: v.to(device) for k, v in keep.items()}, n_img, text


def n_valid_patches(inputs: dict) -> int:
    """パディングを除いた、実際の画像のパッチ数（位置が -1 のものはパディング）。"""
    return int((inputs["image_position_ids"][0, :, 0] >= 0).sum())


# ---------------------------------------------------------------------------
# HF の本来の経路で生成（品質の正解）
# ---------------------------------------------------------------------------
def generate_hf(model, inputs: dict, max_new: int) -> dict:
    """貪欲に生成する。1 回目は画像ごと全部を食わせ、2 回目からは選んだトークンを 1 つずつ食わせる。

    止めるトークン（<turn|> など）が出たら止める。最後に選んだ 1 つは食わせない（端末の手順と同じ）。
    """
    import torch
    sync = (lambda: torch.cuda.synchronize()) if inputs["input_ids"].is_cuda else (lambda: None)
    gen, pkv = [], None
    t0 = time.time()
    feed = dict(inputs)
    t_first = None
    while True:
        with torch.no_grad():
            o = model(**feed, past_key_values=pkv, use_cache=True)
        pkv = o.past_key_values
        nid = int(o.logits[0, -1].float().argmax())
        if t_first is None:
            sync()
            t_first = time.time() - t0
        gen.append(nid)
        if nid in C.STOP_IDS or len(gen) >= max_new:
            break
        feed = {"input_ids": torch.tensor([[nid]], device=inputs["input_ids"].device)}
    sync()
    return {"ids": gen, "t_first": round(t_first, 3), "t_total": round(time.time() - t0, 3)}


def decode(proc, ids: list[int]) -> str:
    return proc.tokenizer.decode([i for i in ids if i not in C.STOP_IDS], skip_special_tokens=False).strip()


# ---------------------------------------------------------------------------
# 端末と同じ入れ方（配線の検算・値域の確認で使う）
# ---------------------------------------------------------------------------
def image_features(model, inputs: dict):
    """画像エンコーダ（ViT 16 層）→ 3×3 平均プーリング → ×√768 → 射影層（RMSNorm＋768→1536）。

    戻り値は [N, 1536]（N＝画像トークン数。パディングの分は HF が取り除く）。
    これが端末で「プロセス A（画像エンコーダ）」が出して、プロセス B（文章側）へ渡すものになる。
    """
    import torch
    with torch.no_grad():
        out = model.model.get_image_features(inputs["pixel_values"], inputs["image_position_ids"], return_dict=True)
    f = out.pooler_output
    return f.reshape(-1, f.shape[-1])


def device_style_embeds(model, input_ids, feats):
    """端末の文章グラフに渡す 2 つの入力を、HF の部品で作る。

    inputs_embeds [1,T,1536]:
        文章の位置 → 埋め込み表を引く（HF の embed_tokens。√1536 倍まで含む）
        画像の位置 → image_features() の出力をそのまま入れる（√1536 倍はしない。HF の masked_scatter と同じ）
    ple_tok [1,T,35,256]:
        PLE の表を引く。画像の位置は ID を pad（0）に置き換えてから引く（HF の Gemma4Model.forward と同じ）

    端末の入口の量子化の範囲で切り詰める実験は check_wiring.py の clip_inputs / entry_clip で行う（全位置に掛ける）。
    """
    import torch
    lm = model.model.language_model
    ids = input_ids[0]
    img_mask = ids == C.IMAGE_TOKEN_ID
    assert int(img_mask.sum()) == feats.shape[0], "画像トークンの数と特徴の数が合わない"
    ids_pad = torch.where(img_mask, torch.full_like(ids, C.PAD_ID), ids)[None]
    with torch.no_grad():
        emb = lm.embed_tokens(ids_pad).clone()
        emb[0, img_mask] = feats.to(emb.dtype)
        ple_tok = lm.get_per_layer_inputs(ids_pad, None)
    return emb, ple_tok, img_mask


def device_style_step(model, emb_t, ple_t, pkv, need_logits: bool = True):
    """文章側を 1 トークン（または数トークン）進めて logits を返す。

    need_logits=False のときは出力層を掛けずに (None, KV) を返す。1 トークンずつのプロンプト処理では最後の位置の
    logits しか使わないので、途中は省く（出力層は 262144 行を毎回その場で戻すので重い。KV には影響しない）。

    端末の decode グラフと同じく、文章側（language_model）だけを通す。PLE の射影（project_per_layer_inputs）は
    language_model の中で inputs_embeds と ple_tok から計算される（端末のグラフの入口でも同じ計算をしている）。
    最後の正規化の後に出力層を掛け、logits の softcap（Gemma の決まり）を掛ける。
    """
    import torch
    lm = model.model.language_model
    with torch.no_grad():
        out = lm(inputs_embeds=emb_t, per_layer_inputs=ple_t, past_key_values=pkv, use_cache=True)
        if not need_logits:
            return None, out.past_key_values
        logits = model.lm_head(out.last_hidden_state)
        cap = model.config.get_text_config().final_logit_softcapping
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap
    return logits[0], out.past_key_values


def generate_device_style(model, emb, ple, max_new: int, prefill: str = "step", transform=None) -> dict:
    """端末と同じ入れ方で貪欲に生成する（2026-09-28 追加）。

    prefill="step":  プロンプトも 1 トークンずつ食わせる。**端末の文章グラフと同じ流し方**（いまは 1 回 1 トークン）
    prefill="batch": プロンプトを一度に食わせる（HF の generate_hf と同じ流し方。差は入れ方だけで、logits は完全に一致する）

    生成に入ってからは、選んだトークンの埋め込みと PLE を表から引いて 1 つずつ食わせる（どちらの prefill でも同じ）。
    transform(e, p) -> (e, p) を渡すと、食わせる直前の inputs_embeds と ple_tok に掛ける（切り詰めの実験用）。
    止めるトークンが出たら止め、最後に選んだ 1 つは食わせない（generate_hf と同じ）。

    ### なぜ要るか

    check_wiring で、SRQ ありでは「一度に」と「1 トークンずつ」で logits の 1 位が 77〜89% しか一致しなかった
    （8bit の丸めの境目で転ぶ。SRQ なしでは 100% 一致）。端末は 1 トークンずつなので、**その流し方で答えの品質が
    保たれるか**を、生成そのもので確かめる（EdgeSLM_AndroidGemma4 P2 では、1 位の一致 75% でも生成は保たれた）。
    """
    import torch
    lm = model.model.language_model
    dev = emb.device
    sync = (lambda: torch.cuda.synchronize()) if emb.is_cuda else (lambda: None)
    tf = transform or (lambda e, p: (e, p))
    t0 = time.time()
    e, p = tf(emb, ple)
    if prefill == "step":
        pkv, T = None, e.shape[1]
        assert T > 0, "プロンプトが空"
        for t in range(T):
            logits, pkv = device_style_step(model, e[:, t:t + 1], p[:, t:t + 1], pkv, need_logits=(t == T - 1))
    elif prefill == "batch":
        logits, pkv = device_style_step(model, e, p, None)
    else:
        raise ValueError(prefill)
    nid = int(logits[-1].float().argmax())
    sync()
    t_first = time.time() - t0
    gen = []
    while True:
        gen.append(nid)
        if nid in C.STOP_IDS or len(gen) >= max_new:
            break
        ids = torch.tensor([[nid]], device=dev)
        with torch.no_grad():
            e1 = lm.embed_tokens(ids)
            p1 = lm.get_per_layer_inputs(ids, None)
        e1, p1 = tf(e1, p1)
        logits, pkv = device_style_step(model, e1, p1, pkv)
        nid = int(logits[-1].float().argmax())
    sync()
    return {"ids": gen, "t_first": round(t_first, 3), "t_total": round(time.time() - t0, 3)}


def generate_step(model, inputs: dict, max_new: int) -> dict:
    """ref_vlm の --feed step 用: 画像の特徴を取り出し、端末と同じ入れ方・流し方（1 トークンずつ）で生成する。

    t_first には画像エンコーダの時間も含む（generate_hf と同じ数え方）。
    """
    t0 = time.time()
    feats = image_features(model, inputs)
    emb, ple, _ = device_style_embeds(model, inputs["input_ids"], feats)
    t_vis = time.time() - t0
    g = generate_device_style(model, emb, ple, max_new, prefill="step")
    g["t_first"] = round(g["t_first"] + t_vis, 3)
    g["t_total"] = round(g["t_total"] + t_vis, 3)
    return g


def generate_with_features(model, inputs: dict, feats, max_new: int, prefill: str = "batch") -> dict:
    """画像の特徴を**外から与えて**生成する（2026-09-28 追加）。

    feats [M,1536] を画像の位置の inputs_embeds に入れ、文章側だけで生成する（device_style_embeds と同じ入れ方）。
    Phase 1 の check_wiring で、HF の特徴を入れれば HF の本来の経路と logits がビット単位で一致すると確かめてある。
    使い道: 自前の配線（vit_graph）の特徴・量子化した画像エンコーダの特徴・端末の特徴で、答えの品質がどう変わるかを見る。
    """
    emb, ple, _ = device_style_embeds(model, inputs["input_ids"], feats)
    return generate_device_style(model, emb, ple, max_new, prefill=prefill)


def tensor_stats(x) -> dict[str, Any]:
    """値域の要約（最小・最大・分位点・RMS）。

    分位点は、要素が多いときは 20 万個を抜き出して求める（torch.quantile の大きさの上限のため）。
    抜き出しは乱数の種を固定して、何度動かしても同じ値になるようにする。最小・最大・RMS は全要素から求める。
    """
    import torch
    x = x.detach().float().flatten()
    if x.numel() > 200000:
        g = torch.Generator().manual_seed(0)
        idx = torch.randperm(x.numel(), generator=g)[:200000].to(x.device)
        xs = x[idx]
    else:
        xs = x
    q = torch.quantile(xs, torch.tensor([0.001, 0.01, 0.5, 0.99, 0.999], device=x.device))
    return {"min": round(float(x.min()), 5), "max": round(float(x.max()), 5), "rms": round(float(x.pow(2).mean().sqrt()), 5),
            "p0.1": round(float(q[0]), 5), "p1": round(float(q[1]), 5), "p50": round(float(q[2]), 5),
            "p99": round(float(q[3]), 5), "p99.9": round(float(q[4]), 5)}
