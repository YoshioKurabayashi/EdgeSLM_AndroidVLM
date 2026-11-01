# -*- coding: utf-8 -*-
"""Phase 1: 「文章側のグラフは作り直さずに使える」を PC で検算し、画像の特徴の値域を調べる（2026-09-27、09-28 改）。

    python -m app.pipeline.check_wiring                      # 既定の 4 問・70 トークン・SRQ あり→なし
    python -m app.pipeline.check_wiring --srq off            # SRQ なしだけ（切り詰めの影響を丸めの揺れ抜きで測る）
    python -m app.pipeline.check_wiring --ids car_02a --budget 280

## 何を確かめるか（3 つ）

### 1. 配線が同じか（A ↔ B）

- A: HF の本来の経路。`pixel_values` を渡し、HF の中で画像の特徴を埋め込みに差し込む（masked_scatter）
- B: 端末と同じ入れ方。画像の特徴を自分で取り出し、`inputs_embeds`（画像の位置＝特徴、文章の位置＝埋め込み表）と
  `ple_tok`（画像の位置＝pad の行）を作って、**文章側（language_model）だけ**に渡す。形は A と同じ（全トークンを一度に）

A と B の logits が一致すれば、「画像の位置の入れ方」「PLE の扱い」「マスク」「位置」の理解が正しいことになる。
形が同じなので、SRQ ありでも（丸めの境目で転ぶことなく）ほぼ一致するはず。

### 2. 1 トークンずつでも同じか（B ↔ C）

- C: B と同じ入力を、端末の decode グラフと同じく **1 トークンずつ** KV キャッシュを持ち越して流す

B ↔ C の差は「行列の形が違うことによる浮動小数の足し算の順序の差」だけ。SRQ なしならほぼ 0。
SRQ ありでは、微小な差が 8bit の格子の境目で転ぶことがある（EdgeSLM_AndroidGemma4 P2 で見た現象）ので、
差の大きさと 1 位のトークンの一致率を記録する。

### 3. 画像の特徴が、端末のグラフの入口の目盛りに収まるか（値域と切り詰めの影響）

端末の入口（inputs_embeds・ple_tok・ple_all）の量子化の目盛りは**文章だけで較正**してある（onnx_scales.py で読む）。
画像の位置の値がその範囲を超える割合を数え、さらに**範囲で切り詰めたとき答えがどう変わるか**を PC で試す
（端末で起きることの近似。16bit の丸めそのものは十分細かいので、切り詰めだけを真似る）。

★2026-09-28 の修正: 初回（wiring.json）は値域と切り詰めを SRQ ありの回にだけ付けていたため、切り詰めの効果（1〜2 個）が
SRQ の丸めの揺れ（B↔C と同じ 0.1 前後）に埋もれて判定できなかった（results/P1_20260928_pc_eval.md 3.1 節）。
いまは SRQ あり・なしの両方で測る。**切り詰めの判定には SRQ なしの回を使う**。

## 結果

`results/P1_check_wiring/wiring_b{budget}_srq-{both|on|off}.json`（数値）。記録の md は人が書く。
初回の結果（2026-09-27）は `wiring.json` として残してある（上書きしない）。

結果のまとめ（`results/P1_20260928_pc_eval.md` 2・8.1 節、StudyMemo 6.2〜6.4 節）:
- A↔B は差 0（√1536 を掛けない・PLE は pad の行・因果マスク・連番の位置、という読みは正しい）
- SRQ なしの B↔C は最大 2.6e-5（1 トークンずつにしても KV キャッシュと位置の数え方は正しい）
- 入口の範囲の外に出るのは 1 画像あたり 1〜2 個で、SRQ なしで切り詰めても答えは同じ → 入口の較正のやり直しは不要

動かす venv は Phase 1 の EdgeSLM_MultiHead_Multi（HF の Gemma 4 を CUDA で動かす）。入口の範囲は onnx_scales で端末のグラフから読む。
"""
from __future__ import annotations

import argparse
import contextlib
import json
import time

from PIL import Image

from app.pipeline import hf_vlm as H
from app.pipeline import onnx_scales
from app.pipeline import vlm_common as C

RES = C.RESULTS / "P1_check_wiring"
# 既定の 4 問（車・合成の掲示・In-the-Wild・汎用 VQA の 4 つの出どころから 1 問ずつ）
DEFAULT_IDS = "car_02a,syn_notice_La,wild_10,vg_00"


def compare(a, b, img_span: tuple[int, int] | None = None) -> dict:
    """logits [T,V] 同士を位置ごとに比べる。相対 L2（‖a−b‖/‖a‖）と、1 位のトークンが同じか。

    rel_l2_argmax は差が最大だった位置。img_span（画像トークンの最初と最後の位置）を渡すと、
    その位置が「画像より前の文章」「画像」「画像より後の文章」のどれかも記録する
    （初回の結果で「最大の差は共通の system の文で起きている」と推定したのを確かめるため）。
    a・b は [T,V]（T = プロンプトの長さ、V = 語彙 262,144）の float の logits。a を基準にした相対 L2 を位置ごとに取る。
    """
    rel = (a - b).norm(dim=-1) / a.norm(dim=-1).clamp_min(1e-12)
    top = (a.argmax(-1) == b.argmax(-1)).float()
    am = int(rel.argmax()) if float(rel.max()) > 0 else None       # 差がまったく無いときは位置を記録しない
    r = {"rel_l2_max": float(rel.max()), "rel_l2_argmax": am, "rel_l2_mean": float(rel.mean()),
         "rel_l2_last": float(rel[-1]), "top1_agree": float(top.mean()), "top1_last_same": bool(top[-1] > 0.5),
         "max_abs": float((a - b).abs().max())}
    if img_span is not None:
        r["argmax_where"] = (None if am is None else
                             "text_before" if am < img_span[0] else "image" if am <= img_span[1] else "text_after")
    return r


@contextlib.contextmanager
def entry_clip(model, ranges: dict | None):
    """端末の入口の量子化で起きる「範囲の外は切り詰め」を真似る。

    inputs_embeds と ple_tok は呼ぶ側で切り詰める。ple_all（PLE の射影＋表の行）は language_model の中で
    計算されるので、project_per_layer_inputs を一時的に包んで出力を切り詰める。
    """
    if ranges is None:
        yield
        return
    # 文章側の PLE の射影の関数を、一時的に「入力を prescaled の範囲で切る → 元の計算 → ple_all の範囲で切る」関数に差し替える
    lm = model.model.language_model
    orig = lm.project_per_layer_inputs
    pa = ranges.get("ple_all_QuantizeLinear")
    ps = ranges.get("inputs_embeds_prescaled_QuantizeLinear")
    # prescaled（inputs_embeds × 1536^-0.5）の範囲は、inputs_embeds に直すと √1536 倍。
    # 端末ではこの切り詰めは PLE の枝にだけ効く（残差の本流には効かない）ので、PLE の射影の中でだけ切る
    ps_lo, ps_hi = (ps["lo"] * C.TEXT_HIDDEN ** 0.5, ps["hi"] * C.TEXT_HIDDEN ** 0.5) if ps else (None, None)

    def clipped(inputs_embeds, per_layer_inputs=None):
        # PLE の射影の代わりに呼ばれる: 入力を prescaled の範囲で切ってから元の計算をし、出力（ple_all）も範囲で切る
        x = inputs_embeds.clamp(ps_lo, ps_hi) if ps else inputs_embeds
        y = orig(x, per_layer_inputs)
        return y.clamp(pa["lo"], pa["hi"]) if pa else y

    lm.project_per_layer_inputs = clipped
    try:
        yield
    finally:
        lm.project_per_layer_inputs = orig


def clip_inputs(emb, ple, ranges: dict | None):
    """inputs_embeds と ple_tok を、端末の入口の範囲で切り詰める（全位置。端末も全位置に同じ目盛りを使う）。"""
    if ranges is None:
        return emb, ple
    e, p = ranges.get("inputs_embeds_QuantizeLinear"), ranges.get("ple_tok_QuantizeLinear")
    if e is not None:
        emb = emb.clamp(e["lo"], e["hi"])
    if p is not None:
        ple = ple.clamp(p["lo"], p["hi"])
    return emb, ple


# 引数: emb・ple = 端末と同じ入れ方の inputs_embeds と ple_tok、ranges = 入口の範囲（None なら切り詰めない）。戻り値: 答えの文字列。
def greedy_device_style(model, proc, emb, ple, ranges, max_new: int = 48) -> str:
    """端末と同じ入れ方で、プロンプトを一度に食わせてから 1 トークンずつ生成する（切り詰めの有無で比べる用）。

    プロンプトは一度に食わせる（速さのため）。SRQ なしでは一度に・1 トークンずつの差は 1e-5 以下なので、
    切り詰めの比較には影響しない。生成しながら食わせるトークンにも同じ切り詰めを掛ける（端末も全位置に同じ目盛り）。
    """
    with entry_clip(model, ranges):
        g = H.generate_device_style(model, emb, ple, max_new, prefill="batch",
                                    transform=lambda e, p: clip_inputs(e, p, ranges))
    return H.decode(proc, g["ids"])


# 値の並び x のうち、範囲 rng（lo〜hi）の外にある割合（範囲が無ければ None）。
def outside(x, rng: dict | None) -> float | None:
    if rng is None:
        return None
    return round(float(((x < rng["lo"]) | (x > rng["hi"])).float().mean()), 6)


def run_one(model, proc, it: dict, budget: int, system: str, device: str, ranges: dict | None,
            do_values: bool) -> dict:
    """1 問について、A（HF の本来の経路）・B（端末と同じ入れ方を一度に）・C（1 トークンずつ）の logits を比べる。

    引数: it = 評価の一覧の 1 問、budget = 画像トークン数、system = system の文、ranges = 端末の入口の範囲、
    do_values = 画像の位置の値域・範囲の外の割合・切り詰めの影響も測るか。戻り値: 比べの結果の辞書（JSON の runs の 1 つ）。
    """
    import torch
    # ① 端末と同じ固定の格子に収めて、HF のプロセッサで入力を作る（画像トークンの数が予算と同じかも確かめる）
    img = H.prepare_image(Image.open(C.EVAL_DIR / it["image"]), "fixed", budget)
    inputs, n_img, _ = H.build_inputs(proc, img, it["question"], budget, system, device, expect_exact=True)
    T = int(inputs["input_ids"].shape[1])
    img_pos = (inputs["input_ids"][0] == C.IMAGE_TOKEN_ID).nonzero().flatten()
    span = (int(img_pos[0]), int(img_pos[-1]))                                 # 画像トークンの最初と最後の位置
    t0 = time.time()
    with torch.no_grad():
        ref = model(**inputs).logits[0].float()                                  # A
    # ② 画像の特徴を取り出し、端末と同じ入れ方の inputs_embeds・ple_tok を作る（画像の位置は特徴そのもの、PLE は pad の行）
    feats = H.image_features(model, inputs)
    emb, ple, img_mask = H.device_style_embeds(model, inputs["input_ids"], feats)
    bat, _ = H.device_style_step(model, emb, ple, None)                         # B（一度に）
    bat = bat.float()
    steps, pkv = [], None
    for t in range(T):                                                          # C（1 トークンずつ）
        lg, pkv = H.device_style_step(model, emb[:, t:t + 1], ple[:, t:t + 1], pkv)
        steps.append(lg[-1].float())
    stp = torch.stack(steps)
    # ③ 3 通りを 2 つずつ比べる（A↔B は入れ方、B↔C は 1 トークンずつの持ち越し、A↔C は全体）
    r = {"id": it["id"], "T": T, "n_image_tokens": n_img, "image_span": list(span), "sec": round(time.time() - t0, 1),
         "A_vs_B": compare(ref, bat, span), "B_vs_C": compare(bat, stp, span), "A_vs_C": compare(ref, stp, span)}

    if do_values:
        # ④ 値域: 文章の位置と画像の位置に分けて、入口の 3 点（inputs_embeds・ple_tok・ple_all）の値の要約と範囲の外の割合を取る
        lm = model.model.language_model
        with torch.no_grad():
            ple_all = lm.project_per_layer_inputs(emb, ple)
        tm = ~img_mask
        g = lambda k: None if ranges is None else ranges.get(k)
        r["values"] = {
            "inputs_embeds_text": H.tensor_stats(emb[0, tm]),
            "inputs_embeds_image": H.tensor_stats(emb[0, img_mask]),
            "ple_tok_image(pad)": H.tensor_stats(ple[0, img_mask]),
            "ple_all_text": H.tensor_stats(ple_all[0, tm]),
            "ple_all_image": H.tensor_stats(ple_all[0, img_mask]),
            "outside_rate": {
                "inputs_embeds_text": outside(emb[0, tm], g("inputs_embeds_QuantizeLinear")),
                "inputs_embeds_image": outside(emb[0, img_mask], g("inputs_embeds_QuantizeLinear")),
                "prescaled_image": outside(emb[0, img_mask] * C.TEXT_HIDDEN ** -0.5,
                                           g("inputs_embeds_prescaled_QuantizeLinear")),
                "ple_tok_image(pad)": outside(ple[0, img_mask], g("ple_tok_QuantizeLinear")),
                "ple_all_text": outside(ple_all[0, tm], g("ple_all_QuantizeLinear")),
                "ple_all_image": outside(ple_all[0, img_mask], g("ple_all_QuantizeLinear")),
            },
        }
        if ranges is not None:
            # 切り詰めの影響: プロンプト最後の位置の logits と、生成した答え
            with entry_clip(model, ranges):
                e, p = clip_inputs(emb, ple, ranges)
                clp, _ = H.device_style_step(model, e, p, None)
            a0 = greedy_device_style(model, proc, emb, ple, None)
            a1 = greedy_device_style(model, proc, emb, ple, ranges)
            r["clip_effect"] = {
                "prompt_logits_vs_noclip": compare(bat, clp.float(), span),
                "answer_noclip": a0, "answer_clip": a1, "answer_same": a0 == a1,
            }
    return r


def main():
    """既定の 4 問を SRQ あり → なし（--srq で選ぶ）の順に流し、A・B・C の比べと値域・切り詰めの影響を JSON に書く。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", default=DEFAULT_IDS)
    ap.add_argument("--budget", type=int, default=70, choices=sorted(C.FIXED_GRIDS))
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--srq", default="both", choices=["both", "on", "off"],
                    help="both＝SRQ あり→なしの順に両方（既定）、on＝ありだけ、off＝なしだけ（読み込み時に外す）")
    ap.add_argument("--out", default="", help="結果のファイル名（既定 wiring_b{budget}_srq-{srq}.json）")
    a = ap.parse_args()
    C.utf8_stdout()

    man = json.loads((C.EVAL_DIR / "manifest.json").read_text(encoding="utf-8"))
    by_id = {it["id"]: it for it in man["items"]}
    items = [by_id[i] for i in a.ids.split(",")]
    # ① 端末の文章グラフ（かたまり A）があれば、入口の量子化の点の範囲を読む（無ければ値域の比べを省く）
    ranges = None
    if C.DEVICE_CHUNK_A_QDQ.exists():
        ranges = onnx_scales.read_quant_points(C.DEVICE_CHUNK_A_QDQ, onnx_scales.ENTRY_NAMES)
        print("端末の入口の範囲:", {k: (v["lo"], v["hi"]) if v else None for k, v in ranges.items()})
        missing = [k for k, v in ranges.items() if v is None]
        if missing:
            print(f"  ★見つからない量子化の点（その点の比較・切り詰めは省く）: {missing}")
    else:
        print(f"（{C.DEVICE_CHUNK_A_QDQ} が無いので、値域の比較は省く）")

    out_file = RES / (a.out or f"wiring_b{a.budget}_srq-{a.srq}.json")
    out = {"budget": a.budget, "device": a.device, "srq_mode": a.srq, "entry_ranges": ranges, "runs": []}
    RES.mkdir(parents=True, exist_ok=True)
    passes = {"both": [True, False], "on": [True], "off": [False]}[a.srq]
    model, proc = H.load(a.device, srq=passes[0])       # off だけのときは読み込み時に外す
    for srq in passes:
        if not srq and passes[0]:
            # SRQ を外す: 活性のスケールを 0 にする（読み直すより速い。hf_vlm.load と同じやり方）
            print(f"SRQ を外した（0 にしたスケール {H.disable_srq(model)} 本）", flush=True)
        for it in items:
            # 値域と切り詰めは SRQ あり・なしの両方で測る（切り詰めの判定には SRQ なしを使う）
            r = run_one(model, proc, it, a.budget, man["system_prompt"], a.device, ranges, do_values=True)
            r["srq"] = srq
            out["runs"].append(r)
            ab, bc = r["A_vs_B"], r["B_vs_C"]
            print(f"[SRQ={'あり' if srq else 'なし'}] {it['id']:14s} T={r['T']}  A-B 相対L2 max {ab['rel_l2_max']:.2e}"
                  f" 1位一致 {ab['top1_agree']:.3f} | B-C max {bc['rel_l2_max']:.2e}"
                  f"（位置 {bc['rel_l2_argmax']}・{bc['argmax_where']}） 1位一致 {bc['top1_agree']:.3f}"
                  f"  ({r['sec']}s)", flush=True)
            if "values" in r:
                print("    範囲の外の割合:", r["values"]["outside_rate"], flush=True)
            if "clip_effect" in r:
                ce = r["clip_effect"]
                pl = ce["prompt_logits_vs_noclip"]
                print(f"    切り詰めの影響: 最後の位置の相対L2 {pl['rel_l2_last']:.2e}  答えが同じ {ce['answer_same']}", flush=True)
                if not ce["answer_same"]:
                    print(f"    切り詰めなし: {ce['answer_noclip'][:60]}\n    切り詰めあり: {ce['answer_clip'][:60]}", flush=True)
            C.write_json_atomic(out_file, out)
    print(f"-> {out_file}")


if __name__ == "__main__":
    main()
