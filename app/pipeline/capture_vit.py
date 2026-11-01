# -*- coding: utf-8 -*-
"""Phase 2-1: HF の画像エンコーダの「正解」と、量子化の較正に使う入力を取っておく（2026-09-28）。

    python -m app.pipeline.capture_vit                        # 70 トークン（21×30 パッチ）
    python -m app.pipeline.capture_vit --budget 280           # 280 トークン（42×60 パッチ。Phase 2-6 用）
    python -m app.pipeline.capture_vit --portrait             # 縦長の格子（30×21）

**Phase 1 と同じ venv（EdgeSLM_MultiHead_Multi）で動かす。** ONNX は使わない。
**既定は CPU で計算する**（`--device cpu`）。書き出しの確認（export_vit）は CPU で動かすので、正解も CPU で取っておくと
GPU の足し算の順序の違い（Phase 1 で 1e-5 前後の差になった）が混ざらない。画像エンコーダだけなので CPU でも数分。
書き出し（export_vit.py）は ONNX の入った EdgeSLM_Android の venv で動かすので、正解をファイルで受け渡す
（EdgeSLM_AndroidGemma4 の capture_states → export_chunk と同じ分け方）。

## 取るもの

| ファイル | 中身 |
|---|---|
| `work/capture/vit_{tag}.npz` | 評価用の画像（既定 16 枚）の入力 pixel_values [E,N,768]、HF の出力（射影層の後）SRQ あり・なし [E,M,1536]、先頭の数枚（既定 2 枚）の途中の値（パッチの埋め込み・16 層の各出力・プーリングの後） |
| `work/capture/vit_{tag}_calib.npz` | 較正用の画像（既定 64 枚。評価用とは別の画像）の pixel_values [C,N,768]。キーはグラフの入力名 |
| `work/capture/vit_{tag}_meta.json` | 画像の一覧・格子・各層の値の範囲（残差の最大値・RMS・99.9% 点）・時間 |

tag は `b70_21x30` のように「画像トークン数_格子」。

## 画像の選び方（機械的に固定）

manifest の画像（重複を除く）を出どころ（vg・wild・car・synth）ごとに分け、各出どころから等間隔に 4 枚ずつを評価用、
残りから等間隔に最大 64 枚を較正用にする。前処理は端末と同じ固定の格子（hf_vlm.prepare_image の fixed）。

## 値の範囲を取る理由

HF の pooler に「×√768 で fp16 の範囲（65504）を超えうる」とある。つまり**最後の層の残差はかなり大きい**はず。
16bit の量子化は 1 本の目盛りで最小〜最大を表すので、巨大な値があると小さな値が潰れる（EdgeSLM_Android StudyMemo 3 章）。
どの層でどれだけ大きいかを先に見ておく（Phase 2-2 の設計の材料）。

## 使う段・入力と出力

- Phase 2-1（基本設計_Phase2.md、StudyMemo 7.6・7.7 節「確かめ方」）。export_vit check（自前 ↔ HF）・quantize_vit の較正・
  eval_qdq_vit acc（量子化した版 ↔ HF）が、ここで取った npz を読む
- 入力: `work/eval/manifest.json` の画像、HF の Gemma 4（vlm_common.HF_DIR、読むだけ）
- 出力: 上の表の 3 ファイル
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict

import numpy as np
from PIL import Image

from app.pipeline import vlm_common as C

# 評価の出どころ 4 つ（JA-VG-VQA-500・JA-VLM-Bench-In-the-Wild・車室内外の写真（Unsplash）・合成画像。StudyMemo 5.1 節）
SETS = ("vg", "wild", "car", "synth")
PROMPT = "この画像を説明してください。"          # 画像の前処理にしか使わない（文章側は動かさない）


def pick_images(man: dict, n_eval_per_set: int, n_calib: int) -> tuple[list[dict], list[dict]]:
    """評価用と較正用の画像を選ぶ（重複しない）。戻り値は {id, set, image} のリスト。"""
    # ① 同じ画像を 2 回数えないように、出どころごとに初出の画像だけを並べる
    seen, by_set = set(), defaultdict(list)
    for it in man["items"]:
        if it["image"] in seen:
            continue
        seen.add(it["image"])
        by_set[it["set"]].append({"id": it["id"], "set": it["set"], "image": it["image"]})
    # ② 各出どころの中で、端から端まで等間隔に n_eval_per_set 枚を評価用にする（乱数を使わず、誰が流しても同じ選び方）
    ev, rest = [], []
    for s in SETS:
        L = by_set[s]
        idx = set(np.linspace(0, len(L) - 1, min(n_eval_per_set, len(L))).round().astype(int).tolist())
        ev += [x for i, x in enumerate(L) if i in idx]
        rest += [x for i, x in enumerate(L) if i not in idx]
    # ③ 残り（評価用に選ばれなかった画像）から等間隔に最大 n_calib 枚を較正用にする
    sel = np.linspace(0, len(rest) - 1, min(n_calib, len(rest))).round().astype(int)
    cal = [rest[i] for i in sorted(set(sel.tolist()))]
    return ev, cal


def stats(x) -> dict:
    """値の範囲の要約（numpy）。

    戻り値: {"max_abs": 絶対値の最大, "rms": 二乗平均の平方根, "p999_abs": 絶対値の 99.9% 点}。
    最大だけでなく 99.9% 点も取るのは、一部の巨大な値（外れ値）と大半の値の大きさを分けて見るため（16bit の目盛りの設計の材料）。
    """
    a = np.abs(x.reshape(-1))
    return {"max_abs": round(float(a.max()), 4), "rms": round(float(np.sqrt((x.astype(np.float64) ** 2).mean())), 4),
            "p999_abs": round(float(np.quantile(a, 0.999)), 4)}


def main():
    """評価用の画像で HF の画像エンコーダを動かし、入力・出力・途中の値・値の範囲を npz と JSON に保存する。

    引数: --budget・--portrait（格子）・--n-eval-per-set（出どころごとの評価用の枚数、既定 4 → 16 枚）・--n-calib（較正用、既定 64）・
    --n-debug（途中の値も残す枚数、既定 2）・--device（既定 cpu）。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=70, choices=sorted(C.FIXED_GRIDS))
    ap.add_argument("--portrait", action="store_true", help="縦長の格子（PORTRAIT_GRIDS）")
    ap.add_argument("--n-eval-per-set", type=int, default=4)
    ap.add_argument("--n-calib", type=int, default=64)
    ap.add_argument("--n-debug", type=int, default=2, help="途中の値（16 層の各出力など）も保存する枚数")
    ap.add_argument("--device", default="cpu", choices=["cuda", "cpu"],
                    help="既定 cpu（export_vit の確認と同じ条件にするため）")
    a = ap.parse_args()
    C.utf8_stdout()

    import torch
    from app.pipeline import hf_vlm as H
    from app.pipeline.vit_graph import grid_positions

    # ① 格子から N（パッチ数）と M（画像トークン数 = N を 3×3 のプーリングで割った数）を決め、画像を選ぶ
    grid = (C.PORTRAIT_GRIDS if a.portrait else C.FIXED_GRIDS)[a.budget]
    gh, gw = grid
    tag = f"b{a.budget}_{gh}x{gw}"
    N, M = gh * gw, (gh // C.POOL) * (gw // C.POOL)
    man = json.loads((C.EVAL_DIR / "manifest.json").read_text(encoding="utf-8"))
    ev, cal = pick_images(man, a.n_eval_per_set, a.n_calib)
    print(f"格子 {gh}×{gw}（{N} パッチ → {M} トークン）  評価用 {len(ev)} 枚・較正用 {len(cal)} 枚", flush=True)

    # ② HF のモデルを SRQ ありで読み込む。expect_pos は自前の配線が前提にするパッチの位置の表
    t0 = time.time()
    model, proc = H.load(a.device, srq=True)
    expect_pos = grid_positions(gh, gw).numpy()

    # 1 枚の前処理: 固定の格子に収めてから HF のプロセッサで入力を作り、パッチの位置が想定どおりかを確かめる
    def prep(entry):
        img = Image.open(C.EVAL_DIR / entry["image"])
        size = [img.width, img.height]
        # 端末と同じ固定の格子に収める（縦長の格子も同じ関数。透明部分の扱いや EXIF の向きを Phase 1 とそろえる）
        img_in = H.prepare_image(img, "fixed", a.budget, grid=grid)
        inputs, n_img, _ = H.build_inputs(proc, img_in, PROMPT, a.budget, None, a.device, expect_exact=True)
        pos = inputs["image_position_ids"][0].cpu().numpy()
        # ★パッチの並びと位置が、自前の配線の前提（n = y·gw + x）と同じか確かめる。違えば配線を直す必要がある
        assert pos.shape == (N, 2) and (pos == expect_pos).all(), "パッチの位置が想定（左上から右へ・行ごとに下へ）と違う"
        return inputs, size

    # 較正用: 前処理だけ（モデルは動かさない）
    calib = np.zeros((len(cal), N, C.PATCH_DIM), np.float32)
    for j, e in enumerate(cal):
        inputs, _ = prep(e)
        calib[j] = inputs["pixel_values"][0].float().cpu().numpy()
    print(f"較正用の入力 {calib.shape}（{time.time() - t0:.0f}s）", flush=True)

    # 評価用: SRQ あり → なし の順（SRQ は外すと戻せない）
    vt = model.model.vision_tower
    E, D = len(ev), min(a.n_debug, len(ev))
    pix = np.zeros((E, N, C.PATCH_DIM), np.float32)
    feats = {m: np.zeros((E, M, C.TEXT_HIDDEN), np.float32) for m in ("srq", "nosrq")}
    dbg = {m: {"patch_embed": np.zeros((D, N, C.VISION_HIDDEN), np.float32),
               "layers": np.zeros((D, C.VIT_LAYERS, N, C.VISION_HIDDEN), np.float32),
               "pooled": np.zeros((D, M, C.VISION_HIDDEN), np.float32)} for m in ("srq", "nosrq")}
    layer_stats = {m: [[] for _ in range(C.VIT_LAYERS)] for m in ("srq", "nosrq")}
    embed_stats = {m: [] for m in ("srq", "nosrq")}
    sizes = []

    # ③ 途中の値はフック（forward の後に呼ばれる関数）で拾う: パッチの埋め込み・16 層の各出力・プーリングの後
    grab = {}
    hooks = [vt.patch_embedder.register_forward_hook(lambda m, i, o: grab.__setitem__("patch_embed", o)),
             vt.pooler.register_forward_hook(lambda m, i, o: grab.__setitem__("pooled", o[0]))]
    for li, ly in enumerate(vt.encoder.layers):
        hooks.append(ly.register_forward_hook(
            lambda m, i, o, li=li: grab.__setitem__(f"layer_{li}", o[0] if isinstance(o, tuple) else o)))

    # ④ 同じ評価用の画像を SRQ あり → なしの 2 回流す。値の範囲は毎枚、途中の値そのものは先頭の D 枚だけ残す（大きいため）
    for mode in ("srq", "nosrq"):
        if mode == "nosrq":
            print(f"SRQ を外した（0 にしたスケール {H.disable_srq(model)} 本）", flush=True)
        for j, e in enumerate(ev):
            inputs, size = prep(e)
            if mode == "srq":
                pix[j] = inputs["pixel_values"][0].float().cpu().numpy()
                sizes.append(size)
            grab.clear()
            f = H.image_features(model, inputs)                       # [M,1536]
            feats[mode][j] = f.float().cpu().numpy()
            for li in range(C.VIT_LAYERS):
                layer_stats[mode][li].append(stats(grab[f"layer_{li}"][0].float().cpu().numpy()))
            embed_stats[mode].append(stats(grab["patch_embed"][0].float().cpu().numpy()))
            if j < D:
                dbg[mode]["patch_embed"][j] = grab["patch_embed"][0].float().cpu().numpy()
                for li in range(C.VIT_LAYERS):
                    dbg[mode]["layers"][j, li] = grab[f"layer_{li}"][0].float().cpu().numpy()
                pooled = grab["pooled"]
                dbg[mode]["pooled"][j] = pooled.reshape(-1, M, C.VISION_HIDDEN)[0].float().cpu().numpy()
        print(f"  {mode}: {E} 枚（{time.time() - t0:.0f}s）", flush=True)
    for h in hooks:
        h.remove()

    # ⑤ 保存: 評価用（入力・正解・途中の値）と較正用（入力だけ）の npz、画像の一覧と値の範囲の JSON
    C.CAPTURE.mkdir(parents=True, exist_ok=True)
    out = C.CAPTURE / f"vit_{tag}.npz"
    np.savez(out, ids=np.array([e["id"] for e in ev]), pixel_values=pix, pos=expect_pos,
             feat_srq=feats["srq"], feat_nosrq=feats["nosrq"],
             **{f"dbg_{m}_{k}": v for m in dbg for k, v in dbg[m].items()})
    cal_path = C.CAPTURE / f"vit_{tag}_calib.npz"
    np.savez(cal_path, pixel_values=calib)

    # 画像ごとの範囲をまとめる: 最大は全画像の最大、RMS は平均
    def agg(lst):
        return {"max_abs": max(s["max_abs"] for s in lst), "rms": round(float(np.mean([s["rms"] for s in lst])), 4),
                "p999_abs": max(s["p999_abs"] for s in lst)}
    meta = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "budget": a.budget, "grid": [gh, gw], "N": N, "M": M,
            "portrait": a.portrait, "hf_dir": str(C.HF_DIR),
            "eval": [dict(e, size=s) for e, s in zip(ev, sizes)], "calib": cal, "n_debug": D,
            "range_patch_embed": {m: agg(embed_stats[m]) for m in embed_stats},
            "range_layers": {m: [agg(layer_stats[m][li]) for li in range(C.VIT_LAYERS)] for m in layer_stats},
            "range_feat": {m: stats(feats[m]) for m in feats}, "sec": round(time.time() - t0, 1)}
    C.write_json_atomic(C.CAPTURE / f"vit_{tag}_meta.json", meta)

    # ⑥ 層ごとの範囲を表示する（最後の層ほど残差が大きくなるかを見る。モジュールの説明「値の範囲を取る理由」）
    print("\n残差（各層の出力）の範囲（SRQ あり、評価用の全画像での最大・RMS の平均）:")
    print(f"  パッチの埋め込み  max|x| {meta['range_patch_embed']['srq']['max_abs']:9.2f}  RMS {meta['range_patch_embed']['srq']['rms']:.3f}")
    for li, s in enumerate(meta["range_layers"]["srq"]):
        print(f"  層 {li:2d}            max|x| {s['max_abs']:9.2f}  RMS {s['rms']:8.3f}  99.9% {s['p999_abs']:8.2f}")
    print(f"  出力（射影層の後） max|x| {meta['range_feat']['srq']['max_abs']:.3f}")
    print(f"-> {out}（{out.stat().st_size / 2**20:.0f}MiB）\n-> {cal_path}（{cal_path.stat().st_size / 2**20:.0f}MiB）")


if __name__ == "__main__":
    main()
