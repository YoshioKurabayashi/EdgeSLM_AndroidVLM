# -*- coding: utf-8 -*-
"""EdgeSLM_AndroidVLM の共通の置き場所と定数（2026-09-27）。

ここに書くのは「どのファイルがどこにあるか」「モデルの決まった数字」と、どのスクリプトでも使う小さな補助
（出力の文字コード・JSON の書き込み）だけ。import しただけで何かが起きることはない（副作用なし）。

（2026-10-18 追記）使い方: 各道具が `from app.pipeline import vlm_common as C` で読む。直接は動かさない。
数字の解説は StudyMemo（EdgeSLM_AndroidVLM）01「VLM の構造」（部品の大きさ）・03「NPU に載せるための設計」（格子の表）。
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 置き場所
# ---------------------------------------------------------------------------
# このファイルは app/pipeline/ にあるので、2つ上がプロジェクトの根（EdgeSLM_AndroidVLM/）
ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "work"          # 大きな生成物（画像・中間データ）。記録ではない
RESULTS = ROOT / "results"    # 検証の記録（Phase ごと）
EVAL_DIR = WORK / "eval"      # 評価セットの画像と一覧（manifest.json）

# モデルは EdgeSLM_AndroidGemma4 が取得したものを「読むだけ」で使う（複製しない。2.4GB あるため）。
# 本プロジェクトからは書き込まない。場所が変わったときは環境変数で差し替えられるようにしておく。
GEMMA4_ROOT = ROOT.parent / "EdgeSLM_AndroidGemma4"
HF_DIR = Path(os.environ.get(
    "VLM_HF_DIR", GEMMA4_ROOT / "work" / "hf" / "gemma-4-E2B-it-qat-mobile-transformers"))

# 端末で動いている Gemma 4 の文章グラフ（かたまり A＝入口＋層0〜8）の量子化済み ONNX。
# 入口（inputs_embeds など）の量子化の目盛り＝「端末のグラフが受け付けられる値の範囲」を読むのに使う。
DEVICE_CHUNK_A_QDQ = GEMMA4_ROOT / "work" / "ops" / "g4c0_8h_L256" / "g4c0_8h_L256_final_qdq.onnx"

# HF のキャッシュを C: に書かせない（Gemma4 と同じ理由。C: の残りが少ない）
HF_HOME = WORK / "hf_home"

# Phase 2 以降の生成物
CAPTURE = WORK / "capture"    # HF から取った正解・較正用の入力（npz）
OPS = WORK / "ops"            # 書き出した ONNX・量子化済み ONNX・context binary（グラフごとのフォルダ）

# ---------------------------------------------------------------------------
# モデルの数字（config.json から。Phase 0 の棚卸しで確認したもの）
# ---------------------------------------------------------------------------
TEXT_HIDDEN = 1536            # 文章側の hidden（inputs_embeds の次元）
NUM_LAYERS = 35               # 文章側の層の数
PLE_DIM = 256                 # PLE（層ごとの埋め込み）の次元
PAD_ID = 0                    # text_config.pad_token_id。画像トークンの PLE はこの行を使う（HF の実装どおり）
IMAGE_TOKEN_ID = 258880       # 画像の中身が入る位置の目印（<|image|>）
BOI_TOKEN_ID = 255999         # 画像の始まり
EOI_TOKEN_ID = 258882         # 画像の終わり

VISION_HIDDEN = 768           # 画像エンコーダ（ViT）の hidden
PATCH = 16                    # 1 パッチ = 16×16 ピクセル
POOL = 3                      # 3×3 のパッチを平均して 1 つの画像トークンにする
SUPPORTED_SOFT_TOKENS = (70, 140, 280, 560, 1120)   # HF の画像前処理が受け付ける「画像トークン数の上限」

# 解像度を固定するときの格子（パッチの縦×横）。端末では形を固定しないとグラフが作れない（動的形状は不可）。
# 3 の倍数（プーリングで割り切れる）かつ 縦×横 = 画像トークン数×9 になるように選んだ。横長（約 1.43:1）。
FIXED_GRIDS = {
    70: (21, 30),     # 336×480 ピクセル、630 パッチ
    140: (30, 42),    # 480×672 ピクセル、1,260 パッチ
    280: (42, 60),    # 672×960 ピクセル、2,520 パッチ
}

# 縦長の格子（2026-09-28 追加）。横長の格子に縦長の写真を入れると使える画素が半分以下になり、280 で品質が落ちた
# （results/P1_20260928_pc_eval.md 8.3 節）。スマホの縦撮りを読むときはこちらのグラフを使う
PORTRAIT_GRIDS = {k: (w, h) for k, (h, w) in FIXED_GRIDS.items()}

# 画像エンコーダ（vision_config）の数字。Phase 2 の自前の配線（vit_graph.py）が使う
VIT_LAYERS = 16
VIT_HEADS = 12                # KV も 12（GQA ではない）
VIT_HEAD_DIM = 64
VIT_INTER = 3072              # MLP の中間
VIT_EPS = 1e-6                # rms_norm_eps（射影層の RMSNorm も同じ値）
VIT_ROPE_THETA = 100.0        # 2 次元 RoPE の θ
PATCH_DIM = 3 * PATCH * PATCH  # 1 パッチの値の数 = 768。並びは (py, px, c)（HF の convert_image_to_patches）

# 生成を止めるトークン（EdgeSLM_AndroidGemma4 の ref_qat.py と同じ: <turn|>, <eos>, <|tool_response>）
STOP_IDS = (106, 1, 50)


def ensure_hf_env() -> None:
    """transformers / huggingface_hub を import する前に呼ぶ。キャッシュの置き場所を固定する。"""
    os.environ.setdefault("HF_HOME", str(HF_HOME))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def utf8_stdout() -> None:
    """標準出力を UTF-8 にする。

    日本語版 Windows では、出力をファイルやパイプに流すと cp932 で書こうとし、
    cp932 に無い文字（「↔」や、モデルが出した珍しい文字）で UnicodeEncodeError になって止まる。
    """
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def _json_default(o):
    """json.dumps が知らない型（numpy の数・配列、Path など）を普通の型にする。"""
    if hasattr(o, "item") and getattr(o, "size", 1) == 1:
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def write_json_atomic(path: Path, obj) -> None:
    """JSON を一時ファイルに書いてから置き換える。書いている途中で止めても、前の内容が壊れない
    （途中から再開する仕組みが、壊れたファイルを読んで止まらないようにするため）。
    numpy の数や Path が混ざっていても止まらないよう、普通の型に直して書く。"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=_json_default), encoding="utf-8")
    os.replace(tmp, path)
