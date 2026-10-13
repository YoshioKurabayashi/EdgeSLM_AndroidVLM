# -*- coding: utf-8 -*-
"""EdgeSLM_AndroidGemma4 の `gemma4_common` と**同じ名前**の置き場所を、本プロジェクト用に置く（Phase 5、2026-09-29）。

Phase 5 では Gemma4 の文章グラフの道具（`gemma4_graph.py`・`qat_inject.py`・`fix_float_inputs.py`・`merge_chunks.py`）を
**変えずに複製**して使う。それらは `from app.pipeline import gemma4_common as G` で置き場所を読むので、同じ名前をここに用意する。

- `ROOT`・`WORK`・`RESULTS`: **本プロジェクト**（書き込むのはこちらだけ）
- `HF_DIR`（モデル）・`G4_WORK`（Gemma4 が取ってある状態 `capture/`・表 `tables/`・1 トークン版の量子化済み ONNX `ops/`）:
  **Gemma4 のプロジェクトを読むだけ**（引き継ぎ一覧 1 節）
"""
from __future__ import annotations

from app.pipeline import vlm_common as C

ROOT = C.ROOT
WORK = C.WORK
RESULTS = C.RESULTS

HF_REPO = "google/gemma-4-E2B-it-qat-mobile-transformers"
HF_DIR = C.HF_DIR
HF_HOME = C.HF_HOME
HF_FILES = [
    "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
    "chat_template.jinja", "processor_config.json", "preprocessor_config.json", "README.md",
    "model.safetensors",
]

# Gemma4 のプロジェクトの作業場所（読むだけ）
G4_WORK = C.GEMMA4_ROOT / "work"
G4_CAPTURE = G4_WORK / "capture"        # 1 トークン版を作るときに HF から取った状態（丸めあり、L=256）
G4_TABLES = G4_WORK / "tables"          # 端末の表（埋め込み 2bit・PLE 4bit・位置）
G4_OPS = G4_WORK / "ops"                # 1 トークン版の量子化済み ONNX（g4full_L256 など）
