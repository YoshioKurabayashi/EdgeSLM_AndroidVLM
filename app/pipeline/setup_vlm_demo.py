# -*- coding: utf-8 -*-
"""Phase 3-2: VLM デモの準備。端末の VLM 検証アプリ（org.edgeslm.vlm）の置き場に、文章側のモデル・表・トークナイザと
画像エンコーダを置き、画面の設定 `jobs/vlm_demo.json` を書く（2026-09-29）。

    python -m app.pipeline.setup_vlm_demo              # 端末を USB でつないで（どの venv でもよい。$PY2 を想定）
    python -m app.pipeline.setup_vlm_demo --dry-run    # 写すものと容量だけ表示
    python -m app.pipeline.setup_vlm_demo --text t16   # Phase 5: 文章側を 16 トークン版（まとめ読み）にする（PC から送る）
    python -m app.pipeline.setup_vlm_demo --text t16 --system "…"   # system の文も差し替える

## 何をするか

| もの | どこから | 大きさ |
|---|---|---|
| 文章側のモデル（Gemma 4、窓 256）: EPContext と context binary | **端末の中で** Gemma4 のアプリの置き場（`/sdcard/Android/data/org.edgeslm.gemma4/files`）から写す | 約 1.5GB |
| 埋め込み表（2bit）・PLE の表（4bit）・スケール・位置の表 | 同上 | 約 1.3GB |
| トークナイザ（vocab.bin・merges.bin・specials.tsv） | 同上 | 数 MB |
| 画像エンコーダ（SRQ なしの版） | PC から送る（device_vit と同じ。済んでいれば送らない） | 約 170MB |
| 設定 `jobs/vlm_demo.json` | PC で作って送る | — |

- Gemma4 のアプリの置き場は**読むだけ**（端末の中の `cp`。PC を通さないので速い）。Gemma4 のアプリもプロジェクトも変えない
- 文章側のモデルの場所・表の名前は、Gemma4 の会話アプリの設定（`../EdgeSLM_AndroidGemma4/jobs/chat_models.json` の `gemma4-l256`）から取る
- 写す前に端末の空き容量を確かめる。同じ大きさのファイルが既にあれば写さない（2 回目からは速い）
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from app.pipeline import device_vit as D
from app.pipeline import vlm_common as C

G4_DEV = f"/sdcard/Android/data/{D.G4_PKG}/files"
G4_CHAT_CONFIG = C.GEMMA4_ROOT / "jobs" / "chat_models.json"
TEXT_KEY = "gemma4-l256"
TOKENIZER_FILES = ("vocab.bin", "merges.bin", "specials.tsv", "tokenizer_meta.json")
SUGGESTIONS = ["この画像を説明してください。", "何が写っていますか？", "書かれている文字を読んでください。"]
IO = C.WORK / "device_vlm"


def dev_size(path: str) -> int:
    """端末のファイルの大きさ（無ければ −1）。"""
    out = D.adb("shell", f"stat -c %s {path} 2>/dev/null || echo -1", check=False).strip().splitlines()
    try:
        return int(out[-1])
    except (ValueError, IndexError):
        return -1


def text_job() -> dict:
    cfg = json.loads(G4_CHAT_CONFIG.read_text(encoding="utf-8"))
    m = next((m for m in cfg["models"] if m["key"] == TEXT_KEY), None)
    if m is None:
        raise SystemExit(f"{G4_CHAT_CONFIG} に {TEXT_KEY} が無い")
    return m["job"]


def needed_files(job: dict) -> list[str]:
    """文章側で使うファイル（files からの相対パス）。context binary の名前は EPContext の中から読む。"""
    rels = [job["model"], job["embed_table"], job["embed_scales"], job["ple_table"], job["ple_scales"],
            *job["rope_tables"].values()]
    IO.mkdir(parents=True, exist_ok=True)
    local = IO / Path(job["model"]).name
    D.adb("pull", f"{G4_DEV}/{job['model']}", str(local))
    names = sorted(set(re.findall(rb"[A-Za-z0-9_.\-]+\.bin", local.read_bytes())))
    if not names:
        raise SystemExit(f"{job['model']} の中に context binary の名前が見つからない")
    d = str(Path(job["model"]).parent).replace("\\", "/")
    rels += [f"{d}/{n.decode()}" for n in names]
    rels += [f"tokenizer/{n}" for n in TOKENIZER_FILES]
    return rels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=70)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--text", default="t1", choices=["t1", "t16"],
                    help="文章側のモデル。t1 = Gemma4 の 1 トークン版（既定）、t16 = Phase 5 の 16 トークン版（build_t16 で作ったもの）")
    ap.add_argument("--system", default="", help="system の文を差し替える（既定は評価と同じ文）")
    a = ap.parse_args()
    C.utf8_stdout()
    D.ensure_app()
    if D.G4_PKG not in D.adb("shell", f"pm path {D.G4_PKG} || true", check=False):
        raise SystemExit(f"{D.G4_PKG}（Gemma4 の会話アプリ）が端末に無い。文章側のモデルの写し元が無い")
    job = text_job()
    rels = needed_files(job)
    plan, total = [], 0
    for rel in rels:
        src, dst = f"{G4_DEV}/{rel}", f"{D.DEV}/{rel}"
        s = dev_size(src)
        if s < 0:
            if rel.endswith("tokenizer_meta.json"):
                continue                                   # 無くてもよい（読むのは vocab・merges・specials）
            raise SystemExit(f"写し元に無い: {src}")
        same = dev_size(dst) == s
        plan.append((rel, s, same))
        total += 0 if same else s
    free_kb = D.adb("shell", f"df -k {D.DEV} | tail -1", check=False).split()
    free = int(free_kb[3]) * 1024 if len(free_kb) > 3 and free_kb[3].isdigit() else -1
    print(f"  写すもの {len(plan)} 個（新しく写す {sum(1 for p in plan if not p[2])} 個、{total / 2**30:.2f}GiB）、"
          f"端末の空き {free / 2**30:.1f}GiB", flush=True)
    for rel, s, same in plan:
        print(f"    {'（同じものがある）' if same else '写す':10s} {rel}  {s / 2**20:.1f}MiB")
    if a.dry_run:
        return
    if free >= 0 and total > free - 512 * 2**20:
        raise SystemExit("端末の空きが足りない（512MiB の余裕を残す）")
    for rel, s, same in plan:
        if same:
            continue
        d = str(Path(rel).parent).replace("\\", "/")
        D.adb("shell", f"mkdir -p {D.DEV}/{d} && cp {G4_DEV}/{rel} {D.DEV}/{rel}", timeout=1800)
        got = dev_size(f"{D.DEV}/{rel}")
        if got != s:
            raise SystemExit(f"写したファイルの大きさが合わない: {rel}（{got} / {s}）")
        print(f"  写した: {rel}", flush=True)
    # 画像エンコーダ（SRQ なしの版）
    tag, grid = D.tag_for(a.budget, False)
    vit_rel = D.push_models(tag, ["nosrq"])["nosrq"]
    # 画面の設定
    man = json.loads((C.EVAL_DIR / "manifest.json").read_text(encoding="utf-8"))
    text = dict(job)
    if a.text == "t16":
        # Phase 5: 16 トークン版（表・トークナイザは 1 トークン版と同じ。モデルだけ PC から送る）
        text["model"] = D.push_t16()
    cfg = {"_note": "VLM デモ（VlmActivity）の設定。setup_vlm_demo.py が書く。文章側の表は Gemma4 の会話アプリの gemma4-l256 と同じ。"
                    "text.model が 16 トークン版（g4t16_W256）ならまとめ読み（Phase 5）",
           "text": text, "text_kind": a.text, "text_t1": job, "vit_model": vit_rel, "grid": list(grid),
           "n_image_tokens": (grid[0] // C.POOL) * (grid[1] // C.POOL),
           "system_prompt": a.system or man["system_prompt"], "suggestions": SUGGESTIONS, "max_new": 96,
           "stop_ids": list(C.STOP_IDS)}
    p = IO / "vlm_demo.json"
    p.write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    D.push_many([(p, "jobs/vlm_demo.json")], "設定")
    print(f"-> 端末の {D.DEV}/jobs/vlm_demo.json（文章側 {a.text}: {text['model']}。ランチャーの「VLM デモ」で開ける）")


if __name__ == "__main__":
    main()
