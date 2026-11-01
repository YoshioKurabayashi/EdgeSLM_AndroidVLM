# -*- coding: utf-8 -*-
"""Phase 1: 評価セットの画像を集めて、1 つの一覧（manifest.json）にまとめる（2026-09-27）。

    python -m app.pipeline.fetch_eval_data            # 取得（済んだものは飛ばす）
    python -m app.pipeline.fetch_eval_data --check    # 取得済みの一覧を数えるだけ（通信しない）

## 何をするか

| 出どころ | 取り方 | 置き場所 |
|---|---|---|
| JA-VG-VQA-500（test 先頭 40 画像） | Hugging Face の datasets-server（行を JSON で返す API）から問題と画像の URL を得て、画像を取得 | work/eval/images/vg/ |
| JA-VLM-Bench-In-the-Wild（50 問） | 同上 | work/eval/images/wild/ |
| 車両画像（Unsplash 13 枚） | images.unsplash.com から幅 1600 に縮小した JPEG を取得 | work/eval/images/car/ |
| 合成画像（15 問・13 枚） | synth_images.py で作成済みのもの（無ければ作る） | work/eval/images/synth/ |

できるもの: `work/eval/manifest.json`（1 問 = 1 件。画像のパス・質問・判定の語・出どころ・ライセンス）

## なぜ datasets-server を使うか

parquet を読むには pyarrow が要るが、借りている venv（EdgeSLM_MultiHead_Multi）には入っていない。
venv のパッケージは変えない約束なので、**標準ライブラリ（urllib）だけ**で取れる JSON の API を使う。

## 使う段

- 評価セットの準備（StudyMemo 5.1 節）。問題の定義は eval_defs.py、合成画像は synth_images.py、判定の語は judge.py
- 動かす venv は Phase 1 の EdgeSLM_MultiHead_Multi（$PY1）。合計 126 問（VG 40・Wild 50・車 21・合成 15）
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

from app.pipeline import eval_defs as D
from app.pipeline import synth_images
from app.pipeline import vlm_common as C
from app.pipeline.judge import answer_to_keywords

# 画像の置き場と一覧のファイル、通信のときの名乗り（User-Agent）、datasets-server の行を返す API
IMG_DIR = C.EVAL_DIR / "images"
MANIFEST = C.EVAL_DIR / "manifest.json"
UA = {"User-Agent": "EdgeSLM_AndroidVLM/0.1 (research; contact: local)"}
ROWS_API = "https://datasets-server.huggingface.co/rows"


def http_get(url: str, timeout: int = 60, retries: int = 3) -> bytes:
    """GET して中身を返す。一時的な失敗に備えて数回やり直す。"""
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as e:          # noqa: BLE001（通信の失敗は種類を問わずやり直す）
            last = e
            time.sleep(2 * (k + 1))
    raise RuntimeError(f"取得に失敗: {url} ({last})")


def fetch_rows(dataset: str, split: str, n: int) -> list[dict]:
    """datasets-server の /rows から先頭 n 行を取る（1 回 100 行まで）。"""
    # 100 行ずつ offset をずらして取る（返ってこなくなったら打ち切る）。各行は {"row": {...}} の形で返る
    rows, off = [], 0
    while off < n:
        q = urllib.parse.urlencode({"dataset": dataset, "config": "default", "split": split,
                                    "offset": off, "length": min(100, n - off)})
        js = json.loads(http_get(f"{ROWS_API}?{q}"))
        got = [r["row"] for r in js["rows"]]
        if not got:
            break
        rows += got
        off += len(got)
    return rows[:n]


# 画像を保存する（既にあって 0 バイトでなければ取り直さない。2 回目からは通信しない）。
def save_image(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(http_get(url))


# URL の拡張子が .png なら .png、それ以外は .jpg として保存する。
def _ext(url: str) -> str:
    p = urllib.parse.urlparse(url).path.lower()
    return ".png" if p.endswith(".png") else ".jpg"


# ---------------------------------------------------------------------------
# 出どころごとの取得
# ---------------------------------------------------------------------------
# JA-VG-VQA-500 の先頭 40 画像の最初の 1 問ずつ。正解の文から judge.answer_to_keywords で判定の語を作る。
# 戻り値: manifest の 1 件の辞書（id・set・tags・image・question・answer_ref・keywords・max_new・source・license）の並び。
def collect_vg() -> list[dict]:
    rows = fetch_rows(D.VG_DATASET, D.VG_SPLIT, D.VG_N_IMAGES)
    items = []
    for i, r in enumerate(rows):
        qa = r["qas"][0]                                  # 各画像の最初の 1 問（選び方を機械的に固定）
        src = (r.get("image") or {}).get("src") or r["url"]
        path = IMG_DIR / "vg" / f"vg_{i:02d}_{r['image_id']}{_ext(src)}"
        try:
            save_image(src, path)
        except RuntimeError:
            save_image(r["url"], path)                     # datasets-server が駄目なら元の Visual Genome の URL
        items.append({"id": f"vg_{i:02d}", "set": "vg", "tags": ["general"],
                      "image": str(path.relative_to(C.EVAL_DIR)).replace("\\", "/"),
                      "question": qa["question"], "answer_ref": qa["answer"],
                      "keywords": answer_to_keywords(qa["answer"]), "max_new": D.DEFAULT_MAX_NEW,
                      "source": {"dataset": D.VG_DATASET, "image_id": r["image_id"], "url": r["url"],
                                 "qa_id": qa.get("qa_id")},
                      "license": D.VG_LICENSE})
    return items


# JA-VLM-Bench-In-the-Wild の 50 問。判定の語は eval_defs.WILD_KEYWORDS（行番号で引く）。同じ画像の問題は画像を 1 回だけ保存する。
def collect_wild() -> list[dict]:
    rows = fetch_rows(D.WILD_DATASET, D.WILD_SPLIT, 50)
    items, cache = [], {}
    for i, r in enumerate(rows):
        # 行の並びが定義を作ったときと同じか（正解の文の書き出しで照合）。違えば判定の語がずれるので止める
        if not r["answer"].startswith(D.WILD_ANSWER_HEAD[i]):
            raise SystemExit(f"JA-VLM-Bench-In-the-Wild の {i} 行目が想定と違う: {r['answer'][:30]!r}。"
                             "eval_defs.WILD_KEYWORDS を見直すこと")
        tags, kw = D.WILD_KEYWORDS[i]
        src = (r.get("image") or {}).get("src") or r["image_url"]
        # 同じ画像に複数の質問がある（page_url が同じ）→ 画像は 1 回だけ保存する
        key = r["page_url"]
        if key not in cache:
            path = IMG_DIR / "wild" / f"wild_{len(cache):02d}{_ext(src)}"
            save_image(src, path)
            cache[key] = path
        path = cache[key]
        items.append({"id": f"wild_{i:02d}", "set": "wild", "tags": tags,
                      "image": str(path.relative_to(C.EVAL_DIR)).replace("\\", "/"),
                      "question": r["question"], "answer_ref": r["answer"], "keywords": kw,
                      "max_new": D.DEFAULT_MAX_NEW,
                      "source": {"dataset": D.WILD_DATASET, "page_url": r["page_url"], "image_url": r["image_url"]},
                      "license": D.WILD_LICENSE})
    return items


# 車の写真 13 枚（Unsplash から幅 1600 で取得）。1 枚に複数の問題があれば id の末尾に a・b・c を付ける。
def collect_car() -> list[dict]:
    items = []
    for c in D.CAR_IMAGES:
        url = f"https://images.unsplash.com/{c['photo']}?w=1600&q=85&fm=jpg"
        path = IMG_DIR / "car" / f"{c['id']}.jpg"
        save_image(url, path)
        for k, q in enumerate(c["qs"]):
            items.append({"id": f"{c['id']}{'abc'[k]}", "set": "car", "tags": q["tags"],
                          "image": str(path.relative_to(C.EVAL_DIR)).replace("\\", "/"),
                          "question": q["q"], "keywords": q["kw"], "max_new": q.get("max_new", D.DEFAULT_MAX_NEW),
                          "source": {"page": c["page"], "url": url, "desc": c["desc"]},
                          "license": D.CAR_LICENSE})
    return items


# 合成画像（無ければ synth_images.build で作る）。問題が 1 つの画像は id の末尾に何も付けない。
def collect_synth() -> list[dict]:
    meta = synth_images.build(force=False)
    items = []
    for s in D.SYNTH_SPECS:
        for k, q in enumerate(s["qs"]):
            items.append({"id": f"{s['id']}{'' if len(s['qs']) == 1 else 'abc'[k]}", "set": "synth", "tags": q["tags"],
                          "image": meta[s["id"]]["path"], "question": q["q"],
                          "keywords": q.get("kw"), "ref_text": q.get("ref_text"),
                          "max_new": q.get("max_new", D.DEFAULT_MAX_NEW),
                          "source": {"spec": s["id"], "font": meta[s["id"]]["font"]},
                          "license": D.SYNTH_LICENSE})
    return items


def main():
    """4 つの出どころから問題を集めて work/eval/manifest.json に書く（--check は数えるだけ）。system の文も一覧に入れる。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="通信せず、manifest の件数だけ数える")
    a = ap.parse_args()
    C.utf8_stdout()
    if a.check:
        m = json.loads(MANIFEST.read_text(encoding="utf-8"))
        from collections import Counter
        print(Counter(it["set"] for it in m["items"]), "計", len(m["items"]))
        missing = [it["image"] for it in m["items"] if not (C.EVAL_DIR / it["image"]).exists()]
        print("画像が無い:", missing or "なし")
        return
    all_items = []
    for name, fn in (("合成", collect_synth), ("車両", collect_car), ("VG", collect_vg), ("Wild", collect_wild)):
        t0 = time.time()
        got = fn()
        print(f"  {name}: {len(got)} 問（{time.time() - t0:.0f}s）", flush=True)
        all_items += got
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps({"created": time.strftime("%Y-%m-%d %H:%M:%S"), "system_prompt": D.SYSTEM_PROMPT,
                                    "items": all_items}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"計 {len(all_items)} 問 -> {MANIFEST}")


if __name__ == "__main__":
    main()
