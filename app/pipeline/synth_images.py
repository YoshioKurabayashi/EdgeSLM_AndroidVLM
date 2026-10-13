# -*- coding: utf-8 -*-
"""OCR・画面・文書の合成画像を作る（2026-09-27）。

    python -m app.pipeline.synth_images            # work/eval/images/synth/*.png を作る（既にあれば作り直さない）
    python -m app.pipeline.synth_images --force    # 作り直す

## なぜ合成するか

公開データの写真では「正解の文字列」が厳密に決まらないことが多い（ぼけ・一部が隠れる・読み方の揺れ）。
合成なら正解が 1 つに決まり、**文字の大きさだけを変えた対（_L / _S）**を作れる。
これで「画像トークン数（＝解像度）を減らすと、どの大きさの文字から読めなくなるか」を切り分けられる。

## 作り方の約束

- キャンバスは 960×672（280 トークンの固定格子 42×60 パッチと同じ大きさ）。70 トークンでは 480×336 に縮小される
- フォントは Noto Sans CJK JP を優先（無ければ Windows の游ゴシック・メイリオ）。**フォントが違うと画像が変わる**ので、
  一度作った画像は作り直さない（`--force` のときだけ作り直す）。どのフォントで作ったかは manifest に残す
- 文面は架空（実在の案内・店舗・人物は使わない）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from app.pipeline import eval_defs as D
from app.pipeline import vlm_common as C

OUT = C.EVAL_DIR / "images" / "synth"

# フォントの候補（上から順に探す）。.ttc は複数の書体が入っているので index を指定する
FONT_CANDIDATES = [
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc", 0),       # Linux（Noto Sans CJK JP Bold）
    ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 0),
    ("C:/Windows/Fonts/YuGothB.ttc", 0),                               # Windows（游ゴシック Bold）
    ("C:/Windows/Fonts/meiryob.ttc", 0),                               # Windows（メイリオ Bold）
    ("C:/Windows/Fonts/meiryo.ttc", 0),
    ("C:/Windows/Fonts/msgothic.ttc", 0),
]

# 見た目の型（背景色・文字色）。車載の画面・警告・看板・文書などを模したもの
STYLES = {
    "screen": {"bg": (18, 32, 64), "fg": (240, 244, 255)},        # 紺の画面に白字（ナビ・エアコン）
    "warning": {"bg": (10, 10, 10), "fg": (255, 176, 0)},         # 黒地に橙（警告表示）
    "sign_yellow": {"bg": (250, 210, 20), "fg": (20, 20, 20)},    # 黄色の看板に黒字（工事）
    "board": {"bg": (0, 90, 60), "fg": (255, 255, 255)},          # 緑の案内板に白字
    "document": {"bg": (250, 250, 245), "fg": (25, 25, 25)},      # 白い紙に黒字
    "noren": {"bg": (150, 20, 30), "fg": (255, 255, 255)},        # 赤地に白字（のれん風、縦書き）
}


def find_font() -> tuple[str, int]:
    for path, idx in FONT_CANDIDATES:
        if Path(path).exists():
            return path, idx
    raise FileNotFoundError("日本語フォントが見つからない。FONT_CANDIDATES に追加すること")


def _font(path: str, idx: int, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size=size, index=idx)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font, max_w: int) -> list[str]:
    """日本語は単語の区切りが無いので、1 文字ずつ足して幅を超えたら改行する。"""
    lines, cur = [], ""
    for ch in text:
        if draw.textlength(cur + ch, font=font) > max_w and cur:
            lines.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        lines.append(cur)
    return lines


def render(spec: dict, font_path: str, font_idx: int) -> Image.Image:
    st = STYLES[spec["style"]]
    img = Image.new("RGB", (D.SYNTH_W, D.SYNTH_H), st["bg"])
    dr = ImageDraw.Draw(img)

    if spec.get("vertical"):
        # 縦書き: 1 文字ずつ縦に並べる（中央揃え）
        text, size = spec["lines"][0]
        f = _font(font_path, font_idx, size)
        step = int(size * 1.15)
        y = (D.SYNTH_H - step * len(text)) // 2
        for ch in text:
            w = dr.textlength(ch, font=f)
            dr.text(((D.SYNTH_W - w) / 2, y), ch, font=f, fill=st["fg"])
            y += step
        return img

    if spec.get("text_block"):
        # 段落: 左右に余白をとって折り返す（左揃え）
        size = spec["block_size"]
        f = _font(font_path, font_idx, size)
        lines = _wrap(dr, spec["text_block"], f, D.SYNTH_W - 120)
        step = int(size * 1.6)
        y = (D.SYNTH_H - step * len(lines)) // 2
        for ln in lines:
            dr.text((60, y), ln, font=f, fill=st["fg"])
            y += step
        return img

    # 行ごとに大きさを変えて中央揃えで縦に並べる
    fonts = [(_font(font_path, font_idx, sz), txt, sz) for txt, sz in spec["lines"]]
    heights = [int(sz * 1.35) for _, _, sz in fonts]
    y = (D.SYNTH_H - sum(heights)) // 2
    for (f, txt, sz), h in zip(fonts, heights):
        w = dr.textlength(txt, font=f)
        dr.text(((D.SYNTH_W - w) / 2, y), txt, font=f, fill=st["fg"])
        y += h
    return img


def build(force: bool = False) -> dict:
    """合成画像を作り、{id: {"path", "font"}} を返す。既にあるものは作り直さない。"""
    OUT.mkdir(parents=True, exist_ok=True)
    meta_path = OUT / "synth_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    font_path, font_idx = None, None
    for spec in D.SYNTH_SPECS:
        p = OUT / f"{spec['id']}.png"
        if p.exists() and not force and spec["id"] in meta:
            continue
        if font_path is None:
            font_path, font_idx = find_font()
        render(spec, font_path, font_idx).save(p)
        meta[spec["id"]] = {"path": str(p.relative_to(C.EVAL_DIR)).replace("\\", "/"),
                            "font": Path(font_path).name}
        print(f"  作成 {p.name}（{Path(font_path).name}）")
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    meta = build(a.force)
    print(f"合成画像 {len(meta)} 枚 -> {OUT}")


if __name__ == "__main__":
    main()
