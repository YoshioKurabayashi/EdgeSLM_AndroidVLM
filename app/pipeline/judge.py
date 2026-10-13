# -*- coding: utf-8 -*-
"""答えの自動判定（2026-09-27）。

## 方針

小型の VLM は答え方が揺れる（「6台です」「６台」「自転車が6台あります」）。完全一致では正解を落としすぎるので、
**「本質の語が含まれているか」**で判定する。ただし次の 2 点で甘くなりすぎないようにする。

1. 文字の表記ゆれは正規化してから比べる（全角→半角、空白、波ダッシュ）
2. 数字は「別の数字の一部」にならないよう正規表現で書く（例: 「200円」は「1,200円」の中にも現れる）

自動判定は**一次判定**であって、最終的な結論は答えの文を人が読んで確かめる（結果の JSON に答えを全部残す）。

## 判定の種類

- keywords: グループのリスト。各グループは言い換えの候補（どれか 1 つが含まれればそのグループは当たり）。
  全グループが当たれば ok=True。当たったグループの割合を score に入れる（部分点）
- ref_text: 正解の文との「文字の再現率」（正解の文字のうち、答えに同じ順で現れた割合＝LCS の長さ÷正解の長さ）。
  0.9 以上を ok とする（句読点の揺れ程度は許す）
- どちらも無い: 判定しない（ok=None）
"""
from __future__ import annotations

import re
import unicodedata


def normalize(s: str) -> str:
    """比較の前に表記をそろえる。

    - NFKC: 全角英数字→半角（「６台」→「6台」、「Ｂ」→「B」）、半角カナ→全角カナ
    - 行頭の箇条書きの番号（「1. 」「2) 」）を消す（2026-09-28 追加。正解「4つ」が、「5つ…1. … 4. …」の番号「4.」に
      当たって正解扱いになっていた）。「- 4. 」「**4.** 」のような Markdown の飾り付きも消す。
      番号の後に空白があるものだけ消すので、行頭の「3.5m」や「4.人参」は残る
    - 数字と数字の間の空白は「/」に置き換える（「9時　05　35」が「9時0535」にくっつかないように）
    - それ以外の空白（半角・全角）を消す
    - 波ダッシュ類をそろえる（〜 ～ ~）
    - 英字は小文字にそろえる（BOOK-OFF / Book-Off）
    """
    s = unicodedata.normalize("NFKC", s)
    s = re.sub(r"(?m)^[ \t]*(?:[-*+][ \t]+)?(?:\*\*)?[0-9]{1,2}[.)](?:\*\*)?[ \t]+", "", s)
    s = s.replace("〜", "~").replace("～", "~")
    s = re.sub(r"(?<=\d)\s+(?=\d)", "/", s)
    s = re.sub(r"\s+", "", s)
    return s.lower()


def _hit(cand: str, ans_n: str) -> bool:
    """候補 1 つが、正規化済みの答えに含まれるか。`re:` で始まる候補は正規表現として扱う。

    正規表現は**正規化した後の答え**（半角・空白なし・英字は小文字）に掛かる。英字を含む正規表現は小文字で書くこと。
    """
    if cand.startswith("re:"):
        return re.search(cand[3:], ans_n) is not None
    return normalize(cand) in ans_n


def judge_keywords(answer: str, groups: list[list[str]]) -> dict:
    ans_n = normalize(answer)
    hits = [any(_hit(c, ans_n) for c in g) for g in groups]
    return {"ok": all(hits), "score": round(sum(hits) / len(hits), 4), "hits": hits}


def lcs_len(a: str, b: str) -> int:
    """最長共通部分列の長さ（文字単位）。OCR の「読み上げた文」がどれだけ正解の文を含むかに使う。"""
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0]
        for j, cb in enumerate(b):
            cur.append(prev[j] + 1 if ca == cb else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def judge_ref_text(answer: str, ref: str, ok_th: float = 0.9) -> dict:
    # 句読点は読み上げで落ちやすいので、比べる前に取り除く（文字の読み取りの力だけを見る）
    strip = lambda s: re.sub(r"[、。，．,.「」『』（）()・:：]", "", normalize(s))
    a, r = strip(answer), strip(ref)
    rec = lcs_len(r, a) / max(len(r), 1)
    return {"ok": rec >= ok_th, "score": round(rec, 4), "hits": None}


def judge(answer: str, item: dict) -> dict:
    """評価項目 1 つを判定する。item は manifest の 1 件（keywords / ref_text を持つ）。"""
    if item.get("ref_text"):
        return judge_ref_text(answer, item["ref_text"])
    if item.get("keywords"):
        return judge_keywords(answer, item["keywords"])
    return {"ok": None, "score": None, "hits": None}


# ---------------------------------------------------------------------------
# JA-VG-VQA-500 の正解文 → 判定の語
# ---------------------------------------------------------------------------
# 正解は「晴天」「6台」「ピザがあります。」「ソファの上です。」のように文の形がまちまち。
# 語尾（です・ます・があります 等）と句読点を落とした「芯」を 1 つの語として使う。
# ★否定の語尾（ません・いません）は落とさない。落とすと「ありません」が「あり」になり、「あります」と区別できなくなる
# ★芯が 1 文字になるときは、その 1 文字が漢字か数字のときだけ落とす（「鳥です」→「鳥」、「1です」→「1」は落とし、
#   「います」→「い」は落とさない）。2026-09-28 修正: 前は 1 文字なら一律に落とさず、「鳥です」のまま探して正解を落としていた
_TAILS = ["があります", "がいます", "をしています", "しています", "でした", "です", "ました", "ます",
          "がある", "がいる"]


def _ok_core(rest: str) -> bool:
    """語尾を落とした残り（芯）として使ってよいか。2 文字以上、または漢字・数字 1 文字。"""
    if len(rest) >= 2:
        return True
    if len(rest) != 1:
        return False
    c = ord(rest)
    kanji = (0x4E00 <= c <= 0x9FFF or 0x3400 <= c <= 0x4DBF or 0xF900 <= c <= 0xFAFF or rest in "々〆〇")
    return kanji or rest.isdigit()


def answer_to_keywords(ans: str) -> list[list[str]]:
    core = unicodedata.normalize("NFKC", ans).strip()
    core = re.sub(r"[。．.!！?？、]+$", "", core)
    changed = True
    while changed:                      # 語尾が重なっていることがある（「〜しています。」）ので繰り返し落とす
        changed = False
        for t in _TAILS:
            # 芯が短すぎる（ひらがな 1 文字など）なら落とさない。どんな答えにも含まれてしまうため
            if core.endswith(t) and _ok_core(core[: -len(t)]):
                core = core[: -len(t)]
                changed = True
    core = core.strip()
    # 数字＋短い助数詞（「6台」「0枚」「3.5m」）は、数字が別の数字の一部にならない正規表現にする。
    # 「10時30分」のように数字が 2 つ以上あるものは、そのままの文字列で比べる
    m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(\D{0,3})", core)
    if m:
        num = re.escape(m.group(1))
        return [[f"re:(?<![0-9.]){num}(?![0-9]|\\.[0-9])"]]
    return [[core]]


if __name__ == "__main__":
    # 簡単な自己テスト: python -m app.pipeline.judge
    assert judge_keywords("６台あります。", answer_to_keywords("6台"))["ok"]
    assert not judge_keywords("16台あります。", answer_to_keywords("6台"))["ok"]
    assert judge_keywords("晴天です", answer_to_keywords("晴天"))["ok"]
    assert answer_to_keywords("ピザがあります。") == [["ピザ"]]
    assert judge_keywords("最大料金は1,200円です", [["re:(?<![0-9,])200(?![0-9])"]])["ok"] is False
    assert judge_keywords("30分200円です", [["re:(?<![0-9,])200(?![0-9])"]])["ok"] is True
    assert judge_keywords("9時5分と9時35分です", [["re:(?<![0-9])0?5(?![0-9])"], ["re:(?<![0-9])35(?![0-9])"]])["ok"]
    assert not judge_keywords("9時35分です", [["re:(?<![0-9])0?5(?![0-9])"], ["re:(?<![0-9])35(?![0-9])"]])["ok"]
    assert judge_keywords("「ＢＯＯＫ－ＯＦＦ」です", [["BOOK-OFF"]])["ok"]
    # 空白で区切られた数字がくっつかないこと（合成の時刻表と同じ書き方）
    assert judge_keywords("9時　05　35", [["re:(?<![0-9])0?5(?![0-9])"], ["re:(?<![0-9])35(?![0-9])"]])["ok"]
    # 否定の語尾を落とさないこと
    assert answer_to_keywords("ありません") == [["ありません"]]
    assert not judge_keywords("雨は降っています", answer_to_keywords("雨は降っていません"))["ok"]
    assert answer_to_keywords("居ません") == [["居ません"]]
    # 小数・時刻
    assert judge_keywords("3.5メートルです", answer_to_keywords("3.5m"))["ok"]
    assert not judge_keywords("13.5メートル", answer_to_keywords("3.5m"))["ok"]
    assert answer_to_keywords("10時30分") == [["10時30分"]]
    # 2026-09-28: 箇条書きの番号を数字として拾わない
    assert not judge_keywords("画像には**5つ**の容器があります。\n\n1. バナナ\n2. 小麦\n3. 煮込み\n4. 人参\n5. ブロッコ",
                              answer_to_keywords("４つ"))["ok"]
    assert judge_keywords("器は4つです。\n1. 茶碗", answer_to_keywords("４つ"))["ok"]
    assert judge_keywords("3.5mです", answer_to_keywords("3.5m"))["ok"]            # 行頭の小数は消さない
    assert not judge_keywords("5つあります\n- 4. 人参\n**4.** 人参", answer_to_keywords("４つ"))["ok"]  # 飾り付きの番号
    # 2026-09-28: 漢字・数字 1 文字の芯
    assert answer_to_keywords("鳥です。") == [["鳥"]]
    assert judge_keywords("水辺にいるカモメのような鳥が写っています。", answer_to_keywords("鳥です。"))["ok"]
    assert answer_to_keywords("います") == [["います"]]
    assert judge_keywords("1頭の動物が見えます", answer_to_keywords("1です"))["ok"]
    r = judge_ref_text("本日は道路工事のため国道20号線の一部区間で片側交互通行となります",
                       "本日は道路工事のため、国道20号線の一部区間で片側交互通行となります。通行の際は誘導員の指示に従ってください。")
    assert 0.5 < r["score"] < 0.7, r
    print("judge: ok")
