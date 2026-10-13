# -*- coding: utf-8 -*-
"""Phase 1: PC で画像つきの日本語の品質を測る（2026-09-27、09-28 改）。

    python -m app.pipeline.ref_vlm run                                   # 既定: native × 70/140/280、全問
    python -m app.pipeline.ref_vlm run --prep fixed --budgets 70         # 端末と同じ固定の格子で 70 トークン
    python -m app.pipeline.ref_vlm run --prep fixed --budgets 70 --feed step   # 端末と同じ流し方（1 トークンずつ）
    python -m app.pipeline.ref_vlm run --prep fixed --budgets 70 --vision own  # 画像エンコーダを自前の配線（vit_graph）に
    python -m app.pipeline.ref_vlm run --sets car,synth --limit 5        # 一部だけ（動作確認）
    python -m app.pipeline.ref_vlm rejudge                               # 保存済みの答えを、いまの判定で付け直す
    python -m app.pipeline.ref_vlm report                                # 集計して results/P1_ref_vlm/summary.md を作る

## 何を測るか

1. **画像トークン数（70 / 140 / 280）で品質がどう変わるか。** 端末では待ち時間が画像トークン数にほぼ比例する
   （いまの文章グラフは 1 回 1 トークン、約 77ms）。どこまで減らせるかを、用途ごとに決める材料にする
2. **前処理を端末と同じ固定の格子（fixed）にしたときの差。** HF の既定（native）は画像ごとに縦横比を保つが、
   端末では形を固定する必要がある。余白を黒で埋めること（レターボックス）の影響を見る
3. 1 問ごとの時間（最初のトークンまで・全体）。**PC の時間は端末の見込みには使えない**（参考値）
4. （09-28 追加）**流し方の差。** `--feed hf`（既定）は HF の本来の経路でプロンプトを一度に食わせる。
   `--feed step` は端末と同じ入れ方で、プロンプトも 1 トークンずつ食わせる（hf_vlm.generate_device_style）。
   SRQ ありでは 2 つの logits の 1 位が 77〜89% しか一致しない（check_wiring）ので、答えの品質が保たれるかを見る。
   report は同じ条件の hf と step を並べて比べる
5. （09-28 追加）**画像エンコーダの差し替え。** `--vision hf`（既定）は HF の画像エンコーダ。`own` は自前の配線（vit_graph.py、
   SRQ あり＝QAT の丸めを真似る）、`own_nosrq` は同じく SRQ なし。特徴を文章側の画像の位置に入れて生成する
   （hf_vlm.generate_with_features）。自前の特徴と HF の特徴の cos も 1 問ごとに記録する。
   **なぜ要るか:** 画像エンコーダは SRQ の 8bit の丸めにとても敏感で、計算の順序が違うだけで特徴が大きく変わる
   （2-1: 自前 SRQ あり ↔ HF SRQ あり で cos 最小 0.84）。端末の特徴も同じくらい変わるはずなので、
   「特徴が 1 割〜3 割変わっても答えの品質は保たれるか」を先に確かめる（Phase 1 の「全体の点で比べる」と同じ考え方）
   `npz:<パス>` は、別に計算しておいた特徴（eval_qdq_vit feats の量子化した画像エンコーダ、端末の出力など）を使う。
   npz は images（画像の相対パス）と feats [K,M,1536] を持つ

## 手順（EdgeSLM_AndroidGemma4 の ref_qat.py と同じ考え方）

- 重みは QAT の値、活性は SRQ（8bit の丸め）あり＝端末で目指す答え。`--no-srq` で外せる
- 貪欲（argmax）、止めるトークンで止める、最大トークン数は問題ごと（既定 64）
- 1 問ごとに結果のファイルへ書く。途中で止めても、次に同じ条件で動かせば終わった問題は飛ばす

## 結果

`results/P1_ref_vlm/{prep}_{budget}{_nosrq}{_step}{_v<vision>}.json`（1 条件 1 ファイル）と `summary.md`（report で作る）

## 判定を直したとき（rejudge）

report は集計のたびに、いまの judge.py で判定を付け直す（ファイルごとに判定の版が混ざらないように）。
`rejudge` はそれを JSON と manifest.json に書き戻すためのもの: すべての結果ファイルの judge 欄を書き換え、
汎用 VQA（vg）の判定の語は、正解の文から answer_to_keywords で作り直す
（車・合成・Wild の語は eval_defs.py の手書きなので触らない。vg の語を手で足しても rejudge で上書きされる）。

## 流し方の比較で「最初のトークンが同じか」も見たいとき

2026-09-28 より前の結果には生成したトークンの ID（gen_ids）が無い。`--refill-ids` を付けて同じ条件を動かすと、
gen_ids の無い問題だけやり直す（貪欲なので答えは同じになるはず）。
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from app.pipeline import vlm_common as C
from app.pipeline.judge import answer_to_keywords, judge

RES = C.RESULTS / "P1_ref_vlm"
MANIFEST = C.EVAL_DIR / "manifest.json"


def load_manifest() -> dict:
    if not MANIFEST.exists():
        raise SystemExit(f"{MANIFEST} が無い。先に python -m app.pipeline.fetch_eval_data を動かすこと")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def vision_label(vision: str) -> str:
    """ファイル名・表の見出し用の短い名前（npz:work/feats/vit_b70_21x30_srq.npz → npz-vit_b70_21x30_srq）。"""
    return "npz-" + Path(vision[4:]).stem if vision.startswith("npz:") else vision


def out_path(prep: str, budget: int, srq: bool, feed: str = "hf", vision: str = "hf") -> Path:
    return RES / (f"{prep}_{budget}{'' if srq else '_nosrq'}{'' if feed == 'hf' else '_' + feed}"
                  f"{'' if vision == 'hf' else '_v' + vision_label(vision)}.json")


def current_items(man: dict) -> dict:
    """manifest の問題を id で引く辞書。汎用 VQA（vg）の判定の語は、いまの answer_to_keywords で作り直す
    （report が「いまの判定」で集計するため。manifest への書き戻しは rejudge が行う）。"""
    out = {}
    for it in man["items"]:
        if it["set"] == "vg" and it.get("answer_ref"):
            it = dict(it, keywords=answer_to_keywords(it["answer_ref"]))
        out[it["id"]] = it
    return out


def run_name(d: dict) -> str:
    """結果ファイルの条件を表の見出しにする（例: fixed/70、native/280/step）。feed の無い古いファイルは hf。"""
    return (f"{d['prep']}/{d['budget']}{'' if d['srq'] else '/nosrq'}{'' if d.get('feed', 'hf') == 'hf' else '/' + d['feed']}"
            f"{'' if d.get('vision', 'hf') == 'hf' else '/v:' + vision_label(d['vision'])}")


def run(a) -> None:
    from app.pipeline import hf_vlm as H
    man = load_manifest()
    items = man["items"]
    if a.sets:
        items = [it for it in items if it["set"] in a.sets.split(",")]
    if a.only:
        items = [it for it in items if it["id"] in a.only.split(",")]
    if a.limit:
        # 各 set から先頭 limit 問ずつ（動作確認用）
        cnt, sel = defaultdict(int), []
        for it in items:
            if cnt[it["set"]] < a.limit:
                sel.append(it)
                cnt[it["set"]] += 1
        items = sel
    budgets = [int(b) for b in a.budgets.split(",")]
    # 使えない画像トークン数は、モデルを読む前に弾く（読み込みに 1 分ほどかかるため）
    ok_b = C.FIXED_GRIDS if a.prep == "fixed" else C.SUPPORTED_SOFT_TOKENS
    bad = [b for b in budgets if b not in ok_b]
    if bad:
        raise SystemExit(f"画像トークン数 {bad} は prep={a.prep} では使えない（使えるもの: {sorted(ok_b)}）")
    if a.vision not in ("hf", "own", "own_nosrq") and not a.vision.startswith("npz:"):
        raise SystemExit(f"--vision は hf / own / own_nosrq / npz:<パス> のどれか（{a.vision}）")
    if a.vision != "hf" and a.prep != "fixed":
        raise SystemExit("--vision own/own_nosrq/npz は固定の格子（--prep fixed）でだけ使える（格子を固定して作った特徴のため）")
    npz_feats, vision_src = None, None
    if a.vision.startswith("npz:"):
        zp = Path(a.vision[4:])
        zp = zp if zp.is_absolute() else C.ROOT / zp
        with np.load(zp) as z:                          # ★配列は 1 回だけ読む（z["feats"] は読むたびにファイル全体を読み直す）
            F = z["feats"]
            npz_feats = {str(p): F[k] for k, p in enumerate(z["images"])}
            npz_tag = str(z["tag"]) if "tag" in z.files else None
        # 特徴を作った格子が、この実行の格子と同じか（縦長の格子の npz を横長の実行に使うなどを防ぐ）
        for b in budgets:
            gh, gw = C.FIXED_GRIDS[b]
            if npz_tag is not None and npz_tag != f"b{b}_{gh}x{gw}":
                raise SystemExit(f"npz の格子（{npz_tag}）が --budgets {b} の格子（b{b}_{gh}x{gw}）と違う")
        st_ = zp.stat()
        vision_src = {"path": str(zp), "size": st_.st_size, "mtime": int(st_.st_mtime)}
        print(f"  画像の特徴: {zp}（{len(npz_feats)} 枚、格子 {npz_tag}）", flush=True)
    srq = not a.no_srq
    model, proc = H.load(a.device, srq=srq)
    RES.mkdir(parents=True, exist_ok=True)
    gen_fn = H.generate_hf if a.feed == "hf" else H.generate_step
    if a.vision != "hf":
        import torch
    if a.vision in ("own", "own_nosrq"):
        from app.pipeline import vit_graph as V
        V.SRQ = (a.vision == "own")

    for budget in budgets:
        path = out_path(a.prep, budget, srq, a.feed, a.vision)
        vit = None
        if a.vision in ("own", "own_nosrq"):
            vit = V.VitGraph(V.QatFile(), C.FIXED_GRIDS[budget]).to(a.device).eval()
            print(f"  画像エンコーダ: 自前の配線（SRQ {'あり' if V.SRQ else 'なし'}）", flush=True)
        done = {}
        if path.exists():
            prev = json.loads(path.read_text(encoding="utf-8"))
            # 特徴のファイルを作り直していたら、前の結果に続けない（古い特徴の答えと混ざるため）
            if vision_src and prev.get("vision_src") not in (None, vision_src):
                raise SystemExit(f"{path.name} は別の特徴のファイル（{prev.get('vision_src')}）で作った結果。"
                                 "消すか名前を変えてから動かし直すこと")
            for r in prev["results"]:
                done[r["id"]] = r
        if a.refill_ids:
            # 生成したトークンの ID（gen_ids）を保存していない古い結果は、やり直す（貪欲なので答えは同じになるはず）
            done = {k: v for k, v in done.items() if v.get("gen_ids")}
        results = list(done.values())
        print(f"== prep={a.prep} budget={budget} SRQ={srq} feed={a.feed} vision={a.vision}  {len(items)} 問"
              f"（済み {len(done)}）", flush=True)
        for it in items:
            if it["id"] in done:
                continue
            img = Image.open(C.EVAL_DIR / it["image"])
            img_in = H.prepare_image(img, a.prep, budget)
            inputs, n_img, _ = H.build_inputs(proc, img_in, it["question"], budget, man["system_prompt"], a.device,
                                              expect_exact=(a.prep == "fixed"))
            extra = {}
            if a.vision == "hf":
                g = gen_fn(model, inputs, it.get("max_new", 64))
            else:
                t0v = time.time()
                if vit is not None:
                    with torch.no_grad():
                        feats = vit(inputs["pixel_values"].float())[0]                # [M,1536]
                else:
                    if it["image"] not in npz_feats:
                        raise SystemExit(f"npz に {it['image']} の特徴が無い")
                    feats = torch.from_numpy(npz_feats[it["image"]]).to(a.device)
                    if feats.shape[0] != n_img:
                        raise SystemExit(f"npz の特徴の数 {feats.shape[0]} と画像トークン数 {n_img} が合わない")
                t_vis = time.time() - t0v                                              # 画像エンコーダの時間
                hf_f = H.image_features(model, inputs).float()
                extra["feat_cos_vs_hf"] = round(float(torch.nn.functional.cosine_similarity(
                    feats.flatten(), hf_f.flatten(), dim=0)), 6)
                g = H.generate_with_features(model, inputs, feats, it.get("max_new", 64),
                                             prefill="step" if a.feed == "step" else "batch")
                g["t_first"] = round(g["t_first"] + t_vis, 3)
                g["t_total"] = round(g["t_total"] + t_vis, 3)
            ans = H.decode(proc, g["ids"])
            jd = judge(ans, it)
            r = {"id": it["id"], "set": it["set"], "tags": it["tags"], "question": it["question"],
                 "answer": ans, "answer_ref": it.get("answer_ref"), "judge": jd,
                 "n_image_tokens": n_img, "prompt_len": int(inputs["input_ids"].shape[1]),
                 "n_patches": H.n_valid_patches(inputs),              # パディングを除いたパッチ数
                 "gen_tokens": len(g["ids"]), "gen_ids": g["ids"], "t_first": g["t_first"], "t_total": g["t_total"],
                 "image_size": [img.width, img.height], **extra}
            results.append(r)
            # ★先に書いてから表示する（表示で止まっても結果は残る。止まった問題を毎回やり直して止まる、を避ける）
            C.write_json_atomic(path, {"prep": a.prep, "budget": budget, "srq": srq, "feed": a.feed, "vision": a.vision,
                                       "vision_src": vision_src, "device": a.device,
                                       "system_prompt": man["system_prompt"], "results": results})
            mark = {True: "○", False: "×", None: "－"}[jd["ok"]]
            print(f"  {mark} {it['id']:16s} [{n_img}tok {g['t_first']:.1f}s] {it['question'][:24]} → {ans[:60]}",
                  flush=True)
        print(f"-> {path}", flush=True)


# ---------------------------------------------------------------------------
# 集計
# ---------------------------------------------------------------------------
def _agg(rows: list[dict]) -> str:
    """正解数/判定数（部分点の平均）の文字列。"""
    j = [r for r in rows if r["judge"]["ok"] is not None]
    if not j:
        return "－"
    ok = sum(1 for r in j if r["judge"]["ok"])
    sc = sum(r["judge"]["score"] for r in j) / len(j)
    return f"{ok}/{len(j)}（{sc:.2f}）"


def report(_a) -> None:
    files = sorted(RES.glob("*.json"))
    if not files:
        raise SystemExit("結果が無い")
    runs = []
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        d["_file"] = f.name
        runs.append(d)
    runs.sort(key=lambda d: (d["prep"], d["budget"], not d["srq"], d.get("feed", "hf") != "hf",
                             d.get("vision", "hf") != "hf", d.get("vision", "hf")))
    # 判定は集計のたびに、いまの judge.py と manifest で付け直す（ファイルごとに判定の版が混ざらないように）
    items = current_items(load_manifest())
    for d in runs:
        for r in d["results"]:
            if r["id"] in items:
                r["judge"] = judge(r["answer"], items[r["id"]])
    names = [run_name(d) for d in runs]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        print(f"★同じ条件の結果ファイルが複数ある（ファイル名を付けて区別する）: {sorted(dup)}")
        names = [f"{n}({d['_file']})" if n in dup else n for n, d in zip(names, runs)]

    lines = ["# P1 — PC での画像つき日本語評価（集計）", "",
             f"作成: {time.strftime('%Y-%m-%d %H:%M')}。値は「正解数/自動判定した数（部分点の平均）」。",
             "自動判定はキーワードによる一次判定。最終判断は答えの文を読んで行う（各 JSON に全文あり）。",
             "判定は集計のたびに、いまの judge.py で付け直している（各 JSON に保存された判定は使わない）。", ""]

    # 出どころ（set）別
    lines += ["## 出どころ別", "", "| 出どころ | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for s in ("vg", "wild", "car", "synth"):
        row = [_agg([r for r in d["results"] if r["set"] == s]) for d in runs]
        lines.append(f"| {s} | " + " | ".join(row) + " |")
    row = [_agg(d["results"]) for d in runs]
    lines.append("| **全体** | " + " | ".join(row) + " |")

    # タグ別（用途の切り口）
    lines += ["", "## タグ別", "", "| タグ | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    tags = sorted({t for d in runs for r in d["results"] for t in r["tags"]})
    for t in tags:
        row = [_agg([r for r in d["results"] if t in r["tags"]]) for d in runs]
        lines.append(f"| {t} | " + " | ".join(row) + " |")

    # 時間（PC の参考値）
    lines += ["", "## 時間（PC の参考値。端末の見込みには使わない）", "",
              "| 条件 | 問題数 | 画像トークン（平均） | プロンプト長（平均） | 最初のトークンまで（中央値） | 生成トークン（平均） |",
              "|---|---|---|---|---|---|"]
    for nm, d in zip(names, runs):
        rs = d["results"]
        if not rs:
            continue
        tf = sorted(r["t_first"] for r in rs)[len(rs) // 2]
        lines.append(f"| {nm} | {len(rs)} | {sum(r['n_image_tokens'] for r in rs) / len(rs):.0f} | "
                     f"{sum(r['prompt_len'] for r in rs) / len(rs):.0f} | {tf:.2f}s | "
                     f"{sum(r['gen_tokens'] for r in rs) / len(rs):.1f} |")

    # 基準（HF の流し方・HF の画像エンコーダ）との比較: 流し方（step）や画像エンコーダ（own など）を変えた結果を、
    # 同じ前処理・画像トークン数・SRQ の基準と並べる
    def is_base(b):
        return b.get("feed", "hf") == "hf" and b.get("vision", "hf") == "hf"
    pairs = []
    for d in runs:
        if is_base(d):
            continue
        base = [b for b in runs if is_base(b) and (b["prep"], b["budget"], b["srq"]) ==
                (d["prep"], d["budget"], d["srq"])]
        if len(base) > 1:
            print(f"★{d['_file']} と比べる HF の結果が複数ある。{base[0]['_file']} を使う")
        if base:
            pairs.append((base[0], d))
    if pairs:
        lines += ["", "## 基準（HF の流し方・HF の画像エンコーダ）との比較", "",
                  "同じ問題どうしで比べる。「最初のトークンが同じ」は gen_ids を保存している結果どうしのときだけ数える。"
                  "「特徴の cos」は画像エンコーダを差し替えたときの、HF の特徴との cos（最小 / 中央値）。", "",
                  "| 条件 | 基準 | 問題数 | 答えが完全に同じ | 最初のトークンが同じ | ×→○ | ○→× | 基準の点 | この条件の点 | 特徴の cos |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for b, d in pairs:
            bb = {r["id"]: r for r in b["results"]}
            common = [r for r in d["results"] if r["id"] in bb]
            if not common:
                continue
            same = sum(1 for r in common if r["answer"] == bb[r["id"]]["answer"])
            with_ids = [r for r in common if r.get("gen_ids") and bb[r["id"]].get("gen_ids")]
            first = (f"{sum(1 for r in with_ids if r['gen_ids'][0] == bb[r['id']]['gen_ids'][0])}/{len(with_ids)}"
                     if with_ids else "－")
            up = sum(1 for r in common if bb[r["id"]]["judge"]["ok"] is False and r["judge"]["ok"] is True)
            dn = sum(1 for r in common if bb[r["id"]]["judge"]["ok"] is True and r["judge"]["ok"] is False)
            cs = sorted(r["feat_cos_vs_hf"] for r in common if "feat_cos_vs_hf" in r)
            fc = f"{cs[0]:.3f} / {cs[len(cs) // 2]:.3f}" if cs else "－"
            lines.append(f"| {run_name(d)} | {run_name(b)} | {len(common)} | {same}/{len(common)} | {first} | {up} | {dn} | "
                         f"{_agg([bb[r['id']] for r in common])} | {_agg(common)} | {fc} |")

    # 条件によって正誤が変わった問題（人が読むための一覧）
    lines += ["", "## 条件によって正誤が変わった問題", ""]
    by_id = defaultdict(dict)
    for nm, d in zip(names, runs):
        for r in d["results"]:
            by_id[r["id"]][nm] = r
    for iid, per in sorted(by_id.items()):
        oks = {nm: r["judge"]["ok"] for nm, r in per.items()}
        if len({v for v in oks.values() if v is not None}) > 1:
            q = next(iter(per.values()))["question"]
            lines.append(f"- **{iid}** {q}")
            for nm, r in per.items():
                mark = {True: "○", False: "×", None: "－"}[r["judge"]["ok"]]
                lines.append(f"  - {mark} {nm}: {r['answer'][:80]}".replace("\n", " ⏎ "))
    out = RES / "summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:40]))
    print(f"-> {out}")


def rejudge(_a) -> None:
    """保存済みの答えを、いまの judge.py で判定し直す（モデルは動かさない）。"""
    man = load_manifest()
    n_kw = 0
    for it in man["items"]:
        if it["set"] == "vg" and it.get("answer_ref"):
            kw = answer_to_keywords(it["answer_ref"])
            if kw != it.get("keywords"):
                print(f"  判定の語を更新 {it['id']}: {it.get('keywords')} → {kw}")
                it["keywords"] = kw
                n_kw += 1
    if n_kw:
        man["keywords_refreshed"] = time.strftime("%Y-%m-%d %H:%M:%S")
        C.write_json_atomic(MANIFEST, man)
    print(f"manifest: vg の判定の語を {n_kw} 件更新")
    by_id = {it["id"]: it for it in man["items"]}
    for f in sorted(RES.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        up, dn = [], []
        for r in d["results"]:
            it = by_id.get(r["id"])
            if it is None:
                continue
            new = judge(r["answer"], it)
            old = r["judge"]["ok"]
            if old is False and new["ok"] is True:
                up.append(r["id"])
            elif old is True and new["ok"] is False:
                dn.append(r["id"])
            r["judge"] = new
        d["rejudged"] = time.strftime("%Y-%m-%d %H:%M:%S")
        C.write_json_atomic(f, d)
        print(f"{f.name:28s} ×→○ {len(up):2d} {up}\n{'':28s} ○→× {len(dn):2d} {dn}")
    print("-> report で集計し直すこと")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "rejudge", "report"])
    ap.add_argument("--prep", default="native", choices=["native", "fixed"])
    ap.add_argument("--budgets", default="70,140,280")
    ap.add_argument("--sets", default="", help="vg,wild,car,synth のうち回すもの（空なら全部）")
    ap.add_argument("--only", default="", help="回す問題の id（カンマ区切り）")
    ap.add_argument("--limit", type=int, default=0, help="各 set から先頭 N 問だけ（動作確認用）")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--no-srq", action="store_true")
    ap.add_argument("--refill-ids", action="store_true",
                    help="gen_ids（生成したトークンの ID）の無い古い結果をやり直す（流し方の比較で最初のトークンを比べるため）")
    ap.add_argument("--vision", default="hf",
                    help="画像エンコーダ: hf（既定）、own＝自前の配線（SRQ あり）、own_nosrq＝自前の配線（SRQ なし）、"
                         "npz:<パス>＝別に計算した特徴（eval_qdq_vit feats・端末の出力）")
    ap.add_argument("--feed", default="hf", choices=["hf", "step"],
                    help="hf＝HF の本来の経路でプロンプトを一度に（既定）、step＝端末と同じ入れ方で 1 トークンずつ")
    a = ap.parse_args()
    C.utf8_stdout()
    {"run": run, "rejudge": rejudge, "report": report}[a.cmd](a)


if __name__ == "__main__":
    main()
