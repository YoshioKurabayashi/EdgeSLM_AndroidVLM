# -*- coding: utf-8 -*-
"""16 トークン版の NPU のずれの切り分け: **同じトークンを「まとまりの中の行 r」と「1 行だけ」で比べる**（2026-10-01）。

    python -m app.pipeline.probe_rows                    # 既定: 3 問 × 5 行（10 分前後）
    python -m app.pipeline.probe_rows --items syn_nav_S --rows 0,6,12

**EdgeSLM_Android の venv（$PY2）で動かす**。端末は USB でつなぐ。端末の 16 トークン版（models/ops/g4t16_W256_*）を使う（変換は要らない）。

## なぜ（2026-10-01 の probe_t16 の基準・probe_l0 の plain から）

- **層 0〜14 の K/V のずれは、1 トークン版でも 16 トークン版でもほぼ同じ**だった（同じ位置の行で、層 0 の K が文字の行 1.5〜2%・画像の行 10〜63%、
  層 1 で 10〜20%。行ごとの値までほぼ一致）。→ この部分は 16 行の形のせいではなく、NPU と PC の CPU の計算の違い（1 トークン版にもある）
- ところが **logits（答えの 1 語目を決める値）のずれは、16 トークン版のプロンプトの最後の回で 38〜61%、1 トークン版で 14〜33%** と 2 倍。
  16 トークン版でも、答えを 1 語ずつ作る回（本物 1 行＝0 行目）は 13% で小さい
- → 16 トークン版だけが悪くなるのは、**層 15〜34（層 13・14 の K/V を使い回す層）か、行を選ぶ所（logit_sel）で、本物の行が複数あるとき**と疑う

## この道具がすること

プロンプトの最後のまとまり（n 行）について、同じトークン（行 r）を 2 通りで通す。数学的には同じ計算になる:

| 通し方 | 入力 |
|---|---|
| **まとまりの中**（in） | 最後のまとまりの n 行をそのまま。logit_sel で行 r を選ぶ |
| **1 行だけ**（alone） | 行 0〜r−1 を past に入れ（K/V は PC の CPU で作ったもの）、行 r のトークンだけを 0 行目に置く（本物 1 行） |

どちらも端末の NPU と PC の CPU（同じ 16 トークン版の量子化済みグラフ）に通し、logits と層 0〜14 の K/V（行 r）のずれを比べる。
- in だけが大きくずれる（r が大きいほど） → まとまりの中の複数行の計算（層 15〜34 の注意など）か行の選び方に原因
- in と alone が同じくらい → 行の位置は関係ない（別の原因）

結果: `results/P5_t16/probe_rows.json`

## ★`--self`: 端末の NPU どうしで比べる（2026-10-03 に追加）

    python -m app.pipeline.probe_rows --self                 # 元の 16 トークン版
    python -m app.pipeline.probe_rows --self --model t16f    # 書き直した版

**なぜ**: 上の比べ方には 2 つの弱点があった（2026-10-03 に分かった）。

1. **比べる相手の PC の CPU が、画像の行では正しくない**。量子化済みグラフの正規化の途中の値（x²・平均・1/√）の目盛りは文章だけで決めたもので、
   画像の行では範囲からはみ出して切れる。NPU は正規化を 1 つの演算（RmsNorm）にまとめて計算するので切れない（重みのない正規化の出力の
   行ごとの二乗平均が NPU は 1.000、CPU は 0.56〜0.93）。つまり「NPU と CPU の差」は NPU の誤差とは限らない
2. 「まとまりの中」は、同じまとまりの手前の行の K/V を **NPU は NPU の値、CPU は CPU の値**で使う。「1 行だけ」は**どちらも CPU の値**を使う。
   なので「まとまりの中の方が差が大きい」のは、手前の行の差が積み重なっただけでも起きる（まとめ読みの不具合とは限らない）

**すること**: CPU を基準にせず、**NPU の「まとまりの中の行 r」と NPU の「1 行だけ」**を比べる。「1 行だけ」の手前の行（行 0〜r−1）の K/V は、
**NPU がまとまりの中で作った値**を past に入れる。こうすると 2 つは数学的に同じ計算で、入力も同じ（どちらも NPU の値）なので、
差があればそれは **NPU の中で、行をまとめたことだけによる違い**。層 0〜14 の K/V も行 r どうしで比べるので、どの層から違うかが分かる。
同じことを PC の CPU どうしでもやり（CPU の「まとまりの中」と CPU の「1 行だけ」）、丸めの違いだけで出る差の大きさ（物差しの目盛り）にする。

結果: `results/P5_t16/probe_rows_self.json`（`--model` を付けると `probe_rows_self_<種類>.json`）
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from app.pipeline import device_vit as D
from app.pipeline import vlm_common as C
from app.pipeline.diag_t16 import IMG, T16_QDQ, TABLES, Tables, load_items, load_results, old_prompt
from app.pipeline.probe_t16 import Cpu, Device, Feeder, rel

RES = C.RESULTS / "P5_t16"
DEFAULT_ITEMS = ["syn_nav_S", "car_12a", "vg_06"]


def logit_cmp(ln: np.ndarray, lc: np.ndarray) -> dict:
    ln, lc = np.asarray(ln, np.float64).reshape(-1), np.asarray(lc, np.float64).reshape(-1)
    tn, tc = np.argsort(-ln)[:5], np.argsort(-lc)[:5]
    return {"logits_rel": round(rel(ln, lc), 5), "top1_npu": int(tn[0]), "top1_cpu": int(tc[0]),
            "top1_same": bool(tn[0] == tc[0]), "top5_overlap": len(set(tn.tolist()) & set(tc.tolist())),
            "margin_cpu": round(float(lc[tc[0]] - lc[tc[1]]), 3)}


def kv_row(npu: dict, cpu: dict, row: int, T: int = 16) -> dict:
    """層 0〜14 の新しい K/V の、まとまりの中の行 row の相対 L2"""
    out = {}
    for name in sorted([k for k in cpu if k.startswith("present_")], key=lambda k: (int(k.rsplit("_", 1)[1]), k)):
        b = np.asarray(cpu[name], np.float64)
        a = np.asarray(npu[name], np.float64)
        lo = b.shape[2]
        i = lo - T + row
        _, kind, li = name.split("_")
        out[f"{kind}{li}"] = round(rel(a[:, :, i, :], b[:, :, i, :]), 5)
    return out


def pair_cmp(a: dict, row_a: int, b: dict, row_b: int, T: int = 16) -> dict:
    """同じトークンを 2 通りで通した出力どうしを比べる（a の行 row_a ↔ b の行 row_b。b を基準にする）。logits と層 0〜14 の新しい K/V"""
    la, lb = np.asarray(a["logits"], np.float64).reshape(-1), np.asarray(b["logits"], np.float64).reshape(-1)
    ta, tb = np.argsort(-la)[:5], np.argsort(-lb)[:5]
    kv = {}
    for name in sorted([k for k in b if k.startswith("present_")], key=lambda k: (int(k.rsplit("_", 1)[1]), k)):
        x, y = np.asarray(a[name], np.float64), np.asarray(b[name], np.float64)
        lo = y.shape[2]
        _, kind, li = name.split("_")
        kv[f"{kind}{li}"] = round(rel(x[:, :, lo - T + row_a, :], y[:, :, lo - T + row_b, :]), 6)
    return {"logits_rel": round(rel(la, lb), 6), "top1_a": int(ta[0]), "top1_b": int(tb[0]), "top1_same": bool(ta[0] == tb[0]),
            "top5_overlap": len(set(ta.tolist()) & set(tb.tolist())), "kv": kv}


def self_check(a, iid, cpu, fd, tabs, R, want, f_in, o_in, t16_rel, rows_all, last0) -> list[dict]:
    """NPU の「まとまりの中の行 r」↔ NPU の「1 行だけ」（手前の行の K/V は NPU がまとまりの中で作った値）。CPU どうしでも同じことをする"""
    n = len(R)
    feeds_in, cpu_in = [], []
    for r in want:
        f = {k_: v.copy() for k_, v in f_in.items()}
        f["logit_sel"][:] = 0.0
        f["logit_sel"][0, r, 0] = 1.0
        feeds_in.append(f)
        cpu_in.append(o_in if r == n - 1 else cpu.run(f))

    def alone_feeds(ctx: dict) -> list[dict]:
        """行 0〜r−1 の K/V を ctx（まとまりの中の出力）から past に入れ、行 r だけを 0 行目に置く"""
        out = []
        for r in want:
            g = Feeder(cpu.ins, tabs)
            g.past = {k_: v.copy() for k_, v in fd.past.items()}
            g.n_valid, g.abs_pos = fd.n_valid, fd.abs_pos
            if r:
                g.advance(ctx, r)
            out.append(g.feed([R[r]]))
        return out

    cpu_al = [cpu.run(f) for f in alone_feeds(o_in)]                   # CPU: 手前の行は CPU の値
    dev = Device(t16_rel, f"self_{iid}")
    npu_in, _ = dev.run(feeds_in, [f"in{r}" for r in want], keep=True)
    npu_al, _ = dev.run(alone_feeds(npu_in[0]), [f"al{r}" for r in want], keep=False)   # NPU: 手前の行は NPU の値
    recs = []
    for i, r in enumerate(want):
        rec = {"row": r, "image": rows_all[last0 + r][0] == IMG,
               "npu": pair_cmp(npu_in[i], r, npu_al[i], 0), "cpu": pair_cmp(cpu_in[i], r, cpu_al[i], 0)}
        recs.append(rec)
        x, y = rec["npu"], rec["cpu"]
        print(f"    行 {r:>2}: NPU どうし logits の差 {x['logits_rel'] * 100:6.2f}%（1 位 {'同じ' if x['top1_same'] else '違う'}）"
              f" 層0の K {x['kv']['k0'] * 100:5.2f}% 層7 {x['kv']['k7'] * 100:5.2f}% 層14 {x['kv']['k14'] * 100:5.2f}%"
              f" ／ CPU どうし logits {y['logits_rel'] * 100:6.2f}%（1 位 {'同じ' if y['top1_same'] else '違う'}）"
              f" 層14 の K {y['kv']['k14'] * 100:5.2f}%", flush=True)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=",".join(DEFAULT_ITEMS))
    ap.add_argument("--rows", default="", help="比べる行（カンマ区切り）。既定は 0・n/4・n/2・3n/4・n−1")
    ap.add_argument("--model", default="t16", choices=list(D.T16_KINDS),
                    help="端末の 16 トークン版（t16f = 正規化の平均と PLE の tanh を書き直した版、t16t = PLE の tanh だけ書き直した版。"
                         "比べる相手の CPU は元の量子化済みグラフのまま）")
    ap.add_argument("--self", dest="self_check", action="store_true",
                    help="端末の NPU どうし（まとまりの中 ↔ 1 行だけ。手前の行は NPU の値）で比べる。CPU を基準にしない")
    a = ap.parse_args()
    C.utf8_stdout()
    items_all, res = load_items(), load_results()
    sel = [x for x in a.items.split(",") if x]
    tabs = Tables(TABLES)
    D.ensure_app()
    t16_rel = D.push_t16(D.T16_KINDS[a.model])
    out_file = RES / f"probe_rows{'_self' if a.self_check else ''}{'' if a.model == 't16' else '_' + a.model}.json"
    D.adb("shell", f"am force-stop {D.PKG}")
    time.sleep(2.0)
    print(f"  端末の空きメモリ {D.wait_mem()}MiB", flush=True)
    cpu = Cpu(T16_QDQ)
    out = {"items": []}
    t0 = time.time()
    for iid in sel:
        it = items_all[iid]
        ids = old_prompt(it["tail"])
        if len(ids) != res["dev16"][iid]["prompt_len"]:
            raise SystemExit(f"{iid}: プロンプトの長さが評価の記録と合わない")
        feats = np.fromfile(it["feat"], np.float32).reshape(-1, tabs.H)
        rows_all, k = [], 0
        for tid in ids:
            if tid == IMG:
                rows_all.append((tid, feats[k]))
                k += 1
            else:
                rows_all.append((tid, None))
        # ---- 最後のまとまりの手前まで、PC の CPU で進める（入力はすべて CPU で作る＝NPU と CPU に同じものを渡す） ----
        fd = Feeder(cpu.ins, tabs)
        fd.reset()
        last0 = ((len(ids) - 1) // fd.T) * fd.T
        for c in range(0, last0, fd.T):
            o = cpu.run(fd.feed(rows_all[c:c + fd.T]))
            fd.advance(o, fd.T)
        R = rows_all[last0:]
        n = len(R)
        want = [int(x) for x in a.rows.split(",") if x] if a.rows else sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})
        want = [r for r in want if 0 <= r < n]
        print(f"\n  {iid}: プロンプト {len(ids)} 個、最後のまとまり {last0}〜{len(ids) - 1}（{n} 行）、比べる行 {want}", flush=True)
        # まとまりの中（行 r を選ぶ）
        f_in = fd.feed(R)
        o_in = cpu.run(f_in)                                   # 行 n−1 を選んだもの（K/V はどの行を選んでも同じ）
        if a.self_check:
            recs = self_check(a, iid, cpu, fd, tabs, R, want, f_in, o_in, t16_rel, rows_all, last0)
            out["items"].append({"id": iid, "prompt_len": len(ids), "last_chunk": [last0, len(ids) - 1], "rows": recs})
            C.write_json_atomic(out_file, out)
            continue
        feeds, tags, cpu_outs = [], [], []
        for r in want:
            f = {k_: v.copy() for k_, v in f_in.items()}
            f["logit_sel"][:] = 0.0
            f["logit_sel"][0, r, 0] = 1.0
            feeds.append(f)
            tags.append(("in", r))
            cpu_outs.append(o_in if r == n - 1 else cpu.run(f))
        # 1 行だけ（行 0〜r−1 を past に入れて、行 r を 0 行目に）
        for r in want:
            g = Feeder(cpu.ins, tabs)                         # 表（大きい）は共有し、past・位置だけ写す
            g.past = {k_: v.copy() for k_, v in fd.past.items()}
            g.n_valid, g.abs_pos = fd.n_valid, fd.abs_pos
            if r:
                g.advance(o_in, r)                             # まとまりの中で CPU が作った行 0〜r−1 の K/V
            f = g.feed([R[r]])
            feeds.append(f)
            tags.append(("alone", r))
            cpu_outs.append(cpu.run(f))
        print(f"    PC の CPU で {len(feeds)} 回ぶん（{time.time() - t0:.0f}s）。端末へ", flush=True)
        dev = Device(t16_rel, f"rows_{iid}")
        on, _ = dev.run(feeds, [f"{m}{r}" for m, r in tags], keep=False)
        recs = []
        for (mode, r), o_n, o_c in zip(tags, on, cpu_outs):
            rr = r if mode == "in" else 0
            rec = {"mode": mode, "row": r, "image": rows_all[last0 + r][0] == IMG, **logit_cmp(o_n["logits"], o_c["logits"]),
                   "kv": kv_row(o_n, o_c, rr), "ort_run_ms": o_n.get("_ort_run_ms")}
            recs.append(rec)
        by = {(x["mode"], x["row"]): x for x in recs}
        for r in want:
            x, y = by[("in", r)], by[("alone", r)]
            print(f"    行 {r:>2}{'（画像）' if x['image'] else ''}: logits の差 まとまりの中 {x['logits_rel'] * 100:6.2f}%"
                  f"（1 位 {'同じ' if x['top1_same'] else '違う'}）／1 行だけ {y['logits_rel'] * 100:6.2f}%"
                  f"（1 位 {'同じ' if y['top1_same'] else '違う'}）  層14 の K の差 {x['kv']['k14'] * 100:5.1f}% ／ {y['kv']['k14'] * 100:5.1f}%",
                  flush=True)
        out["items"].append({"id": iid, "prompt_len": len(ids), "last_chunk": [last0, len(ids) - 1], "rows": recs})
        C.write_json_atomic(out_file, out)
    D.adb("shell", f"am force-stop {D.PKG}")
    if a.self_check:
        rows = [x for it in out["items"] for x in it["rows"]]
        md_ = lambda xs: round(float(np.median(xs)) * 100, 3) if xs else None
        keys = sorted(rows[0]["npu"]["kv"], key=lambda k: (int(k[1:]), k[0])) if rows else []
        s = {"n": len(rows)}
        for side in ("npu", "cpu"):
            r0 = [x[side] for x in rows if x["row"] == 0]
            rp = [x[side] for x in rows if x["row"] > 0]
            s[side] = {"logits_rel_median_row0": md_([x["logits_rel"] for x in r0]),
                       "logits_rel_median_rowpos": md_([x["logits_rel"] for x in rp]),
                       "top1_same": sum(x["top1_same"] for x in r0 + rp),
                       "kv_median_rowpos": {k: md_([x["kv"][k] for x in rp]) for k in keys},
                       "kv_median_row0": {k: md_([x["kv"][k] for x in r0]) for k in keys}}
        out["summary"] = s
        C.write_json_atomic(out_file, out)
        for side, lab in (("npu", "端末の NPU どうし"), ("cpu", "PC の CPU どうし")):
            v = s[side]
            print(f"\n  ■ {lab}（まとまりの中 ↔ 1 行だけ）: logits の差の中央値 0 行目 {v['logits_rel_median_row0']}%・"
                  f"1 行目以降 {v['logits_rel_median_rowpos']}%、1 位が同じ {v['top1_same']}/{s['n']}")
            print("    1 行目以降の K の差（層ごと、%）: " + " ".join(f"{k[1:]}:{v['kv_median_rowpos'][k]}" for k in keys if k[0] == "k"))
        print("  読み方: 2 つは同じ計算なので、差は「行をまとめたことだけによる違い」。NPU どうしの差が CPU どうしの差（丸めの違いだけで出る大きさ）と"
              "同じくらい → まとめ読みに不具合は無い。NPU だけ大きい → その層から先に、行をまとめたときだけの違いがある")
        print(f"-> {out_file}（{(time.time() - t0) / 60:.1f} 分）")
        return
    ins = [x for it in out["items"] for x in it["rows"] if x["mode"] == "in"]
    al = [x for it in out["items"] for x in it["rows"] if x["mode"] == "alone"]
    md = lambda xs: float(np.median(xs)) * 100 if xs else float("nan")
    s = {"in_logits_rel_median": md([x["logits_rel"] for x in ins]), "alone_logits_rel_median": md([x["logits_rel"] for x in al]),
         "in_top1_same": sum(x["top1_same"] for x in ins), "alone_top1_same": sum(x["top1_same"] for x in al), "n": len(ins),
         "in_row0_logits_rel_median": md([x["logits_rel"] for x in ins if x["row"] == 0]),
         "in_rowpos_logits_rel_median": md([x["logits_rel"] for x in ins if x["row"] > 0])}
    out["summary"] = s
    C.write_json_atomic(out_file, out)
    print(f"\n  ■ logits の差の中央値: まとまりの中 {s['in_logits_rel_median']:.1f}%（0 行目 {s['in_row0_logits_rel_median']:.1f}%、"
          f"1 行目以降 {s['in_rowpos_logits_rel_median']:.1f}%）／1 行だけ {s['alone_logits_rel_median']:.1f}%")
    print(f"  ■ 1 位が同じ: まとまりの中 {s['in_top1_same']}/{s['n']}、1 行だけ {s['alone_top1_same']}/{s['n']}")
    print("  読み方: まとまりの中だけが大きい（とくに 1 行目以降）→ 複数行のときの層 15〜34 か行の選び方が原因。同じくらい → 行の位置は関係ない")
    print(f"-> {out_file}（{(time.time() - t0) / 60:.1f} 分）")


if __name__ == "__main__":
    main()
