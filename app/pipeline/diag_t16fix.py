# -*- coding: utf-8 -*-
"""書き直した 16 トークン版（g4t16f）が、PC の CPU で元の版と同じ計算になっていない理由を調べる（2026-10-02）。

    python -m app.pipeline.diag_t16fix          # 5〜10 分。メモリ 6GB 前後（グラフ 1.1GB を 1 つずつ読む）

**EdgeSLM_Android の venv（$PY2）で動かす**。端末は使わない。変換（build_context.sh）と同時に動かしてもよい。

## なぜ（2026-10-02 の build_t16fix の書き直しの段）

小さいグラフ（層 0・1）では、書き直す前と後の PC の CPU の差は 0.000% だった（`probe_fix`）。
ところが全体（35 層）では **logits の差が 18%、層 12〜14 の K/V で 5〜7%** 出た。考えられるのは 2 つ:

| 仮説 | 中身 | この道具で見えること |
|---|---|---|
| A. 書き直しの誤り | どこかの正規化・tanh で、書き直した計算が元と違う（形・重み・目盛りの取り違え） | その演算の**入力は同じなのに出力が大きく違う**所がある |
| B. 丸めの境目のゆらぎが育つ | 平均の計算の順序が変わる（足してから割る ↔ 1/N を掛けて足す）と、浮動小数の差が 1e-7 ほど出る。16bit に丸めるとき、ちょうど境目にある値だけ 1 目盛り（LSB）ずれ、35 層かけて大きくなる | 平均の丸めた後の値の違いは**ちょうど 1 目盛り**。元の版の入力をほんの少し（1e-6）揺らしただけでも同じくらい logits が変わる |

B なら、書き直しは正しく、変換したものはそのまま使える（量子化したグラフがもともと小さな揺れに敏感、という別の発見になる）。
A なら、変換をやめて書き直しを直す。

## すること

1. 元のグラフから「見張る値」を選ぶ: 正規化の平均（242 個）の入力・丸める前・丸めた後、tanh（70 個）の入力・出力・丸めた後
2. 元と書き直した版にその値を出力として足し、同じ入力（build_t16fix の確かめと同じ syn_nav_S の最初の 2 回ぶん）で PC の CPU に通す
3. 元の版をもう 1 回（同じ入力で結果が毎回同じか）、入力の埋め込みを相対 1e-6 だけ揺らした元の版（揺れへの敏感さ）も通す
4. ORT のグラフ最適化を切った場合（DISABLE_ALL）も、元と書き直した版で比べる（最適化の違いのせいでないか）
5. 見張る値をグラフの順に並べ、**入力が同じなのに出力が違う最初の所**と、丸めた後の違いが何目盛りかを出す

結果: `results/P5_t16/diag_t16fix.json`
"""
from __future__ import annotations

import gc
import json
import time

import numpy as np

from app.pipeline import vlm_common as C
from app.pipeline.build_t16fix import DST, RES, SRC


def rel(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def watch_list(m) -> list[dict]:
    """元のグラフから見張る値を、グラフの順（ONNX の演算は並びが計算の順）に選ぶ"""
    from onnx import numpy_helper
    inits = {t.name: t for t in m.graph.initializer}
    cons: dict[str, list] = {}
    for n in m.graph.node:
        for x in n.input:
            cons.setdefault(x, []).append(n)

    def q_dq_after(t):
        """t → Q → DQ の DQ の出力名と、Q の目盛り（1 目盛りの大きさ）"""
        qs = [n for n in cons.get(t, []) if n.op_type == "QuantizeLinear"]
        if len(qs) != 1:
            return None, None
        dqs = [n for n in cons.get(qs[0].output[0], []) if n.op_type == "DequantizeLinear"]
        sc = inits.get(qs[0].input[1])
        lsb = float(numpy_helper.to_array(sc).reshape(-1)[0]) if sc is not None else None
        return (dqs[0].output[0] if dqs else None), lsb

    out, i_mean, i_tanh = [], 0, 0
    for n in m.graph.node:
        if n.op_type not in ("ReduceMean", "Tanh"):
            continue
        kind = "mean" if n.op_type == "ReduceMean" else "tanh"
        idx = i_mean if kind == "mean" else i_tanh
        dq, lsb = q_dq_after(n.output[0])
        out.append({"kind": kind, "idx": idx, "node": n.name, "in": n.input[0], "out": n.output[0], "dq": dq, "lsb": lsb})
        if kind == "mean":
            i_mean += 1
        else:
            i_tanh += 1
    return out


def make_feeds(ins: dict, run_first):
    """build_t16fix.equiv_check と同じ入力（syn_nav_S の最初の 32 行 = 2 回ぶん）。2 回目の past は元の版の 1 回目の出力"""
    from app.pipeline.diag_t16 import IMG, TABLES, Tables, load_items, old_prompt
    from app.pipeline.probe_t16 import Feeder
    tabs = Tables(TABLES)
    it = load_items()["syn_nav_S"]
    ids = old_prompt(it["tail"])
    feats = np.fromfile(it["feat"], np.float32).reshape(-1, tabs.H)
    rows, k = [], 0
    for tid in ids[:32]:
        if tid == IMG:
            rows.append((tid, feats[k]))
            k += 1
        else:
            rows.append((tid, None))
    fd = Feeder(ins, tabs)
    fd.reset()
    f0 = fd.feed(rows[:16])
    o0 = run_first(f0)
    fd.advance(o0, 16)
    f1 = fd.feed(rows[16:32])
    return [f0, f1], o0


def session(path, extra: list[str], opt: str):
    """グラフに見張る値を出力として足して ORT の CPU で開く（グラフ 1.1GB は 2GB 未満なのでバイト列で渡せる）"""
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper
    t0 = time.time()
    m = onnx.load(str(path))
    have = {o for n in m.graph.node for o in n.output}
    names = [t for t in extra if t in have]
    miss = [t for t in extra if t not in have]
    base_out = [o.name for o in m.graph.output]
    m.graph.output.extend([helper.make_tensor_value_info(t, TensorProto.FLOAT, None) for t in names if t not in base_out])
    so = ort.SessionOptions()
    so.graph_optimization_level = {"basic": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                                   "none": ort.GraphOptimizationLevel.ORT_DISABLE_ALL}[opt]
    blob = m.SerializeToString()
    del m
    gc.collect()
    s = ort.InferenceSession(blob, so, providers=["CPUExecutionProvider"])
    del blob
    gc.collect()
    print(f"  開いた {path.name}（最適化 {opt}、見張る値 {len(names)} 本、無い {len(miss)} 本、{time.time() - t0:.0f}s）", flush=True)
    return s, miss


def run(s, feed: dict) -> dict:
    outs = [o.name for o in s.get_outputs()]
    return dict(zip(outs, s.run(None, feed)))


def main():
    import onnx
    C.utf8_stdout()
    t_all = time.time()
    if not DST.exists():
        raise SystemExit(f"{DST} が無い（先に build_t16fix --steps rewrite）")
    m = onnx.load(str(SRC), load_external_data=False)
    W = watch_list(m)
    del m
    gc.collect()
    extra = []
    for w in W:
        extra += [w["in"], w["out"]] + ([w["dq"]] if w["dq"] else [])
    extra = list(dict.fromkeys(extra))
    print(f"  見張る値: 平均 {sum(w['kind'] == 'mean' for w in W)} 個・tanh {sum(w['kind'] == 'tanh' for w in W)} 個（{len(extra)} 本）", flush=True)

    res: dict[str, list[dict]] = {}
    # ---- 元の版（最適化 basic = Cpu と同じ設定）: 基準・もう 1 回・揺らした入力 ----
    s, _ = session(SRC, extra, "basic")
    ins = {v.name: [int(x) for x in v.shape] for v in s.get_inputs()}
    feeds, _ = make_feeds(ins, lambda f: run(s, f))
    res["orig"] = [run(s, f) for f in feeds]
    res["orig_again"] = [run(s, f) for f in feeds]
    rng = np.random.default_rng(0)
    pert = []
    for f in feeds:
        g = {k: v.copy() for k, v in f.items()}
        e = g["inputs_embeds"]
        e += (1e-6 * float(np.sqrt(np.mean(e.astype(np.float64) ** 2))) * rng.standard_normal(e.shape)).astype(np.float32)
        pert.append(g)
    res["orig_pert1e-6"] = [run(s, f) for f in pert]
    del s
    gc.collect()
    # ---- 書き直した版（basic）----
    s, miss = session(DST, extra, "basic")
    res["fix"] = [run(s, f) for f in feeds]
    del s
    gc.collect()
    # ---- 最適化なし（元・書き直し）は 1 回目の入力だけ ----
    s, _ = session(SRC, extra, "none")
    res["orig_noopt"] = [run(s, feeds[0])]
    del s
    gc.collect()
    s, _ = session(DST, extra, "none")
    res["fix_noopt"] = [run(s, feeds[0])]
    del s
    gc.collect()

    # ---- 比べる ----
    def cmp(tag: str, ref: str = "orig") -> dict:
        b = 0
        A, B = res[tag][b], res[ref][b]
        lg = {f"logits_b{bb}": round(rel(res[tag][bb]["logits"], res[ref][bb]["logits"]), 6) for bb in range(len(res[tag]))}
        rows = []
        for w in W:
            r = {"kind": w["kind"], "idx": w["idx"], "node": w["node"]}
            for key in ("in", "out", "dq"):
                t = w[key]
                if t and t in A and t in B:
                    r[key] = rel(A[t], B[t])
            if w["dq"] and w["dq"] in A and w["dq"] in B and w["lsb"]:
                d = np.abs(np.asarray(A[w["dq"]], np.float64) - np.asarray(B[w["dq"]], np.float64)) / w["lsb"]
                r["dq_n_diff"] = int((d > 0.5).sum())
                r["dq_max_lsb"] = round(float(d.max()), 2)
                r["n"] = int(d.size)
            rows.append(r)
        # 入力が同じ（1e-6 未満）なのに、丸めた後が 2 目盛り以上違う（丸めた後が無い所は丸める前が 1e-4 以上違う）最初の所＝A の印。
        # 書き直した PLE の tanh は「出力」が書き直した版では丸めた後の値になるので、丸めた後で比べる
        def jump(r):
            if "dq_max_lsb" in r:
                return r["dq_max_lsb"] > 1.5
            return r.get("out", 0) > 1e-4
        first_out = next((r for r in rows if r.get("in", 1) < 1e-6 and jump(r)), None)
        first_dq = next((r for r in rows if r.get("in", 1) < 1e-6 and r.get("dq_n_diff", 0) > 0), None)
        first_in = next((r for r in rows if r.get("in", 0) > 1e-6), None)
        return {"logits": lg, "first_out_jump_same_in": first_out, "first_dq_flip_same_in": first_dq,
                "first_input_diverged": first_in, "rows": rows}

    out = {"watch": len(W), "missing_in_fix": miss[:20],
           "fix_vs_orig": cmp("fix"), "orig_again_vs_orig": cmp("orig_again"),
           "pert1e-6_vs_orig": cmp("orig_pert1e-6"),
           "fix_noopt_vs_orig_noopt": cmp("fix_noopt", "orig_noopt"), "orig_noopt_vs_orig": cmp("orig_noopt"),
           "sec": round(time.time() - t_all, 1)}
    out_file = RES / "diag_t16fix.json"
    C.write_json_atomic(out_file, out)

    # ---- 表示 ----
    print("\n  ■ logits の差（元の版の基準と比べて。b0 = 1 回目、b1 = 2 回目）")
    for k, lab in (("orig_again_vs_orig", "元をもう 1 回         "), ("pert1e-6_vs_orig", "元・入力を 1e-6 揺らす"),
                   ("fix_vs_orig", "書き直した版         "), ("orig_noopt_vs_orig", "元・最適化なし       "),
                   ("fix_noopt_vs_orig_noopt", "書き直し↔元（最適化なし）")):
        print(f"    {lab}: " + "、".join(f"{kk} {v * 100:.4f}%" for kk, v in out[k]["logits"].items()))
    fx = out["fix_vs_orig"]
    print("\n  ■ 書き直した版: 見張る値の差（グラフの順、1 回目の入力。入力 → 丸める前 → 丸めた後、目盛りがずれた数）")
    shown = 0
    for r in fx["rows"]:
        if shown >= 40:
            break
        if r.get("in", 0) > 0 or r.get("out", 0) > 1e-6 or r.get("dq_n_diff", 0) > 0:
            print(f"    {r['kind']:>4} {r['idx']:>3} {r['node'][:40]:<40} 入力 {r.get('in', float('nan')):.2e}  "
                  f"丸める前 {r.get('out', float('nan')):.2e}  丸めた後 {r.get('dq', float('nan')):.2e}  "
                  f"ずれた目盛り {r.get('dq_n_diff', '-')}/{r.get('n', '-')}（最大 {r.get('dq_max_lsb', '-')} 目盛り）")
            shown += 1
    for key, lab in (("first_out_jump_same_in", "入力が同じなのに丸めた後が 2 目盛り以上違う最初の所（A の印）"),
                     ("first_dq_flip_same_in", "入力が同じなのに丸めた後が違う最初の所（B の印。1 目盛りなら境目のゆらぎ）"),
                     ("first_input_diverged", "入力がずれ始めた最初の所")):
        r = fx[key]
        print(f"  ■ {lab}: " + (f"{r['kind']} {r['idx']}（{r['node']}）丸める前 {r.get('out', 0):.2e}・"
                                 f"ずれた目盛り {r.get('dq_n_diff', '-')}（最大 {r.get('dq_max_lsb', '-')}）" if r else "無し"))
    print("  読み方: 書き直しの差が「元・入力を 1e-6 揺らす」と同じくらいで、A の印が無い → B（書き直しは正しい。変換はそのまま使える）。"
          "A の印がある → その演算の書き直しが誤り（変換をやめて直す）")
    print(f"-> {out_file}（{(time.time() - t_all) / 60:.1f} 分）")


if __name__ == "__main__":
    main()
