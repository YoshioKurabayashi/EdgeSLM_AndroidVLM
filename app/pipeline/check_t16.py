# -*- coding: utf-8 -*-
"""Phase 5-1: 繋いだ 16 トークン版（量子化済み QDQ）を PC の ORT（CPU）で動かし、1 トークン版の QDQ・HF と比べる（2026-09-29）。

    python -m app.pipeline.check_t16                # 4 窓、行 15（と 1 窓だけ行 7）
    python -m app.pipeline.check_t16 --n 8

**EdgeSLM_Android の venv（$PY2）で動かす**。変換（4 時間）の前に、量子化・書き込み・繋ぎ方が正しいかを CPU で確かめる
（EdgeSLM_AndroidGemma4 の `ort_full.py` と同じ考え方）。

## 比べ方

Gemma4 が取ってある状態（8 会話・245 位置、HF で丸めあり）の、同じ会話の連続 16 位置 i〜i+15 を使う。

| | 16 トークン版（こちら） | 1 トークン版（Gemma4 の `g4full_L256_final_qdq.onnx`、読むだけ） |
|---|---|---|
| 入力 | 位置 i〜i+15 を 1 回で（past は位置 i の後ろ 240 スロット）、`logit_sel` で行 r を選ぶ | 位置 i+r を 1 回（past は取ってある位置 i+r の past） |
| 出力 | 位置 i+r の logits | 位置 i+r の logits |

- 違いは「この回の中の K/V を、16 トークン版は自分の量子化済みグラフで作り、1 トークン版は HF の値を使う」ことと、量子化の目盛り（較正が別）だけ
- HF の正解（丸めあり、`head_L256.npz` の `ref_argmax`・`ref_top_ids`）とも比べる。1 トークン版の Gemma4 の記録では 1 位の一致 6/8〜14/16
- 見るもの: 1 位の一致、上位 5 の重なり、logits の相対 L2（16 ↔ 1 トークン版）

結果: `results/P5_t16/check_t16.json`
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from app.pipeline import build_t16 as B
from app.pipeline import gemma4_common as G
from app.pipeline import vlm_common as C

T1_QDQ = G.G4_OPS / "g4full_L256" / "g4full_L256_final_qdq.onnx"


def session(path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    t0 = time.time()
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    print(f"  読み込み {path.name} {time.time() - t0:.0f}s", flush=True)
    return s


def inputs_t16(caps: dict, i: int, r: int, names: list[str]) -> dict:
    """16 トークン版の入力（1 回ぶん）。build_t16.make_inputs と同じ作り方。"""
    a, b = caps["A"], caps["B"]
    feed = {}
    for nm in names:
        if nm in ("inputs_embeds", "ple_tok", "cos_s", "sin_s", "cos_g", "sin_g"):
            feed[nm] = np.concatenate([a[nm][i + t] for t in range(B.T)], axis=0)[None].astype(np.float32)
        elif nm == "attn_mask":
            feed[nm] = B.mask_rows(B.past_valid(a["attn_mask"][i]))[None]
        elif nm.startswith("past_"):
            src = a if int(nm.rsplit("_", 1)[1]) <= 8 else b
            feed[nm] = src[nm][i][:, B.W - B.P:, :][None].astype(np.float32)
        elif nm == "logit_sel":
            s = np.zeros((1, B.T, 1), np.float32)
            s[0, r, 0] = 1.0
            feed[nm] = s
        else:
            raise SystemExit(f"16 トークン版の入力 {nm} の作り方を知らない")
    return feed


def inputs_t1(caps: dict, j: int, names: list[str]) -> dict:
    """1 トークン版の入力（位置 j、取ってある値そのまま）。"""
    a, b = caps["A"], caps["B"]
    feed = {}
    for nm in names:
        if nm in ("inputs_embeds", "ple_tok", "cos_s", "sin_s", "cos_g", "sin_g", "attn_mask"):
            feed[nm] = a[nm][j][None].astype(np.float32)
        elif nm.startswith("past_"):
            src = a if int(nm.rsplit("_", 1)[1]) <= 8 else b
            feed[nm] = src[nm][j][None].astype(np.float32)
        else:
            raise SystemExit(f"1 トークン版の入力 {nm} の作り方を知らない")
    return feed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4, help="窓の数")
    ap.add_argument("--no-t1", action="store_true", help="1 トークン版と比べない（HF とだけ比べる。速い）")
    a = ap.parse_args()
    C.utf8_stdout()
    full = B.ops_dir(B.full_name()) / f"{B.full_name()}_final_qdq.onnx"
    if not full.exists():
        raise SystemExit(f"{full} が無い。先に build_t16 --steps merge")
    caps = {"A": B.Capture("chunk_0_8_L256.npz"), "B": B.Capture("chunk_9_17_L256.npz")}
    head = B.Capture("head_L256.npz")
    win = B.windows(caps["A"])
    picks = [win[int(k)] for k in np.linspace(0, len(win) - 1, a.n).astype(int)]
    cases = [(i, B.T - 1) for i in picks] + [(picks[0], 7)]

    # ---- 16 トークン版 ----
    s16 = session(full)
    n16 = [v.name for v in s16.get_inputs()]
    o16 = [v.name for v in s16.get_outputs()]
    li = o16.index("logits")
    got16 = {}
    for i, r in cases:
        t0 = time.time()
        lg = np.asarray(s16.run([o16[li]], inputs_t16(caps, i, r, n16))[0]).reshape(-1)
        got16[(i, r)] = lg
        print(f"  16 トークン版: 窓 i={i} 行 {r} → 1 位 {int(lg.argmax())}（{time.time() - t0:.1f}s）", flush=True)
    del s16

    # ---- 1 トークン版 ----
    got1 = {}
    if not a.no_t1:
        if not T1_QDQ.exists():
            print(f"  ★1 トークン版の QDQ が無い（{T1_QDQ}）。HF とだけ比べる", flush=True)
        else:
            s1 = session(T1_QDQ)
            n1 = [v.name for v in s1.get_inputs()]
            o1 = [v.name for v in s1.get_outputs()]
            l1 = o1.index("logits")
            for i, r in cases:
                t0 = time.time()
                lg = np.asarray(s1.run([o1[l1]], inputs_t1(caps, i + r, n1))[0]).reshape(-1)
                got1[(i, r)] = lg
                print(f"  1 トークン版: 位置 {i + r} → 1 位 {int(lg.argmax())}（{time.time() - t0:.1f}s）", flush=True)
            del s1

    # ---- 比べる ----
    rows = []
    for i, r in cases:
        j = i + r
        lg = got16[(i, r)]
        top5 = set(int(x) for x in np.argsort(-lg)[:5])
        hf_top5 = set(int(x) for x in head["ref_top_ids"][j][:5])
        row = {"i": int(i), "row": int(r), "pos": int(caps["A"]["pos"][j]),
               "t16_top1": int(lg.argmax()), "hf_top1": int(head["ref_argmax"][j]),
               "t16_vs_hf_top1": bool(int(lg.argmax()) == int(head["ref_argmax"][j])),
               "t16_vs_hf_top5": len(top5 & hf_top5)}
        if (i, r) in got1:
            l1 = got1[(i, r)]
            t1top5 = set(int(x) for x in np.argsort(-l1)[:5])
            row.update({"t1_top1": int(l1.argmax()),
                        "t16_vs_t1_top1": bool(int(lg.argmax()) == int(l1.argmax())),
                        "t16_vs_t1_top5": len(top5 & t1top5),
                        "t16_vs_t1_rel": float(np.linalg.norm(lg - l1) / max(np.linalg.norm(l1), 1e-12)),
                        "t1_vs_hf_top1": bool(int(l1.argmax()) == int(head["ref_argmax"][j]))})
        rows.append(row)
        print(f"  位置 {row['pos']:>3}（窓 {i} 行 {r}）: 16↔HF 1位 {row['t16_vs_hf_top1']} 上位5 {row['t16_vs_hf_top5']}/5"
              + (f" / 16↔1 1位 {row['t16_vs_t1_top1']} 上位5 {row['t16_vs_t1_top5']}/5 相対L2 {row['t16_vs_t1_rel']:.3f}"
                 f"（1↔HF 1位 {row['t1_vs_hf_top1']}）" if "t1_top1" in row else ""), flush=True)
    out = {"model": str(full), "rows": rows,
           "t16_vs_hf_top1": float(np.mean([r["t16_vs_hf_top1"] for r in rows])),
           "t16_vs_hf_top5": float(np.mean([r["t16_vs_hf_top5"] for r in rows]))}
    if got1:
        out.update({"t16_vs_t1_top1": float(np.mean([r["t16_vs_t1_top1"] for r in rows])),
                    "t16_vs_t1_top5": float(np.mean([r["t16_vs_t1_top5"] for r in rows])),
                    "t1_vs_hf_top1": float(np.mean([r["t1_vs_hf_top1"] for r in rows]))})
    B.RES.mkdir(parents=True, exist_ok=True)
    C.write_json_atomic(B.RES / "check_t16.json", out)
    print(f"\n  16↔HF: 1 位の一致 {out['t16_vs_hf_top1']:.2f}、上位 5 の重なり {out['t16_vs_hf_top5']:.1f}/5"
          + (f" / 16↔1 トークン版: 1 位の一致 {out['t16_vs_t1_top1']:.2f}、上位 5 {out['t16_vs_t1_top5']:.1f}/5"
             f"（1 トークン版↔HF 1 位 {out['t1_vs_hf_top1']:.2f}）" if got1 else "")
          + f"\n-> {B.RES / 'check_t16.json'}")


if __name__ == "__main__":
    main()
