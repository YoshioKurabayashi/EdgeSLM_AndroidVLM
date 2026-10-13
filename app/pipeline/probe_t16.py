# -*- coding: utf-8 -*-
"""HF との一致の切り分けの 2 段目: **同じ入力**を端末の NPU と PC の CPU に通して比べる（2026-09-30）。

    python -m app.pipeline.probe_t16                    # 既定: 7 問（20〜30 分。端末の画面が job ごとに前に出る）
    python -m app.pipeline.probe_t16 --items car_12a,vg_33 --decode 0     # 短く試す（5 分前後）
    python -m app.pipeline.probe_t16 --fresh            # 前の結果を捨てて最初から（既定は済んだ問題を飛ばす）
    python -m app.pipeline.probe_t16 --no-t1            # 1 トークン版の基準を取らない
    python -m app.pipeline.probe_t16 --model t16f --items syn_train,vg_21 --decode 12 --decode-all
                                                        # ★書き直した版（t16f・t16t）で。端末の NPU も PC の CPU もその版にする。
                                                        #   結果は probe_t16_<種類>.json。2026-10-02 夜に追加（t16f の「画像が無い」誤りの調べ）

**EdgeSLM_Android の venv（$PY2）で動かす**（onnxruntime・numpy・adb）。端末は USB でつなぎ、VLM 検証アプリ（org.edgeslm.vlm）を
入れておく。デモ画面が開いていても最初に止める（同じプロセスで 1.5GB のモデルを 2 つ開けないため）。

## ここまでに分かっていること（`diag_t16`、`results/P5_t16/diag_t16.json`）

同じ量子化済みの 16 トークン版を PC の CPU で動かすと、答えの 1 トークン目が HF と 13/16 一致した（端末は 5/16）。
つまりグラフ（量子化の目盛り）は正しく、**ずれは端末での実行で起きている**。端末での実行には 2 つの部品がある:

| 部品 | 中身 | 疑い方 |
|---|---|---|
| (B1) NPU の計算 | QAIRT の変換結果（context binary）を HTP が整数で計算する。T=16 の形でだけ別の計算の仕方（HMX の経路など）が選ばれている可能性 | 同じ入力で NPU と CPU の出力が大きく違う |
| (B2) C++ の入れ方 | `job.cpp` の `generate_kv` が、16 行の埋め込み・PLE・位置・マスク・画像の特徴の並び・KV のずらし方を作る | NPU の計算が正しいのに、端末の答えだけ違う |

## この道具がすること

diag_t16 と**同じ入れ方を Python で作り**（diag_t16.Runner16 と同じ手順）、1 回ぶんの入力をファイルにして、
端末では **generic の `infer` の job**（`generate_kv` を通らない。入力のファイルをそのまま NPU に渡すだけ）で動かす。
同じ入力を PC の CPU（ONNX Runtime、同じ量子化済みグラフ）にも通す。

```
プロンプトの 16 個ずつ（と答えの 1 個ずつ）について:
    入力（Python で作る）──┬─→ 端末の NPU（infer の job）──→ 出力 A ──→ ★past を A で進める（端末だけで続けた状態）
                           └─→ PC の CPU（ORT）          ──→ 出力 B
    A と B を比べる: logits・各層の新しい K/V（層 0〜14）・行ごと
```

ここから 2 つのことが分かる:

1. **1 回ぶんの NPU の誤差**（同じ入力なので、差はすべて NPU の計算による）。層ごと（K/V は層ごとに出ているので、
   どの層から離れ始めるかが分かる）・行ごと（16 行のどこかだけ壊れていないか）・画像の行を含む回と文字だけの回
2. **Python の入れ方で NPU だけで続けた答え**（past を NPU の出力で進める）。これを端末の評価の答え（C++ の入れ方）と比べる:
   - NPU で続けた答え ＝ 端末の答え（C++） → C++ の入れ方は Python と同じ。**原因は NPU の計算（B1）**
   - NPU で続けた答え ＝ PC の CPU・HF の答えで、端末の答えだけ違う → **原因は C++ の入れ方（B2）**

基準として、**1 トークン版**（HF との一致 97/126 だった方）でも、プロンプトの最後の 1 個を同じように NPU と CPU で比べる
（past は上の「NPU で続けた状態」から最後の 1 個を除いたもの）。16 トークン版の誤差がこれより大きければ、T=16 の変換を疑う。

## 端末での動かし方

- 入力は `files/inputs/probe_t16/`、出力は `files/out/probe_t16/` に置き、1 回ごとに消して作り直す（送ったあと md5 で確かめる）
- **keep_session** でモデルを開いたままにする（1.5GB を毎回開くと 1 回 5〜8 秒かかる）。画面は job ごとに前に出る（hold）
- job の間に OS に止められたら、次の job で開き直す（落ちた job は 30 秒待って 2 回までやり直す）。最後にアプリを止める

結果: `results/P5_t16/probe_t16.json`（1 問ごとに書き足す。途中で止めても同じコマンドで続きから）
"""
from __future__ import annotations

import argparse
import gc
from concurrent.futures import ThreadPoolExecutor
import json
import shutil
import time
from pathlib import Path

import numpy as np

from app.pipeline import device_vit as D
from app.pipeline import gemma4_common as G
from app.pipeline import vlm_common as C
from app.pipeline.diag_t16 import (IMG, MASK_NEG, OCR_ITEMS, PAD, STOP, T16_GRAPHS, T16_QDQ, TABLES, Tables,
                                   load_items, load_results, model_qdq, old_prompt, prefix_len)

# 1 トークン版（基準）: PC の量子化済みグラフは EdgeSLM_AndroidGemma4 のもの（読むだけ）、端末は Phase 3 で VLM 検証アプリに写したもの
T1_QDQ = G.G4_OPS / "g4full_L256" / "g4full_L256_final_qdq.onnx"
T1_REL = "models/ops/g4full_L256_epctx.onnx"
# 既定の 7 問: 文字を読み違えた 2 問・diag_t16 で「PC の CPU は HF と同じ、端末だけ違う」だった 4 問・全部同じだった 1 問（対照）
DEFAULT_ITEMS = ["syn_nav_S", "syn_vertical", "car_12a", "vg_33", "syn_parking", "wild_40", "vg_06"]
RES = C.RESULTS / "P5_t16"
IO = C.WORK / "probe_t16"
DEV_IN = "inputs/probe_t16"           # 端末（files/ の下）
DEV_OUT = "out/probe_t16"
DIAG = RES / "diag_t16.json"


# ---------------------------------------------------------------------------
# 入れ方（diag_t16.Runner16 と同じ手順を「入力を作る」と「出力で past を進める」に分けたもの。T は 16 でも 1 でもよい）
# ---------------------------------------------------------------------------
class Feeder:
    def __init__(self, ins: dict, tabs: Tables):
        self.ins, self.tabs = ins, tabs
        self.T = int(ins["inputs_embeds"][1])               # 1 回に入れる行（16 トークン版 16、1 トークン版 1）
        self.W = int(ins["attn_mask"][-1])                  # 注意の窓（256）
        self.kv = sorted([n for n in ins if n.startswith("past_")], key=lambda n: (int(n.rsplit("_", 1)[1]), n))
        self.P = int(ins[self.kv[0]][2])                    # past のスロット数（16 トークン版 240、1 トークン版 256）
        # ★2 つの作りがある（2026-09-30、1 回目の基準で「past 256 ＋ 行 1 が窓 256 と合わない」で止まったため対応）:
        #   - 16 トークン版: past 240 ＋ 新しい 16 行 ＝ 窓 256。マスクは [past | 新しい行]、present は新しい行（層 13・14 は窓全体）
        #   - 1 トークン版（Gemma4 の g4full_L256）: past 256 ＝ 窓 256。グラフの中で past を 1 つ左へずらして右端に新しい行を足し、
        #     マスクはその窓（[past の 2 個目から, 新しい行]）に掛かる。present は窓全体。C++ は present をそのまま次の past にし、
        #     マスクは「末尾の有効な数（新しい行を含む）」だけ開ける（job.cpp の 1 トークンの経路）
        self.shift = (self.P == self.W)
        if self.shift and self.T != 1:
            raise SystemExit(f"past {self.P} ＝ 窓 {self.W} の作りは 1 行（T=1）だけ想定（T={self.T}）")
        if not self.shift and self.P + self.T != self.W:
            raise SystemExit(f"past {self.P} ＋ 行 {self.T} が窓 {self.W} と合わない")
        known = {"inputs_embeds", "ple_tok", "attn_mask", "logit_sel", *tabs.rope, *self.kv}
        unknown = sorted(set(ins) - known)
        if unknown:
            raise SystemExit(f"入力 {unknown} の作り方を知らない（グラフの入力が想定と違う）")
        self.reset()

    def reset(self):
        self.past = {n: np.zeros(self.ins[n], np.float32) for n in self.kv}
        self.n_valid = 0
        self.abs_pos = 0

    def feed(self, rows: list[tuple[int, np.ndarray | None]]) -> dict:
        """rows: 本物の行 n 個（ID と、画像の位置なら特徴の 1 行）。端末の C++ の複数トークンの経路と同じ入力を作る"""
        tb, T, P, n = self.tabs, self.T, self.P, len(rows)
        z = lambda name: np.zeros(self.ins[name], np.float32)
        emb, ple, mask = z("inputs_embeds"), z("ple_tok"), np.full(self.ins["attn_mask"], MASK_NEG, np.float32)
        rope = {k: z(k) for k in tb.rope if k in self.ins}
        pos_last = 0
        for r, (tid, feat) in enumerate(rows):
            pos = min(self.abs_pos + r, tb.rope_rows - 1)
            if feat is not None:                           # 画像の位置: 特徴の 1 行をそのまま（√1536 倍しない）、PLE は pad の行
                emb[0, r] = feat
                ple[0, r] = tb.ple(PAD)
            else:
                emb[0, r] = tb.embed(tid)
                ple[0, r] = tb.ple(tid)
            for k in rope:
                rope[k][0, r, 0] = tb.rope[k][pos]
            pos_last = pos
        for r in range(n, T):                              # 空の行: 位置は最後の本物の行と同じ、埋め込み・PLE は 0
            for k in rope:
                rope[k][0, r, 0] = tb.rope[k][pos_last]
        if self.shift:                                     # 1 トークン版: 窓の末尾の有効な数（新しい行を含む）だけ開ける
            nvi = min(self.n_valid + 1, self.W)
            mask[0, 0, 0, self.W - nvi:self.W] = 0.0
        else:
            nv = min(self.n_valid, P)
            for r in range(T):
                rr = r if r < n else 0                     # 空の行のマスクは行 0 と同じ
                mask[0, r, 0, P - nv:P] = 0.0              # past の有効な右端
                mask[0, r, 0, P:P + rr + 1] = 0.0          # この回の自分より前（自分を含む）
        feed = {"inputs_embeds": emb, "ple_tok": ple, "attn_mask": mask, **rope, **self.past}
        if "logit_sel" in self.ins:                        # 最後の本物の行を選ぶ（16 トークン版）
            sel = z("logit_sel")
            sel[0, n - 1, 0] = 1.0
            feed["logit_sel"] = sel
        return {k: v for k, v in feed.items() if k in self.ins}

    def advance(self, outs: dict, n: int):
        """past を n 個左へずらし、present の新しい行（後ろから T 行目から n 行）を右端に書く"""
        if self.shift:                                     # 1 トークン版: present（窓全体）がそのまま次の past
            for name in self.kv:
                self.past[name] = np.asarray(outs[name.replace("past_", "present_")], np.float32).reshape(self.ins[name])
            self.n_valid = min(self.n_valid + n, self.W)
            self.abs_pos += n
            return
        for name in self.kv:
            pr = outs[name.replace("past_", "present_")]
            lo = pr.shape[2]
            new = pr[:, :, lo - self.T:lo - self.T + n, :].astype(np.float32)
            pb = self.past[name]
            self.past[name] = np.concatenate([pb[:, :, n:, :], new], axis=2) if n < self.P else new[:, :, -self.P:, :]
        self.n_valid = min(self.n_valid + n, self.P)
        self.abs_pos += n


# ---------------------------------------------------------------------------
# PC の CPU と端末の NPU
# ---------------------------------------------------------------------------
class Cpu:
    """同じ量子化済みグラフを PC の ONNX Runtime（CPU、diag_t16 と同じ設定）で動かす"""

    def __init__(self, path: Path):
        import onnxruntime as ort
        if not path.exists():
            raise SystemExit(f"{path} が無い")
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        t0 = time.time()
        self.s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        print(f"  PC の CPU に読み込み {path.name}（{time.time() - t0:.0f}s）", flush=True)
        self.ins = {v.name: [int(x) for x in v.shape] for v in self.s.get_inputs()}
        self.outs = [v.name for v in self.s.get_outputs()]

    def run(self, feed: dict) -> dict:
        return dict(zip(self.outs, self.s.run(None, feed)))


class Device:
    """端末の NPU で動かす（generic の infer の job。C++ の generate_kv の入れ方は通らない）"""

    def __init__(self, model_rel: str, tag: str):
        self.model_rel, self.tag = model_rel, tag
        self.n = 0
        self.first = None                                  # 最初の結果（開けたか・strict）

    def run(self, feeds: list[dict], ids: list[str], keep: bool = True) -> tuple[list[dict], dict]:
        """feeds: 1 回ぶんの入力（名前 → 配列）のリスト。1 つの job にまとめて送り、出力（名前 → 配列）のリストを返す"""
        # ---- 入力をファイルにして送る（フォルダごと 1 回の push。送ったあと全部 md5 で確かめる）----
        loc = IO / "push" / Path(DEV_IN).name
        shutil.rmtree(loc, ignore_errors=True)
        loc.mkdir(parents=True)
        batches, want = [], {}
        for b, (bid, feed) in enumerate(zip(ids, feeds)):
            ins = {}
            for name, arr in feed.items():
                fn = f"b{b}_{name}.bin"
                np.ascontiguousarray(arr, np.float32).tofile(loc / fn)
                ins[name] = f"{DEV_IN}/{fn}"
                want[f"{DEV_IN}/{fn}"] = D.md5(loc / fn)
            batches.append({"id": bid, "inputs": ins})
        # ★送り先のフォルダは adb shell の mkdir で作ってから、ファイルを並べて 1 回の push で入れる。
        #   adb push にフォルダを作らせると、Android 13 の files/ の下では「remote secure_mkdirs failed: Operation not permitted」で
        #   止まる（2026-09-30 の 1 回目）。adb shell の mkdir は通る（device_vit の push_many・infer_images と同じやり方）
        D.adb("shell", f"rm -rf {D.DEV}/{DEV_IN} {D.DEV}/{DEV_OUT}; mkdir -p {D.DEV}/{DEV_IN} {D.DEV}/{DEV_OUT}")
        names = sorted({Path(k).name for k in want})
        # ★Windows のコマンド行は約 32,000 文字まで（2026-10-01: probe_rows で 10 回ぶん 380 個を 1 回で送ろうとして
        #   「WinError 206 ファイル名または拡張子が長すぎます」）。40 個ずつに分けて送る
        for i in range(0, len(names), 40):
            try:
                D.adb("push", *[str(loc / fn) for fn in names[i:i + 40]], f"{D.DEV}/{DEV_IN}/")
            except RuntimeError as e:
                print(f"    （まとめての送信に失敗: {str(e).strip()[:300]}。1 個ずつ送る）", flush=True)
        got = dev_md5_dir()
        bad = [k for k in want if got.get(k) != want[k]]
        if bad:
            if len(bad) < len(want) or self.n == 0:
                print(f"    （md5 が合わない {len(bad)} 個を 1 個ずつ送り直す）", flush=True)
            for k in bad:
                D.adb("push", str(loc / Path(k).name), f"{D.DEV}/{k}")
            got = dev_md5_dir()
            bad = [k for k in bad if got.get(k) != want[k]]
        if bad:
            # 何が起きているかを見せる（送り先の中身・端末の空き容量）
            info = D.adb("shell", f"ls -la {D.DEV}/inputs | head -20; echo ---; ls {D.DEV}/{DEV_IN} | head -5; echo ---; df -h /sdcard | tail -2",
                         check=False)
            raise SystemExit(f"端末に送った入力の md5 が合わない（{len(bad)} 個。例 {bad[:3]}）\n{info}")
        # ---- 動かす ----
        self.n += 1
        job = {"job_id": f"probe_{self.tag}_{self.n:04d}", "task": "infer", "model": self.model_rel, "ep": D.EP,
               "keep_session": keep, "batches": batches, "output": {"dir": DEV_OUT}}
        res = run_job_retry(job)
        if res.get("_kind") != "result":
            raise SystemExit(f"端末でモデルを開けなかった（{res.get('_kind')}）: {json.dumps(res, ensure_ascii=False)[:800]}")
        if self.first is None:
            self.first = {**D.describe_line(res), "session_open_ms": res.get("session_open_ms")}
        # ---- 出力を取ってくる ----
        local = IO / "pull"
        shutil.rmtree(local, ignore_errors=True)
        local.mkdir(parents=True)
        D.adb("pull", f"{D.DEV}/{DEV_OUT}", str(local))                # → pull/probe_t16/
        outs = []
        for rec in res.get("records", []):
            if "error" in rec:
                raise SystemExit(f"端末の推論でエラー（{rec.get('id')}）: {rec['error']}")
            d = {"_ort_run_ms": rec.get("ort_run_ms")}
            for o in rec["outputs"]:
                if o.get("type", "float32") != "float32":
                    raise SystemExit(f"出力 {o['name']} の型が float32 でない（{o.get('type')}）")
                a = np.fromfile(local / Path(DEV_OUT).name / Path(o["path"]).name, np.float32)
                shape = [int(x) for x in o["shape"]]
                if a.size != int(np.prod(shape)):
                    raise SystemExit(f"出力 {o['name']} の大きさが合わない（{a.size} と {shape}）")
                d[o["name"]] = a.reshape(shape)
            outs.append(d)
        if len(outs) != len(feeds):
            raise SystemExit(f"端末の出力が {len(outs)} 回ぶん（{len(feeds)} 回ぶん送った）")
        return outs, res


def dev_md5_dir() -> dict:
    """端末の入力の置き場（files/inputs/probe_t16/）のファイルの md5 を、短いコマンド 1 回で全部取る（ファイル名を並べると
    Windows のコマンド行の長さの上限を超えることがある）。キーは files/ からの相対パス"""
    txt = D.adb("shell", f"md5sum {D.DEV}/{DEV_IN}/* 2>/dev/null || true", check=False)
    out = {}
    for ln in txt.splitlines():
        parts = ln.split()
        if len(parts) == 2 and parts[1].startswith(D.DEV + "/"):
            out[parts[1][len(D.DEV) + 1:]] = parts[0]
    return out


def run_job_retry(job: dict, retries: int = 2, cooldown: int = 30) -> dict:
    """OS に止められたら、アプリを止めて少し待ち、同じ job をやり直す（開いたままのモデルは消えるので、次で開き直す）"""
    for attempt in range(retries + 1):
        try:
            return D.run_job(job, timeout_s=900, hold=True)
        except D.DeviceJobFailed as e:
            print(f"    ！{e}", flush=True)
            if attempt >= retries:
                raise SystemExit(f"{retries + 1} 回続けて落ちたので止める。同じコマンドをもう一度打てば、済んだ問題を飛ばして続ける")
            print(f"    {cooldown} 秒待ってやり直す", flush=True)
            D.adb("shell", f"am force-stop {D.PKG}")
            time.sleep(cooldown)
            D.wait_mem()
    raise AssertionError


# ---------------------------------------------------------------------------
# 比べる
# ---------------------------------------------------------------------------
def rel(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def layer_of(name: str) -> tuple[int, str]:
    """present_k_12 → (12, "k")"""
    _, kind, li = name.split("_")
    return int(li), kind


def cmp_step(npu: dict, cpu: dict, T: int, n: int, W: int) -> dict:
    """同じ入力での NPU（npu）と CPU（cpu、正解の側）の出力を比べる"""
    ln = np.asarray(npu["logits"], np.float64).reshape(-1)
    lc = np.asarray(cpu["logits"], np.float64).reshape(-1)
    tn, tc = np.argsort(-ln)[:5], np.argsort(-lc)[:5]
    out = {
        "logits_rel": round(rel(ln, lc), 5),
        "logits_cos": round(float(ln @ lc / max(np.linalg.norm(ln) * np.linalg.norm(lc), 1e-300)), 6),
        "top1_npu": int(tn[0]), "top1_cpu": int(tc[0]), "top1_same": bool(tn[0] == tc[0]),
        "top5_overlap": len(set(tn.tolist()) & set(tc.tolist())),
        "top5_npu": [int(x) for x in tn], "top5_npu_logit": [round(float(ln[x]), 3) for x in tn],
        "top5_cpu": [int(x) for x in tc], "top5_cpu_logit": [round(float(lc[x]), 3) for x in tc],
        "margin_npu": round(float(ln[tn[0]] - ln[tn[1]]), 3), "margin_cpu": round(float(lc[tc[0]] - lc[tc[1]]), 3),
        "npu_logit_of_cpu_top1": round(float(ln[tc[0]]), 3),
    }
    kv, last_row, passthru = {}, {}, {}
    row_acc = np.zeros(n)
    names = sorted([k for k in cpu if k.startswith("present_")], key=lambda k: layer_of(k))
    for name in names:
        a = np.asarray(npu[name], np.float64)
        b = np.asarray(cpu[name], np.float64)
        lo = b.shape[2]
        na, nb = a[:, :, lo - T:lo - T + n, :], b[:, :, lo - T:lo - T + n, :]
        li, kind = layer_of(name)
        key = f"{kind}{li}"
        kv[key] = round(rel(na, nb), 5)
        # 行ごと（ヘッドと次元をまとめた相対 L2）
        dr = np.sqrt(((na - nb) ** 2).sum(axis=(0, 1, 3)))
        br = np.maximum(np.sqrt((nb ** 2).sum(axis=(0, 1, 3))), 1e-12)
        row_acc += dr / br
        last_row[key] = round(float(dr[-1] / br[-1]), 5)
        if lo == W and lo - T > 0:                         # 層 13・14 は窓全体を出す: past の部分はそのまま戻ってくるはず
            pa, pb = a[:, :, :lo - T, :], b[:, :, :lo - T, :]
            if np.linalg.norm(pb) > 0:
                passthru[key] = round(rel(pa, pb), 5)
    out.update(kv=kv, kv_last_row=last_row, row_rel_mean=[round(float(x), 5) for x in row_acc / max(len(names), 1)],
               passthru=passthru)
    return out


def median(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.median(xs)), 5) if xs else None


def summarize(items: list[dict], t1: dict) -> dict:
    s = {"n_items": len(items)}
    have = [r for r in items if r.get("npu_chain_first") is not None]
    cnt = lambda f: sum(1 for r in have if f(r))
    s["chain"] = {
        "npu_eq_dev16": cnt(lambda r: r["npu_chain_first"] == r["dev16_first"]),
        "npu_eq_cpu16": cnt(lambda r: r.get("cpu16_first") is not None and r["npu_chain_first"] == r["cpu16_first"]),
        "npu_eq_hf": cnt(lambda r: r["npu_chain_first"] == r["hf_first"]),
        "dev16_eq_hf": cnt(lambda r: r["dev16_first"] == r["hf_first"]),
        "n_with_cpu16": cnt(lambda r: r.get("cpu16_first") is not None),
    }
    steps = [st for r in items for st in r["steps"]]
    prompt = [st for st in steps if st["kind"] == "prompt"]
    last = [r["steps"][r["n_prompt_steps"] - 1] for r in items]
    s["step"] = {
        "n": len(steps), "top1_same": sum(st["top1_same"] for st in steps),
        "logits_rel_median": median([st["logits_rel"] for st in steps]),
        "logits_rel_median_last_prompt": median([st["logits_rel"] for st in last]),
        "top1_same_last_prompt": sum(st["top1_same"] for st in last),
    }
    keys = sorted(prompt[0]["kv"], key=lambda k: (int(k[1:]), k[0])) if prompt else []
    img = [st for st in prompt if st["n_img"] > 0]
    txt = [st for st in prompt if st["n_img"] == 0]
    s["layer_t16"] = {k: median([st["kv"][k] for st in prompt]) for k in keys}
    s["layer_t16_image_steps"] = {k: median([st["kv"][k] for st in img]) for k in keys}
    s["layer_t16_text_steps"] = {k: median([st["kv"][k] for st in txt]) for k in keys}
    s["layer_t16_last_row"] = {k: median([st["kv_last_row"][k] for st in last]) for k in keys}
    full = [st for st in prompt if st["n"] == 16]
    s["row_t16"] = [median([st["row_rel_mean"][r] for st in full]) for r in range(16)] if full else []
    if t1:
        rows = list(t1.values())
        k1 = sorted(rows[0]["kv"], key=lambda k: (int(k[1:]), k[0]))
        s["t1"] = {"n": len(rows), "top1_same": sum(r["top1_same"] for r in rows),
                   "logits_rel_median": median([r["logits_rel"] for r in rows]),
                   "layer": {k: median([r["kv"][k] for r in rows]) for k in k1}}
    return s


# ---------------------------------------------------------------------------
# 本体
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default="", help="問題（カンマ区切り）。既定は " + ",".join(DEFAULT_ITEMS))
    ap.add_argument("--decode", type=int, default=8, help="文字を読み違えた 2 問で、答えを何トークン先まで NPU で続けるか（0 で作らない）")
    ap.add_argument("--no-t1", action="store_true", help="1 トークン版の基準を取らない")
    ap.add_argument("--fresh", action="store_true", help="前の結果を捨てて最初から")
    ap.add_argument("--model", default="t16", choices=list(T16_GRAPHS),
                    help="16 トークン版の種類（t16 = 元、t16f・t16t = build_t16fix で書き直した版）。端末の NPU と PC の CPU の両方を"
                         "その版にする。t16 以外は結果を probe_t16_<種類>.json に分け、1 トークン版の基準は取らない")
    ap.add_argument("--decode-all", action="store_true", help="選んだ全部の問題で、答えを --decode トークン先まで NPU で続ける")
    a = ap.parse_args()
    C.utf8_stdout()
    sel = [x for x in a.items.split(",") if x] or DEFAULT_ITEMS
    items_all, res = load_items(), load_results(a.model)
    qdq = model_qdq(a.model)
    if not qdq.exists():
        raise SystemExit(f"{qdq} が無い")
    miss = [i for i in sel if i not in items_all or i not in res["dev16"]]
    if miss:
        raise SystemExit(f"評価の job・記録に無い問題: {miss}")
    # PC の CPU だけで続けた答え（diag_t16）。t16 以外は、その版の記録（あれば）を使う
    diag_file = DIAG if a.model == "t16" else next(
        (p for p in (RES / f"diag_t16_{a.model}_dec.json", RES / f"diag_t16_{a.model}.json") if p.exists()), RES / "_none_")
    diag = {r["id"]: r for r in json.loads(diag_file.read_text(encoding="utf-8"))["rows"]} if diag_file.exists() else {}
    out_path = RES / f"probe_t16{'' if a.model == 't16' else '_' + a.model}.json"
    prev = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() and not a.fresh else {}
    done = {r["id"]: r for r in prev.get("items", []) if r["id"] in sel}
    t1 = {k: v for k, v in prev.get("t1", {}).items() if k in sel}
    (IO / "t1").mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)
    print(f"  問題 {len(sel)}（前の結果 {len(done)} 問は飛ばす）: {', '.join(sel)}", flush=True)

    tabs = Tables(TABLES)
    D.ensure_app()
    meta = {"t16_model": str(qdq), "t1_model": str(T1_QDQ), "decode": a.decode, "model": a.model}
    items = [done[i] for i in sel if i in done]

    def save():
        C.write_json_atomic(out_path, {**meta, "items": items, "t1": t1, "summary": summarize(items, t1)})

    todo = [i for i in sel if i not in done]
    if todo:
        t16_rel = D.push_t16(D.T16_KINDS[a.model])         # 端末に 16 トークン版があるか（md5 で確かめ、無ければ送る）
        D.adb("shell", f"am force-stop {D.PKG}")          # デモ画面などがモデルを持っていたら放す
        time.sleep(2.0)
        print(f"  端末の空きメモリ {D.wait_mem()}MiB", flush=True)
        cpu = Cpu(qdq)
        fd = Feeder(cpu.ins, tabs)
        dev = Device(t16_rel, a.model)
        # PC の CPU（1 回 8 秒ほど）と端末（job のやり取りで 5 秒ほど）を同時に動かす。同じ入力を渡すだけなので結果は変わらない
        pool = ThreadPoolExecutor(1)
        t_all = time.time()
        for k, iid in enumerate(todo):
            it = items_all[iid]
            ids = old_prompt(it["tail"])
            if len(ids) != res["dev16"][iid]["prompt_len"]:
                raise SystemExit(f"{iid}: 作ったプロンプト {len(ids)} 個が評価の記録（{res['dev16'][iid]['prompt_len']} 個）と合わない")
            feats = np.fromfile(it["feat"], np.float32).reshape(-1, tabs.H)
            t0 = time.time()
            fd.reset()
            steps, img_next, ln = [], 0, None
            print(f"  {k + 1}/{len(todo)} {iid}（プロンプト {len(ids)} 個 = {-(-len(ids) // fd.T)} 回）", flush=True)
            for c in range(0, len(ids), fd.T):
                chunk = []
                for tid in ids[c:c + fd.T]:
                    if tid == IMG:
                        chunk.append((tid, feats[img_next]))
                        img_next += 1
                    else:
                        chunk.append((tid, None))
                feed = fd.feed(chunk)
                fut = pool.submit(cpu.run, feed)
                (o_npu,), _ = dev.run([feed], ["p"])
                o_cpu = fut.result()
                st = cmp_step(o_npu, o_cpu, fd.T, len(chunk), fd.W)
                st.update(kind="prompt", start=c, n=len(chunk), n_img=sum(1 for t, _ in chunk if t == IMG),
                          ort_run_ms=o_npu.get("_ort_run_ms"))
                steps.append(st)
                print(f"    {c:>3}〜{c + len(chunk) - 1:<3}（画像 {st['n_img']:>2}）logits の差 {st['logits_rel'] * 100:5.2f}%  "
                      f"1 位 {'同じ' if st['top1_same'] else '違う'}  K/V の差 層0 {st['kv']['k0'] * 100:.2f}%・"
                      f"層14 {st['kv']['k14'] * 100:.2f}%（NPU {st['ort_run_ms'] or 0:.0f}ms）", flush=True)
                # ★past は NPU の出力で進める（端末だけで続けた状態を作る。端末の generate_kv と同じ）
                fd.advance(o_npu, len(chunk))
                ln = np.asarray(o_npu["logits"]).reshape(-1)
            n_prompt = len(steps)
            # 1 トークン版の基準の材料: プロンプトの最後の 1 個の手前までの past（NPU で続けた状態）と最後の 1 個
            m = fd.n_valid - 1
            if a.model == "t16":                           # 書き直した版では基準を取らないので、元の版の材料を上書きしない
                np.savez(IO / "t1" / f"{iid}.npz", last_id=np.int64(ids[-1]), m=np.int64(m), pos=np.int64(fd.abs_pos - 1),
                         **{nm: fd.past[nm][:, :, fd.P - fd.n_valid:fd.P - 1, :] for nm in fd.kv})
            # 答えを NPU だけで続ける（文字を読み違えた 2 問）
            gen = [int(ln.argmax())]
            n_dec = a.decode if (iid in OCR_ITEMS or a.decode_all) else 0
            while len(gen) < 1 + n_dec and gen[-1] not in STOP:
                feed = fd.feed([(gen[-1], None)])
                fut = pool.submit(cpu.run, feed)
                (o_npu,), _ = dev.run([feed], ["d"])
                o_cpu = fut.result()
                st = cmp_step(o_npu, o_cpu, fd.T, 1, fd.W)
                st.update(kind="decode", start=fd.abs_pos, n=1, n_img=0, ort_run_ms=o_npu.get("_ort_run_ms"))
                steps.append(st)
                fd.advance(o_npu, 1)
                gen.append(int(np.asarray(o_npu["logits"]).reshape(-1).argmax()))
            d16, d1, hf = (res[x][iid]["gen_ids"] for x in ("dev16", "dev1", "hf"))
            dg = diag.get(iid, {})
            rec = {"id": iid, "prompt_len": len(ids), "n_prompt_steps": n_prompt,
                   "npu_chain_first": gen[0], "cpu16_first": dg.get("cpu16_first"), "dev16_first": d16[0],
                   "dev1_first": d1[0], "hf_first": hf[0],
                   "npu_chain_gen": gen, "dev16_gen": d16[:len(gen) + 4], "hf_gen": hf[:len(gen) + 4],
                   "cpu16_gen": dg.get("cpu16_gen"),
                   "prefix_npu_dev16": prefix_len(gen, d16), "prefix_npu_hf": prefix_len(gen, hf),
                   "steps": steps, "sec": round(time.time() - t0, 1)}
            items.append(rec)
            eq = lambda x, y: "＝" if x == y else "≠"
            c16 = rec["cpu16_first"]
            print(f"    → 1 トークン目: NPU で続けた {gen[0]} {eq(gen[0], d16[0])} 端末(C++) {d16[0]}、"
                  f"{eq(gen[0], c16)} PC の CPU {c16}、{eq(gen[0], hf[0])} HF {hf[0]}"
                  + (f"。続き: 端末(C++) と {rec['prefix_npu_dev16']} 個目まで同じ（HF とは {rec['prefix_npu_hf']}）" if n_dec else "")
                  + f"（{rec['sec']:.0f}s）", flush=True)
            save()
        print(f"  16 トークン版 {len(todo)} 問（{time.time() - t_all:.0f}s）。端末のモデルを放す", flush=True)
        D.adb("shell", f"am force-stop {D.PKG}")
        pool.shutdown()
        del cpu
        gc.collect()

    # ---- 基準: 1 トークン版で、プロンプトの最後の 1 個を同じ入力で NPU と CPU に ----
    need = [i for i in sel if i not in t1 and (IO / "t1" / f"{i}.npz").exists()]
    if not a.no_t1 and a.model == "t16" and need:
        have = D.adb("shell", f"ls {D.DEV}/{T1_REL} 2>/dev/null || true", check=False).strip()
        if not have.endswith(".onnx"):
            print(f"  端末に 1 トークン版（{T1_REL}）が無いので基準は飛ばす（setup_vlm_demo で写す）", flush=True)
        else:
            time.sleep(10.0)                               # 前のモデルが片付くのを待つ
            print(f"  基準（1 トークン版）{len(need)} 問。端末の空きメモリ {D.wait_mem()}MiB", flush=True)
            cpu1 = Cpu(T1_QDQ)
            f1 = Feeder(cpu1.ins, tabs)
            feeds, cpu_outs = [], []
            for iid in need:
                z = np.load(IO / "t1" / f"{iid}.npz")
                mm = int(z["m"])
                f1.reset()
                for nm in f1.kv:
                    if nm not in z:
                        raise SystemExit(f"1 トークン版の入力 {nm} が 16 トークン版の past に無い")
                    f1.past[nm][:, :, f1.P - mm:f1.P, :] = z[nm]
                f1.n_valid, f1.abs_pos = mm, int(z["pos"])
                feed = f1.feed([(int(z["last_id"]), None)])
                feeds.append(feed)
                cpu_outs.append(cpu1.run(feed))
            dev1 = Device(T1_REL, "t1")
            outs, _ = dev1.run(feeds, need, keep=False)
            for iid, o_npu, o_cpu in zip(need, outs, cpu_outs):
                t1[iid] = cmp_step(o_npu, o_cpu, f1.T, 1, f1.W)
                t1[iid]["ort_run_ms"] = o_npu.get("_ort_run_ms")
                print(f"    {iid:14s} logits の差 {t1[iid]['logits_rel'] * 100:5.2f}%  1 位 {'同じ' if t1[iid]['top1_same'] else '違う'}", flush=True)
            meta["t1_device"] = dev1.first
            D.adb("shell", f"am force-stop {D.PKG}")
            save()

    save()
    s = summarize(items, t1)
    ch, stp = s["chain"], s["step"]
    n = s["n_items"]
    print(f"\n  ■ NPU で続けた答え（Python の入れ方）の 1 トークン目: 端末の評価（C++ の入れ方）と {ch['npu_eq_dev16']}/{n} 一致、"
          f"PC の CPU と {ch['npu_eq_cpu16']}/{ch['n_with_cpu16']}、HF と {ch['npu_eq_hf']}/{n}（端末の評価と HF は {ch['dev16_eq_hf']}/{n}）")
    print(f"  ■ 同じ入力での 1 回ぶん（{stp['n']} 回）: 1 位が同じ {stp['top1_same']}/{stp['n']}、logits の差の中央値 "
          f"{(stp['logits_rel_median'] or 0) * 100:.2f}%（プロンプトの最後の回 {(stp['logits_rel_median_last_prompt'] or 0) * 100:.2f}%）")
    if s.get("layer_t16"):
        lay = s["layer_t16"]
        print("  ■ 層ごとの K の差（16 トークン版、中央値）: " + " ".join(f"{k[1:]}:{lay[k] * 100:.1f}%" for k in lay if k[0] == "k"))
        if s.get("t1"):
            l1, lr = s["t1"]["layer"], s["layer_t16_last_row"]
            print("  ■ プロンプトの最後の 1 行どうしで比べる（同じ位置・同じ過去）")
            print("  　 16 トークン版                     : " + " ".join(f"{k[1:]}:{lr[k] * 100:.1f}%" for k in lr if k[0] == "k"))
            print("  　 1 トークン版（基準）              : " + " ".join(f"{k[1:]}:{l1[k] * 100:.1f}%" for k in l1 if k[0] == "k")
                  + f"（logits の差 {(s['t1']['logits_rel_median'] or 0) * 100:.2f}%、1 位が同じ {s['t1']['top1_same']}/{s['t1']['n']}）")
    print("  読み方: NPU で続けた答えが端末の評価と同じなら、C++ の入れ方は正しく、原因は NPU の計算（層ごとの差で場所を見る）。"
          "\n          NPU で続けた答えが PC の CPU・HF と同じで、端末の評価だけ違うなら、原因は C++ の入れ方")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
