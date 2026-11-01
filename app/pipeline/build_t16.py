# -*- coding: utf-8 -*-
"""Phase 5-1/5-2: 文章グラフの「1 回で 16 トークン読む」版（`g4t16_W256`）を作る（2026-09-29）。

    python -m app.pipeline.build_t16                                   # 書き出し → 自己確認 → 量子化 → 書き込み → 繋ぐ（2〜3 時間）
    python -m app.pipeline.build_t16 --steps equiv --only A            # 自己確認だけ（数分。最初にこれで配線を確かめる）
    python -m app.pipeline.build_t16 --steps convert                   # 変換（QAIRT、4 時間前後。夜に回す）
    python -m app.pipeline.build_t16 --only C,D                        # かたまりを絞る
    python -m app.pipeline.build_t16 --redo                            # 済んだ段もやり直す（既定では済んだ段は飛ばす）

**途中で止まっても、同じコマンドをもう一度打てば済んだ段は飛ばして続きから動く**（2026-09-29: 1 回目は量子化の途中で
画面を閉じてしまい、`forrtl: error (200): program aborting due to window-CLOSE event` で止まった）。
済んだかどうかは、段ごとの出力ファイル（書き出しの ONNX・較正の入力、量子化の QDQ、書き込み後の final_qdq）と、
自己確認の結果（results/P5_t16/build.json）で見る。

**EdgeSLM_Android の venv（$PY2）で動かす**（torch・onnx・onnxruntime が要る）。HF（transformers）は使わない。

## 何をするか（EdgeSLM_AndroidGemma4 の `build_full.py` と同じ手順を、T=16 で）

| 段（--steps） | 中身 | 道具 |
|---|---|---|
| export | かたまりごとに、較正の入力（下）を作り、T=16 のグラフを書き出す（`g4t_graph.DecodeChunkT`）。出力層 H は 1 トークン版の完成品を複製 | このファイル |
| equiv | **自己確認**: 同じ入力で「1 トークン版を 16 回」と「T=16 版を 1 回」を浮動小数で比べる（A・C・E。マスク・窓・行の選択の約束の確かめ） | このファイル |
| prepare | 較正して QDQ に（`a16w8_rhs8`） | p2b_ops（複製済み） |
| inject | QAT の整数とスケール・8bit の丸めを書き込む → 量子化されない入力に Q/DQ（B〜E）→ C-8 の右辺を 8bit に | qat_inject・fix_float_inputs・downgrade_actact_rhs（複製、変更なし） |
| merge | 6 つを 1 つのグラフに繋ぐ | merge_chunks（複製、変更なし） |
| convert | QAIRT で v68 の context binary → EPContext の ONNX で包む（embed=False） | scripts/build_context.sh・wrap_epctx |

かたまり: A（入口＋層 0〜8）・B（9〜17）・C（18〜22）・D（23〜28）・E（29〜34、出口で 1 行を選ぶ）・H（出力層）。1 トークン版と同じ分け方。
**H は作らない**: E の出口で 1 行を選ぶので、H は 1 トークン版とまったく同じ形になる。Gemma4 で作り実機で動いている
`../EdgeSLM_AndroidGemma4/work/ops/g4head_L256/g4head_L256_final_qdq.onnx`（md5 `5c829673e08298e800e420ca6f49cfc7`、単体のファイル）を複製して使う。

## 較正の入力（Gemma4 が取ってある状態から作る。読むだけ）

`../EdgeSLM_AndroidGemma4/work/capture/*_L256.npz`（8 会話・245 位置。HF で丸めありの状態を 1 トークンずつ取ったもの）から、
**同じ会話の連続する 16 位置**（125 通り）を選び、1 回ぶんの入力に並べる:

| 入力 | 作り方 |
|---|---|
| inputs_embeds・ple_tok・hidden_in・ple_all・cos/sin | 位置 i〜i+15 の値を行に並べる（ple_all はかたまり A の正解 `ref_ple_all` から） |
| past_k/v（[1,1,240,hd]） | 位置 i の past（256 スロット、右詰め）の**後ろ 240 スロット** |
| shared_k/v_13・14（C〜E、[1,1,256,hd]） | **位置 i+15 の窓**（past 240 ＋ この回の 16 と同じ中身になる） |
| attn_mask（[1,16,1,256]） | 行 t: past の有効な右端 pv 個（[240−pv, 240)）と、この回の 0〜t 行目（[240, 240+t]）を 0、ほかは −60。pv は位置 i までのトークン数（最大 240） |
| logit_sel（E、[1,16,1]） | 1 行だけ 1（較正では行をずらして全部の行を使う） |

- 1 トークン版の較正（`export_chunk`）と同じく、代表 64 回ぶん
- 注意: Gemma4 の `bench_prefill.make_inputs` のマスクは、時間を測るためだけのもので past の一部を塞いでいた（中身の正しさは問わなかった）。
  ここでは端末の C++ と同じ正しいマスクを作る（`mask_rows`）

## 名前

かたまり `g4t16_c0_8h`・`g4t16_c9_17`・…・`g4t16_head`、繋いだもの `g4t16_W256`（`work/ops/` の下）。
結果の記録は `results/P5_t16/`。

## 使う段・記録

- Phase 5（基本設計_Phase5.md、StudyMemo 10.3〜10.6 節）。グラフの中身は g4t_graph.py、確かめの ② は check_t16.py
- equiv の結果（①）は `results/P5_t16/build.json` の `equiv`（相対 L2 ≤ 1.3×10⁻⁶。StudyMemo 10.6 節）
- 端末へ送るのは convert の出力 `work/ops/g4t16_W256/g4t16_W256_a16w8_rhs8_epctx.onnx` ＋ `build/ctx_*/*.bin`（setup_vlm_demo --text t16）
- 2026-10-02 に層 0〜14 の部品を書き換えた版を build_t16fix.py で作った（StudyMemo 10.7 節の 6 段目）。このファイルは元の版
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from app.pipeline import gemma4_common as G
from app.pipeline import vlm_common as C

# 量子化の道具は別のプロセスで動かす（同じ Python で -m を呼ぶ）。T・W・P・マスクの値は端末の job.cpp と同じ約束
PY = sys.executable
SCHEME = "a16w8_rhs8"
T = 16
W = 256
P = W - T                       # past のスロット数（240）
MASK_NEG = -60.0
RES = C.RESULTS / "P5_t16"
GIT_BASH = os.environ.get("GIT_BASH", r"C:\Program Files\Git\bin\bash.exe")
# かたまりの分け方（1 トークン版と同じ）。cap は Gemma4 が取ってある状態のファイル名、head_ple は入口（PLE の射影を中で作る）、
# select は出口で 1 行を選ぶかたまり（E）
CHUNKS = [
    {"name": "A", "layers": (0, 8), "head_ple": True, "cap": "chunk_0_8_L256.npz"},
    {"name": "B", "layers": (9, 17), "cap": "chunk_9_17_L256.npz"},
    {"name": "C", "layers": (18, 22), "cap": "chunk_18_22_L256.npz"},
    {"name": "D", "layers": (23, 28), "cap": "chunk_23_28_L256.npz"},
    {"name": "E", "layers": (29, 34), "cap": "chunk_29_34_L256.npz", "select": True},
    {"name": "H", "head": True},          # 1 トークン版の完成品を複製（copy_head）
]
G4_HEAD = G.G4_OPS / "g4head_L256" / "g4head_L256_final_qdq.onnx"
G4_HEAD_MD5 = "5c829673e08298e800e420ca6f49cfc7"


# かたまりの名前（例: g4t16_c0_8h。末尾の h は入口のかたまり）。full_name は繋いだもの、ops_dir は置き場。
def chunk_name(c: dict) -> str:
    if c.get("head"):
        return f"g4t{T}_head"
    lo, hi = c["layers"]
    return f"g4t{T}_c{lo}_{hi}{'h' if c.get('head_ple') else ''}"


def full_name() -> str:
    # 繋いだグラフの名前（g4t16_W256）。ops_dir はその置き場（work/ops/<名前>/）
    return f"g4t{T}_W{W}"


def ops_dir(nm: str) -> Path:
    return C.OPS / nm


# ---------------------------------------------------------------------------
# 較正の入力
# ---------------------------------------------------------------------------
class Capture:
    """Gemma4 が取ってある状態（npz）を、キーごとに必要なときだけ読む（1 本 数百 MB〜数 GB あるので丸ごと読まない）。"""

    def __init__(self, fname: str):
        # fname = Gemma4 の work/capture/ の中のファイル名。np.load は配列を開くだけで、中身はキーを引いたときに読む
        self.path = G.G4_CAPTURE / fname
        if not self.path.exists():
            raise SystemExit(f"取ってある状態が無い: {self.path}（EdgeSLM_AndroidGemma4 の build_full --steps capture で作られたもの）")
        self.z = np.load(self.path)
        self.files = set(self.z.files)
        self._cache: dict[str, np.ndarray] = {}

    # キーを初めて引いたときだけ npz から読み、覚えておく。drop で忘れる（大きな past を使い終えたらメモリを返すため）
    def __getitem__(self, k: str) -> np.ndarray:
        if k not in self._cache:
            self._cache[k] = self.z[k]
        return self._cache[k]

    def drop(self, k: str):
        # 覚えた配列を忘れる（大きな past を使い終えたらメモリを返す）
        self._cache.pop(k, None)


def windows(cap: Capture) -> list[int]:
    """同じ会話で位置が 1 ずつ続く 16 位置の、先頭の添字。"""
    # 会話の番号（dialog）が 16 個とも同じで、位置（pos）が 1 ずつ増えている所だけを窓にできる（8 会話・245 位置から 125 通り）
    pos, dia = cap["pos"], cap["dialog"]
    return [i for i in range(len(pos) - T + 1)
            if (dia[i:i + T] == dia[i]).all() and (np.diff(pos[i:i + T]) == 1).all()]


def past_valid(mask_row: np.ndarray) -> int:
    """1 トークン版のマスク（位置 i、[1,1,256]）から「位置 i より前の有効なトークン数」を読む（開いている数 − 自分の 1 つ）。"""
    # 1 トークン版のマスクは「右詰めで有効な数＋自分」の所が 0。0 の数から自分の 1 つを引く
    return int((mask_row.reshape(-1) == 0).sum()) - 1


def mask_rows(pv: int, n: int = T) -> np.ndarray:
    """T 行ぶんのマスク [T,1,W]。本物の行 t（< n）: past の右端 pv 個と、この回の 0〜t 行目を開ける。空の行は 0 行目と同じ。"""
    # pv = past の有効な数（0〜240 に収める）、n = 本物の行の数。行 t は past の右端 pv 個（[P−pv, P)）と
    # この回の 0〜t 行目（[P, P+t]）を 0 にする。空の行は 0 行目と同じにする（出力は使わない。NaN を出さないためだけ）
    pv = max(0, min(pv, P))
    m = np.full((T, 1, W), MASK_NEG, np.float32)
    for t in range(T):
        tt = t if t < n else 0
        m[t, 0, P - pv:P] = 0.0
        m[t, 0, P:P + tt + 1] = 0.0
    return m


def make_inputs(c: dict, names: list[str], idxs: list[int], cap: Capture, cap_a: Capture | None,
                sel_rows: list[int] | None = None) -> dict:
    """かたまり c の T=16 の入力を、窓の先頭 idxs ぶん作る（[n, ...]）。"""
    # 引数: names = グラフの入力の名前、idxs = 窓の先頭の添字、cap = このかたまりの取ってある状態、cap_a = かたまり A のもの
    # （ple_all の正解を借りる）、sel_rows = logit_sel で選ぶ行（省くと 0, 1, …, 15 を順に回して全部の行を較正に使う）。
    # 戻り値: {入力の名前: [len(idxs), …] の float32}。作り方はモジュールの説明「較正の入力」の表のとおり。
    out = {}
    stack_keys = {"inputs_embeds", "ple_tok", "hidden_in", "cos_s", "sin_s", "cos_g", "sin_g"}
    for nm in names:
        if nm in stack_keys:
            arr = cap[nm]
            out[nm] = np.stack([np.concatenate([arr[i + t] for t in range(T)], axis=0) for i in idxs]).astype(np.float32)
        elif nm == "ple_all":
            arr = cap_a["ref_ple_all"]
            out[nm] = np.stack([np.concatenate([arr[i + t] for t in range(T)], axis=0) for i in idxs]).astype(np.float32)
        elif nm == "attn_mask":
            mk = cap["attn_mask"]
            out[nm] = np.stack([mask_rows(past_valid(mk[i])) for i in idxs]).astype(np.float32)
        elif nm.startswith("past_"):
            arr = cap[nm]                                   # [N,1,256,hd]
            out[nm] = np.stack([arr[i][:, W - P:, :] for i in idxs]).astype(np.float32)   # 後ろ 240 スロット
            cap.drop(nm)
        elif nm.startswith("shared_"):
            arr = cap[nm]                                   # [N,1,256,hd]（位置ごとの窓）
            out[nm] = np.stack([arr[i + T - 1] for i in idxs]).astype(np.float32)          # 位置 i+15 の窓
            cap.drop(nm)
        elif nm == "logit_sel":
            rows_ = sel_rows if sel_rows is not None else [k % T for k in range(len(idxs))]
            s = np.zeros((len(idxs), T, 1), np.float32)
            for k, r in enumerate(rows_):
                s[k, r, 0] = 1.0
            out[nm] = s
        else:
            raise SystemExit(f"{c['name']}: 入力 {nm} の作り方を知らない")
    return out


# ---------------------------------------------------------------------------
# 書き出し
# ---------------------------------------------------------------------------
def export(c: dict, calib_max: int = 64) -> dict:
    """かたまり c の較正の入力を作り、T=16 のグラフを ONNX に書き出す（段 export）。

    引数: c = CHUNKS の 1 つ、calib_max = 較正に使う窓の数（既定 64。1 トークン版の較正と同じ）。
    戻り値: 名前・較正の回数・窓の数・入出力の形・ノード数。ONNX・較正の npz・spec.json を work/ops/<名前>/ に書く。
    """
    import torch
    from app.pipeline import g4t_graph as GT
    nm = chunk_name(c)
    d = ops_dir(nm)
    d.mkdir(parents=True, exist_ok=True)
    cap = Capture(c["cap"])
    cap_a = Capture(CHUNKS[0]["cap"]) if c["name"] != "A" else None
    lay = list(range(c["layers"][0], c["layers"][1] + 1))
    m = GT.build_chunk(lay, head_ple=bool(c.get("head_ple")), select_row=bool(c.get("select")))
    names, onames = m.input_names(), m.output_names()
    # ① 125 通りの窓から等間隔に最大 64 個を選び、較正の入力を作って npz に保存する
    win = windows(cap)
    sel = [win[int(k)] for k in np.linspace(0, len(win) - 1, min(calib_max, len(win))).astype(int)]
    t0 = time.time()
    ins = make_inputs(c, names, sel, cap, cap_a)
    calib = d / f"{nm}_calib.npz"
    np.savez(calib, **ins)
    print(f"  較正の入力 {len(sel)} 回ぶん（窓 {len(win)} 通りから）-> {calib.name}（{calib.stat().st_size / 2**20:.0f}MiB、"
          f"{time.time() - t0:.0f}s）", flush=True)
    # ② 1 回ぶんの入力で一度流して出力の形を表示し（主な出力だけ）、その入力で ONNX に書き出す（opset 21、重みは外部ファイル）
    args = tuple(torch.from_numpy(ins[k][:1]) for k in names)
    with torch.no_grad():
        outs = m(*args)
    print(f"  入力 {len(names)} 本 / 出力 {len(outs)} 本: " +
          ", ".join(f"{o}{tuple(v.shape)}" for o, v in zip(onames, outs) if o in ("hidden_out", "logits", "ple_all")
                    or o.endswith(("_0", "_13", "_14", "_18"))), flush=True)
    path = d / f"{nm}.onnx"
    t0 = time.time()
    torch.onnx.export(m, args, str(path), input_names=names, output_names=onames,
                      opset_version=21, dynamo=True, external_data=True)
    import onnx
    mo = onnx.load(str(path), load_external_data=False)
    io = {"inputs": [(v.name, [x.dim_value for x in v.type.tensor_type.shape.dim]) for v in mo.graph.input],
          "outputs": [(v.name, [x.dim_value for x in v.type.tensor_type.shape.dim]) for v in mo.graph.output]}
    print(f"  書き出し {time.time() - t0:.0f}s  ノード {len(mo.graph.node)}  演算 {sorted({n.op_type for n in mo.graph.node})}",
          flush=True)
    # ③ 量子化の道具（p2b_ops）が読む spec。文章グラフの重みは 4bit（QAT の INT4）なので weight_bits は 4
    spec = {"model": str(path).replace("\\", "/"),
            "note": f"Gemma 4 E2B {c['name']} T={T} W={W}（Phase 5 のまとめ読み）",
            "per_channel_weights": True, "weight_bits": 4,
            "calib_npz": str(calib).replace("\\", "/")}
    (d / f"{nm}.spec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"name": nm, "calib_samples": len(sel), "windows": len(win), "io": io, "nodes": len(mo.graph.node)}


def copy_head() -> dict:
    """出力層のかたまり（H）は 1 トークン版の完成品を複製する（読むだけの元は変えない）。"""
    import hashlib
    import shutil
    nm = chunk_name(CHUNKS[-1])
    dst = ops_dir(nm) / f"{nm}_final_qdq.onnx"
    if not G4_HEAD.exists():
        raise SystemExit(f"1 トークン版の出力層が無い: {G4_HEAD}")
    md5 = hashlib.md5(G4_HEAD.read_bytes()).hexdigest()
    if md5 != G4_HEAD_MD5:
        print(f"  ★{G4_HEAD.name} の md5 が記録と違う（{md5}）。Gemma4 で作り直された可能性", flush=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(G4_HEAD, dst)
    print(f"  出力層: {G4_HEAD} を複製 -> {dst}（md5 {md5}）", flush=True)
    return {"src": str(G4_HEAD), "md5": md5, "dst": str(dst)}


# ---------------------------------------------------------------------------
# 自己確認（浮動小数、PC）
# ---------------------------------------------------------------------------
# 相対 L2（b が正解）。float64 で計算する。
def rel(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def equiv(c: dict, n_win: int = 2) -> dict:
    """同じ入力で「1 トークン版（gemma4_graph.DecodeChunk）を 16 回」と「T=16 版（DecodeChunkT）」を比べる。

    - 1 トークン版の 1 回目の past は取ってある位置 i の past（256）。2 回目からは**自分の present**（窓全体）を次の past にする
      （＝ T=16 版がこの回の中で自分で K/V を作るのと同じ条件）
    - C〜E（K/V を使い回す層だけ）の shared は、1 トークン版は位置 i+t の窓、T=16 版は位置 i+15 の窓（取ってある HF の値）
    - 見るもの: hidden_out の行 t どうし（と、A なら新しい K/V）。E は行を選ぶので、0・7・15 行目を選んだ 3 回で比べる。
      浮動小数どうしなので 1e-5 前後になるはず
    """
    # ① 同じ重み（QatFile）で 1 トークン版と T=16 版の両方を組む
    import torch
    from app.pipeline import gemma4_graph as M
    from app.pipeline import g4t_graph as GT
    cap = Capture(c["cap"])
    cap_a = Capture(CHUNKS[0]["cap"]) if c["name"] != "A" else None
    lay = list(range(c["layers"][0], c["layers"][1] + 1))
    qf = M.QatFile()
    m1 = M.DecodeChunk(qf, lay, with_ple_proj=bool(c.get("head_ple"))).eval()
    mt = GT.DecodeChunkT(qf, lay, with_ple_proj=bool(c.get("head_ple")), select_row=bool(c.get("select"))).eval()
    win = windows(cap)
    rows_out = []
    for i in [win[int(k)] for k in np.linspace(0, len(win) - 1, n_win).astype(int)]:
        # ---- 1 トークン版を 16 回 ----
        names1 = m1.input_names()
        pv = past_valid(cap["attn_mask"][i])
        cur = {k: cap[k][i][None].astype(np.float32) for k in names1 if k.startswith("past_")}   # [1,1,256,hd]
        h1, new1 = [], []
        for t in range(T):
            feed = {}
            for k in names1:
                if k in ("inputs_embeds", "ple_tok", "hidden_in", "cos_s", "sin_s", "cos_g", "sin_g"):
                    feed[k] = cap[k][i + t][None].astype(np.float32)
                elif k == "ple_all":
                    feed[k] = cap_a["ref_ple_all"][i + t][None].astype(np.float32)
                elif k == "attn_mask":
                    mk = np.full((1, 1, 1, W), MASK_NEG, np.float32)
                    mk[..., W - min(pv + t + 1, W):] = 0.0               # 1 トークン版: 右端の有効な数だけ開ける
                    feed[k] = mk
                elif k.startswith("past_"):
                    feed[k] = cur[k]
                elif k.startswith("shared_"):
                    feed[k] = cap[k][i + t][None].astype(np.float32)
            with torch.no_grad():
                o1 = m1(*[torch.from_numpy(feed[k]) for k in names1])
            on1 = dict(zip(m1.output_names(), [o.numpy() for o in o1]))
            h1.append(on1["hidden_out"][0, 0])
            new1.append({x[len("past_"):]: on1["present_" + x[len("past_"):]][:, :, W - 1, :] for x in cur})
            for k in list(cur):
                cur[k] = on1["present_" + k[len("past_"):]]              # 窓全体 → 次の past
        # ---- T=16 版 ----
        # ② T=16 版を 1 回流し、行ごとの hidden と新しい K/V を 1 トークン版の 16 回ぶんと比べる。E は 1 行だけ出すので 3 回選び直す
        worst_h, worst_k = 0.0, 0.0
        for r in ((0, 7, T - 1) if c.get("select") else (None,)):
            ins = make_inputs(c, mt.input_names(), [i], cap, cap_a, sel_rows=[r] if r is not None else None)
            with torch.no_grad():
                ot = mt(*[torch.from_numpy(ins[k]) for k in mt.input_names()])
            on_t = dict(zip(mt.output_names(), [o.numpy() for o in ot]))
            ht = on_t["hidden_out"]
            if r is not None:                                            # 選んだ 1 行 [1,1,H]
                worst_h = max(worst_h, rel(ht[0, 0], h1[r]))
            else:
                for t in range(T):
                    worst_h = max(worst_h, rel(ht[0, t], h1[t]))
                    for x in new1[t]:
                        pt = on_t["present_" + x]                        # 新しい 16 行（13・14 は窓 256）
                        worst_k = max(worst_k, rel(pt[:, :, pt.shape[2] - T + t, :], new1[t][x]))
        rows_out.append({"i": int(i), "pos": int(cap["pos"][i]), "past_valid": pv,
                         "hidden_rel_max": worst_h, "newkv_rel_max": worst_k if new1[0] else None})
        print(f"  窓 i={i}（位置 {int(cap['pos'][i])}〜、past {pv}）: hidden の相対L2 最大 {worst_h:.2e}"
              + (f"、新しい K/V {worst_k:.2e}" if new1[0] else "") + ("（0・7・15 行目を選んで）" if c.get("select") else ""),
              flush=True)
    # ③ 合否は相対 L2 が 1e-3 未満か（浮動小数どうしなので本来は 1e-5 前後。実測は 1.3×10⁻⁶ 以下）
    ok = all(r["hidden_rel_max"] < 1e-3 and (r["newkv_rel_max"] is None or r["newkv_rel_max"] < 1e-3) for r in rows_out)
    print(f"  → {'一致（T=16 版の窓・マスク・行の選択の約束は 1 トークン版と同じ計算）' if ok else '★一致しない'}", flush=True)
    return {"rows": rows_out, "ok": ok}


# ---------------------------------------------------------------------------
# 量子化・書き込み・繋ぐ・変換（build_full と同じ呼び方）
# ---------------------------------------------------------------------------
# 外のコマンド（量子化の道具など）を動かし、標準出力・標準エラーをログに書く。画面には最後の 3 行だけ。失敗したら止める。
def run(cmd: list, log: Path, env: dict | None = None) -> None:
    print("  $ " + " ".join(str(x) for x in cmd[-6:]), flush=True)
    t0 = time.time()
    log.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([str(x) for x in cmd], cwd=str(C.ROOT), capture_output=True, text=True,
                       encoding="utf-8", errors="replace", env=env)
    log.write_text((r.stdout or "") + "\n---- stderr ----\n" + (r.stderr or ""), encoding="utf-8")
    tail = "\n".join((r.stdout or "").strip().splitlines()[-3:])
    print(f"    {time.time() - t0:.0f}s  {tail}", flush=True)
    if r.returncode != 0:
        print((r.stderr or "")[-1500:])
        raise SystemExit(f"失敗: {' '.join(str(x) for x in cmd[-4:])}（ログ {log}）")


# かたまりの書き込み後の完成品（final_qdq）のパス。
def final_path(c: dict) -> Path:
    nm = chunk_name(c)
    return ops_dir(nm) / f"{nm}_final_qdq.onnx"


# 書き出しが済んでいるか: 記録（build.json）に export があり、ONNX・spec・較正の npz の 3 つがそろっていること。
def export_done(c: dict, rec: dict) -> bool:
    nm = chunk_name(c)
    d = ops_dir(nm)
    return bool(rec.get(c["name"], {}).get("export")) and all(
        (d / f).exists() for f in (f"{nm}.onnx", f"{nm}.spec.json", f"{nm}_calib.npz"))


def prepare_done(c: dict) -> bool:
    """量子化（p2b_ops prepare）が済んでいるか。書き込み（inject）が始まっていれば `_qdq_ort.onnx` に名前が変わっている。"""
    nm = chunk_name(c)
    d = ops_dir(nm)
    if final_path(c).exists() or (d / f"{nm}_{SCHEME}_qdq_ort.onnx").exists():
        return True
    st = d / f"status_prepare_{SCHEME}.json"
    if (d / f"{nm}_{SCHEME}_qdq.onnx").exists() and st.exists():
        try:
            return bool(json.loads(st.read_text(encoding="utf-8"))["stages"]["S4_quantize"]["ok"])
        except Exception:
            return False
    return False


def prepare(c: dict, logs: Path):
    """かたまり c を較正して QDQ にする（段 prepare。p2b_ops を別のプロセスで動かす）。

    p2b_ops は C-1 の指摘があると失敗で終わるが、入口でないかたまりの「量子化されない入力」の指摘だけなら想定どおりなので通す。
    """
    if c.get("head"):
        return                                  # 出力層は複製（量子化済み）
    nm = chunk_name(c)
    try:
        run([PY, "-m", "app.pipeline.p2b_ops", "prepare", "--work", "work/ops", "--name", nm, "--scheme", SCHEME],
            logs / f"t16_prepare_{c['name']}.log")
    except SystemExit:
        # ★入口でないかたまりは `ple_all`（→Gather）と層13・14 の K/V（`shared_*`、→Transpose/Reshape）が量子化されない（C-1）。
        #   この指摘**だけ**なら想定どおりで、inject の後の fix_float_inputs が「作る側と同じ目盛り」の Q/DQ で解消する（build_full と同じ）
        st = json.loads((ops_dir(nm) / f"status_prepare_{SCHEME}.json").read_text(encoding="utf-8"))
        vio = st.get("checks", {}).get("C-1", {}).get("violations", [])
        ok_names = {"ple_all", "shared_k_13", "shared_v_13", "shared_k_14", "shared_v_14"}
        other = [v for v in vio if v.get("tensor") not in ok_names]
        if not vio or other or not st["stages"]["S4_quantize"]["ok"]:
            raise
        print(f"    C-1 の指摘は量子化されない入力だけ（{sorted({v['tensor'] for v in vio})}）→ この後 Q/DQ を入れて解消する",
              flush=True)


def inject(c: dict, logs: Path):
    """QAT の値を書き込み（qat_inject）、量子化されない入力に Q/DQ を入れ（fix_float_inputs）、C-8 の右辺を 8bit にする（段 inject）。

    書き込みの前に、量子化の出力の名前を `_qdq_ort.onnx` に変える（qat_inject が同じ名前 `_qdq.onnx` に書くため。prepare_done はこの名前で済みを見る）。
    """
    if c.get("head"):
        return
    nm = chunk_name(c)
    d = ops_dir(nm)
    ort_qdq = d / f"{nm}_{SCHEME}_qdq_ort.onnx"
    if not ort_qdq.exists():
        (d / f"{nm}_{SCHEME}_qdq.onnx").rename(ort_qdq)
    lay = f"{c['layers'][0]}-{c['layers'][1]}"
    args = ["--qdq", str(ort_qdq), "--prepped", str(d / f"{nm}_prepped.onnx"), "--layers", lay,
            "--out", str(d / f"{nm}_{SCHEME}_qdq.onnx")]
    if c.get("head_ple"):
        args += ["--ple-proj"]
    run([PY, "-m", "app.pipeline.qat_inject", *args], logs / f"t16_inject_{c['name']}.log")
    src = d / f"{nm}_{SCHEME}_qdq.onnx"
    if not c.get("head_ple"):
        # 参照は自分より前のかたまりだけ（A が ple_all、B が層13・14 の K/V＝窓全体を作る）
        refs = []
        for r in CHUNKS[:2]:
            if r["name"] == c["name"]:
                continue
            rn = chunk_name(r)
            refs += ["--ref", str(ops_dir(rn) / f"{rn}_final_qdq.onnx")]
        run([PY, "-m", "app.pipeline.fix_float_inputs", "--qdq", str(src), "--out",
             str(d / f"{nm}_{SCHEME}_qdq_fixin.onnx"), *refs], logs / f"t16_fixin_{c['name']}.log")
        src = d / f"{nm}_{SCHEME}_qdq_fixin.onnx"
    run([PY, "-m", "app.pipeline.downgrade_actact_rhs", "--qdq", str(src),
         "--prepped", str(d / f"{nm}_prepped.onnx"), "--out", str(d / f"{nm}_final_qdq.onnx")],
        logs / f"t16_downgrade_{c['name']}.log")


def merge(logs: Path) -> Path:
    """6 つのかたまりの final_qdq を 1 つのグラフに繋ぎ（merge_chunks）、端末の C++ が当てにしている入出力の形を確かめる。

    確かめる形: inputs_embeds [1,16,1536]・attn_mask [1,16,1,256]・logit_sel [1,16,1]・logits [1,1,262144]・past_k が 15 本で長さ 240。
    past_k が 15 本なのは、K/V を自分で持つのが層 0〜14 だけだから（15〜34 は層 13・14 を使い回す）。
    """
    parts = [str(ops_dir(chunk_name(c)) / f"{chunk_name(c)}_final_qdq.onnx") for c in CHUNKS]
    missing = [p for p in parts if not Path(p).exists()]
    if missing:
        raise SystemExit(f"かたまりの final_qdq が無い: {missing}")
    out = ops_dir(full_name()) / f"{full_name()}_final_qdq.onnx"
    run([PY, "-m", "app.pipeline.merge_chunks", "--chunks", ",".join(parts), "--out", str(out)], logs / "t16_merge.log")
    import onnx
    mo = onnx.load(str(out), load_external_data=False)
    io = {"inputs": [(v.name, [x.dim_value for x in v.type.tensor_type.shape.dim]) for v in mo.graph.input],
          "outputs": [(v.name, [x.dim_value for x in v.type.tensor_type.shape.dim]) for v in mo.graph.output]}
    # 端末の C++ が当てにしている形を確かめる
    ins = dict(io["inputs"])
    outs = dict(io["outputs"])
    want = {"inputs_embeds": [1, T, C.TEXT_HIDDEN], "attn_mask": [1, T, 1, W], "logit_sel": [1, T, 1]}
    bad = [f"{k}: {ins.get(k)} ≠ {v}" for k, v in want.items() if ins.get(k) != v]
    if outs.get("logits") != [1, 1, 262144]:
        bad.append(f"logits: {outs.get('logits')}")
    pk = [k for k in ins if k.startswith("past_k_")]
    if len(pk) != 15 or any(ins[k][2] != P for k in pk):
        bad.append(f"past_k の数・長さ（{len(pk)} 本、長さ {sorted({ins[k][2] for k in pk})}）")
    print(f"  繋いだグラフ: 入力 {len(ins)} 本・出力 {len(outs)} 本。"
          + ("形は想定どおり" if not bad else "★想定と違う: " + "; ".join(bad)), flush=True)
    (ops_dir(full_name()) / "io.json").write_text(json.dumps(io, ensure_ascii=False, indent=1), encoding="utf-8")
    if bad:
        raise SystemExit("繋いだグラフの形が想定と違う")
    return out


def convert(logs: Path) -> dict:
    """繋いだグラフを QAIRT で v68 の context binary にし、検査（C-1〜C-5）してから EPContext の ONNX で包む（段 convert）。

    時間は 4 時間前後（1 トークン版は 4 時間 7 分）。context binary の大きさは NPU の上限（約 1,710MiB）に収まる必要がある。
    """
    from app.pipeline import verify_artifacts, wrap_epctx
    FULL = full_name()
    d = ops_dir(FULL)
    final = d / f"{FULL}_final_qdq.onnx"
    if not final.exists():
        raise SystemExit(f"{final} が無い。先に --steps merge")
    if not Path(GIT_BASH).exists():
        raise SystemExit(f"Git Bash が無い: {GIT_BASH}")
    # ★`QAIRT_INT4_PATCH=1` が要る（変換器の INT4 の不具合を実行時に 1 行直す。文章グラフの重みは 4bit。無いと重みが壊れる）
    env = {**os.environ, "QAIRT_INT4_PATCH": "1"}
    t0 = time.time()
    print("  build_context.sh（S5〜S7）…　4 時間前後かかる（1 トークン版は 4 時間 7 分）", flush=True)
    from app.pipeline.build_vit_ctx import run_streaming
    rc = run_streaming([GIT_BASH, "scripts/build_context.sh", str(final), str(d / "build"), FULL], logs / "t16_convert.log", env)
    bins = sorted((d / "build").glob("ctx_*/*.bin"))
    out = {"sec": round(time.time() - t0, 1), "rc": rc}
    if rc != 0 or not bins:
        raise SystemExit(f"変換に失敗（終了コード {rc}）。{d / 'build'} の *_convert.log / *_gen.log を見ること")
    meta = d / "build" / f"{FULL}_meta.json"
    cmdlog = d / "build" / f"{FULL}_commands.txt"
    cmds = cmdlog.read_text(encoding="utf-8", errors="replace").splitlines() if cmdlog.exists() else []
    ver = verify_artifacts.verify_all(final, meta, cmds)
    out["verify"] = [{"check": x["check"], "pass": bool(x.get("pass")), "detail": str(x.get("detail", ""))} for x in ver["checks"]]
    for x in out["verify"]:
        print(f"  {'合格' if x['pass'] else '★不合格'}  {x['check']}: {x['detail']}", flush=True)
    info = wrap_epctx.run(bins[0], final, d / f"{FULL}_{SCHEME}_epctx.onnx", FULL, embed=False)
    out.update({"ctx_bin": str(bins[0]), "ctx_mib": round(bins[0].stat().st_size / 2**20, 1),
                "epctx": str(d / f"{FULL}_{SCHEME}_epctx.onnx"), "all_pass": bool(ver["all_pass"])})
    print(f"  context binary {out['ctx_mib']}MiB（1 トークン版 1,467MiB。NPU の上限 約 1,710MiB）/ EPContext {info['bytes']}B / "
          f"{out['sec'] / 3600:.1f} 時間", flush=True)
    return out


def main():
    """--steps の段を、かたまりごとに順に流す。済んだ段は飛ばし、段ごとに results/P5_t16/build.json に書く。

    自己確認（equiv）は A（入口）・C（K/V を使い回す層だけのかたまり）・E（行の選択）の 3 つだけで行い、通らなければ量子化に進まない。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", default="export,equiv,prepare,inject,merge")
    ap.add_argument("--only", default="", help="かたまりを絞る（A,B,C,D,E,H）")
    ap.add_argument("--calib-max", type=int, default=64)
    ap.add_argument("--equiv-windows", type=int, default=2)
    ap.add_argument("--redo", action="store_true", help="済んだ段もやり直す")
    a = ap.parse_args()
    C.utf8_stdout()
    steps = a.steps.split(",")
    chunks = [c for c in CHUNKS if not a.only or c["name"] in a.only.split(",")]
    logs = C.WORK / "logs"
    RES.mkdir(parents=True, exist_ok=True)
    res_path = RES / "build.json"
    rec = json.loads(res_path.read_text(encoding="utf-8")) if res_path.exists() else {}
    t_all = time.time()
    for c in chunks:
        key = c["name"]
        rec.setdefault(key, {})
        if "export" in steps:
            if c.get("head"):
                print(f"\n[{key}] 出力層は 1 トークン版の完成品を複製", flush=True)
                rec[key]["copy"] = copy_head()
            elif export_done(c, rec) and not a.redo:
                print(f"\n[{key}] 書き出しは済んでいる（飛ばす）", flush=True)
            else:
                print(f"\n[{key}] 書き出し（T={T}）", flush=True)
                rec[key]["export"] = export(c, a.calib_max)
                rec[key].pop("equiv", None)                  # 書き出し直したら自己確認もやり直す
            C.write_json_atomic(res_path, rec)
        if "equiv" in steps and key in ("A", "C", "E"):
            if rec[key].get("equiv", {}).get("ok") and not a.redo:
                print(f"[{key}] 自己確認は済んでいる（飛ばす）", flush=True)
            else:
                print(f"\n[{key}] 自己確認（1 トークン版 ×16 ↔ T=16 版 ×1、浮動小数）", flush=True)
                rec[key]["equiv"] = equiv(c, a.equiv_windows)
                C.write_json_atomic(res_path, rec)
                if not rec[key]["equiv"]["ok"]:
                    raise SystemExit(f"[{key}] 自己確認に通らない。量子化に進まない")
        if c.get("head"):
            continue
        if "prepare" in steps:
            if prepare_done(c) and not a.redo:
                print(f"[{key}] 量子化は済んでいる（飛ばす）", flush=True)
            else:
                print(f"\n[{key}] 量子化（較正）", flush=True)
                prepare(c, logs)
        if "inject" in steps:
            if final_path(c).exists() and not a.redo:
                print(f"[{key}] 書き込みは済んでいる（飛ばす）", flush=True)
            else:
                print(f"\n[{key}] QAT の値と 8bit の丸めを書き込む", flush=True)
                inject(c, logs)
    if "merge" in steps:
        print("\n[全体] 繋ぐ", flush=True)
        rec["merge"] = {"out": str(merge(logs))}
        C.write_json_atomic(res_path, rec)
    if "convert" in steps:
        print("\n[全体] QAIRT で変換（時間がかかる）", flush=True)
        rec["convert"] = convert(logs)
        C.write_json_atomic(res_path, rec)
    print(f"\nおわり（{(time.time() - t_all) / 60:.0f} 分）-> {res_path}")


if __name__ == "__main__":
    main()
