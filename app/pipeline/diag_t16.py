# -*- coding: utf-8 -*-
"""Phase 5 の後: 16 トークン版で「HF の答えとの一致」が下がった理由を切り分ける（2026-09-30）。

    python -m app.pipeline.diag_t16                 # 既定: 16 問（最初の 1 トークン）＋ 文字を読み違えた 2 問は 24 トークン先まで（20〜30 分）
    python -m app.pipeline.diag_t16 --n-mis 6 --n-match 2 --decode 0   # 短く試す（10 分弱）
    python -m app.pipeline.diag_t16 --fresh         # 前の結果を捨てて最初から（既定は済んだ問題を飛ばす）

**EdgeSLM_Android の venv（$PY2）で動かす**（onnxruntime と numpy だけ。HF・スマホは使わない）。

## 何を切り分けるか

126 問で、16 トークン版（端末）の答えは HF（PC、同じ画像の特徴）と 1 トークン目が同じ問題が 78/126 で、1 トークン版（端末）の 97/126 より少なかった。
考えられる原因は 2 つ:

| 原因 | 中身 | この道具で見えること |
|---|---|---|
| (A) 16 トークン版の**量子化の目盛り** | 重みは 1 トークン版とビット単位で同じだが、途中の活性の 16bit の目盛りは T=16 の形で較正し直した | PC の CPU で**同じ量子化済みグラフ**（`g4t16_W256_final_qdq.onnx`）を動かしても、端末と同じ答えになる（＝端末は正しく動いていて、グラフそのものが HF から離れている） |
| (B) **端末での実行**（NPU の計算・C++ の複数トークンの経路） | 変換（QAIRT）や NPU の整数の計算、C++ の入れ方に違いがある | PC の CPU の答えが端末と違い、PC の方が HF に近い |

そこで、評価と**まったく同じプロンプト・同じ画像の特徴**を、PC の ONNX Runtime（CPU）で 16 トークン版の量子化済みグラフに流し、
端末（16 トークン版）・HF・端末（1 トークン版）の答えと比べる。

## 入れ方（端末の C++ `generate_kv` の複数トークンの経路と同じ）

- プロンプト: 評価（system 文は前の「画像についての質問に、日本語で簡潔に答えてください。」）と同じ ID。
  最新の評価の job（`work/device_vit/vlm_eval_*.json`、system 文は新しい方）から質問の部分（`<image|>` の後ろ）を取り、
  前の system 文の頭（端末のログで確認した 23 個）と画像の 72 個を付け直す。**長さが評価の記録の `prompt_len` と一致することを確かめる**
- 画像の特徴: 評価で端末に送ったファイル（`work/device_vlm/feats/f_XXX.bin`、端末の画像エンコーダの出力 [70,1536]）
- 埋め込み・PLE・位置の表: EdgeSLM_AndroidGemma4 の `work/tables/`（端末に写したものと同じ。読むだけ）
- 16 個ずつ読む。本物の行は埋め込み（画像の位置は特徴の 1 行、PLE は pad の行）、空の行は 0。マスクは行ごと（past の有効な右端と、この回の自分より前）、
  値は −60。`logit_sel` は最後の本物の行。回したら past を n 個左へずらし、present の新しい行（後ろから 16 行目から）を右端に書く
- 答えを作るとき（`--decode`）は 1 個ずつ（本物 1 行＋空 15 行）

## 見るもの

- 1 トークン目: PC(16) と 端末(16) が同じか → 同じなら端末は正しい（原因は A）、違うなら B を疑う
- PC(16)↔HF・端末(16)↔HF・端末(1)↔HF の一致（同じ問題どうしで）
- 1 位と 2 位の logit の差（接戦かどうか）
- 文字を読み違えた 2 問（`syn_nav_S`「新宿」→「新倉」、`syn_vertical`「営業」→「宮業」）は 24 トークン先まで作り、端末の答えと同じ所で読み違えるか

結果: `results/P5_t16/diag_t16.json`（1 問ごとに書き足す。途中で止めても同じコマンドで続きから）

## 書き直した版で（2026-10-02 夜に追加）

    python -m app.pipeline.diag_t16 --model t16f --items syn_train,vg_21 --decode 12 --decode-all

- `--model t16f`・`t16t`: `build_t16fix` で書き直した版の量子化済みグラフを PC の CPU で動かす。端末(16) の欄はその版の評価（あれば）
- `--decode-all`: 選んだ全部の問題で `--decode` トークン先まで作る。直した版（t16f）で増えた「画像が提供されていません」という答えが、
  PC の CPU だけで解かせても出るか（＝グラフの性質か、NPU での計算のせいか）を見るため
- 結果は `diag_t16_<種類>.json`、`--decode-all` のときは `…_dec.json` に分ける（元の記録を上書きしない）

## 結果と、その後の訂正（StudyMemo 10.7 節）

- 2026-09-30（1 段目）: PC の CPU で動かした同じグラフは HF とほとんど同じ答えを出し、文字も読めた（「新宿駅西口」「営業中」）。
  端末だけが離れていた → このときは「原因は B（端末での実行）」と読み、probe_t16（2 段目）に進んだ
- ★2026-10-18 追記: この読みは 2026-10-03 の 7 段目で取り消した。PC の CPU も画像の行では正規化の途中の目盛りが切れていて正しくない
  （文章だけで較正した目盛りを画像の行がはみ出す）。答えの 1 トークン目は半分が「画像」で始まるので、一致の数だけでは CPU が正しいとは言えない。
  上の「見るもの」の読み方（PC(16) が HF に近ければ B を疑う）は、PC の CPU が正しいことを前提にしている点に注意
- 決定（2026-10-03）: 文章側は元の 16 トークン版で確定（`results/P5_20261003_t16_accuracy.md`）
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from app.pipeline import gemma4_common as G
from app.pipeline import vlm_common as C

# 端末の C++ と同じ定数。IMG・BOI・EOI は画像のトークン・画像の始まり・終わりの ID（vlm_common と同じ値）、PAD は PLE を引く行。
# STOP は答えを止める ID（vlm_common.STOP_IDS と同じ 3 つ）。マスクの −60 は「ほぼ −∞」（−3.4e38 は 16bit に量子化できない。C-9）
T = 16                       # 1 回に入れる行
W = 256                      # 注意の窓
P = W - T                    # past のスロット数（240）
IMG, BOI, EOI, PAD = 258880, 255999, 258882, 0
MASK_NEG = -60.0
STOP = {1, 50, 106}
# 前の system 文（「画像についての質問に、日本語で簡潔に答えてください。」）の頭。Phase 3 の画面のログ（vlm prompt）と評価の job で確認した並び
OLD_HEAD = [2, 105, 9731, 107, 32613, 144509, 70622, 237000, 236951, 94951, 237007, 239309, 241910, 237000,
            238843, 19230, 15142, 236924, 106, 107, 105, 2364, 107]
OCR_ITEMS = ["syn_nav_S", "syn_vertical"]      # 16 トークン版で文字を読み違えた 2 問

T16_QDQ = C.WORK / "ops" / "g4t16_W256" / "g4t16_W256_final_qdq.onnx"
# 16 トークン版の種類（2026-10-02 夜に追加）: t16 = 元、t16f = 正規化の平均と PLE の tanh を書き直した版、t16t = PLE の tanh だけ書き直した版
T16_GRAPHS = {"t16": "g4t16_W256", "t16f": "g4t16f_W256", "t16t": "g4t16t_W256"}


def model_qdq(model: str) -> Path:
    """種類 → PC の CPU で動かす量子化済みグラフ"""
    n = T16_GRAPHS[model]
    return C.WORK / "ops" / n / f"{n}_final_qdq.onnx"
# 読む場所: 埋め込み・PLE・位置の表（Gemma4 のもの）、評価の job と端末に送った画像の特徴、評価の結果、この道具の結果
TABLES = G.G4_TABLES
JOBS = C.WORK / "device_vit"
FEATS = C.WORK / "device_vlm" / "feats"
REF = C.RESULTS / "P1_ref_vlm"
RES = C.RESULTS / "P5_t16"
RESULT_FILES = {
    "dev16": "fixed_70_device_t16_vnpz-vit_b70_21x30_nosrq_dev.json",     # 端末・16 トークン版（前の system 文）
    "dev1": "fixed_70_device_vnpz-vit_b70_21x30_nosrq_dev.json",          # 端末・1 トークン版（Phase 3）
    "hf": "fixed_70_vnpz-vit_b70_21x30_nosrq_dev.json",                   # PC の HF（同じ画像の特徴）
}


# ---------------------------------------------------------------------------
# 表（端末の C++ と同じ読み方）
# ---------------------------------------------------------------------------
# 端末の C++ と同じ読み方で、埋め込み（2bit）・PLE（4bit）・RoPE の表を読む。大きな表は memmap で必要な行だけ読む。
# tables.json に表のファイル名・行と列の数・スケールのファイル・倍率（埋め込みは √1536、PLE は 16）が書いてある。
class Tables:
    def __init__(self, d: Path):
        # d = 表の置き場（Gemma4 の work/tables/）。tables.json から、埋め込み（2bit）・PLE（4bit）の表とスケール、
        # RoPE の表（局所・全体）の名前と形を読む。大きな表は memmap（触った行だけ読む）
        meta = json.loads((d / "tables.json").read_text(encoding="utf-8"))
        e, p, r = meta["embed"], meta["ple"], meta["rope"]
        self.H = e["cols"]
        self.emb_q2 = np.memmap(d / e["file"], np.uint8, "r", shape=(e["rows"], e["packed_cols"]))
        self.emb_sc = np.fromfile(d / e["scale_file"], np.float32)
        self.emb_mul = float(e["embed_scale"])                      # √1536
        self.L, self.D = p["layers"], p["dim"]
        self.ple_q4 = np.memmap(d / p["file"], np.uint8, "r", shape=(p["rows"], p["packed_cols"]))
        self.ple_sc = np.fromfile(d / p["scale_file"], np.float32).reshape(p["rows"], self.L)
        self.ple_mul = float(p["embed_scale"])                      # 16
        self.rope = {k: np.fromfile(d / f, np.float32).reshape(r["rows"], r["dims"][k]) for k, f in r["files"].items()}
        self.rope_rows = r["rows"]

    def embed(self, tid: int) -> np.ndarray:
        """2bit の埋め込み: uint8 の下位から 2bit ずつ 4 個、値は 0..3 から 2 を引く × 行のスケール × √1536"""
        row = np.asarray(self.emb_q2[tid])
        q = np.stack([(row >> s) & 3 for s in (0, 2, 4, 6)], axis=1).reshape(-1).astype(np.float32) - 2.0
        return q * (self.emb_sc[tid] * self.emb_mul)

    def ple(self, tid: int) -> np.ndarray:
        """4bit の PLE: uint8 の下位 4bit → 上位 4bit、値は 0..15 から 8 を引く × (行, 層) のスケール × 16。[35,256]"""
        row = np.asarray(self.ple_q4[tid])
        q = np.stack([row & 15, row >> 4], axis=1).reshape(self.L, self.D).astype(np.float32) - 8.0
        return q * (self.ple_sc[tid][:, None] * self.ple_mul)


# ---------------------------------------------------------------------------
# 16 トークン版を CPU で動かす（C++ の複数トークンの経路と同じ手順）
# ---------------------------------------------------------------------------
# 16 トークン版の量子化済みグラフを PC の ORT（CPU）で動かし、端末の C++ の複数トークンの経路と同じ手順で past を進める。
# past は層ごとに [1,1,240,hd]、n_valid は past の有効な数（右詰め）、abs_pos は次に入れるトークンの位置（RoPE の表の行）。
class Runner16:
    def __init__(self, path: Path, tabs: Tables):
        # path = 16 トークン版の量子化済みグラフ。ORT の CPU（最適化は basic）で開き、past_* を層の順に並べて、
        # 対になる present_* の出力番号と logits の出力番号を覚える
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        t0 = time.time()
        self.s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        print(f"  読み込み {path.name} {time.time() - t0:.0f}s", flush=True)
        self.tabs = tabs
        self.ins = {v.name: v.shape for v in self.s.get_inputs()}
        self.outs = [v.name for v in self.s.get_outputs()]
        self.kv = sorted([n for n in self.ins if n.startswith("past_")], key=lambda n: (int(n.rsplit("_", 1)[1]), n))
        self.out_of = {n: self.outs.index(n.replace("past_", "present_")) for n in self.kv}
        self.li = self.outs.index("logits")
        self.reset()

    # 新しい問題の前に past を 0、有効な数と位置を 0 に戻す
    def reset(self):
        self.past = {n: np.zeros(self.ins[n], np.float32) for n in self.kv}
        self.n_valid = 0
        self.abs_pos = 0

    def run(self, rows: list[tuple[int, np.ndarray | None]]) -> np.ndarray:
        """rows: 本物の行 n 個（ID と、画像の位置なら特徴の 1 行）。選んだ最後の本物の行の logits を返し、past と位置を進める"""
        # ① 本物の行の入力を作る（文字は表から、画像の位置は特徴の 1 行。PLE は pad の行）。空の行は 0 のまま
        tb, n = self.tabs, len(rows)
        emb = np.zeros((1, T, tb.H), np.float32)
        ple = np.zeros((1, T, tb.L, tb.D), np.float32)
        rope = {k: np.zeros((1, T, 1, v.shape[1]), np.float32) for k, v in tb.rope.items()}
        pos_last = 0
        for r, (tid, feat) in enumerate(rows):
            pos = min(self.abs_pos + r, tb.rope_rows - 1)
            if feat is not None:                       # 画像の位置: 特徴の 1 行をそのまま（√1536 倍しない）、PLE は pad の行
                emb[0, r] = feat
                ple[0, r] = tb.ple(PAD)
            else:
                emb[0, r] = tb.embed(tid)
                ple[0, r] = tb.ple(tid)
            for k, v in tb.rope.items():
                rope[k][0, r, 0] = v[pos]
            pos_last = pos
        for r in range(n, T):                          # 空の行: 位置は最後の本物の行と同じ
            for k, v in tb.rope.items():
                rope[k][0, r, 0] = v[pos_last]
        # ② 行ごとのマスク（past の有効な右端 nv 個と、この回の 0〜r 行目を開ける）と、最後の本物の行を選ぶ logit_sel
        mask = np.full((1, T, 1, W), MASK_NEG, np.float32)
        nv = min(self.n_valid, P)
        for r in range(T):
            rr = r if r < n else 0                     # 空の行は行 0 と同じ
            mask[0, r, 0, P - nv:P] = 0.0
            mask[0, r, 0, P:P + rr + 1] = 0.0
        sel = np.zeros((1, T, 1), np.float32)
        sel[0, n - 1, 0] = 1.0
        feed = {"inputs_embeds": emb, "ple_tok": ple, "attn_mask": mask, "logit_sel": sel, **rope, **self.past}
        feed = {k: v for k, v in feed.items() if k in self.ins}
        # ③ 1 回流して、past を進める。present の新しい行は「後ろから 16 行目」から始まる（層 13・14 は窓全体 256 行、ほかは 16 行の形）
        out = self.s.run(None, feed)
        for name in self.kv:                           # past を n 個左へずらし、新しい行を右端に
            pr = out[self.out_of[name]]
            lo = pr.shape[2]
            new = pr[:, :, lo - T:lo - T + n, :]
            pb = self.past[name]
            self.past[name] = np.concatenate([pb[:, :, n:, :], new], axis=2) if n < P else new[:, :, -P:, :]
        self.n_valid = min(self.n_valid + n, P)
        self.abs_pos += n
        return np.asarray(out[self.li]).reshape(-1)


# ---------------------------------------------------------------------------
# プロンプトと画像の特徴（評価と同じもの）
# ---------------------------------------------------------------------------
def load_items() -> dict:
    """評価の job から、問題ごとの「質問の部分の ID」と「画像の特徴のファイル」を取る"""
    items = {}
    for jf in sorted(JOBS.glob("vlm_eval_*.json")):
        j = json.loads(jf.read_text(encoding="utf-8"))
        for dg in j.get("dialogs", []):
            ids = dg["turns"][0]
            if EOI not in ids or dg["id"] in items:
                continue
            tail = ids[ids.index(EOI) + 1:]
            items[dg["id"]] = {"tail": tail, "feat": FEATS / Path(dg["image_features"]).name}
    return items


# 前の system 文の頭（23 個）＋画像の始まり＋画像 70 個＋画像の終わり＋質問の部分、で評価と同じ ID の並びを作る（画像まわりで 72 個）
def old_prompt(tail: list[int]) -> list[int]:
    return OLD_HEAD + [BOI] + [IMG] * 70 + [EOI] + tail


def load_results(model: str = "t16") -> dict:
    """評価の記録。model が t16 以外なら、「端末・16 トークン版」をその版の評価（あれば）に差し替える"""
    out = {}
    files = dict(RESULT_FILES)
    alt = f"fixed_70_device_{model}_vnpz-vit_b70_21x30_nosrq_dev.json"
    if model != "t16" and (REF / alt).exists():
        files["dev16"] = alt
    for k, fn in files.items():
        d = json.loads((REF / fn).read_text(encoding="utf-8"))
        out[k] = {r["id"]: r for r in d["results"]}
    return out


# 調べる問題を選ぶ: 文字を読み違えた 2 問＋「端末(16)と HF の 1 トークン目が違う」問題から n_mis 問＋「同じ」問題から n_match 問（対照）。
# どちらも等間隔に選ぶ（乱数を使わない）。--items で直接指定したときはそれだけ。
def pick(items: dict, res: dict, n_mis: int, n_match: int, explicit: list[str]) -> list[str]:
    if explicit:
        return explicit
    ids = [i for i in res["dev16"] if i in items and i in res["hf"] and i in res["dev1"]]
    mis = [i for i in ids if res["dev16"][i]["gen_ids"][:1] != res["hf"][i]["gen_ids"][:1] and i not in OCR_ITEMS]
    mat = [i for i in ids if res["dev16"][i]["gen_ids"][:1] == res["hf"][i]["gen_ids"][:1] and i not in OCR_ITEMS]
    take = lambda xs, n: [xs[int(k)] for k in np.linspace(0, len(xs) - 1, n).round().astype(int)] if n > 0 and xs else []
    return OCR_ITEMS + take(mis, n_mis) + take(mat, n_match)


# 2 つの ID の並びが頭から何個同じか（読み違える所を見つけるため）。
def prefix_len(a: list[int], b: list[int]) -> int:
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]:
        k += 1
    return k


def main():
    """選んだ問題を 1 問ずつ PC の CPU で解き、端末(16)・端末(1)・HF の答えと 1 トークン目と続きを比べる。1 問ごとに結果を書き足す。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mis", type=int, default=10, help="端末(16)と HF の 1 トークン目が違った問題から選ぶ数")
    ap.add_argument("--n-match", type=int, default=4, help="同じだった問題から選ぶ数（比べるための対照）")
    ap.add_argument("--decode", type=int, default=24, help="文字を読み違えた 2 問で、何トークン先まで作るか（0 で作らない）")
    ap.add_argument("--items", default="", help="問題を直接指定（カンマ区切り）")
    ap.add_argument("--fresh", action="store_true", help="前の結果を捨てて最初から")
    ap.add_argument("--model", default="t16", choices=list(T16_GRAPHS),
                    help="PC の CPU で動かす 16 トークン版（t16 = 元、t16f・t16t = build_t16fix で書き直した版）")
    ap.add_argument("--decode-all", action="store_true",
                    help="選んだ全部の問題で --decode トークン先まで作る（「画像が無い」と答えるかを見る。結果は …_dec.json に分ける）")
    a = ap.parse_args()
    C.utf8_stdout()
    qdq = model_qdq(a.model)
    if not qdq.exists():
        raise SystemExit(f"{qdq} が無い")
    items, res = load_items(), load_results(a.model)
    sel = pick(items, res, a.n_mis, a.n_match, [x for x in a.items.split(",") if x])
    out_path = RES / ("diag_t16" + ("" if a.model == "t16" else f"_{a.model}") + ("_dec" if a.decode_all else "") + ".json")
    done = {}
    if out_path.exists() and not a.fresh:
        done = {r["id"]: r for r in json.loads(out_path.read_text(encoding="utf-8")).get("rows", [])}
    print(f"  問題 {len(sel)}（前の結果 {len([i for i in sel if i in done])} 問は飛ばす）: {', '.join(sel)}", flush=True)

    tabs = Tables(TABLES)
    run16 = Runner16(qdq, tabs)
    rows = [done[i] for i in sel if i in done]
    t_all = time.time()
    for k, iid in enumerate(sel):
        if iid in done:
            continue
        it = items[iid]
        # ① プロンプトを作り直し、長さが評価の記録と同じか・画像の特徴が 70 行かを確かめる（違えば比べる意味がないので止める）
        ids = old_prompt(it["tail"])
        plen = res["dev16"][iid]["prompt_len"]
        if len(ids) != plen:
            raise SystemExit(f"{iid}: 作ったプロンプト {len(ids)} 個が評価の記録（{plen} 個）と合わない")
        feats = np.fromfile(it["feat"], np.float32).reshape(-1, tabs.H)
        if feats.shape[0] != 70:
            raise SystemExit(f"{iid}: 画像の特徴が {feats.shape[0]} 行（70 のはず）")
        t0 = time.time()
        run16.reset()
        img_next = 0
        logits = None
        # ② 16 個ずつ読む。最後の回の logits が答えの 1 つ目
        for c in range(0, len(ids), T):
            chunk = []
            for tid in ids[c:c + T]:
                if tid == IMG:
                    chunk.append((tid, feats[img_next]))
                    img_next += 1
                else:
                    chunk.append((tid, None))
            logits = run16.run(chunk)
        top = np.argsort(-logits)[:5]
        gen = [int(top[0])]
        # ③ 文字の 2 問（と --decode-all のとき）は、1 個ずつ（本物 1 行＋空 15 行）答えの続きを作る
        n_dec = a.decode if (iid in OCR_ITEMS or a.decode_all) else 0
        while len(gen) < 1 + n_dec and gen[-1] not in STOP:
            lg = run16.run([(gen[-1], None)])
            gen.append(int(lg.argmax()))
        d16, d1, hf = (res[x][iid]["gen_ids"] for x in ("dev16", "dev1", "hf"))
        row = {
            "id": iid, "prompt_len": len(ids),
            "cpu16_first": gen[0], "dev16_first": d16[0], "dev1_first": d1[0], "hf_first": hf[0],
            "cpu16_top5": [int(x) for x in top], "cpu16_top5_logit": [round(float(logits[x]), 3) for x in top],
            "cpu16_margin": round(float(logits[top[0]] - logits[top[1]]), 3),
            "cpu16_logit_of_dev16": round(float(logits[d16[0]]), 3), "cpu16_logit_of_hf": round(float(logits[hf[0]]), 3),
            "cpu16_gen": gen,
            "prefix_cpu16_dev16": prefix_len(gen, d16) if len(gen) > 1 else None,
            "prefix_cpu16_hf": prefix_len(gen, hf) if len(gen) > 1 else None,
            "prefix_dev16_hf": prefix_len(d16, hf), "prefix_dev1_hf": prefix_len(d1, hf),
            "sec": round(time.time() - t0, 1),
        }
        rows.append(row)
        mark = lambda x, y: "＝" if x == y else "≠"
        print(f"  {k + 1:>2}/{len(sel)} {iid:16s} PC16{mark(gen[0], d16[0])}端末16  PC16{mark(gen[0], hf[0])}HF  "
              f"端末16{mark(d16[0], hf[0])}HF  端末1{mark(d1[0], hf[0])}HF  差 {row['cpu16_margin']:.2f}"
              + (f"  続き: PC16 と端末16 が {row['prefix_cpu16_dev16']} トークン目まで同じ（HF とは {row['prefix_cpu16_hf']}）" if n_dec else "")
              + f"（{row['sec']:.0f}s）", flush=True)
        RES.mkdir(parents=True, exist_ok=True)
        C.write_json_atomic(out_path, {"model": str(qdq), "rows": rows})

    # ④ 1 トークン目の一致の数と、1 位と 2 位の logit の差（接戦かどうか）の中央値をまとめる
    # ---- まとめ ----
    n = len(rows)
    agree = lambda f: sum(1 for r in rows if f(r))
    s = {
        "n": n,
        "cpu16_eq_dev16": agree(lambda r: r["cpu16_first"] == r["dev16_first"]),
        "cpu16_eq_hf": agree(lambda r: r["cpu16_first"] == r["hf_first"]),
        "dev16_eq_hf": agree(lambda r: r["dev16_first"] == r["hf_first"]),
        "dev1_eq_hf": agree(lambda r: r["dev1_first"] == r["hf_first"]),
        "margin_median": float(np.median([r["cpu16_margin"] for r in rows])) if rows else None,
        "margin_median_when_cpu16_ne_hf": float(np.median([r["cpu16_margin"] for r in rows if r["cpu16_first"] != r["hf_first"]]))
        if any(r["cpu16_first"] != r["hf_first"] for r in rows) else None,
    }
    C.write_json_atomic(out_path, {"model": str(qdq), "summary": s, "rows": rows})
    print(f"\n  {n} 問（{time.time() - t_all:.0f}s）: 1 トークン目が PC(16)＝端末(16) {s['cpu16_eq_dev16']}/{n}、"
          f"PC(16)＝HF {s['cpu16_eq_hf']}/{n}、端末(16)＝HF {s['dev16_eq_hf']}/{n}、端末(1)＝HF {s['dev1_eq_hf']}/{n}")
    print("  読み方: PC(16)＝端末(16) がほぼ全部なら、端末は正しく動いていて、ずれは 16 トークン版の量子化済みグラフそのもの（原因 A）。"
          "\n          PC(16) と端末(16) が多く違い、PC(16) の方が HF に近ければ、端末での実行（原因 B）を疑う")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
