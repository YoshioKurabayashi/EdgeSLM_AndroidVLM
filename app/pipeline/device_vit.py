# -*- coding: utf-8 -*-
"""Phase 2-4 / 2-5: 画像エンコーダ（v68 の context binary）を端末の NPU で動かし、PC と比べる（2026-09-29）。

    python -m app.pipeline.device_vit check                    # 端末がつながっているか・アプリが入っているか
    python -m app.pipeline.device_vit acc                      # 評価用 16 枚: 端末 ↔ PC の QDQ ↔ HF（SRQ なし・ありの 2 版）
    python -m app.pipeline.device_vit bench                    # 1 枚の時間（ウォームアップ 3 回のあと 20 回）
    python -m app.pipeline.device_vit mem                      # NPU のメモリ（dma-buf）・RSS・ロードの時間
    python -m app.pipeline.device_vit coexist                  # Gemma4 の会話アプリ（文章側 1,467MiB）と同時に開けるか
    python -m app.pipeline.device_vit feats --variant srq      # 2-5: 全 108 枚を端末で → work/feats/vit_b70_21x30_srq_dev.npz
    （共通: --variants nosrq,srq  --budget 70）

**EdgeSLM_Android の venv（$PY2）で動かす**（PC の QDQ を ORT で計算するため）。端末には先に VLM 検証アプリ
（`android_app/`、`org.edgeslm.vlm`）を入れておくこと。

## 端末の動かし方（EdgeSLM_AndroidGemma4 と同じ）

アプリは画面を持たない入口（RunnerActivity）に job（JSON）を渡すと、Foreground Service の中で C++ の runner が動き、
結果を `files/out/last_result.json` に書く。PC は adb で job と入力を送り、結果を待って取りに行く。

| task | 使い道 | 返ってくるもの |
|---|---|---|
| `infer` | 実データの入力で動かし、出力を生バイト列で書く | 出力のファイル・1 回の時間（ort_run_ms）・`describe`（strict＝CPU に逃げずに開けたか） |
| `bench` | 同じ入力（乱数）で繰り返し動かす | ウォームアップと本番の時間の生の値・中央値・p90 |
| `memprobe` | 何も載せていない状態からモデルを開き、1 回動かす | 開く前後・動かした後の dma-buf（NPU のメモリ）・RSS・ロード中の最大 RSS・開く時間 |

adb の小さな道具（adb・md5・push の検証）は EdgeSLM_AndroidGemma4/app/pipeline/device_small.py から、
結果の待ち方は probe_two_proc.py から写した（DoC/引き継ぎ一覧.md）。

## 何を見るか（基本設計 Phase 2 の Q9〜Q11）

| 問い | 見るもの | 目安 |
|---|---|---|
| Q9 全部 NPU で動くか | `describe.strict_no_cpu_fallback`（CPU への逃げ道を禁じて開けたか）・`backend` | true・"all nodes" |
| Q10 端末の値は PC の真似と合うか | **SRQ なしの版**: 端末 ↔ PC の QDQ の相対 L2（Gemma4 の文章側は 1.8%）、端末 ↔ HF（SRQ なし）の cos | 相対 L2 が数 % 以内、cos ≥ 0.99 |
| 　SRQ ありの版 | 記録だけ（SRQ の揺れで PC の真似とも離れうる）。品質は 2-5（feats → ref_vlm）で見る | — |
| Q11 時間とメモリ | bench の中央値、memprobe の dma-buf、文章側と同時に開けるか | 見込み 0.2〜0.5 秒・190〜215MiB |

結果は `results/P2_vit_device/`、端末とやり取りしたファイルは `work/device_vit/`。

## 入力と出力・使った記録（2026-10-18 追記）

- 入力: build_vit_ctx が作った EPContext の ONNX と context binary（`<OPS>/vit_<tag>/`）、
  capture_vit が作った評価用 16 枚の pixel_values と HF の特徴（`<CAPTURE>/vit_<tag>.npz`）、
  feats では評価セットの manifest.json の全画像。tag は「b<予算>_<縦>x<横>」（例 b70_21x30）。
- 出力: `results/P2_vit_device/<tag>_<コマンド>.json`（例 b70_21x30_acc.json）、feats は
  `work/feats/vit_<tag>_<版>_dev.npz`（ref_vlm --vision npz:… に渡す）。
- 記録: results/P2_20260929_vit_device.md（例: 直した後の端末 ↔ HF の cos 最小 0.991、1 枚 159〜195ms、NPU 152MiB）。
  解説は StudyMemo 08「端末でだけ壊れる」・09「端末でつなぐ」。
- この adb・run_job の部分は EdgeSLM_AndroidVLMSpec の device_spec などからも import して使う（StudyMemo VLMSpec 6.1）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from app.pipeline import vlm_common as C

# ---- 端末とアプリの名前・置き場（PC は Windows。adb の場所は決め打ち）----
ADB = r"D:\Android\sdk\platform-tools\adb.exe"
PKG = "org.edgeslm.vlm"
# namespace（Kotlin のパッケージ）は org.edgeslm.demo のまま。短い `.RunnerActivity` は org.edgeslm.vlm.… と
# 解釈されて見つからないので、完全な名前で書く（Gemma4 で踏んだ）
ACT = f"{PKG}/org.edgeslm.demo.RunnerActivity"
# ★画面を前に出したまま job を動かす入口（Phase 3 で追加、2026-09-29）。1.5GB のモデルを開く job はこちらを使う
#   （画面が裏だと、開く間の一時的な 3.1GB で OS に止められることがある。RunnerHoldActivity.kt の説明）
HOLD_ACT = f"{PKG}/org.edgeslm.demo.RunnerHoldActivity"
ACTION = "org.edgeslm.demo.RUN"
DEV = f"/sdcard/Android/data/{PKG}/files"
# QNN EP の設定は Gemma4 と同じ（offload_graph_io_quantization=0: 入出力の float ↔ 整数の変換を NPU 側で行う）
EP = {"provider": "QNN", "backend": "libQnnHtp.so",
      "options": {"offload_graph_io_quantization": "0", "qnn_context_priority": "high"},
      "session": {"intra_op_num_threads": 1, "log_severity_level": 3}}
# 同時に開く相手（文章側）。Gemma4 の会話アプリ。起動すると既定のモデル（1,467MiB）を読み込んで保持する
G4_PKG = "org.edgeslm.gemma4"
G4_CHAT = f"{G4_PKG}/org.edgeslm.demo.ChatActivity"

# 結果（JSON・logcat）と、端末とやり取りしたファイル（job・入力の .bin・取ってきた出力）の置き場
RES = C.RESULTS / "P2_vit_device"
IO = C.WORK / "device_vit"
Q10_REL_GUIDE = 0.05       # SRQ なしの版の「端末 ↔ PC の QDQ」相対 L2 の目安（Gemma4 の文章側 1.8% の数倍まで）


# ---------------------------------------------------------------------------
# adb の小さな道具（EdgeSLM_AndroidGemma4/app/pipeline/device_small.py から）
# ---------------------------------------------------------------------------
def adb(*args, timeout=900, check=True) -> str:
    """adb を呼び、標準出力を文字列で返す。check=True なら失敗（終了コード ≠ 0）で RuntimeError。
    timeout は秒（既定 900 秒＝大きいモデルの push に足りる長さ）。
    """
    r = subprocess.run([ADB, *args], capture_output=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"adb {' '.join(args)}: {r.stderr.decode('utf-8', 'replace')}")
    return r.stdout.decode("utf-8", "replace")


def md5(p: Path) -> str:
    """PC のファイルの md5（8MiB ずつ読む。1.5GB のモデルでもメモリを食わない）。端末の md5sum と比べて送り直しを省くのに使う。"""
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(8 << 20), b""):
            h.update(c)
    return h.hexdigest()


def dev_md5s(rels: list[str]) -> dict:
    """端末のファイルの md5 を 1 回の adb でまとめて取る（無いファイルは入らない）。"""
    if not rels:
        return {}
    # 「md5sum a b c 2>/dev/null || true」: 無いファイルがあっても止めない。出力の各行は「md5 フルパス」
    txt = adb("shell", "md5sum " + " ".join(f"{DEV}/{r}" for r in rels) + " 2>/dev/null || true", check=False)
    out = {}
    for ln in txt.splitlines():
        parts = ln.split()
        if len(parts) == 2 and parts[1].startswith(DEV + "/"):
            out[parts[1][len(DEV) + 1:]] = parts[0]
    return out


def push_many(pairs: list[tuple[Path, str]], label: str):
    """(PC のファイル, 端末の相対パス) を送る。端末に同じ md5 のものがあれば送らない。送ったあと md5 で確かめる。"""
    want = {rel: md5(src) for src, rel in pairs}
    have = dev_md5s(list(want))
    todo = [(src, rel) for src, rel in pairs if have.get(rel) != want[rel]]
    dirs = sorted({str(Path(rel).parent).replace("\\", "/") for _, rel in todo})
    if dirs:
        adb("shell", "mkdir -p " + " ".join(f"{DEV}/{d}" for d in dirs))
    t0 = time.time()
    for src, rel in todo:
        adb("push", str(src), f"{DEV}/{rel}")
    if todo:
        got = dev_md5s([rel for _, rel in todo])
        bad = [rel for _, rel in todo if got.get(rel) != want[rel]]
        if bad:
            raise RuntimeError(f"送ったファイルの md5 が合わない: {bad[:5]}")
    print(f"  {label}: {len(pairs)} 個（新しく送った {len(todo)} 個、{time.time() - t0:.0f}s）", flush=True)


def wake():
    """端末の画面を点け、ロック画面を外す（画面を前に出す job・会話アプリの起動の前に呼ぶ）。"""
    adb("shell", "input keyevent KEYCODE_WAKEUP", check=False)
    adb("shell", "wm dismiss-keyguard", check=False)


class DeviceJobFailed(SystemExit):
    """端末の job が結果を返さなかった。SystemExit の仲間なので、捕まえなければ今までどおりそこで止まる。
    捕まえれば「少し待ってやり直す」ができる（device_vlm eval）。"""


class DeviceProcessDied(DeviceJobFailed):
    """端末のアプリのプロセスが落ちた（OS の lowmemorykiller に止められた等）。"""


class DeviceJobTimeout(DeviceJobFailed):
    """決めた時間までに結果が返らなかった（プロセスは生きているが止まっている等）。"""


def mem_available_mib() -> int:
    """端末の空きメモリ（/proc/meminfo の MemAvailable、MiB）。読めなければ −1。"""
    for ln in adb("shell", "cat /proc/meminfo", check=False).splitlines():
        if ln.startswith("MemAvailable:"):
            try:
                return int(ln.split()[1]) // 1024
            except (ValueError, IndexError):
                return -1
    return -1


def wait_mem(min_mib: int = 3072, timeout_s: int = 90) -> int:
    """空きメモリが min_mib 以上になるまで待つ（最大 timeout_s 秒。届かなくても先へ進む）。
    ★前の job の文章側のモデル（約 1.6GB）が片付く前に次を開くと、一瞬だけ 2 つ分のメモリが要り、OS に止められる
      （2026-09-29 の device_vlm eval: 43 問目からの job が読み込み中に lowmemorykiller に止められた。理由は "device is not responding"）"""
    t0 = time.time()
    m = mem_available_mib()
    while 0 <= m < min_mib and time.time() - t0 < timeout_s:
        time.sleep(3)
        m = mem_available_mib()
    waited = time.time() - t0
    if waited >= 3:
        print(f"    （空きメモリを待った: {waited:.0f}s → {m}MiB）", flush=True)
    return m


def run_job(job: dict, timeout_s: int = 900, activity: str = ACT, hold: bool = False) -> dict:
    """job を送って動かし、結果（last_result.json）を待つ。モデルを開けなかったとき（last_describe / last_error）も拾う。
    hold=True: 画面を前に出したまま動かす（RunnerHoldActivity）。大きいモデルを開く job で OS に止められにくくする"""
    if hold:
        activity = HOLD_ACT
    # ① job の JSON を PC に書き（後から見返せる）、端末の jobs/ へ送る。前の結果のファイルと logcat は消しておく
    #   （古い last_result.json を今回の結果と取り違えないため）
    IO.mkdir(parents=True, exist_ok=True)
    jp = IO / f"{job['job_id']}.json"
    jp.write_text(json.dumps(job, ensure_ascii=False, indent=1), encoding="utf-8")
    adb("shell", f"mkdir -p {DEV}/jobs {DEV}/out")
    adb("push", str(jp), f"{DEV}/jobs/{jp.name}")
    adb("shell", f"rm -f {DEV}/out/last_result.json {DEV}/out/last_describe.json {DEV}/out/last_error.json")
    adb("logcat", "-c", check=False)
    if hold:
        wake()                                             # 画面を点けてロックを外す（前に出すため）
    # ② アプリの入口（Activity）を起動し、job のパスを渡す。アプリは Foreground Service で job を動かす
    r = subprocess.run([ADB, "shell", f"am start -n {activity} -a {ACTION} --es job jobs/{jp.name}"],
                       capture_output=True, timeout=60)
    started = r.stdout.decode("utf-8", "replace") + r.stderr.decode("utf-8", "replace")   # am の誤りは stderr に出ることがある
    if "Error" in started or "does not exist" in started:
        # 入口が無い（アプリが古い）など。job は動いていないので、待たずに止める
        raise SystemExit(f"端末で job を起動できなかった（{activity}）: {started.strip()[:300]}\n"
                         f"  RunnerHoldActivity が無いなら、Phase 3 のアプリをビルドし直して入れること")
    # ③ 2 秒ごとに結果を見る。結果が無ければアプリのプロセスが生きているかを見張る（落ちたらすぐ止める）
    pkg = activity.split("/")[0]
    t0 = time.time()
    n_poll, pid_seen, gone, last_beat = 0, False, 0, t0
    while time.time() - t0 < timeout_s:
        time.sleep(2)
        n_poll += 1
        # 先に結果を見る（すぐ終わる job は、プロセスを見る前に結果が出ている。結果があれば「落ちた」と間違えない）
        j = _read_result(job)
        if j is not None:
            j["_wall_s"] = round(time.time() - t0, 1)
            return j
        # ★アプリのプロセスの見張り。落ちると結果は二度と書かれないので、待ち続けずにすぐ止める
        #   - 最初の 30 秒は 2 秒ごと、そのあとは 10 秒ごと
        #   - 「見えていたのに 2 回続けて無い」か「20 秒たっても一度も見えない」なら落ちたとみなす
        #   （2026-09-29: モデルの読み込み中、開始 10 秒前後で OS に止められた。10 秒ごとの見張りでは一度も見ないまま消えたので
        #     「落ちた」と気づけず、20 分待って時間切れになった）
        el = time.time() - t0
        if el < 30 or n_poll % 5 == 0:
            pid = adb("shell", f"pidof {pkg} || true", check=False).strip()
            if pid:
                pid_seen, gone = True, 0
                if time.time() - last_beat >= 60:          # 1 分ごとに「まだ動いている」ことを見せる
                    last_beat = time.time()
                    print(f"    …端末で実行中（{el:.0f}s、pid {pid}）", flush=True)
            else:
                gone += 1
                if (pid_seen and gone >= 2) or (not pid_seen and el >= 20):
                    j = _read_result(job)                  # 落ちる直前に書き終えていたら、それを使う
                    if j is not None:
                        j["_wall_s"] = round(time.time() - t0, 1)
                        return j
                    log = save_logcat(f"{job['job_id']}_died")
                    # OS に止められた記録（lowmemorykiller の "Kill '<パッケージ>' (pid) … reason: …"）があれば一緒に見せる
                    kill = [ln.split("lowmemorykiller:", 1)[-1].strip() for ln in log.splitlines()
                            if "lowmemorykiller" in ln and f"Kill '{pkg}'" in ln]
                    why = f"。OS のメモリ不足で止められた: {kill[-1]}" if kill else ""
                    raise DeviceProcessDied(f"端末のアプリのプロセスが落ちた（{el:.0f}s）{why}。logcat を {RES} に保存した")
    # ④ 時間切れ: logcat を残して DeviceJobTimeout（呼ぶ側が捕まえればやり直せる）
    save_logcat(f"{job['job_id']}_timeout")
    raise DeviceJobTimeout(f"端末の結果が {timeout_s} 秒で返らない。logcat を {RES} に保存した")


def _read_result(job: dict) -> dict | None:
    """端末の out/ から結果（last_result / last_describe / last_error）を読む。まだ無い・書いている途中なら None。"""
    for kind in ("result", "describe", "error"):
        txt = adb("shell", f"cat {DEV}/out/last_{kind}.json 2>/dev/null || true", check=False)
        if not txt.strip():
            continue
        try:
            j = json.loads(txt)
        except json.JSONDecodeError:
            continue                                      # 書いている途中
        # 結果に別の job の id が入っていたら、前の job の残りなので読まない
        if kind == "result" and j.get("job_id") not in (None, job["job_id"]):
            continue
        j["_kind"] = kind
        return j
    return None


def save_logcat(name: str) -> str:
    """logcat を保存する。2 つ書く:
    - `{name}_logcat.txt`: タグで絞らず最後の 5000 行（OS に kill された記録・QNN のエラーも入るように）
    - `{name}_logcat_edgeslm.txt`: このアプリのタグと lowmemorykiller だけ（長く待つと他のアプリの行で押し出されるので、絞った方も残す）
    ※ logcat の置き場は端末の中で一定の大きさを回して使う。1 時間待つと最初の方の行は消えている（2026-09-29 の 1 回目が空だった理由とみられる）"""
    RES.mkdir(parents=True, exist_ok=True)
    log = adb("logcat", "-d", "-t", "5000", check=False)
    (RES / f"{name}_logcat.txt").write_text(log, encoding="utf-8")
    # このアプリのタグと、OS がアプリを止めた記録（lowmemorykiller の I 以上。"Kill '…'" の行）。
    # タグで絞ると置き場の全体から拾える（最後の 5000 行より前に止められていても残る）
    mine = adb("logcat", "-d", "-s", "EdgeSLMRunner:*", "EdgeSLMVlm:*", "EdgeSLMChat:*", "EdgeSLM:*",
               "lowmemorykiller:I", check=False)
    (RES / f"{name}_logcat_edgeslm.txt").write_text(mine, encoding="utf-8")
    return log + "\n" + mine


def tag_for(budget: int, portrait: bool) -> tuple[str, tuple[int, int]]:
    """画像トークンの予算（budget）と向きから、モデルの名前の印 tag と格子（パッチの縦×横）を返す。
    例: budget=70・横長 → ("b70_21x30", (21, 30))。格子の表は vlm_common.FIXED_GRIDS / PORTRAIT_GRIDS。
    """
    grid = (C.PORTRAIT_GRIDS if portrait else C.FIXED_GRIDS)[budget]
    return f"b{budget}_{grid[0]}x{grid[1]}", grid


def model_files(tag: str, variant: str) -> tuple[Path, Path]:
    """EPContext の ONNX と、それが参照する context binary（embed=False なので別ファイル。同じ場所に置く）。"""
    d = C.OPS / f"vit_{tag}"
    epctx = d / f"vit_{tag}_{variant}_epctx.onnx"
    if not epctx.exists():
        raise SystemExit(f"{epctx} が無い。先に build_vit_ctx を動かすこと")
    # EPContext の属性 ep_cache_context に binary のファイル名が文字列で入っている。onnx を読まずに、
    # 同じ場所にある候補のうち、その名前が ONNX の中に現れるものを選ぶ（onnx の無い環境でも確かめられるように）
    raw = epctx.read_bytes()
    ref = next((c for c in sorted(d.glob(f"vit_{tag}_{variant}_ctx*.bin")) if c.name.encode() in raw), None)
    if ref is None:
        raise SystemExit(f"{epctx.name} が参照する context binary が {d} に無い")
    return epctx, ref


def push_models(tag: str, variants: list[str]) -> dict:
    """版（nosrq / srq）ごとの EPContext と context binary を端末の models/vit/ へ送る（同じ md5 なら送らない）。
    戻り値: 版 → 端末での EPContext の相対パス（job の "model" に書く）。
    """
    rels, pairs = {}, []
    for v in variants:
        epctx, bin_ = model_files(tag, v)
        rels[v] = f"models/vit/{epctx.name}"
        pairs += [(epctx, rels[v]), (bin_, f"models/vit/{bin_.name}")]
    push_many(pairs, "モデル（EPContext と context binary）")
    return rels


# ---------------------------------------------------------------------------
# Phase 5: 16 トークン版の文章モデル（build_t16 が作る。work/ops/g4t16_W256/）
# ---------------------------------------------------------------------------
T16_NAME = "g4t16_W256"
# ★NPU のずれの直し（2026-10-02、build_t16fix）: 正規化の平均を行列の掛け算に・PLE の tanh を 1 行の形に書き換えた版。入出力は同じ
T16F_NAME = "g4t16f_W256"
# ★tanh だけ直した版（2026-10-02 夜、build_t16fix --variant t）: PLE の tanh を 1 行の形にするだけ。正規化は元のまま（HTP の RmsNorm にまとまる）。
#   t16f は 126 問で「画像が無い」と答える誤りが増えた（0 → 12 問）ので、16 行のときだけ起きる不具合（tanh）だけを直した版を別に作る
T16T_NAME = "g4t16t_W256"
# --text・--model の名前 → グラフの名前
T16_KINDS = {"t16": T16_NAME, "t16f": T16F_NAME, "t16t": T16T_NAME}


def t16_files(name: str = T16_NAME) -> tuple[Path, Path]:
    """16 トークン版の EPContext の ONNX と、それが参照する context binary。"""
    d = C.OPS / name
    epctx = d / f"{name}_a16w8_rhs8_epctx.onnx"
    if not epctx.exists():
        raise SystemExit(f"{epctx} が無い。先に build_t16 --steps convert（直した版は build_t16fix）を動かすこと")
    raw = epctx.read_bytes()
    cands = sorted(d.glob("*.bin")) + sorted((d / "build").glob("ctx_*/*.bin"))
    ref = next((c for c in cands if c.name.encode() in raw), None)
    if ref is None:
        raise SystemExit(f"{epctx.name} が参照する context binary が {d} に無い")
    return epctx, ref


def push_t16(name: str = T16_NAME) -> str:
    """16 トークン版を端末の models/ops/ に送る（同じものがあれば送らない）。端末での EPContext の場所を返す。"""
    epctx, b = t16_files(name)
    push_many([(epctx, f"models/ops/{epctx.name}"), (b, f"models/ops/{b.name}")], "16 トークン版の文章モデル（約 1.5GB）")
    return f"models/ops/{epctx.name}"


def text_job(kind: str = "t1") -> dict:
    """文章側の job の基本形。t1 = Gemma4 の会話アプリの gemma4-l256（1 トークン版）、t16 = 16 トークン版、
    t16f = 16 トークン版を NPU のずれに合わせて書き直した版（正規化の平均と PLE の tanh）、t16t = PLE の tanh だけ書き直した版（表は同じ）。"""
    cfg = json.loads((C.GEMMA4_ROOT / "jobs" / "chat_models.json").read_text(encoding="utf-8"))
    job = dict(next(m["job"] for m in cfg["models"] if m["key"] == "gemma4-l256"))
    if kind in T16_KINDS:
        job["model"] = push_t16(T16_KINDS[kind])
    elif kind != "t1":
        raise SystemExit(f"--text は t1・{'・'.join(T16_KINDS)}（{kind}）")
    return job


def ensure_app():
    """端末がつながっていて VLM 検証アプリ（org.edgeslm.vlm）が入っているかを確かめ、つながっている端末のシリアルを返す。
    無ければ止める（何をすればよいかを表示する）。
    """
    out = adb("devices")
    devs = [ln.split()[0] for ln in out.splitlines()[1:] if ln.strip().endswith("device")]
    if not devs:
        raise SystemExit("端末が見つからない（USB でつなぎ、USB デバッグを許可すること）")
    if PKG not in adb("shell", f"pm path {PKG} || true", check=False):
        raise SystemExit(f"端末に {PKG} が入っていない。android_app をビルドして adb install すること（README の 2-4）")
    wake()
    # 入れた直後は「停止状態」なので、入口を 1 回開いてファイル置き場を作らせる（job なしなら作って終わる）
    adb("shell", f"am start -n {ACT}")
    time.sleep(2.0)
    return devs[0]


# ---------------------------------------------------------------------------
# 比べる道具
# ---------------------------------------------------------------------------
def compare(a: np.ndarray, b: np.ndarray) -> dict:
    """a（調べる側）と b（正解の側）: 全体の相対 L2（Gemma4 の device_small と同じ定義）、トークンごとの相対 L2 の平均、cos。

    引数: 1 枚分の特徴 [M, 1536]（M = 画像トークン数。b70 なら 70）。最後の軸以外は 1 本にまとめる。
    ★cos は**全体を 1 本のベクトルとして**取る（トークンごとの cos の平均ではない）。
    """
    a = np.asarray(a, np.float64).reshape(-1, a.shape[-1])
    b = np.asarray(b, np.float64).reshape(-1, b.shape[-1])
    tok = np.linalg.norm(a - b, axis=-1) / np.maximum(np.linalg.norm(b, axis=-1), 1e-12)
    return {"rel_l2": float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12)),
            "rel_tok_mean": float(tok.mean()),
            "cos": float((a * b).sum() / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-300))}


def summarize(rows: list[dict]) -> dict:
    """compare の結果（1 枚 1 つ）を並べ、相対 L2 の中央値・最大と、cos の最小・中央値にまとめる（偶数個のときは上側の中央を取る）。"""
    cs = sorted(r["cos"] for r in rows)
    rl = sorted(r["rel_l2"] for r in rows)
    return {"rel_l2_median": rl[len(rl) // 2], "rel_l2_max": rl[-1], "cos_min": cs[0], "cos_median": cs[len(cs) // 2]}


def pc_qdq(tag: str, variant: str, pix: np.ndarray) -> np.ndarray:
    """PC の QDQ（final_qdq を ORT CPU・最適化なしで）＝量子化だけを真似た値。

    引数: pix = [K, 630, 768]（K 枚の pixel_values。630 = 21×30 パッチ、768 = 16×16×3 の画素）。
    戻り値: [K, M, 1536] の画像の特徴。端末の値と比べる「正解の候補」の 1 つ（端末の NPU の誤差だけを見る）。
    """
    from app.pipeline.eval_qdq_vit import model_path, session
    s = session(model_path(tag, variant))
    return np.stack([s.run(None, {"pixel_values": pix[j][None].astype(np.float32)})[0][0] for j in range(len(pix))])


def infer_images(tag: str, variant: str, model_rel: str, pix_rels: list[str], ids: list[str],
                 out_shape: tuple, job_id: str, timeout_s: int = 1800) -> tuple[dict, np.ndarray]:
    """端末で infer を回し、出力を取ってくる。返り値は（結果の JSON、出力 [K,M,1536]）。"""
    # ① 出力の置き場を job ごとに分け、前の出力を消す（古い .bin を読まないため）
    out_dir = f"out/{job_id}"
    adb("shell", f"rm -rf {DEV}/{out_dir}; mkdir -p {DEV}/{out_dir}")
    job = {"job_id": job_id, "task": "infer", "model": model_rel, "ep": EP,
           "batches": [{"id": i, "inputs": {"pixel_values": r}} for i, r in zip(ids, pix_rels)],
           "output": {"dir": out_dir}}
    # ② 1 枚 1 batch の infer job を動かす。describe / error しか返らなければ、モデルを開けなかった
    res = run_job(job, timeout_s=timeout_s)
    if res.get("_kind") != "result":
        raise SystemExit(f"端末でモデルを開けなかった（{res.get('_kind')}）: {json.dumps(res, ensure_ascii=False)[:800]}")
    import shutil
    local = IO / job_id
    shutil.rmtree(local, ignore_errors=True)                 # 前回の出力を読まない
    adb("pull", f"{DEV}/{out_dir}", str(IO))                 # フォルダごと取る（1 枚ずつ pull すると遅い）→ IO/<job_id>/
    # ③ 各 batch の出力 image_embeds（float32 の生の並び）を [M, 1536] に戻して積む。records は ids と同じ順で返る前提
    outs = np.zeros((len(ids), *out_shape), np.float32)
    errors = []
    for k, rec in enumerate(res.get("records", [])):
        if "error" in rec:
            errors.append((rec.get("id"), rec["error"]))
            continue
        o = next(o for o in rec["outputs"] if o["name"] == "image_embeds")
        outs[k] = np.fromfile(local / Path(o["path"]).name, np.float32).reshape(out_shape)
    if errors:
        raise SystemExit(f"端末の推論でエラー: {errors[:3]}")
    return res, outs


def describe_line(res: dict) -> dict:
    """端末の結果から、記録に残す要点だけを抜き出す。
    strict_no_cpu_fallback（CPU に逃げずに開けたか＝Q9）・backend・ctx_retries（context を開き直した回数）・版・かかった秒数。
    """
    d = res.get("describe", {})
    return {"strict_no_cpu_fallback": d.get("strict_no_cpu_fallback"), "backend": d.get("backend"),
            "ctx_retries": d.get("ctx_retries"), "versions": res.get("versions"), "wall_s": res.get("_wall_s")}


# ---------------------------------------------------------------------------
# コマンド
# ---------------------------------------------------------------------------
def cmd_check(a) -> dict:
    """check: 端末とアプリの確認。機種・Android の版・SoC・アプリの版を表示して返す。"""
    serial = ensure_app()
    info = {"serial": serial,
            "model": adb("shell", "getprop ro.product.model").strip(),
            "android": adb("shell", "getprop ro.build.version.release").strip(),
            "soc": adb("shell", "getprop ro.soc.model").strip(),
            "app": adb("shell", f"dumpsys package {PKG} | grep -m1 versionName", check=False).strip()}
    print(f"  端末: {info['model']}（Android {info['android']}、{info['soc']}）、アプリ: {info['app']}")
    return info


def cmd_acc(a) -> dict:
    """acc（Q9・Q10）: 評価用 16 枚を端末の NPU で画像エンコーダに通し、PC の QDQ・HF の特徴と比べる。

    段: ① モデルと 16 枚の入力を送る → ② 版ごとに端末で infer → ③ 同じ 16 枚を PC の QDQ で計算
        → ④ 4 通りの組で compare → ⑤ SRQ なしの版で Q10 の目安を判定、全版で Q9（strict）を出す
    4 通り: dev_vs_qdq（端末の NPU の誤差だけ）・dev_vs_hf（同じ版の HF と）・qdq_vs_hf（量子化の誤差だけ）・
            dev_vs_hf_nosrq（SRQ なしの HF と。SRQ ありの版がどれだけ動くか）
    結果の例: results/P2_20260929_vit_device.md の 2 節・8.2 節（直した後 端末 ↔ HF の cos 最小 0.991）
    """
    from app.pipeline.export_vit import load_capture
    tag, grid = tag_for(a.budget, a.portrait)
    ensure_app()
    variants = a.variants.split(",")
    rels = push_models(tag, variants)
    cap = load_capture(tag)
    ids = [str(i) for i in cap["ids"]]
    pix = cap["pixel_values"].astype(np.float32)                       # [16,630,768]（HF のプロセッサで作ったもの）
    IO.mkdir(parents=True, exist_ok=True)
    pairs = []
    for j, i in enumerate(ids):
        p = IO / f"pix_{tag}_{i}.bin"
        pix[j][None].tofile(p)
        pairs.append((p, f"inputs/vit/pix_{tag}_{i}.bin"))
    push_many(pairs, "入力（評価用 16 枚の pixel_values）")
    # M = 画像トークン数（格子を 3×3 ずつまとめる。21×30 → 7×10 = 70）。端末の出力は 1 枚 [M, 1536]
    M = (grid[0] // C.POOL) * (grid[1] // C.POOL)
    out = {"tag": tag, "n": len(ids), "variants": {}}
    for v in variants:
        print(f"\n== 版: {v}", flush=True)
        res, dev = infer_images(tag, v, rels[v], [r for _, r in pairs], ids, (M, C.TEXT_HIDDEN), f"vit_acc_{tag}_{v}")
        (RES / f"acc_{tag}_{v}_result.json").parent.mkdir(parents=True, exist_ok=True)
        (RES / f"acc_{tag}_{v}_result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        ms = [r["ort_run_ms"] for r in res["records"]]
        print(f"  端末: strict={res['describe'].get('strict_no_cpu_fallback')}  {res['describe'].get('backend')}  "
              f"1 枚 {np.median(ms):.1f}ms（中央値）", flush=True)
        print("  PC の QDQ を計算中…", flush=True)
        q = pc_qdq(tag, v, pix)
        hf_same = cap["feat_srq"] if v == "srq" else cap["feat_nosrq"]
        rows = {"dev_vs_qdq": [], "dev_vs_hf": [], "qdq_vs_hf": [], "dev_vs_hf_nosrq": []}
        # ④ 1 枚ずつ 4 通りの組で比べる（hf_same は同じ版の HF の特徴: srq なら SRQ あり、nosrq なら SRQ なし）
        for j in range(len(ids)):
            rows["dev_vs_qdq"].append(compare(dev[j], q[j]))
            rows["dev_vs_hf"].append(compare(dev[j], hf_same[j]))
            rows["qdq_vs_hf"].append(compare(q[j], hf_same[j]))
            rows["dev_vs_hf_nosrq"].append(compare(dev[j], cap["feat_nosrq"][j]))
        np.savez(IO / f"acc_{tag}_{v}.npz", ids=np.array(ids), dev=dev, qdq=q)
        summ = {k: summarize(r) for k, r in rows.items()}
        out["variants"][v] = {"describe": describe_line(res), "ort_run_ms": ms, "ort_run_ms_median": float(np.median(ms)),
                              "summary": summ, "per_image": {k: [dict(id=i, **x) for i, x in zip(ids, r)] for k, r in rows.items()}}
        for k, s in summ.items():
            print(f"  {k:16s} 相対L2 中央値 {s['rel_l2_median']:.4f}・最大 {s['rel_l2_max']:.4f}  "
                  f"cos 最小 {s['cos_min']:.4f}・中央値 {s['cos_median']:.4f}", flush=True)
    # ⑤ Q10 は SRQ なしの版だけで判定（端末 ↔ PC の QDQ の相対 L2 中央値 ≤ 0.05 かつ 端末 ↔ HF の cos 最小 ≥ 0.99）
    if "nosrq" in out["variants"]:
        s = out["variants"]["nosrq"]["summary"]
        q10 = s["dev_vs_qdq"]["rel_l2_median"] <= Q10_REL_GUIDE and s["dev_vs_hf"]["cos_min"] >= 0.99
        out["q10_nosrq"] = {"pass": bool(q10), "dev_vs_qdq_rel_l2_median": s["dev_vs_qdq"]["rel_l2_median"],
                            "dev_vs_hf_cos_min": s["dev_vs_hf"]["cos_min"]}
        print(f"\n[Q10] SRQ なしの版: 端末 ↔ PC の QDQ 相対L2 中央値 {s['dev_vs_qdq']['rel_l2_median']:.4f}"
              f"（目安 {Q10_REL_GUIDE}、Gemma4 の文章側 0.018）・端末 ↔ HF cos 最小 {s['dev_vs_hf']['cos_min']:.4f}（目安 0.99）"
              f" → {'合格' if q10 else '★目安に届かない'}")
    strict = {v: out["variants"][v]["describe"]["strict_no_cpu_fallback"] for v in out["variants"]}
    out["q9_strict"] = strict
    print(f"[Q9] CPU に逃げずに開けたか: {strict}")
    return out


def cmd_bench(a) -> dict:
    """bench（Q11 の時間）: 乱数の入力で、ウォームアップ 3 回のあと --iters 回動かし、1 枚の時間（ms）の中央値・p90 などを出す。
    記録: results/P2_20260929_vit_device.md 3 節（SRQ なし 中央値 193.0ms・SRQ あり 311.4ms。途中から 2 割遅くなる）
    """
    tag, _ = tag_for(a.budget, a.portrait)
    ensure_app()
    variants = a.variants.split(",")
    rels = push_models(tag, variants)
    out = {"tag": tag, "variants": {}}
    for v in variants:
        job = {"job_id": f"vit_bench_{tag}_{v}", "task": "bench", "model": rels[v], "ep": EP,
               "input": {"kind": "random", "seed": 0}, "protocol": {"warmup": 3, "iterations": a.iters}}
        res = run_job(job, timeout_s=900)
        if res.get("_kind") != "result" or not res.get("ok"):
            raise SystemExit(f"bench が失敗: {json.dumps(res, ensure_ascii=False)[:800]}")
        st = res["ort_run_ms"]
        out["variants"][v] = {"describe": describe_line(res), "warmup_ms": res.get("warmup_ms"),
                              "samples_ms": res.get("samples_ms"), "stats_ms": st}
        print(f"  {v}: 中央値 {st['median']:.1f}ms・p90 {st['p90']:.1f}・最小 {st['min']:.1f}・最大 {st['max']:.1f}"
              f"（{st['n']} 回）  ウォームアップ {[round(x) for x in res.get('warmup_ms', [])]}  "
              f"strict={res['describe'].get('strict_no_cpu_fallback')}", flush=True)
    return out


def memprobe(model_rel: str, job_id: str, hold_ms: int = 0) -> dict:
    """端末で memprobe の job を動かし、モデルを開く前・開いた後・動かした後・落ち着いた後のメモリを MiB で返す。

    引数: model_rel = 端末でのモデルの相対パス、hold_ms = 開いたまま待つ時間（coexist で文章側と同時に開いた状態を保つ）。
    戻り値: ok・open_ms・run_ms・dmabuf_mib（NPU のメモリ）・rss_mib・hwm_open_mib（ロード中の RSS の山）など。
    dma-buf は NPU に渡すメモリ。RSS には入らないので別に数える（RSS だけ見ると NPU の分を見落とす）。
    """
    job = {"job_id": job_id, "task": "memprobe", "models": [model_rel], "hold": hold_ms > 0, "run_once": True,
           "settle_ms": 1000, "sample_ms": 200, "hold_ms": hold_ms, "ep": EP, "output": {"dir": "out"}}
    res = run_job(job, timeout_s=300 + hold_ms // 1000)
    rec = (res.get("records") or [{}])[0]

    def mib(snap, key="dmabuf"):
        """memprobe の記録（snapshot）から MiB を取る。dma-buf はバイト、RSS などは KB で入っている。"""
        snap = snap or {}
        if key == "dmabuf":
            return snap.get("dmabuf", {}).get("bytes", -1) / 2**20
        return snap.get(key, -1) / 1024
    b, o, r, st = rec.get("before"), rec.get("after_open"), rec.get("after_run"), rec.get("settled")
    return {"ok": bool(res.get("ok")) and bool(rec.get("open_ok")), "open_ms": rec.get("open_ms"), "run_ms": rec.get("run_ms"),
            "backend": rec.get("backend"), "error": rec.get("error") or rec.get("run_error"),
            "dmabuf_mib": {"before": mib(b), "after_open": mib(o), "after_run": mib(r), "settled": mib(st)},
            "rss_mib": {"before": mib(b, "vm_rss_kb"), "after_open": mib(o, "vm_rss_kb"), "after_run": mib(r, "vm_rss_kb"),
                        "settled": mib(st, "vm_rss_kb")},
            "hwm_open_mib": mib(o, "vm_hwm_kb"), "mem_available_mib_before": mib(b, "mem_available_kb"),
            "raw": res}


def cmd_mem(a) -> dict:
    """mem（Q11 のメモリ）: 版ごとにアプリを止めてから memprobe を動かし、NPU のメモリ・RSS・開く時間を出す。
    記録: results/P2_20260929_vit_device.md 4 節（dma-buf 152.3MiB・RSS 約 110MiB・開く 551〜733ms）
    """
    tag, _ = tag_for(a.budget, a.portrait)
    ensure_app()
    variants = a.variants.split(",")
    rels = push_models(tag, variants)
    out = {"tag": tag, "variants": {}}
    for v in variants:
        adb("shell", f"am force-stop {PKG}")                  # 前の job の残りを持ち越さない
        time.sleep(1.0)
        m = memprobe(rels[v], f"vit_mem_{tag}_{v}")
        out["variants"][v] = m
        d = m["dmabuf_mib"]
        print(f"  {v}: 開く {m['open_ms']:.0f}ms・1 回目 {m['run_ms']:.0f}ms  dma-buf {d['before']:.0f} → {d['after_open']:.0f}"
              f" → 動かした後 {d['after_run']:.0f}MiB（増えた分 {d['after_run'] - d['before']:.0f}MiB）  "
              f"RSS {m['rss_mib']['after_run']:.0f}MiB（ロード中の山 {m['hwm_open_mib']:.0f}）  {m['backend']}", flush=True)
    return out


def cmd_coexist(a) -> dict:
    """文章側（Gemma4 の会話アプリ、別のアプリ＝別プロセス）がモデルを開いている間に、画像エンコーダを開いて動かす。

    Gemma4 のアプリは起動するだけで何も変えない（画面を開くと既定のモデルを読み込む）。最後に両方とも止める。
    """
    # 段: ① 両方のアプリを止める → ② 会話アプリを起動し、logcat の「chat ready」まで待つ
    #     → ③ 画像エンコーダを別のアプリで開いて hold 秒保つ → ④ 会話アプリのプロセス番号が変わらず、OS の kill も無いかで判定
    #   記録: results/P2_20260929_vit_device.md 9.2 節（同居できた。文章側 1,467MiB ＋ 画像エンコーダ 152MiB）
    tag, _ = tag_for(a.budget, a.portrait)
    ensure_app()
    v = a.variants.split(",")[0]
    rels = push_models(tag, [v])
    if G4_PKG not in adb("shell", f"pm path {G4_PKG} || true", check=False):
        raise SystemExit(f"{G4_PKG}（Gemma4 の会話アプリ）が端末に無い")
    adb("shell", f"am force-stop {PKG}")
    adb("shell", f"am force-stop {G4_PKG}")
    adb("logcat", "-c", check=False)
    wake()
    print(f"  文章側: {G4_CHAT} を開き、モデルの読み込みが終わるのを待つ（最大 {a.wait} 秒）", flush=True)
    adb("shell", f"am start -n {G4_CHAT}")
    # 会話アプリは読み込みが終わると logcat（タグ EdgeSLMChat）に "chat ready: … strict=…"、失敗すると "chat open failed" を出す。
    # 決め打ちで待つと「読み込み前に測ってしまう」ので、この行が出るまで待つ（2026-09-29 の初回は待つだけで確かめていなかった）
    t0, chat_line = time.time(), ""
    while time.time() - t0 < a.wait:
        time.sleep(2)
        lines = [ln for ln in adb("logcat", "-d", "-s", "EdgeSLMChat:*", check=False).splitlines()
                 if "chat ready" in ln or "chat open failed" in ln]
        if lines:
            chat_line = lines[-1]
            break
    print(f"  文章側の読み込み: {chat_line.strip() or '★確認できなかった（時間切れ）'}（{time.time() - t0:.0f}s）", flush=True)
    time.sleep(3)                                              # 読み込み直後の一時的なメモリを落ち着かせる
    pid_g4 = adb("shell", f"pidof {G4_PKG} || true", check=False).strip()
    mem_g4 = adb("shell", f"dumpsys meminfo {G4_PKG} | grep -E 'TOTAL|Graphics|Unknown' | head -5", check=False)
    print(f"  文章側のプロセス: {pid_g4 or '（無し）'}", flush=True)
    print(f"  画像エンコーダ（{v}）を別のアプリで開いて動かす…", flush=True)
    m = memprobe(rels[v], f"vit_coexist_{tag}_{v}", hold_ms=a.hold * 1000)
    # ④ 判定: 文章側の読み込み完了・画像エンコーダが開けた・文章側のプロセス番号が前後で同じ・lowmemorykiller の記録なし
    pid_g4_after = adb("shell", f"pidof {G4_PKG} || true", check=False).strip()
    lmk = [ln for ln in adb("logcat", "-d", check=False).splitlines()
           if "lowmemorykiller" in ln.lower() and (G4_PKG in ln or PKG in ln)]
    chat_ready = "chat ready" in chat_line
    ok = chat_ready and m["ok"] and bool(pid_g4) and pid_g4 == pid_g4_after and not lmk
    out = {"tag": tag, "variant": v, "wait_s": a.wait, "chat_line": chat_line.strip(), "chat_ready": chat_ready,
           "pid_g4_before": pid_g4, "pid_g4_after": pid_g4_after,
           "meminfo_g4": mem_g4, "vit": m, "lmk": lmk[-5:], "pass": bool(ok)}
    d = m["dmabuf_mib"]
    print(f"  画像エンコーダ: 開けた={m['ok']}・開く {m['open_ms']}ms・1 回目 {m['run_ms']}ms・dma-buf 増 "
          f"{d['after_run'] - d['before']:.0f}MiB・{m['backend']}", flush=True)
    print(f"  文章側のプロセス: 前 {pid_g4 or '無し'} → 後 {pid_g4_after or '無し'}"
          f"{'（生きている）' if pid_g4 and pid_g4 == pid_g4_after else '（★落ちた／入れ替わった）'}", flush=True)
    if lmk:
        print(f"  ★OS に kill された記録: {lmk[-1][:200]}", flush=True)
    print(f"\n判定: {'**同時に開けた**' if ok else '★同時には開けなかった（上を見ること）'}")
    adb("shell", f"am force-stop {PKG}")
    adb("shell", f"am force-stop {G4_PKG}")
    return out


def cmd_feats(a) -> dict:
    """2-5: manifest の全画像を端末の NPU で画像エンコーダに通し、ref_vlm に渡せる npz を作る。"""
    from PIL import Image
    from app.pipeline import hf_vlm as H
    from app.pipeline.eval_qdq_vit import check_pixels
    tag, grid = tag_for(a.budget, a.portrait)
    ensure_app()
    v = a.variant
    rels = push_models(tag, [v])
    man = json.loads((C.EVAL_DIR / "manifest.json").read_text(encoding="utf-8"))
    check_pixels(tag, a.budget, grid, man)                     # numpy の前処理が HF と同じか（違えば止まる）
    # 段: ① 評価セットの画像を重複なく並べる（1 枚に複数の問いがあるので。全 108 枚）
    #     → ② PC で端末と同じ前処理（レターボックス → パッチへの並べ替え）をして .bin に → ③ 端末で infer → ④ npz に保存
    images = list(dict.fromkeys(it["image"] for it in man["items"]))
    IO.mkdir(parents=True, exist_ok=True)
    pairs, ids = [], []
    for k, rp in enumerate(images):
        img_in = H.prepare_image(Image.open(C.EVAL_DIR / rp), "fixed", a.budget, grid=grid)
        p = IO / f"all_{tag}_{k:03d}.bin"
        H.pixels_fixed_np(img_in, grid)[None].tofile(p)
        pairs.append((p, f"inputs/vit_all/all_{tag}_{k:03d}.bin"))
        ids.append(f"{k:03d}")
    push_many(pairs, f"入力（全 {len(images)} 枚の pixel_values）")
    M = (grid[0] // C.POOL) * (grid[1] // C.POOL)
    res, dev = infer_images(tag, v, rels[v], [r for _, r in pairs], ids, (M, C.TEXT_HIDDEN), f"vit_feats_{tag}_{v}")
    feats_dir = C.WORK / "feats"
    feats_dir.mkdir(parents=True, exist_ok=True)
    path = feats_dir / f"vit_{tag}_{v}_dev.npz"
    np.savez(path, images=np.array(images), feats=dev, variant=f"{v}_dev", tag=tag)
    ms = [r["ort_run_ms"] for r in res["records"]]
    print(f"-> {path}（{len(images)} 枚、端末 1 枚 {np.median(ms):.1f}ms）")
    # PC の QDQ の特徴（eval_qdq_vit feats の結果）があれば、端末との差も出しておく
    pc_path = feats_dir / f"vit_{tag}_{v}.npz"
    vs_pc = None
    if pc_path.exists():
        with np.load(pc_path) as z:
            pcf = {str(p): z["feats"][k] for k, p in enumerate(z["images"])}
        rows = [compare(dev[k], pcf[rp]) for k, rp in enumerate(images) if rp in pcf]
        vs_pc = summarize(rows)
        print(f"  端末 ↔ PC の QDQ（{len(rows)} 枚）: 相対L2 中央値 {vs_pc['rel_l2_median']:.4f}・cos 最小 {vs_pc['cos_min']:.4f}")
    print(f"  次: & $PY1 -m app.pipeline.ref_vlm run --prep fixed --budgets {a.budget} --vision npz:work/feats/{path.name}")
    return {"tag": tag, "variant": v, "path": str(path), "n_images": len(images), "describe": describe_line(res),
            "ort_run_ms_median": float(np.median(ms)), "vs_pc_qdq": vs_pc}


def main():
    """コマンド（check・acc・bench・mem・coexist・feats）を選んで動かし、結果を results/P2_vit_device/<tag>_<コマンド>.json に書く。
    feats は版の名前も付ける（例 b70_21x30_feats_nosrq.json）。書き込みは write_json_atomic（途中で落ちても壊れた JSON を残さない）。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "acc", "bench", "mem", "coexist", "feats"])
    ap.add_argument("--budget", type=int, default=70, choices=sorted(C.FIXED_GRIDS))
    ap.add_argument("--portrait", action="store_true")
    ap.add_argument("--variants", default="nosrq,srq", help="acc・bench・mem で使う版（coexist は先頭の 1 つ）")
    ap.add_argument("--variant", default="srq", choices=["srq", "nosrq"], help="feats で使う版")
    ap.add_argument("--iters", type=int, default=20, help="bench の回数")
    ap.add_argument("--wait", type=int, default=120, help="coexist: 文章側のモデルの読み込みを待つ最大の秒数")
    ap.add_argument("--hold", type=int, default=10, help="coexist: 画像エンコーダを開いたまま待つ秒数")
    a = ap.parse_args()
    C.utf8_stdout()
    fn = {"check": cmd_check, "acc": cmd_acc, "bench": cmd_bench, "mem": cmd_mem, "coexist": cmd_coexist,
          "feats": cmd_feats}[a.cmd]
    out = fn(a)
    RES.mkdir(parents=True, exist_ok=True)
    tag, _ = tag_for(a.budget, a.portrait)
    name = f"{tag}_{a.cmd}" + (f"_{a.variant}" if a.cmd == "feats" else "")
    C.write_json_atomic(RES / f"{name}.json", out)
    print(f"-> {RES / (name + '.json')}")


if __name__ == "__main__":
    main()
