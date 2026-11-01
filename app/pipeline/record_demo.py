# -*- coding: utf-8 -*-
"""VLM デモの画面を録画して PC に保存する（2026-09-29、Phase 5 の前後で比べるため）。

    python -m app.pipeline.record_demo                 # 録画開始 → デモの画面を前に出す → スマホで操作 → PC で Enter を押すと止まる
    python -m app.pipeline.record_demo --restart       # アプリを一度止めて開き直す（モデルを開くところから映す）
    python -m app.pipeline.record_demo --seconds 120   # 最長の秒数（端末の screenrecord の上限は 180 秒）
    python -m app.pipeline.record_demo --tag before    # ファイル名に付ける目印（例: before / after）

- 既定: デモの画面が前にあればそのまま（続けて撮るとき、モデルを開き直さない）。前に無ければ開く
- 開いたあと、本当にデモの画面が前に出たかを確かめて表示する（ロック画面のままだと前に出ない）

どの venv でもよい（$PY2 を想定）。端末は USB でつないだまま（機内モードでも adb は USB で通じる）。

## 何をするか

1. 端末の状態を見せる: 機内モードか（`settings get global airplane_mode_on`）、空きメモリ
2. 端末の録画（`adb shell screenrecord`）を始める。**音は入らない**（画面だけ）
3. VLM デモの画面を前に出す（`--restart` ならアプリを一度止めてから開くので、モデルを開くところから映る）。
   前に出たか（`dumpsys activity activities` の topResumedActivity）を確かめる
4. PC で Enter を押すか、最長の秒数になったら録画を止める。
   止めるときは端末の screenrecord に SIGINT を送る（`pkill -INT screenrecord`）。PowerShell の Ctrl+C で adb ごと切ると、
   動画のファイルが最後まで書かれず再生できないことがあるため
5. 動画を PC の `results/P3_device_vlm/demo/` に取ってくる。端末にも `/sdcard/Movies/VLMDemo/` に残す（ギャラリーで見られる）
6. 同じ時間の画面のログ（logcat の EdgeSLMVlm）も保存し、写真の読み取り・最初の文字まで・速さの行を表示する（動画の字幕や説明に使える）

## 録画のこつ

- 機内モードにしてから始める（「端末の中だけで動いている」ことが映る）
- ほかのアプリは閉じておく（最近のアプリの一覧から全部消す）。カメラの間にデモが OS に止められにくくなる
- 画面の自動回転は切る（デモの画面は縦固定。カメラアプリだけ横になることがある）
"""
# ---- 2026-10-18 追記: 入力と出力 ----
# 入力: USB でつないだ端末（デモのアプリが入っていること）。PC のキーボードの Enter で録画を止める。
# 出力: results/P3_device_vlm/demo/vlm_demo_<日時>[_<tag>].mp4 と、同じ名前の _log.txt（logcat の EdgeSLMVlm）。
#       端末にも /sdcard/Movies/VLMDemo/ に動画が残る。
# adb の道具（adb・wake・ensure_app・mem_available_mib）は device_vit から借りる。
from __future__ import annotations

import argparse
import subprocess
import threading
import time
from datetime import datetime

from app.pipeline import device_vit as D
from app.pipeline import vlm_common as C

VLM_ACT = f"{D.PKG}/org.edgeslm.demo.VlmActivity"
DEV_DIR = "/sdcard/Movies/VLMDemo"
OUT = C.RESULTS / "P3_device_vlm" / "demo"


def resumed_activity() -> str:
    """いま前にある画面（topResumedActivity / mResumedActivity の行）。読めなければ空"""
    out = D.adb("shell", "dumpsys activity activities | grep -E 'topResumedActivity|mResumedActivity'", check=False)
    return out.strip().splitlines()[0].strip() if out.strip() else ""


def vlm_in_front() -> bool:
    """VLM デモの画面（VlmActivity）がいま前に出ているか。"""
    return "VlmActivity" in resumed_activity()


def main():
    """録画の本体。冒頭の「何をするか」の 1〜6 をこの順に行う（コメントの「# 1.」〜「# 6.」）。
    --seconds は 1〜180 に丸める（端末の screenrecord の上限が 180 秒）。
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=180, help="最長の秒数（1〜180）")
    ap.add_argument("--bitrate", type=float, default=8.0, help="動画のビットレート（Mbps）。8 で 1 分 約 60MB")
    ap.add_argument("--restart", action="store_true", help="アプリを一度止めて開き直す（モデルを開くところから映す）")
    ap.add_argument("--no-launch", action="store_true", help="デモの画面を自動で開かない（自分で開くとき）")
    ap.add_argument("--tag", default="", help="ファイル名に付ける目印（例: before / after）")
    a = ap.parse_args()
    C.utf8_stdout()
    sec = max(1, min(180, a.seconds))
    D.ensure_app()

    # 1. 端末の状態
    air = D.adb("shell", "settings get global airplane_mode_on", check=False).strip()
    mem = D.mem_available_mib()
    print(f"  機内モード: {'オン' if air == '1' else 'オフ（「端末の中だけ」を見せるなら、オンにしてから始める）'}"
          f"、空きメモリ {mem}MiB", flush=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"vlm_demo_{stamp}" + (f"_{a.tag}" if a.tag else "")
    dev_path = f"{DEV_DIR}/{name}.mp4"
    D.adb("shell", f"mkdir -p {DEV_DIR}", check=False)
    D.adb("logcat", "-c", check=False)
    D.wake()

    # 2. 録画を始める（adb の子プロセスとして。終わるまで待たない）
    rec = subprocess.Popen([D.ADB, "shell", "screenrecord", "--time-limit", str(sec),
                            "--bit-rate", str(int(a.bitrate * 1_000_000)), dev_path],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    time.sleep(1.0)
    if rec.poll() is not None:
        raise SystemExit(f"録画を始められなかった: {rec.stdout.read().decode('utf-8', 'replace')[:300]}")

    # 3. デモの画面を前に出す（2026-09-29: 2 本目を --no-launch で撮ったとき、デモの画面が前に無く何も起きなかった
    #    → 既定は「前に無ければ開く」にし、前に出たかを確かめて表示する）
    if not a.no_launch:
        if a.restart:
            D.adb("shell", f"am force-stop {D.PKG}", check=False)
            time.sleep(0.8)
        if a.restart or not vlm_in_front():
            out = D.adb("shell", f"am start -n {VLM_ACT}", check=False)
            if "Error" in out:
                print(f"  ★デモの画面を開けなかった: {out.strip()[:200]}", flush=True)
    front = False
    for _ in range(16):                                    # 最大 8 秒、前に出るのを待つ
        if vlm_in_front():
            front = True
            break
        time.sleep(0.5)
    if front:
        print("  デモの画面が前に出た", flush=True)
    else:
        print(f"  ★デモの画面が前に出ていない（いま前: {resumed_activity()[:120] or '不明'}）。\n"
              f"    スマホの画面を点けてロックを外し、ランチャーから「VLM デモ」を開いてください（録画は続いている）", flush=True)
    t0 = time.time()
    print(f"\n  ● 録画中（最長 {sec} 秒）。スマホでデモを操作してください。終わったら PC でこの画面を選んで Enter", flush=True)

    # 4. Enter か、最長の秒数で止める
    # 入力待ち（input）は止まってしまうので別のスレッドで待ち、Enter が来たら pressed を立てる。本体は 0.3 秒ごとに見る
    pressed = threading.Event()
    threading.Thread(target=lambda: (input(), pressed.set()), daemon=True).start()
    last = -1
    while rec.poll() is None and not pressed.is_set():
        el = int(time.time() - t0)
        if el // 15 != last:                               # 15 秒ごとに経過を出す
            last = el // 15
            print(f"    …{el} 秒", flush=True)
        time.sleep(0.3)
    if rec.poll() is None:
        # SIGINT で止めると、screenrecord が動画のファイルを最後まで書いてから終わる
        D.adb("shell", "pkill -INT screenrecord 2>/dev/null || kill -INT $(pidof screenrecord)", check=False)
        try:
            rec.wait(timeout=15)
        except subprocess.TimeoutExpired:
            rec.kill()
    dur = time.time() - t0
    time.sleep(1.0)                                        # 端末でファイルが閉じられるのを待つ

    # 5. 動画を取ってくる
    OUT.mkdir(parents=True, exist_ok=True)
    local = OUT / f"{name}.mp4"
    D.adb("pull", dev_path, str(local))
    size = local.stat().st_size if local.exists() else 0
    # ギャラリーに出るよう、端末のメディアの一覧に知らせる（失敗しても動画は残る）
    D.adb("shell", f"am broadcast -a android.intent.action.MEDIA_SCANNER_SCAN_FILE -d file://{dev_path}", check=False)

    # 6. 画面のログ
    log = D.adb("logcat", "-d", "-s", "EdgeSLMVlm:*", check=False)
    (OUT / f"{name}_log.txt").write_text(log, encoding="utf-8")
    keys = ("vlm text ready", "vlm image:", "vlm stats:", "vlm answer:", "vlm read:")
    lines = [ln.split("EdgeSLMVlm:", 1)[-1].strip() for ln in log.splitlines() if any(k in ln for k in keys)]

    print(f"\n  ■ 録画を止めた（{dur:.0f} 秒）")
    print(f"  -> {local}（{size / 2**20:.1f}MB）。端末にも {dev_path}")
    if size == 0:
        print("  ★動画を取ってこられなかった。端末の Movies/VLMDemo を見ること")
    if lines:
        print("  画面のログ（写真の読み取り・最初の文字まで・速さ）:")
        for ln in lines:
            print(f"    {ln[:200]}")


if __name__ == "__main__":
    main()
