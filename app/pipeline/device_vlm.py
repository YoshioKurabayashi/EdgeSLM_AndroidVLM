# -*- coding: utf-8 -*-
"""Phase 3: 端末でつないだ VLM の照合（2026-09-29）。

    python -m app.pipeline.device_vlm prep            # Q14: 端末の前処理（C++）↔ PC（PIL）。評価の全 108 枚（どの venv でもよい）
    python -m app.pipeline.device_vlm eval            # Q15: 端末だけで 126 問（**Phase 1 の venv（$PY1）**で動かす）
    python -m app.pipeline.device_vlm uicheck         # Q13・Q16: 画面（VlmActivity）が作るプロンプトの ID・前処理・答え・時間（$PY1）
    python -m app.pipeline.device_vlm holdcheck       # 画面を前に出したまま 1.5GB を開くと OS に止められないか（5 回。どの venv でもよい）
    python -m app.pipeline.device_vlm holdcheck --no-hold   # 比べる: 従来の入口（画面なし）

端末には Phase 3 のアプリ（vlm_prep・画像の特徴を入れる generate_kv・VlmActivity）を入れ、setup_vlm_demo を済ませておくこと。

## prep（Q14）

PC で写真を読み（EXIF の向き・透明は白・RGB。hf_vlm.prepare_image と同じ）、その RGB のバイト列を端末に送って
端末の C++（`vlm_prep`）で pixel_values を作らせ、PC の `hf_vlm.pixels_fixed_np(prepare_image(…))` と比べる。
**同じ RGB から始めるので、ビット単位で一致するはず**（C++ で PIL の BICUBIC を固定小数点まで再現した）。
JPEG の復号（PIL と Android の違い）はここでは比べない（uicheck で見る）。

## eval（Q15）

- プロンプトの ID は PC の HF（`hf_vlm.build_inputs`）で作る＝ Phase 1〜2 の評価と同じ
- 画像の特徴は、2-5 で端末の NPU で作ったもの（`work/feats/vit_b70_21x30_nosrq_dev.npz`）
- 端末の文章側（generate_kv、1 トークンずつ）に、画像ごとに特徴のファイルを渡して解かせる（job の `dialogs` に `image_features`）
- 答えを PC で文字に戻して判定し、`results/P1_ref_vlm/` に ref_vlm と同じ形で書く（`ref_vlm report` の表に並ぶ）
- 28 問ずつの job に分ける。job ごとに結果をファイルに書き足すので、途中で止まっても**同じコマンドをもう一度打てば残りから続く**
  （最初からやり直すときは `--fresh`）
- 端末のアプリが落ちたら（OS のメモリ不足で止められた等）・結果が返らなければ、30 秒待って同じ job を 2 回までやり直す
  （`--retries`・`--cooldown`）
- job と job の間は 10 秒あけ（`--settle`）、さらに端末の空きメモリ（MemAvailable）が 3GiB 以上になるまで最大 90 秒待つ
- ★2026-09-29 の記録: 文章側のモデル（1.5GB）を開く途中（開始 10 秒前後）に、OS の lowmemorykiller に 2 回止められた
  （理由 "device is not responding"＝メモリが足りず端末全体が詰まった。止められたときの優先度は adj 50＝画面の無い Service）。
  開く回数を減らし（14 → 28 問ずつ）、開く前に間をあける。さらに **画面を前に出したまま動かす**（RunnerHoldActivity。
  画面が前なら止められないことが EdgeSLM_AndroidScale P3 で 25/25。古いアプリでは `--no-hold`）

## uicheck（Q13・Q16）

評価の画像 1 枚を端末に置き、画面を「その写真・その質問」で起動する（`--es image … --es question_file …`）。
logcat から、画面が作ったプロンプトの ID（HF と同じか）、前処理の結果（PC の PIL と比べる。Android の JPEG の復号の差が出る）、
答えと時間を取る。
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from pathlib import Path

import numpy as np

from app.pipeline import device_vit as D
from app.pipeline import vlm_common as C

IO = C.WORK / "device_vlm"
RES_REF = C.RESULTS / "P1_ref_vlm"
RES = C.RESULTS / "P3_device_vlm"
FEATS_DEV = C.WORK / "feats" / "vit_b70_21x30_nosrq_dev.npz"
VLM_ACT = f"{D.PKG}/org.edgeslm.demo.VlmActivity"


def load_manifest() -> dict:
    return json.loads((C.EVAL_DIR / "manifest.json").read_text(encoding="utf-8"))


def to_rgb(img):
    """hf_vlm.prepare_image と同じ扱い（EXIF の向き・透明は白）で RGB にする。"""
    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[-1])
        img = bg
    return img.convert("RGB")


# ---------------------------------------------------------------------------
# prep（Q14）
# ---------------------------------------------------------------------------
def cmd_prep(a) -> dict:
    from PIL import Image
    from app.pipeline import hf_vlm as H
    D.ensure_app()
    grid = C.FIXED_GRIDS[a.budget]
    man = load_manifest()
    images = list(dict.fromkeys(it["image"] for it in man["items"]))
    if a.n:
        images = images[:a.n]
    d = IO / "prep"
    d.mkdir(parents=True, exist_ok=True)
    pairs, items, refs = [], [], []
    for k, rp in enumerate(images):
        rgb = to_rgb(Image.open(C.EVAL_DIR / rp))
        arr = np.asarray(rgb, dtype=np.uint8)
        p = d / f"img_{k:03d}.rgb"
        arr.tofile(p)
        pairs.append((p, f"inputs/vlmprep/img_{k:03d}.rgb"))
        items.append({"rgb": f"inputs/vlmprep/img_{k:03d}.rgb", "width": rgb.width, "height": rgb.height,
                      "out": f"out/vlmprep/img_{k:03d}_pix.bin"})
        refs.append(H.pixels_fixed_np(H.prepare_image(rgb, "fixed", a.budget, grid=grid), grid))
    D.push_many(pairs, f"写真の RGB（{len(images)} 枚）")
    D.adb("shell", f"rm -rf {D.DEV}/out/vlmprep; mkdir -p {D.DEV}/out/vlmprep")
    job = {"job_id": "vlm_prep_check", "task": "vlm_prep", "items": items, "grid": list(grid), "patch": C.PATCH}
    res = D.run_job(job, timeout_s=900)
    if res.get("_kind") != "result":
        raise SystemExit(f"端末で前処理が動かなかった: {json.dumps(res, ensure_ascii=False)[:600]}")
    local = IO / "vlmprep"
    shutil.rmtree(local, ignore_errors=True)
    D.adb("pull", f"{D.DEV}/out/vlmprep", str(IO))
    rows, n_exact = [], 0
    for k, (rp, rec) in enumerate(zip(images, res.get("records", []))):
        if "error" in rec:
            rows.append({"image": rp, "error": rec["error"]})
            continue
        dev = np.fromfile(local / f"img_{k:03d}_pix.bin", np.float32).reshape(refs[k].shape)
        diff = np.abs(dev - refs[k])
        exact = bool(diff.max() == 0)
        n_exact += exact
        rows.append({"image": rp, "exact": exact, "max_abs": float(diff.max()), "n_diff": int((diff > 0).sum()),
                     "resize_ms": rec.get("info", {}).get("resize_ms"), "in": rec.get("info", {}).get("in")})
    ms = [r["resize_ms"] for r in rows if r.get("resize_ms") is not None]
    print(f"  端末の前処理 ↔ PC（PIL）: ビット一致 {n_exact}/{len(images)} 枚"
          f"（縮める時間の中央値 {np.median(ms):.1f}ms、最大 {max(ms):.1f}ms）", flush=True)
    for r in rows:
        if not r.get("exact"):
            print(f"    ★{r}")
    return {"n": len(images), "n_exact": n_exact, "pass": n_exact == len(images), "rows": rows}


# ---------------------------------------------------------------------------
# eval（Q15）: 端末だけで 126 問
# ---------------------------------------------------------------------------
def hf_prompt(proc, H, img, question: str, budget: int, system: str):
    img_in = H.prepare_image(img, "fixed", budget)
    inputs, n_img, _ = H.build_inputs(proc, img_in, question, budget, system, "cpu", expect_exact=True)
    return [int(x) for x in inputs["input_ids"][0].tolist()], n_img


def cmd_eval(a) -> dict:
    C.ensure_hf_env()
    from PIL import Image
    from transformers import AutoProcessor
    from app.pipeline import hf_vlm as H
    from app.pipeline.judge import judge
    D.ensure_app()
    man = load_manifest()
    items = man["items"][:a.n] if a.n else man["items"]
    proc = AutoProcessor.from_pretrained(str(C.HF_DIR), local_files_only=True)
    with np.load(FEATS_DEV) as z:
        F = z["feats"]
        feats = {str(p): F[k] for k, p in enumerate(z["images"])}
    text = D.text_job(a.text)                               # t1: 1 トークン版、t16: Phase 5 の 16 トークン版（端末に送る）
    system = a.system or man["system_prompt"]
    # 画像ごとの特徴のファイル（[70,1536] float32）を送る
    d = IO / "feats"
    d.mkdir(parents=True, exist_ok=True)
    img_names = sorted({it["image"] for it in items})
    pairs, rel_of = [], {}
    for k, rp in enumerate(img_names):
        p = d / f"f_{k:03d}.bin"
        feats[rp].astype(np.float32).tofile(p)
        rel_of[rp] = f"inputs/vlmfeat/f_{k:03d}.bin"
        pairs.append((p, rel_of[rp]))
    D.push_many(pairs, f"画像の特徴（端末の NPU で作ったもの、{len(img_names)} 枚）")
    # プロンプトの ID（HF と同じ）
    dialogs, meta = [], {}
    for it in items:
        img = Image.open(C.EVAL_DIR / it["image"])
        ids, n_img = hf_prompt(proc, H, img, it["question"], a.budget, system)
        dialogs.append({"id": it["id"], "turns": [ids], "image_features": rel_of[it["image"]],
                        "max_new": it.get("max_new", 64)})
        meta[it["id"]] = {"it": it, "prompt_len": len(ids), "n_img": n_img, "image_size": [img.width, img.height]}
    # 条件ごとに別のファイル（ref_vlm report の表で見出しが分かれるように feed の名前を変える）
    feed = "device" + ("" if a.text == "t1" else f"_{a.text}") + (f"_{a.system_tag}" if a.system else "")
    out_path = RES_REF / f"fixed_{a.budget}_{feed}_vnpz-vit_b70_21x30_nosrq_dev.json"
    # ★続きから: 前の実行の結果があれば、解き終えた問題は飛ばす（--fresh で最初から）。
    #   途中で止まっても（OS に止められた等）、もう一度同じコマンドを打てば残りだけ解く
    results = []
    if out_path.exists() and not a.fresh:
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        if prev.get("feed") == feed and prev.get("vision") == f"npz:work/feats/{FEATS_DEV.name}":
            want = {dg["id"] for dg in dialogs}
            results = [r for r in prev.get("results", []) if r["id"] in want and (r.get("device") or {}).get("ok")]
    done = {r["id"] for r in results}
    todo = [dg for dg in dialogs if dg["id"] not in done]
    if done:
        print(f"  前の実行で解いた {len(done)} 問は飛ばす。残り {len(todo)} 問（最初からは --fresh）", flush=True)
    t0 = time.time()
    # ★最初に 1 回だけアプリを止める。VLM デモの画面が開いていると、同じプロセスで文章側のモデル（1,467MiB）をもう 1 つ開くことになり、
    #   NPU の上限（プロセスごと約 1,710MiB）を超える（2026-09-29 の 1 回目は 1 時間待って結果が返らなかった）。
    #   job ごとには止めない: RunnerService は job の終わりにモデルを閉じる（nativeDestroy）。止めてすぐ開き直すと、
    #   前のプロセスのメモリが片付く前に次の 1.6GB を読み込むことになり、OS に止められた（2026-09-29 の 2 回目、43 問目から）
    D.adb("shell", f"am force-stop {D.PKG}")
    time.sleep(2.0)
    for c0 in range(0, len(todo), a.chunk):
        chunk = todo[c0:c0 + a.chunk]
        n0 = len(done) + c0
        job = {"job_id": f"vlm_eval_{n0:03d}", "task": "generate_kv", "model": text["model"], "ep": D.EP,
               "mask_neg": -60.0, **{k: v for k, v in text.items() if k != "model"},
               "stop_ids": list(C.STOP_IDS), "reuse_cache": False, "dialogs": chunk}
        for attempt in range(a.retries + 1):
            # ★前の job のモデルが片付くのを待ってから次を開く。モデルを開くとき（読み込み中）が一番メモリを使い、
            #   ここで OS に止められる（2026-09-29 は 2 回とも開始 10 秒前後の読み込み中だった）
            if c0 or attempt:
                time.sleep(a.settle)
            mem = D.wait_mem()
            D.wake()
            print(f"  {n0 + 1}〜{n0 + len(chunk)} 問目を端末で解いている…（{time.time() - t0:.0f}s、端末の空きメモリ {mem}MiB"
                  f"{f'、やり直し {attempt} 回目' if attempt else ''}）", flush=True)
            try:
                res = D.run_job(job, timeout_s=a.job_timeout, hold=not a.no_hold)
                break
            except D.DeviceJobFailed as e:
                # OS に止められた等。少し待ってから同じ job をやり直す（解き終えた分はファイルに書いてある）
                print(f"    ！{e}", flush=True)
                if attempt >= a.retries:
                    raise SystemExit(f"{a.retries + 1} 回続けて落ちたので止める。同じコマンドをもう一度打てば残りから続ける")
                print(f"    {a.cooldown} 秒待ってやり直す", flush=True)
                D.adb("shell", f"am force-stop {D.PKG}")
                time.sleep(a.cooldown)
        if res.get("_kind") != "result" or "dialogs" not in res:
            raise SystemExit(f"端末で解けなかった: {json.dumps(res, ensure_ascii=False)[:800]}")
        for dg in res["dialogs"]:
            iid = dg.get("id")
            tr = (dg.get("turns") or [{}])[0]
            m = meta[iid]
            gen = [int(x) for x in tr.get("new_ids", [])]
            ans = H.decode(proc, gen)
            jd = judge(ans, m["it"])
            step = tr.get("step_ms_stats", {}) or {}
            r = {"id": iid, "set": m["it"]["set"], "tags": m["it"]["tags"], "question": m["it"]["question"],
                 "answer": ans, "answer_ref": m["it"].get("answer_ref"), "judge": jd,
                 "n_image_tokens": m["n_img"], "prompt_len": m["prompt_len"], "gen_tokens": len(gen), "gen_ids": gen,
                 "t_first": round(float(tr.get("prefill_ms", 0.0)) / 1000.0, 3),
                 "t_total": round(float(tr.get("total_ms", 0.0) or 0.0) / 1000.0, 3),
                 "image_size": m["image_size"], "device": {"ok": tr.get("ok"), "error": tr.get("error"),
                                                           "image_tokens_fed": tr.get("image_tokens_fed"),
                                                           "step_ms_median": step.get("median"),
                                                           "prefill_ms": tr.get("prefill_ms"),
                                                           # Phase 5: 最後の読み込みの回（＝答えの 1 つ目）の時間と、まとめ読みの記録
                                                           "first_step_ms": (tr.get("step_ms") or [None])[0],
                                                           "multi_token": tr.get("multi_token")}}
            results.append(r)
            mark = {True: "○", False: "×", None: "－"}[jd["ok"]]
            print(f"    {mark} {iid:16s} [画像 {tr.get('image_tokens_fed')} 位置・{tr.get('prefill_ms', 0) / 1000:.1f}s] "
                  f"{m['it']['question'][:20]} → {ans[:50]}", flush=True)
        RES_REF.mkdir(parents=True, exist_ok=True)
        C.write_json_atomic(out_path, {"prep": "fixed", "budget": a.budget, "srq": True, "feed": feed,
                                       "text_model": text["model"],
                                       "vision": f"npz:work/feats/{FEATS_DEV.name}", "device": "SOG05 NPU",
                                       "system_prompt": system, "results": results})
    ok = [r for r in results if r["judge"]["ok"] is not None]
    n_ok = sum(1 for r in ok if r["judge"]["ok"])
    fed_bad = [r["id"] for r in results if r["device"]["image_tokens_fed"] != r["n_image_tokens"]]
    print(f"\n  端末だけで {n_ok}/{len(ok)}（{time.time() - t0:.0f}s）。画像の特徴を入れた数が合わない問題 {len(fed_bad)}"
          f"\n-> {out_path}（`ref_vlm report` の表に並ぶ）")
    return {"n": len(results), "n_ok": n_ok, "n_judged": len(ok), "image_fed_mismatch": fed_bad,
            "path": str(out_path), "sec": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------
# uicheck（Q13・Q16）
# ---------------------------------------------------------------------------
def cmd_uicheck(a) -> dict:
    C.ensure_hf_env()
    from PIL import Image
    from transformers import AutoProcessor
    from app.pipeline import hf_vlm as H
    D.ensure_app()
    man = load_manifest()
    it = next(x for x in man["items"] if x["id"] == a.item)
    src = C.EVAL_DIR / it["image"]
    ext = src.suffix.lower()
    d = IO / "ui"
    d.mkdir(parents=True, exist_ok=True)
    qf = d / "q.txt"
    qf.write_text(it["question"], encoding="utf-8")
    D.push_many([(src, f"inputs/vlm/test{ext}"), (qf, "inputs/vlm/q.txt")], "写真と質問")
    D.adb("shell", f"am force-stop {D.PKG}")
    D.adb("logcat", "-c", check=False)
    D.wake()
    t0 = time.time()
    D.adb("shell", f"am start -n {VLM_ACT} --es image inputs/vlm/test{ext} --es question_file inputs/vlm/q.txt")
    log = ""
    while time.time() - t0 < a.timeout:
        time.sleep(2)
        log = D.adb("logcat", "-d", "-s", "EdgeSLMVlm:*", check=False)
        if "vlm stats:" in log or "vlm read failed" in log:
            break
    (d / "logcat.txt").write_text(log, encoding="utf-8")
    grab = lambda pat: (re.findall(pat, log) or [None])[-1]
    ids_s = grab(r"vlm prompt first=true n=\d+ ids=([0-9,]+)")
    if ids_s is None:
        raise SystemExit(f"画面の結果が {a.timeout} 秒で出なかった。{d / 'logcat.txt'} を見ること")
    ui_ids = [int(x) for x in ids_s.split(",")]
    proc = AutoProcessor.from_pretrained(str(C.HF_DIR), local_files_only=True)
    img = Image.open(src)
    hf_ids, _ = hf_prompt(proc, H, img, it["question"], a.budget, man["system_prompt"])
    same_ids = ui_ids == hf_ids
    # 前処理: 画面が作った pixel_values と PC（PIL で復号）を比べる（違いは JPEG の復号と、大きい写真の読むときの縮小）
    D.adb("pull", f"{D.DEV}/inputs/vlm/cam_pix.bin", str(d / "cam_pix.bin"))
    grid = C.FIXED_GRIDS[a.budget]
    ref = H.pixels_fixed_np(H.prepare_image(img, "fixed", a.budget, grid=grid), grid)
    ui_pix = np.fromfile(d / "cam_pix.bin", np.float32).reshape(ref.shape)
    diff = np.abs(ui_pix - ref) * 255
    ans = grab(r"vlm answer: (.*)")
    stats = grab(r"vlm stats: (.*)")
    image = grab(r"vlm image: (.*)")
    print(f"  プロンプトの ID: 画面 {len(ui_ids)} 個 ↔ HF {len(hf_ids)} 個 → {'一致' if same_ids else '★違う'}")
    print(f"  前処理（画面 ↔ PC の PIL）: 画素の差の最大 {diff.max():.0f}・平均 {diff.mean():.3f}（0〜255）")
    print(f"  写真の読み取り: {image}")
    print(f"  答え: {ans}")
    print(f"  時間など: {stats}")
    if not same_ids:
        k = next((i for i, (x, y) in enumerate(zip(ui_ids, hf_ids)) if x != y), min(len(ui_ids), len(hf_ids)))
        print(f"    最初に違う位置 {k}: 画面 {ui_ids[max(0, k - 3):k + 3]} / HF {hf_ids[max(0, k - 3):k + 3]}")
    return {"item": a.item, "same_prompt_ids": same_ids, "ui_ids_len": len(ui_ids), "hf_ids_len": len(hf_ids),
            "pix_diff_max_255": float(diff.max()), "pix_diff_mean_255": float(diff.mean()),
            "image_line": image, "answer": ans, "stats_line": stats}


# ---------------------------------------------------------------------------
# holdcheck（画面を前に出したまま開くと、OS に止められにくいか）
# ---------------------------------------------------------------------------
def cmd_holdcheck(a) -> dict:
    """文章側のモデル（1.5GB）を開いて 1 問だけ解く job を、続けて --repeat 回動かす。読み込み中のアプリの優先度
    （/proc/<pid>/oom_score_adj。0 = 画面が前、50 前後 = 画面の無い Service）を 1 秒ごとに記録する。

    - job の中身は eval の 1 回目の job（work/device_vit/vlm_eval_000.json）の先頭の問題をそのまま使う
      （プロンプトの ID も画像の特徴も端末に置いてあるので、HF は要らない。$PY2 でも動く）
    - 既定は画面を前に出す（RunnerHoldActivity）。`--no-hold` で従来の入口（RunnerActivity）と比べられる
    - eval の結果のファイルには触らない
    """
    import threading
    src = D.IO / "vlm_eval_000.json"
    if not src.exists():
        raise SystemExit(f"{src} が無い。先に device_vlm eval を 1 回動かすこと（job の中身をそこから取る）")
    base = json.loads(src.read_text(encoding="utf-8"))
    if a.text != "t1":
        base["model"] = D.text_job(a.text)["model"]              # 16 トークン版で試す（端末に送る）
    hold = not a.no_hold
    mode = "hold" if hold else "nohold"
    D.ensure_app()
    D.adb("shell", f"am force-stop {D.PKG}")
    time.sleep(2.0)
    runs = []
    for k in range(a.repeat):
        if k:
            time.sleep(a.settle)
        mem = D.wait_mem()
        job = dict(base, job_id=f"vlm_holdcheck_{mode}_{k}", dialogs=base["dialogs"][:max(1, a.n or 1)])
        samples, stop = [], threading.Event()

        def sample(t0=time.time()):
            while not stop.is_set():
                pid = D.adb("shell", f"pidof {D.PKG} || true", check=False).split()
                if pid:
                    adj = D.adb("shell", f"cat /proc/{pid[0]}/oom_score_adj 2>/dev/null || true", check=False).strip()
                    samples.append([round(time.time() - t0, 1), pid[0], adj])
                stop.wait(1.0)

        th = threading.Thread(target=sample, daemon=True)
        th.start()
        rec = {"run": k, "mem_available_mib": mem}
        try:
            res = D.run_job(job, timeout_s=a.job_timeout, hold=hold)
            dg = (res.get("dialogs") or [{}])[0]
            tr = (dg.get("turns") or [{}])[0]
            rec.update(ok=res.get("_kind") == "result" and bool(tr.get("ok")), wall_s=res.get("_wall_s"),
                       prefill_ms=tr.get("prefill_ms"), image_tokens_fed=tr.get("image_tokens_fed"))
            # Phase 5: 速さも見る。答えの 1 つ目 = 読み込み（prefill_ms）＋最後の読み込みの回（step_ms[0]）、
            #          2 つ目からは 1 回 1 トークン（step_ms[1:] の中央値）。まとめ読みの記録（回数・1 回の時間）も残す
            sm = [float(x) for x in (tr.get("step_ms") or [])]
            rec.update(prompt_len=tr.get("prompt_len"), new_tokens=len(tr.get("new_ids") or []),
                       first_token_ms=round(float(tr.get("prefill_ms") or 0.0) + (sm[0] if sm else 0.0), 1),
                       decode_ms_median=round(float(np.median(sm[1:])), 1) if len(sm) > 1 else None,
                       multi_token=tr.get("multi_token"))
        except D.DeviceJobFailed as e:
            rec.update(ok=False, error=str(e))
        finally:
            stop.set()
            th.join(timeout=5)
        adjs = [s_[2] for s_ in samples if s_[2] != ""]
        rec["adj_samples"] = samples
        rec["adj_values"] = sorted(set(adjs))
        runs.append(rec)
        print(f"  {k + 1}/{a.repeat}: {'○ 開けた' if rec['ok'] else '× ' + rec.get('error', '失敗')[:120]}"
              f"（空き {mem}MiB、読み込み中の優先度 oom_score_adj = {', '.join(rec['adj_values']) or '読めない'}、"
              f"{rec.get('wall_s')}s）", flush=True)
        if rec["ok"]:
            mt = rec.get("multi_token") or {}
            print(f"      プロンプト {rec.get('prompt_len')} トークン → 答えの 1 つ目まで {rec['first_token_ms'] / 1000:.2f}s"
                  f"（読み込み {mt.get('prefill_runs', rec.get('prompt_len'))} 回）、2 つ目から 1 トークン "
                  f"{rec.get('decode_ms_median')}ms、答え {rec.get('new_tokens')} トークン", flush=True)
    n_ok = sum(1 for r in runs if r["ok"])
    print(f"\n  {mode}: {n_ok}/{len(runs)} 回、止められずに開けた")
    return {"mode": mode, "n_ok": n_ok, "n": len(runs), "runs": runs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "eval", "uicheck", "holdcheck"])
    ap.add_argument("--budget", type=int, default=70, choices=[70])
    ap.add_argument("--n", type=int, default=0, help="prep・eval で使う数（0 = 全部）")
    ap.add_argument("--chunk", type=int, default=28,
                    help="eval: 1 つの job で解く問題の数（1 問 12 秒前後。モデルを開く回数を減らすため 28。落ちても解き終えた job の分は残る）")
    ap.add_argument("--fresh", action="store_true", help="eval: 前の実行の結果を使わず、最初から解く")
    ap.add_argument("--retries", type=int, default=2, help="eval: アプリが落ちた・止まったとき同じ job をやり直す回数")
    ap.add_argument("--cooldown", type=int, default=30, help="eval: 落ちたあと、やり直す前に待つ秒数")
    ap.add_argument("--settle", type=int, default=10, help="eval: job と job の間に待つ秒数（前のモデルが片付くのを待つ）")
    ap.add_argument("--job-timeout", type=int, default=1200, help="eval: 1 つの job の結果を待つ秒数")
    ap.add_argument("--no-hold", action="store_true",
                    help="eval: 画面を前に出さずに動かす（RunnerActivity。古いアプリ用。OS に止められやすい）")
    ap.add_argument("--repeat", type=int, default=5, help="holdcheck: 開いて 1 問解く job を続けて動かす回数")
    ap.add_argument("--text", default="t1", choices=["t1", *D.T16_KINDS],
                    help="eval・holdcheck: 文章側のモデル（t1 = 1 トークン版、t16 = Phase 5 の 16 トークン版、"
                         "t16f = 16 トークン版を NPU のずれに合わせて書き直した版、t16t = PLE の tanh だけ書き直した版）")
    ap.add_argument("--system", default="", help="eval: system の文を差し替える（Phase 5 の「画像が無い」誤りの対策の比較）")
    ap.add_argument("--system-tag", default="sys2", help="eval: --system を使ったときの結果の名前の目印")
    ap.add_argument("--item", default="car_01a", help="uicheck で使う問題")
    ap.add_argument("--timeout", type=int, default=180, help="uicheck: 画面の結果を待つ秒数")
    a = ap.parse_args()
    C.utf8_stdout()
    out = {"prep": cmd_prep, "eval": cmd_eval, "uicheck": cmd_uicheck, "holdcheck": cmd_holdcheck}[a.cmd](a)
    RES.mkdir(parents=True, exist_ok=True)
    name = f"b{a.budget}_{a.cmd}" + (f"_{a.item}" if a.cmd == "uicheck" else "") \
        + (f"_{'nohold' if a.no_hold else 'hold'}" if a.cmd == "holdcheck" else "") \
        + ("" if a.text == "t1" else f"_{a.text}") + (f"_{a.system_tag}" if a.cmd == "eval" and a.system else "")
    C.write_json_atomic(RES / f"{name}.json", out)
    print(f"-> {RES / (name + '.json')}")


if __name__ == "__main__":
    main()
