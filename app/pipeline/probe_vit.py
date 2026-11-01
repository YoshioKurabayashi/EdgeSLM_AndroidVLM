# -*- coding: utf-8 -*-
"""Phase 2-4 の切り分け: 途中の値も出力する「探り用」の画像エンコーダを作り、端末と PC の QDQ を途中ごとに比べる（2026-09-29）。

    python -m app.pipeline.probe_vit build                    # 2 層版（速い）と 16 層版（途中の値つき）を作る（書き出し → 量子化 → 変換）
    python -m app.pipeline.probe_vit build --layers 2         # 2 層版だけ（変換 数分の見込み）
    python -m app.pipeline.probe_vit device                   # 作った版を端末で動かし、途中の値ごとに PC の QDQ と比べる
    （共通: --budget 70。**EdgeSLM_Android の venv（$PY2）で動かす**。変換は build_vit_ctx と同じく Git Bash 経由）

## なぜ要るか（2026-09-29 の device_vit acc の結果）

端末の出力が PC の QDQ とまったく合わなかった（cos 0.05 前後）。しかも端末の出力は
**画像にもトークンの位置にもほとんど依存しない**（画像どうし cos 0.999、トークンどうし cos 1.0）。
CPU に逃げずに全部 NPU で動いてはいる（strict=true）ので「動くが値が壊れる」型。どこで壊れるかを、途中の値を
グラフの出力に足した版で調べる。

- **2 層版**（パッチの埋め込み・層 0・層 1 の出力）: 変換が速い。ここで既に壊れていれば、形の扱い（630 トークンの
  4 次元のテンソル・2 次元 RoPE・q/k/v の正規化など）を疑う
- **16 層版**（パッチの埋め込み・層 0,1,2,4,5,7,10,15・プーリング後・最終出力）: 層 5 から巨大な活性（最大 2,158）が
  現れるので、そこで壊れ始めるなら数値（16bit の範囲・RMSNorm の x²）を疑う。プーリングで壊れるなら 4 次元の reshape を疑う

量子化・変換の手順は本番（quantize_vit・build_vit_ctx）と同じ（SRQ なしの版だけ）。比べる相手は同じ探り用グラフの
PC の QDQ（ORT CPU・最適化なし）と float の ONNX。結果は `results/P2_vit_device/probe_*.json`。

## 結果（2026-09-29、StudyMemo 8.2〜8.5 節・results/P2_20260929_vit_device.md）

- 2 層版の変換は 89 秒（16 層は 56 分）。最初の出力（パッチの埋め込み）で既に端末 ↔ PC の QDQ の相対 L2 が 21.8 → 層も softmax も関係なかった
- 壊れ方を形で読むと、重みのスケール 4.0e-6 未満のチャネルが全部壊れ、4.1e-6 以上は全部正常（C-14）。
  qat_inject_vit にスケールの下限（最大の 1/64）を足して直した
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np

from app.pipeline import vlm_common as C

# 量子化の方式（本番と同じ）、16 層版で途中の値を出す層の番号、結果の置き場
SCHEME = "a16w8_rhs8"
FULL_TAPS = (0, 1, 2, 4, 5, 7, 10, 15)
RES = C.RESULTS / "P2_vit_device"
PROBE_IDS = ("vg_00", "vg_13", "wild_18", "syn_sign")      # 評価用 16 枚から 4 枚（vg_00・vg_13 は HF の途中の値もある）


# 予算からファイル名の札と格子を返す（横長の格子だけ）。
def tag_for(budget: int) -> tuple[str, tuple[int, int]]:
    grid = C.FIXED_GRIDS[budget]
    return f"b{budget}_{grid[0]}x{grid[1]}", grid


# 途中の値を出す層: 16 層版は FULL_TAPS、それより少ない版は全部の層。
def taps_for(n_layers: int) -> list[int]:
    return [i for i in FULL_TAPS if i < n_layers] if n_layers == C.VIT_LAYERS else list(range(n_layers))


# 出力の名前の並び（パッチの埋め込み・層の出力・16 層版だけプーリングの後と最終出力）。グラフの出力の順と同じ。
def out_names(n_layers: int) -> list[str]:
    names = ["patch_embed"] + [f"layer_{i}" for i in taps_for(n_layers)]
    return names + (["pooled", "image_embeds"] if n_layers == C.VIT_LAYERS else [])


# 探り用の版の置き場（work/ops/vitprobe_<札>_L<層の数>/）と、各段の生成物のパス。
def paths(tag: str, n_layers: int) -> dict:
    name = f"vitprobe_{tag}_L{n_layers}"
    d = C.OPS / name
    return {"name": name, "dir": d, "onnx": d / f"{name}.onnx", "spec": d / f"{name}.spec.json",
            "calib": d / f"{name}_calib.npz", "prepped": d / f"{name}_prepped.onnx",
            "qdq_raw": d / f"{name}_{SCHEME}_qdq.onnx", "qdq_ort": d / f"{name}_{SCHEME}_qdq_ort.onnx",
            "qdq": d / f"{name}_nosrq_qdq.onnx", "final": d / f"{name}_final_qdq.onnx",
            "build": d / "build", "epctx": d / f"{name}_epctx.onnx"}


# ---------------------------------------------------------------------------
# 探り用のグラフ（vit_graph の部品をそのまま使い、途中の値も返す）
# ---------------------------------------------------------------------------
# 探り用のグラフを作る。VitGraph の部品をそのまま使い、層を n_layers に切り詰め、途中の値をまとめて出力にする（重みも計算も本番と同じ）。
def make_probe(grid, n_layers: int):
    import torch
    from app.pipeline import vit_graph as V

    # 探り用のグラフの本体（重み・計算は VitGraph と同じで、出力だけ多い）
    class ProbeGraph(torch.nn.Module):
        def __init__(self):
            # 本番と同じ VitGraph を組み、層を n_layers に切り詰める（2 層版は変換が速い）
            super().__init__()
            self.vit = V.VitGraph(V.QatFile(), grid, fold_pool_scale=True).eval()
            self.vit.layers = self.vit.layers[:n_layers]
            self.taps = set(taps_for(n_layers))

        def forward(self, pixel_values):
            # 戻り値: (パッチの埋め込み, 選んだ層の出力…[, プーリングの後, 最終出力]) の並び。out_names と同じ順
            v = self.vit
            h = v.patch_proj(2.0 * (pixel_values - 0.5)) + v.pos_emb       # 本番の run() と同じ式
            outs = [h]
            for i, ly in enumerate(v.layers):
                h = ly(h, v.cos, v.sin)
                if i in self.taps:
                    outs.append(h)
            if n_layers == C.VIT_LAYERS:
                p = v.pool(h)                                             # ×√768 は射影層の eps に吸収済み（本番と同じ）
                outs += [p, v.proj(v.proj_norm(p))]
            return tuple(outs)

    return ProbeGraph().eval()


def export(tag: str, grid, n_layers: int) -> dict:
    """探り用のグラフを ONNX に書き出し、出力の名前と入口の演算（Sub）を確かめて、量子化の spec と較正データを置く。"""
    import onnx
    import torch
    from app.pipeline import vit_graph as V
    from app.pipeline.export_vit import load_capture
    assert not V.SRQ
    P = paths(tag, n_layers)
    P["dir"].mkdir(parents=True, exist_ok=True)
    for old in (P["onnx"], P["dir"] / f"{P['name']}.onnx.data"):
        old.unlink(missing_ok=True)
    m = make_probe(grid, n_layers)
    cap = load_capture(tag)
    x = torch.from_numpy(cap["pixel_values"][0:1].astype(np.float32))
    names = out_names(n_layers)
    t0 = time.time()
    torch.onnx.export(m, (x,), str(P["onnx"]), input_names=["pixel_values"], output_names=names,
                      dynamic_axes=None, opset_version=21, do_constant_folding=True, dynamo=True)
    mo = onnx.load(str(P["onnx"]), load_external_data=False)
    got = [v.name for v in mo.graph.output]
    if got != names:
        raise SystemExit(f"出力の名前が想定と違う: {got}")
    first = [n.op_type for n in mo.graph.node if "pixel_values" in n.input]
    if first != ["Sub"]:
        raise SystemExit(f"pixel_values を最初に使う演算が Sub でない: {first}")
    shutil.copyfile(C.CAPTURE / f"vit_{tag}_calib.npz", P["calib"])
    spec = {"model": str(P["onnx"]).replace("\\", "/"), "note": f"探り用（{n_layers} 層、途中の値を出力に足した版）",
            "per_channel_weights": True, "weight_bits": 4, "calib_npz": str(P["calib"]).replace("\\", "/")}
    P["spec"].write_text(json.dumps(spec, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  書き出し {time.time() - t0:.0f}s、ノード {len(mo.graph.node)}、出力 {names}", flush=True)
    return {"nodes": len(mo.graph.node), "outputs": names}


def quantize(tag: str, n_layers: int) -> dict:
    """quantize_vit と同じ手順（SRQ なし）。QAT の重みの一覧は、この版に含まれる層・部品だけに絞る。"""
    from app.pipeline import downgrade_actact_rhs, p2b_ops, verify_artifacts
    from app.pipeline import qat_inject_vit as Q
    from app.pipeline.vit_graph import PFX
    P = paths(tag, n_layers)
    for old in (P["prepped"], P["qdq_raw"], P["qdq_ort"]):
        old.unlink(missing_ok=True)
    st = p2b_ops.prepare(C.OPS, P["name"], SCHEME, n_random=16)
    if st.get("verdict") != "ready_for_convert":
        raise SystemExit(f"prepare が通らない: {st.get('verdict')}")
    P["qdq_raw"].replace(P["qdq_ort"])
    lin = Q.vit_linears()
    # この版に含まれる層の QAT の重みと、パッチの埋め込み（16 層版なら射影層も）だけを書き込みの対象にする
    keep = {k: v for k, v in lin.items()
            if (".encoder.layers." in k and int(k.split(".encoder.layers.")[1].split(".")[0]) < n_layers)
            or k == PFX + "patch_embedder.input_proj"
            or (k == "model.embed_vision.embedding_projection" and n_layers == C.VIT_LAYERS)}
    inj = Q.inject(P["qdq_ort"], P["prepped"], keep, P["qdq"], srq=False)
    dg = downgrade_actact_rhs.run(P["qdq"], P["prepped"], P["final"])
    if dg["n_rhs"] != 2 * n_layers:
        raise SystemExit(f"活性×活性の右辺の数 {dg['n_rhs']}（想定 {2 * n_layers}）")
    c1 = verify_artifacts.check_c1_no_float_segments(P["final"])
    c5 = verify_artifacts.check_c5_no_batchnorm(P["final"])
    c14 = Q.scale_ratio_report(P["final"])
    print(f"  量子化: 重み {inj['written']} 本・C-8 右辺 {dg['n_rhs']}・C-1 {'合格' if c1.ok else '★不合格 ' + str(c1.get('detail'))}"
          f"・C-5 {'合格' if c5.ok else '★不合格'}・C-14 {'合格' if c14['ok'] else '★注意'}（スケールの最小/最大 {c14['min_ratio']:.4f}）"
          f"・スケールを下限に上げたチャネル {inj.get('scale_floored')}", flush=True)
    if not (c1.ok and c5.ok):
        raise SystemExit("C-1/C-5 に通らない")
    return {"written": inj["written"], "n_rhs": dg["n_rhs"], "C1": bool(c1.ok), "C5": bool(c5.ok), "C14": c14,
            "scale_floored": inj.get("scale_floored")}


def convert(tag: str, n_layers: int) -> dict:
    """build_vit_ctx と同じ手順で変換・検査・包む（重みは INT8 なので INT4 用の回避策は外す）。"""
    from app.pipeline import build_vit_ctx as B
    from app.pipeline import verify_artifacts, wrap_epctx
    P = paths(tag, n_layers)
    shutil.rmtree(P["build"], ignore_errors=True)
    env = dict(os.environ)
    env.pop("QAIRT_INT4_PATCH", None)
    t0 = time.time()
    rc = B.run_streaming([B.GIT_BASH, "scripts/build_context.sh", str(P["final"]), str(P["build"]), P["name"]],
                         C.WORK / "logs" / f"{P['name']}_build.log", env)
    bins = sorted(P["build"].glob("ctx_*/*.bin"))
    if rc != 0 or not bins:
        raise SystemExit(f"変換が失敗（{rc}）。{P['build']} のログを見ること")
    cmds = (P["build"] / f"{P['name']}_commands.txt").read_text(encoding="utf-8", errors="replace").splitlines()
    ver = verify_artifacts.verify_all(P["final"], P["build"] / f"{P['name']}_meta.json", cmds)
    if not ver["all_pass"]:
        raise SystemExit(f"検査に通らない: {[c['check'] for c in ver['checks'] if not c.get('pass')]}")
    info = wrap_epctx.run(bins[0], P["final"], P["epctx"], P["name"], embed=False)
    print(f"  変換 {time.time() - t0:.0f}s、context binary {bins[0].stat().st_size / 2**20:.1f}MiB、"
          f"出力 {[o['name'] for o in info['outputs']]}", flush=True)
    return {"sec": round(time.time() - t0, 1), "ctx_mib": round(bins[0].stat().st_size / 2**20, 1)}


# --layers の版ごとに 書き出し → 量子化 → 変換 を流す。
def cmd_build(a) -> dict:
    tag, grid = tag_for(a.budget)
    out = {}
    for n in [int(x) for x in a.layers.split(",")]:
        print(f"\n== 探り用 {n} 層", flush=True)
        r = {"export": export(tag, grid, n)}
        r["quantize"] = quantize(tag, n)
        r["convert"] = convert(tag, n)
        out[f"L{n}"] = r
    return out


# ---------------------------------------------------------------------------
# 端末で動かして比べる
# ---------------------------------------------------------------------------
def token_sim(x: np.ndarray) -> float:
    """トークンがどれだけ互いに似ているか: 各トークンとトークン平均の cos の平均（全部同じなら 1.0）。"""
    x = np.asarray(x, np.float64).reshape(-1, x.shape[-1])
    m = x.mean(0)
    return float(np.mean(x @ m / np.maximum(np.linalg.norm(x, axis=1) * np.linalg.norm(m), 1e-300)))


def cmd_device(a) -> dict:
    """探り用の版を端末で動かし、途中の値ごとに端末 ↔ PC の QDQ（と QDQ ↔ float）を比べ、最初に大きくずれる所を出す。

    入力は評価用 16 枚のうち PROBE_IDS の 4 枚（capture_vit の pixel_values）。端末へは generic の infer の job で送る。
    """
    from app.pipeline import device_vit as D
    from app.pipeline.eval_qdq_vit import session
    from app.pipeline.export_vit import load_capture
    tag, grid = tag_for(a.budget)
    D.ensure_app()
    cap = load_capture(tag)
    ids = [str(i) for i in cap["ids"]]
    sel = [ids.index(i) for i in PROBE_IDS]
    pairs = []
    for j in sel:
        p = D.IO / f"pix_{tag}_{ids[j]}.bin"
        cap["pixel_values"][j][None].astype(np.float32).tofile(p)
        pairs.append((p, f"inputs/vit/pix_{tag}_{ids[j]}.bin"))
    D.push_many(pairs, f"入力（{len(sel)} 枚）")
    out = {"tag": tag, "images": list(PROBE_IDS), "probes": {}}
    for n in [int(x) for x in a.layers.split(",")]:
        P = paths(tag, n)
        if not P["epctx"].exists():
            print(f"（{P['epctx'].name} が無いので飛ばす。先に build）")
            continue
        print(f"\n== 探り用 {n} 層", flush=True)
        # 包んだ ONNX の中に書かれた名前で、どの .bin が対応するかを探す（同じ置き場に古い .bin が残っていても取り違えない）
        raw = P["epctx"].read_bytes()
        bin_ = next(c for c in sorted(P["dir"].glob(f"{P['name']}_ctx*.bin")) if c.name.encode() in raw)
        model_rel = f"models/vitprobe/{P['epctx'].name}"
        D.push_many([(P["epctx"], model_rel), (bin_, f"models/vitprobe/{bin_.name}")], "モデル")
        job_id = f"probe_{P['name']}"
        out_dir = f"out/{job_id}"
        D.adb("shell", f"rm -rf {D.DEV}/{out_dir}; mkdir -p {D.DEV}/{out_dir}")
        job = {"job_id": job_id, "task": "infer", "model": model_rel, "ep": D.EP,
               "batches": [{"id": ids[j], "inputs": {"pixel_values": r}} for j, (_, r) in zip(sel, pairs)],
               "output": {"dir": out_dir}}
        res = D.run_job(job, timeout_s=900)
        if res.get("_kind") != "result" or not res.get("ok"):
            raise SystemExit(f"端末で動かなかった: {json.dumps(res, ensure_ascii=False)[:800]}")
        local = D.IO / job_id
        shutil.rmtree(local, ignore_errors=True)
        D.adb("pull", f"{D.DEV}/{out_dir}", str(D.IO))
        names = out_names(n)
        dev = {nm: [] for nm in names}
        for rec in res["records"]:
            for o in rec["outputs"]:
                dev[o["name"]].append(np.fromfile(local / Path(o["path"]).name, np.float32).reshape(o["shape"])[0])
        print(f"  端末: strict={res['describe'].get('strict_no_cpu_fallback')}  "
              f"1 枚 {np.median([r['ort_run_ms'] for r in res['records']]):.1f}ms", flush=True)
        # PC: 同じ探り用グラフの QDQ（量子化の真似）と float
        sq, sf = session(P["final"]), session(P["onnx"])
        ref_q = {nm: [] for nm in names}
        ref_f = {nm: [] for nm in names}
        for j in sel:
            x = cap["pixel_values"][j][None].astype(np.float32)
            for nm, y in zip(names, sq.run(None, {"pixel_values": x})):
                ref_q[nm].append(y[0])
            for nm, y in zip(names, sf.run(None, {"pixel_values": x})):
                ref_f[nm].append(y[0])
        # 途中の値ごとに、端末 ↔ QDQ の相対 L2・cos、QDQ ↔ float の相対 L2、トークンの似かた（全部同じ値なら 1.0）、絶対値の最大を並べる
        rows = []
        print(f"  {'途中の値':14s} {'端末↔QDQ 相対L2':>16s} {'cos':>7s} {'QDQ↔float 相対L2':>17s} "
              f"{'トークンの似かた 端末/QDQ':>22s} {'max|x| 端末/QDQ':>18s}")
        for nm in names:
            dq = [D.compare(d, q) for d, q in zip(dev[nm], ref_q[nm])]
            qf = [D.compare(q, f) for q, f in zip(ref_q[nm], ref_f[nm])]
            row = {"name": nm,
                   "dev_vs_qdq_rel_l2": float(np.median([r["rel_l2"] for r in dq])),
                   "dev_vs_qdq_cos": float(np.median([r["cos"] for r in dq])),
                   "qdq_vs_float_rel_l2": float(np.median([r["rel_l2"] for r in qf])),
                   "token_sim_dev": float(np.median([token_sim(d) for d in dev[nm]])),
                   "token_sim_qdq": float(np.median([token_sim(q) for q in ref_q[nm]])),
                   "maxabs_dev": float(np.max([np.abs(d).max() for d in dev[nm]])),
                   "maxabs_qdq": float(np.max([np.abs(q).max() for q in ref_q[nm]])),
                   "per_image_dev_vs_qdq": dq}
            rows.append(row)
            print(f"  {nm:14s} {row['dev_vs_qdq_rel_l2']:16.4f} {row['dev_vs_qdq_cos']:7.4f} {row['qdq_vs_float_rel_l2']:17.4f} "
                  f"{row['token_sim_dev']:10.3f} / {row['token_sim_qdq']:.3f} {row['maxabs_dev']:10.1f} / {row['maxabs_qdq']:.1f}",
                  flush=True)
        np.savez(D.IO / f"{job_id}.npz", **{f"dev_{k}": np.stack(v) for k, v in dev.items()},
                 **{f"qdq_{k}": np.stack(v) for k, v in ref_q.items()})
        bad = next((r["name"] for r in rows if r["dev_vs_qdq_rel_l2"] > 0.2), None)
        print(f"  → 最初に大きくずれる所（相対L2 > 0.2）: {bad or '無し'}", flush=True)
        out["probes"][f"L{n}"] = {"describe": D.describe_line(res), "rows": rows, "first_bad": bad,
                                  "ort_run_ms": [r["ort_run_ms"] for r in res["records"]]}
    return out


def main():
    # build（書き出し → 量子化 → 変換）か device（端末で比べる）を動かし、results/P2_vit_device/probe_<札>_<コマンド>.json に書く
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "device"])
    ap.add_argument("--budget", type=int, default=70, choices=sorted(C.FIXED_GRIDS))
    ap.add_argument("--layers", default="2,16", help="作る／調べる版の層の数（カンマ区切り）")
    a = ap.parse_args()
    C.utf8_stdout()
    out = {"build": cmd_build, "device": cmd_device}[a.cmd](a)
    RES.mkdir(parents=True, exist_ok=True)
    tag, _ = tag_for(a.budget)
    path = RES / f"probe_{tag}_{a.cmd}.json"
    C.write_json_atomic(path, out)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
