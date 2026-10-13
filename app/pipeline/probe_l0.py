# -*- coding: utf-8 -*-
"""16 トークン版の NPU のずれの切り分け 3 段目: **層 0 だけを切り出した小さいグラフ**で、どの計算でずれるかを特定する（2026-10-01）。

    python -m app.pipeline.probe_l0                      # 作る（切り出し → 変換 4 つ、数分〜20 分）→ 端末で比べる（数分）
    python -m app.pipeline.probe_l0 build                # 作るだけ
    python -m app.pipeline.probe_l0 device               # 端末で比べるだけ（作ってあるもの）
    python -m app.pipeline.probe_l0 --variants t16_plain,t1_plain    # 一部だけ
    python -m app.pipeline.probe_l0 build --redo         # 作り直す

**EdgeSLM_Android の venv（$PY2）で動かす**（onnx・onnxruntime・adb）。変換は build_t16 と同じく Git Bash から build_context.sh。
端末は USB でつなぎ、VLM 検証アプリ（org.edgeslm.vlm）を入れておく。

## ここまで（probe_t16、results/P5_t16/probe_t16.json）

C++ の入れ方は正しく、16 トークン版の NPU の計算が、同じ量子化済みグラフの CPU の計算と大きく違う。
層ごとの K/V の差は、文字の行で層 0 の 2% → 層 1 で 23%、画像の行は層 0 から 40%。→ **層 0 の中**でずれ始める。

## この道具がすること

1. **切り出し**: いまの量子化済みグラフ（かたまり A の final_qdq。16 トークン版は本プロジェクト、1 トークン版は
   EdgeSLM_AndroidGemma4 のもの。どちらも読むだけ）から、「層 1 の K/V（present_k_1・present_v_1）を作るのに要る部分」＝
   **入口・層 0 の全部・層 1 の入口の正規化と K/V の行列**だけを抜き出す。量子化の目盛り・重みは元のグラフのまま（作り直さない）
2. 4 つの版を作る:

   | 版 | 元 | 出力 |
   |---|---|---|
   | `t16_plain` | 16 トークン版 | present_k/v_0・present_k/v_1 だけ（**小さいグラフでも同じずれが出るか**を見る） |
   | `t16_taps` | 16 トークン版 | ＋ **途中の値すべて**（入力の丸め・各演算の出力。約 170 本） |
   | `t1_plain`・`t1_taps` | 1 トークン版 | 同上（基準） |

   ★途中の値を出力に足すと、NPU が演算をまとめるやり方が変わり、ずれが消えることがある（そのため plain も作る）
3. QAIRT で変換（build_t16 と同じ build_context.sh、`QAIRT_INT4_PATCH=1`）→ EPContext で包む
4. **入力**: 評価と同じ問題（既定 syn_nav_S）のプロンプトの最初の 3 回ぶん（0〜47 番目。0 回目は system 文、1 回目は文字 8 行＋画像 8 行、
   2 回目は画像 16 行）。past は小さいグラフ（16 トークン版 plain）を PC の CPU で順に回して作る
   - 16 トークン版: 3 回ぶん（16 行ずつ）
   - 1 トークン版: 同じ位置の 8 行（0 回目の 0・15 行目、1 回目の 0・7（文字）・8・15（画像）行目、2 回目の 0・15 行目）。past は同じ中身を右詰め
5. 同じ入力を端末の NPU（generic の infer の job）と PC の CPU（ORT）に通し、**出力ごと・行ごと**に相対 L2 を比べる。
   16 トークン版の行 r と、1 トークン版の同じ位置の 1 行を並べ、**16 トークン版だけが大きくずれる最初の計算**を探す

結果: `results/P5_t16/l0probe.json`（全部）と `results/P5_t16/l0probe_table.md`（途中の値の表）
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from app.pipeline import device_vit as D
from app.pipeline import gemma4_common as G
from app.pipeline import vlm_common as C
from app.pipeline.diag_t16 import IMG, TABLES, Tables, load_items, old_prompt
from app.pipeline.probe_t16 import Cpu, Device, Feeder

SRC = {
    "t16": C.OPS / "g4t16_c0_8h" / "g4t16_c0_8h_final_qdq.onnx",                  # 本プロジェクト（Phase 5）
    "t1": G.G4_OPS / "g4c0_8h_L256" / "g4c0_8h_L256_final_qdq.onnx",              # EdgeSLM_AndroidGemma4（読むだけ）
}
VARIANTS = ["t16_plain", "t1_plain", "t16_taps", "t1_taps"]      # plain を先に（変換が軽く、再現するかが先に分かる）
CUT_OUTPUTS = ["present_k_0", "present_v_0", "present_k_1", "present_v_1"]
RES = C.RESULTS / "P5_t16"
IO = C.WORK / "l0probe"
GIT_BASH = os.environ.get("GIT_BASH", r"C:\Program Files\Git\bin\bash.exe")
T1_ROWS = [(0, 0), (0, 15), (1, 0), (1, 7), (1, 8), (1, 15), (2, 0), (2, 15)]   # 1 トークン版で通す（回, 行）
N_CHUNKS = 3

# 層 0 の演算の名前（書き出しの名前。16 トークン版・1 トークン版で同じ）→ 何の計算か（表に出す）
ROLE = {
    "node_mul": "PLE: 入口×定数", "node_linear": "PLE: 射影（1536→35×256）", "node_view": "PLE: 形を変える",
    "node_pow_1": "PLE: 正規化 x²", "node_mean": "PLE: 正規化 平均", "node_add": "PLE: 正規化 +eps",
    "node_rsqrt": "PLE: 正規化 1/√", "node_mul_1": "PLE: 正規化 ×1/√", "node_mul_2": "PLE: 正規化 ×重み",
    "node_add_1": "PLE: ＋ple_tok", "node_mul_3": "PLE: ×定数（ple_all）", "node_select": "PLE: 層 0 の分を選ぶ",
    "node_pow_2": "入口の正規化 x²", "node_mean_1": "入口の正規化 平均", "node_add_2": "入口の正規化 +eps",
    "node_rsqrt_1": "入口の正規化 1/√", "node_mul_4": "入口の正規化 ×1/√", "node_mul_5": "入口の正規化 ×重み",
    "node_linear_1": "Q の行列", "node_linear_2": "K の行列", "node_linear_3": "V の行列",
    "node_view_1": "Q 形", "node_view_2": "K 形", "node_view_3": "V 形",
    "node_pow_3": "Q 正規化 x²", "node_mean_2": "Q 正規化 平均", "node_add_3": "Q 正規化 +eps", "node_rsqrt_2": "Q 正規化 1/√",
    "node_mul_6": "Q 正規化 ×1/√", "node_mul_7": "Q 正規化 ×重み",
    "node_pow_4": "K 正規化 x²", "node_mean_3": "K 正規化 平均", "node_add_5": "K 正規化 +eps", "node_rsqrt_3": "K 正規化 1/√",
    "node_mul_10": "K 正規化 ×1/√", "node_mul_11": "K 正規化 ×重み",
    "node_pow_5": "V 正規化 x²", "node_mean_4": "V 正規化 平均", "node_add_7": "V 正規化 +eps", "node_rsqrt_4": "V 正規化 1/√",
    "node_mul_14": "V 正規化 ×1/√", "node_mul_15": "V 正規化 ×重み",
    "node_mul_8": "Q 回転 ×cos", "node_slice_1": "Q 回転 半分", "node_slice_2": "Q 回転 半分", "node_neg": "Q 回転 −",
    "node_cat": "Q 回転 つなぐ", "node_mul_9": "Q 回転 ×sin", "node_add_4": "Q 回転後",
    "node_mul_12": "K 回転 ×cos", "node_slice_3": "K 回転 半分", "node_slice_4": "K 回転 半分", "node_neg_1": "K 回転 −",
    "node_cat_1": "K 回転 つなぐ", "node_mul_13": "K 回転 ×sin", "node_add_6": "K 回転後",
    "node_matmul": "注意の点数 Q·Kᵀ", "node_add_8": "注意の点数 ＋マスク", "node_softmax": "softmax", "node_view_4": "softmax 形",
    "node_matmul_1": "注意 ×V", "node_view_6": "注意 形", "node_linear_4": "出力の行列 o_proj",
    "node_pow_6": "注意後の正規化 x²", "node_mean_5": "注意後の正規化 平均", "node_add_9": "注意後の正規化 +eps",
    "node_rsqrt_5": "注意後の正規化 1/√", "node_mul_16": "注意後の正規化 ×1/√", "node_mul_17": "注意後の正規化 ×重み",
    "node_add_10": "残差 1（入口＋注意）",
    "node_pow_7": "MLP 前の正規化 x²", "node_mean_6": "MLP 前の正規化 平均", "node_add_11": "MLP 前の正規化 +eps",
    "node_rsqrt_6": "MLP 前の正規化 1/√", "node_mul_18": "MLP 前の正規化 ×1/√", "node_mul_19": "MLP 前の正規化 ×重み",
    "node_linear_5": "MLP gate の行列", "node_linear_6": "MLP up の行列",
    "node_mul_20": "GELU 0.5x", "node_mul_21": "GELU c·x", "node_mul_22": "GELU c·x²", "node_mul_23": "GELU c·x³",
    "node_add_12": "GELU x+c·x³", "node_mul_24": "GELU ×√(2/π)", "node_tanh": "GELU tanh", "node_add_13": "GELU 1+tanh",
    "node_mul_25": "GELU 出力", "node_mul_26": "gate×up", "node_linear_7": "MLP down の行列",
    "node_pow_8": "MLP 後の正規化 x²", "node_mean_7": "MLP 後の正規化 平均", "node_add_14": "MLP 後の正規化 +eps",
    "node_rsqrt_7": "MLP 後の正規化 1/√", "node_mul_27": "MLP 後の正規化 ×1/√", "node_mul_28": "MLP 後の正規化 ×重み",
    "node_add_15": "残差 2（＋MLP）",
    "node_linear_8": "PLE gate の行列", "node_mul_29": "PLE GELU 0.5x", "node_mul_30": "PLE GELU c·x",
    "node_mul_31": "PLE GELU c·x²", "node_mul_32": "PLE GELU c·x³", "node_add_16": "PLE GELU x+c·x³",
    "node_mul_33": "PLE GELU ×√(2/π)", "node_tanh_1": "PLE GELU tanh", "node_add_17": "PLE GELU 1+tanh",
    "node_mul_34": "PLE GELU 出力", "node_mul_35": "×ple（層 0）", "node_linear_9": "PLE 射影の行列",
    "node_pow_9": "PLE 後の正規化 x²", "node_mean_8": "PLE 後の正規化 平均", "node_add_18": "PLE 後の正規化 +eps",
    "node_rsqrt_8": "PLE 後の正規化 1/√", "node_mul_36": "PLE 後の正規化 ×1/√", "node_mul_37": "PLE 後の正規化 ×重み",
    "node_add_19": "残差 3（＋PLE）", "node_mul_38": "×layer_scalar（層 0 の出力）",
    "node_pow_10": "層1 入口の正規化 x²", "node_mean_9": "層1 入口の正規化 平均", "node_add_20": "層1 入口の正規化 +eps",
    "node_rsqrt_9": "層1 入口の正規化 1/√", "node_mul_39": "層1 入口の正規化 ×1/√", "node_mul_40": "層1 入口の正規化 ×重み",
    "node_linear_11": "層1 K の行列", "node_linear_12": "層1 V の行列",
}


# ---------------------------------------------------------------------------
# 切り出し
# ---------------------------------------------------------------------------
def paths(variant: str) -> dict:
    name = f"l0p_{variant}"
    d = C.OPS / name
    return {"name": name, "dir": d, "final": d / f"{name}_final_qdq.onnx", "build": d / "build",
            "epctx": d / f"{name}_epctx.onnx", "taps": d / f"{name}_taps.json", "log": C.WORK / "logs" / f"{name}_build.log"}


def cut(variant: str) -> dict:
    """元の final_qdq から、層 1 の K/V までに要る部分を抜き出し、出力（plain: present だけ／taps: 途中の値すべて）を付けて保存する"""
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper
    kind, mode = variant.split("_")
    src = SRC[kind]
    if not src.exists():
        raise SystemExit(f"元のグラフが無い: {src}")
    P = paths(variant)
    P["dir"].mkdir(parents=True, exist_ok=True)
    m = onnx.load(str(src))
    g = m.graph
    nodes = list(g.node)
    prod = {o: i for i, n in enumerate(nodes) for o in n.output}
    gin = {v.name: v for v in g.input}
    # ---- 層 1 の K/V から逆にたどって、要る演算を集める ----
    need, stack = set(), list(CUT_OUTPUTS)
    while stack:
        t = stack.pop()
        if t in prod and prod[t] not in need:
            need.add(prod[t])
            stack.extend(nodes[prod[t]].input)
    keep = [nodes[i] for i in sorted(need)]
    used = {x for n in keep for x in n.input}
    inits = [t for t in g.initializer if t.name in used]
    inputs = [gin[x] for x in gin if x in used]
    # ---- 途中の値: 各演算の出力 → Q → DQ の DQ の出力（と、入力の Q → DQ） ----
    cons: dict[str, list] = {}
    for n in keep:
        for x in n.input:
            cons.setdefault(x, []).append(n)
    taps = []                                      # {"tensor", "node", "op", "label"}
    if mode == "taps":
        # ★入力の丸め（入力 → Q → DQ）は出力にしない。QAIRT が Q/DQ を消すとき「入力と出力が同じテンソル」になって
        #   「Squash not possible for op attn_mask_DequantizeLinear」で変換が止まる（2026-10-01 の 1 回目）
        for n in keep:
            if n.op_type in ("QuantizeLinear", "DequantizeLinear"):
                continue
            for q in cons.get(n.output[0], []):
                if q.op_type != "QuantizeLinear":
                    continue
                for dq in cons.get(q.output[0], []):
                    if dq.op_type == "DequantizeLinear":
                        taps.append({"tensor": dq.output[0], "node": n.name, "op": n.op_type, "label": ROLE.get(n.name, n.op_type)})
        # Sqrt は 16 行版と 1 行版で名前の番号が違う → 入力を作る演算の名前で呼ぶ（版をまたいで揃える）
        byname = {n.name: n for n in keep}
        for t in taps:
            n = byname.get(t["node"])
            if n is not None and n.op_type == "Sqrt":
                srcn = n.input[0]
                while srcn in prod and nodes[prod[srcn]].op_type in ("QuantizeLinear", "DequantizeLinear"):
                    srcn = nodes[prod[srcn]].input[0]
                pn = nodes[prod[srcn]].name if srcn in prod else srcn
                t["node"] = f"sqrt({pn})"
                t["label"] = ROLE.get(pn, "").replace("+eps", "√") or "√"
    outs = list(CUT_OUTPUTS) + [t["tensor"] for t in taps if t["tensor"] not in CUT_OUTPUTS]
    # ---- 1 回目: 出力の形を知らないまま作り、ORT で回して形を知る ----
    def build(shapes: dict | None):
        ovi = [helper.make_tensor_value_info(o, TensorProto.FLOAT, shapes[o] if shapes else None) for o in outs]
        gg = helper.make_graph(keep, P["name"], inputs, ovi, initializer=inits)
        mm = helper.make_model(gg, opset_imports=list(m.opset_import))
        mm.ir_version = m.ir_version
        return mm
    tmp = P["dir"] / f"{P['name']}_noshape.onnx"
    onnx.save(build(None), str(tmp))
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s = ort.InferenceSession(str(tmp), so, providers=["CPUExecutionProvider"])
    feed = {v.name: np.zeros([d.dim_value for d in v.type.tensor_type.shape.dim], np.float32) for v in inputs}
    res = s.run(outs, feed)
    shapes = {o: list(r.shape) for o, r in zip(outs, res)}
    del s
    tmp.unlink(missing_ok=True)
    final = build(shapes)
    onnx.save(final, str(P["final"]))
    info = {"src": str(src), "nodes": len(keep), "initializers": len(inits), "inputs": [v.name for v in inputs],
            "outputs": len(outs), "taps": taps, "shapes": shapes, "mib": round(P["final"].stat().st_size / 2**20, 1)}
    P["taps"].write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  切り出し {P['final'].name}: 演算 {len(keep)}・重みなど {len(inits)}・入力 {len(inputs)}・出力 {len(outs)}"
          f"（{info['mib']}MiB）", flush=True)
    return info


def convert(variant: str, redo: bool) -> dict:
    from app.pipeline import verify_artifacts, wrap_epctx
    from app.pipeline.build_vit_ctx import run_streaming
    P = paths(variant)
    bins = sorted((P["build"]).glob("ctx_*/*.bin")) if P["build"].exists() else []
    if (not redo and P["epctx"].exists() and bins and P["epctx"].stat().st_mtime > P["final"].stat().st_mtime):
        print(f"  変換は済んでいる（{P['epctx'].name}）", flush=True)
        return {"skipped": True, "ctx_bin": str(bins[0])}
    for old in bins:
        old.unlink(missing_ok=True)
    if not Path(GIT_BASH).exists():
        raise SystemExit(f"Git Bash が無い: {GIT_BASH}")
    env = {**os.environ, "QAIRT_INT4_PATCH": "1"}          # 文章グラフの重みは 4bit（build_t16 と同じ）
    t0 = time.time()
    rc = run_streaming([GIT_BASH, "scripts/build_context.sh", str(P["final"]), str(P["build"]), P["name"]], P["log"], env)
    bins = sorted((P["build"]).glob("ctx_*/*.bin"))
    out = {"sec": round(time.time() - t0, 1), "rc": rc}
    if rc != 0 or not bins:
        raise SystemExit(f"{variant}: 変換に失敗（終了コード {rc}）。{P['build']} の *_convert.log / *_gen.log を見ること")
    cmdlog = P["build"] / f"{P['name']}_commands.txt"
    cmds = cmdlog.read_text(encoding="utf-8", errors="replace").splitlines() if cmdlog.exists() else []
    try:
        ver = verify_artifacts.verify_all(P["final"], P["build"] / f"{P['name']}_meta.json", cmds)
        out["verify"] = [{"check": x["check"], "pass": bool(x.get("pass"))} for x in ver["checks"]]
        bad = [x["check"] for x in out["verify"] if not x["pass"]]
        print(f"  検査: {'全部合格' if not bad else '★不合格 ' + ', '.join(bad)}（探り用なので記録だけ）", flush=True)
    except Exception as e:                                   # 探り用。検査が動かなくても先へ進む
        out["verify_error"] = str(e)[:300]
    info = wrap_epctx.run(bins[0], P["final"], P["epctx"], P["name"], embed=False)
    out.update({"ctx_bin": str(bins[0]), "ctx_mib": round(bins[0].stat().st_size / 2**20, 2), "epctx_bytes": info["bytes"]})
    print(f"  変換 {out['sec']:.0f}s、context binary {out['ctx_mib']}MiB", flush=True)
    return out


def rewrap(variant: str) -> dict:
    """EPContext を、変換結果（context binary のメタデータ）の入出力どおりに包み直す。

    ★QAIRT は「x² → 平均 → +eps → √ → 1/x → ×」の並びを 1 つの RmsNorm にまとめる。まとめた所の途中の値は、
      出力に指定していても**黙って捨てる**（2026-10-01: taps の 165 本のうち 30 本が消え、端末で
      「ExecuteGraph Inconsistent output sizes」で止まった）。そこで、実際にできたグラフの出力だけを宣言し直す"""
    import onnx
    from onnx import TensorProto, helper
    from app.pipeline import wrap_epctx
    P = paths(variant)
    meta = json.loads((P["build"] / f"{P['name']}_meta.json").read_text(encoding="utf-8"))
    g = meta["info"]["graphs"][0]["info"]

    def vi(t):
        info = t["info"]
        if info["dataType"] != "QNN_DATATYPE_FLOAT_32":
            raise SystemExit(f"{variant}: {info['name']} の型が float32 でない（{info['dataType']}）")
        return helper.make_tensor_value_info(info["name"], TensorProto.FLOAT, list(info["dimensions"]))
    ins, outs = [vi(t) for t in g["graphInputs"]], [vi(t) for t in g["graphOutputs"]]
    iop = P["dir"] / f"{P['name']}_io.onnx"
    onnx.save(helper.make_model(helper.make_graph([], P["name"], ins, outs)), str(iop))
    bins = sorted(P["build"].glob("ctx_*/*.bin"))
    wrap_epctx.run(bins[0], iop, P["epctx"], P["name"], embed=False)
    want = json.loads(P["taps"].read_text(encoding="utf-8"))
    asked = list(CUT_OUTPUTS) + [t["tensor"] for t in want["taps"] if t["tensor"] not in CUT_OUTPUTS]
    got = {o.name for o in outs}
    missing = [x for x in asked if x not in got]
    if missing:
        print(f"  {variant}: 変換で {len(missing)} 本の出力が消えた（RmsNorm などにまとめられた所）。できた {len(outs)} 本で包み直した", flush=True)
    # 変換後のグラフ（DLC）に RmsNorm（まとめた正規化）がいくつあるか（演算の種類の名前が文字列で入っている）
    dlc = P["build"] / f"{P['name']}.dlc"
    n_rms = dlc.read_bytes().count(b"RmsNorm") if dlc.exists() else None
    print(f"  {variant}: 変換後の RmsNorm（正規化を 1 つにまとめたもの）{n_rms} 個（切り出した部分の正規化は 12 個）", flush=True)
    return {"qnn_outputs": len(outs), "missing": missing, "rmsnorm_in_dlc": n_rms}


# ---------------------------------------------------------------------------
# 入力（評価と同じプロンプト・画像の特徴。past は 16 トークン版 plain を CPU で回して作る）
# ---------------------------------------------------------------------------
def graph_inputs(path: Path) -> dict:
    """グラフの入力の名前と形"""
    import onnx
    mo = onnx.load(str(path), load_external_data=False)
    return {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in mo.graph.input}


def make_feeds(item: str, tabs: Tables) -> dict:
    items = load_items()
    if item not in items:
        raise SystemExit(f"評価の job に {item} が無い")
    ids = old_prompt(items[item]["tail"])
    feats = np.fromfile(items[item]["feat"], np.float32).reshape(-1, tabs.H)
    img_idx, k = {}, 0
    for p, t in enumerate(ids):
        if t == IMG:
            img_idx[p] = k
            k += 1
    row = lambda p: (ids[p], feats[img_idx[p]] if p in img_idx else None)
    cpu = Cpu(paths("t16_plain")["final"])
    f16 = Feeder(cpu.ins, tabs)
    hist = {}                                                # ("k"|"v", 層) → [1,1,N,hd]（CPU で作った K/V の行）
    feeds16, meta16 = [], []
    for c in range(N_CHUNKS):
        f16.reset()
        n0 = c * 16
        for nm in f16.kv:                                    # past_k_0 / past_v_0（240 スロット、右詰め）
            kv, ly = nm.split("_")[1], int(nm.split("_")[2])
            h = hist.get((kv, ly))
            if h is not None and h.shape[2]:
                take = h[:, :, -f16.P:, :]
                f16.past[nm][:, :, f16.P - take.shape[2]:, :] = take
        f16.n_valid, f16.abs_pos = min(n0, f16.P), n0
        rows = [row(p) for p in range(n0, n0 + 16)]
        fd = f16.feed(rows)
        feeds16.append(fd)
        meta16.append({"chunk": c, "start": n0, "image_rows": [r for r in range(16) if ids[n0 + r] == IMG]})
        o = cpu.run(fd)
        for nm in ("present_k_0", "present_v_0", "present_k_1", "present_v_1"):
            kv, ly = nm.split("_")[1], int(nm.split("_")[2])
            new = np.asarray(o[nm], np.float32)[:, :, -16:, :]          # 16 トークン版の層 0・1 は新しい 16 行
            hist[(kv, ly)] = new if (kv, ly) not in hist else np.concatenate([hist[(kv, ly)], new], axis=2)
    out = {"ids": ids[:N_CHUNKS * 16], "t16": feeds16, "t16_meta": meta16, "t1": [], "t1_meta": []}
    if paths("t1_plain")["final"].exists():
        f1 = Feeder(graph_inputs(paths("t1_plain")["final"]), tabs)
        for c, r in T1_ROWS:
            p = c * 16 + r
            f1.reset()
            for nm in f1.kv:                                 # past_k/v_0・past_k/v_1（256 スロット、右詰め。中身は位置 0〜p−1）
                kv, ly = nm.split("_")[1], int(nm.split("_")[2])
                if p:
                    f1.past[nm][:, :, f1.P - p:, :] = hist[(kv, ly)][:, :, :p, :]
            f1.n_valid, f1.abs_pos = p, p
            out["t1"].append(f1.feed([row(p)]))
            out["t1_meta"].append({"chunk": c, "row": r, "pos": p, "image": ids[p] == IMG})
    return out


# ---------------------------------------------------------------------------
# 比べる
# ---------------------------------------------------------------------------
def rel(a, b) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    nb = np.linalg.norm(b)
    if nb < 1e-12:
        return 0.0 if np.linalg.norm(a) < 1e-12 else float("inf")
    return float(np.linalg.norm(a - b) / nb)


def row_axis(shape: list, T: int):
    if T == 1:
        return None
    if len(shape) > 1 and shape[1] == T:
        return 1
    if len(shape) > 2 and shape[2] == T:
        return 2
    return None


def compare(variant: str, npu: list, cpu: list, info: dict) -> list:
    """出力ごと・回ごとの相対 L2。16 トークン版は行ごとも"""
    T = 16 if variant.startswith("t16") else 1
    rows = []
    for b, (on, oc) in enumerate(zip(npu, cpu)):
        for name, ref in oc.items():
            if name not in on:
                continue
            a, r = np.asarray(on[name]), np.asarray(ref)
            if a.shape != r.shape:
                rows.append({"b": b, "out": name, "error": f"形が違う {a.shape} {r.shape}"})
                continue
            rec = {"b": b, "out": name, "rel": round(rel(a, r), 6)}
            if name.startswith("present_") and T == 1:       # 1 行版の present は窓全体: 新しい行（最後）だけ
                rec["rel"] = round(rel(a[:, :, -1:, :], r[:, :, -1:, :]), 6)
            ax = row_axis(list(r.shape), T)
            if ax is not None:
                rec["rows"] = [round(rel(np.take(a, [i], axis=ax), np.take(r, [i], axis=ax)), 6) for i in range(T)]
            rows.append(rec)
    return rows


def table(res: dict, feeds: dict) -> tuple[list, str]:
    """途中の値ごとに: 16 行版（文字の行・画像の行）と 1 行版（文字・画像）の相対 L2 の中央値を並べる"""
    def load_taps(v):
        p = paths(v)["taps"]
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"taps": [], "shapes": {}}
    t16, t1 = res.get("t16_taps"), res.get("t1_taps")
    if not t16:
        return [], ""
    j16, j1 = load_taps("t16_taps"), load_taps("t1_taps")
    taps16 = j16["taps"]
    taps1 = {t["node"]: t for t in j1["taps"]}

    def comparable(t16tap, t1tap) -> bool:
        """版をまたいで比べてよいか: 1 行版の形が「16 行版の形の行の軸を 1 にしたもの」と同じとき。
        K/V の窓の組み立て（16 行版は past 240＋新 16、1 行版は past 256 をずらす）は形が違うので自然に外れる"""
        s16, s1 = j16["shapes"].get(t16tap["tensor"]), j1["shapes"].get(t1tap["tensor"])
        if s16 is None or s1 is None:
            return False
        want = list(s16)
        ax = row_axis(s16, 16)
        if ax is not None:
            want[ax] = 1
        return list(s1) == want
    by16 = {}
    for r in t16["compare"]:
        by16.setdefault(r["out"], []).append(r)
    by1 = {}
    for r in (t1 or {}).get("compare", []):
        by1.setdefault(r["out"], []).append(r)
    img_rows = {m["chunk"]: set(m["image_rows"]) for m in feeds["t16_meta"]}
    lines, out = [], []
    for t in taps16:
        recs = by16.get(t["tensor"], [])
        txt, img = [], []
        for r in recs:
            if "rows" in r:
                for i, v in enumerate(r["rows"]):
                    (img if i in img_rows.get(r["b"], set()) else txt).append(v)
            elif "rel" in r:
                txt.append(r["rel"])
        t1t, t1i = [], []
        same = taps1.get(t["node"])
        if same is not None and comparable(t, same):
            for r in by1.get(same["tensor"], []):
                if "rel" in r:
                    (t1i if feeds["t1_meta"][r["b"]]["image"] else t1t).append(r["rel"])
        md = lambda xs: float(np.median(xs)) if xs else None
        fused = t["tensor"] in set(t16.get("missing_on_npu", []))
        row = {"node": t["node"], "op": t["op"], "label": t["label"] + ("（NPU では RmsNorm にまとめられ出力なし）" if fused else ""),
               "t16_text": md(txt), "t16_image": md(img), "t1_text": md(t1t), "t1_image": md(t1i), "fused_on_npu": fused}
        out.append(row)
    f = lambda x: "—" if x is None else f"{x * 100:.2f}%"
    lines.append("| # | 演算 | 何の計算 | 16 行版・文字の行 | 16 行版・画像の行 | 1 行版・文字 | 1 行版・画像 |")
    lines.append("|---:|---|---|---:|---:|---:|---:|")
    for i, r in enumerate(out):
        lines.append(f"| {i} | {r['node']} | {r['label']} | {f(r['t16_text'])} | {f(r['t16_image'])} | {f(r['t1_text'])} | {f(r['t1_image'])} |")
    return out, "\n".join(lines)


def first_jump(rows: list) -> dict | None:
    """16 行版だけが大きくずれる最初の計算（1 行版の 3 倍以上、かつ 1% 以上）"""
    for r in rows:
        for k16, k1 in (("t16_text", "t1_text"), ("t16_image", "t1_image")):
            a, b = r.get(k16), r.get(k1)
            if a is not None and b is not None and a >= 0.01 and a >= 3 * max(b, 1e-4):
                return {**r, "which": k16}
    return None


# ---------------------------------------------------------------------------
# 本体
# ---------------------------------------------------------------------------
def cmd_build(variants: list, redo: bool, rec: dict):
    for v in variants:
        print(f"\n== {v}: 切り出し", flush=True)
        P = paths(v)
        if P["final"].exists() and P["taps"].exists() and not redo:
            print(f"  切り出しは済んでいる（{P['final'].name}）", flush=True)
            info = json.loads(P["taps"].read_text(encoding="utf-8"))
        else:
            info = cut(v)
        rec.setdefault("build", {})[v] = {"cut": {k: info[k] for k in ("src", "nodes", "initializers", "inputs", "outputs", "mib")}}
        print(f"== {v}: 変換（QAIRT）", flush=True)
        try:
            rec["build"][v]["convert"] = convert(v, redo)
        except SystemExit as e:                              # 1 つ失敗しても残りは作る（taps だけ変換できない、などがありうる）
            print(f"  ★{e}", flush=True)
            rec["build"][v]["convert"] = {"error": str(e)}


def cmd_device(variants: list, item: str, rec: dict):
    tabs = Tables(TABLES)
    feeds = make_feeds(item, tabs)
    rec["item"] = item
    rec["t16_meta"], rec["t1_meta"] = feeds["t16_meta"], feeds["t1_meta"]
    D.ensure_app()
    D.adb("shell", f"am force-stop {D.PKG}")
    time.sleep(2.0)
    print(f"  端末の空きメモリ {D.wait_mem()}MiB", flush=True)
    IO.mkdir(parents=True, exist_ok=True)
    res = rec.setdefault("device", {})
    for v in variants:
        P = paths(v)
        if not P["epctx"].exists():
            print(f"  {v}: 変換したものが無いので飛ばす（build を先に）", flush=True)
            continue
        fl = feeds["t16"] if v.startswith("t16") else feeds["t1"]
        if not fl:
            print(f"  {v}: 入力が無いので飛ばす", flush=True)
            continue
        ids = [f"b{i}" for i in range(len(fl))]
        print(f"\n== {v}: PC の CPU と端末の NPU（{len(fl)} 回ぶん）", flush=True)
        rw = rewrap(v)
        cpu = Cpu(P["final"])
        oc = [cpu.run(fd) for fd in fl]
        binf = sorted(P["build"].glob("ctx_*/*.bin"))[0]
        rel_model = f"models/l0probe/{P['epctx'].name}"
        D.push_many([(P["epctx"], rel_model), (binf, f"models/l0probe/{binf.name}")], f"{v} のモデル")
        dev = Device(rel_model, v)
        on, raw = dev.run(fl, ids, keep=False)
        cmp_rows = compare(v, on, oc, {})
        res[v] = {"device": dev.first, "ort_run_ms": [o.get("_ort_run_ms") for o in on], "compare": cmp_rows,
                  "missing_on_npu": rw["missing"], "rmsnorm_in_dlc": rw["rmsnorm_in_dlc"]}
        np.savez(IO / f"{v}_npu.npz", **{f"b{b}__{k}": np.asarray(x) for b, o in enumerate(on) for k, x in o.items() if not k.startswith("_")})
        np.savez(IO / f"{v}_cpu.npz", **{f"b{b}__{k}": np.asarray(x) for b, o in enumerate(oc) for k, x in o.items()})
        # present の要点（小さいグラフでも同じずれが出るか）
        for nm in CUT_OUTPUTS:
            rs = [r for r in cmp_rows if r["out"] == nm and "rel" in r]
            if not rs:
                continue
            if v.startswith("t16"):
                txt = [x for r in rs for i, x in enumerate(r.get("rows", [])) if i not in set(feeds["t16_meta"][r["b"]]["image_rows"])]
                img = [x for r in rs for i, x in enumerate(r.get("rows", [])) if i in set(feeds["t16_meta"][r["b"]]["image_rows"])]
            else:
                txt = [r["rel"] for r in rs if not feeds["t1_meta"][r["b"]]["image"]]
                img = [r["rel"] for r in rs if feeds["t1_meta"][r["b"]]["image"]]
            md = lambda xs: f"{np.median(xs) * 100:.2f}%" if xs else "—"
            print(f"    {nm}: 文字の行 {md(txt)}・画像の行 {md(img)}", flush=True)
    D.adb("shell", f"am force-stop {D.PKG}")
    rows, md_table = table(res, feeds)
    rec["table"] = rows
    rec["first_jump"] = first_jump(rows)
    if md_table:
        (RES / "l0probe_table.md").write_text(
            f"# 層 0 の途中の値: 端末の NPU と PC の CPU の差（相対 L2 の中央値、{item}）\n\n{md_table}\n", encoding="utf-8")
    return feeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="all", choices=["all", "build", "device"])
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--item", default="syn_nav_S")
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    C.utf8_stdout()
    variants = [v for v in a.variants.split(",") if v]
    bad = [v for v in variants if v not in VARIANTS]
    if bad:
        raise SystemExit(f"--variants は {VARIANTS} から（{bad}）")
    RES.mkdir(parents=True, exist_ok=True)
    out_path = RES / "l0probe.json"
    rec = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    t0 = time.time()
    if a.cmd in ("all", "build"):
        cmd_build(variants, a.redo, rec)
        C.write_json_atomic(out_path, rec)
    if a.cmd in ("all", "device"):
        cmd_device(variants, a.item, rec)
        C.write_json_atomic(out_path, rec)
        fj = rec.get("first_jump")
        if rec.get("table"):
            print("\n  ■ 途中の値（1% 以上ずれたもの。全部は results/P5_t16/l0probe_table.md）")
            f = lambda x: "  —  " if x is None else f"{x * 100:6.2f}%"
            print("      演算              何の計算                      16行・文字  16行・画像   1行・文字   1行・画像")
            for r in rec["table"]:
                if any((r[k] or 0) >= 0.01 for k in ("t16_text", "t16_image", "t1_text", "t1_image")):
                    print(f"      {r['node'][:16]:16s}  {r['label'][:26]:26s}  {f(r['t16_text'])}   {f(r['t16_image'])}   "
                          f"{f(r['t1_text'])}   {f(r['t1_image'])}")
        if fj:
            print(f"\n  ■ 16 行版だけが大きくずれる最初の計算: {fj['node']}（{fj['label']}）"
                  f" 16 行版 {fj[fj['which']] * 100:.2f}%、1 行版 {(fj[fj['which'].replace('t16', 't1')] or 0) * 100:.2f}%")
        elif rec.get("table"):
            print("\n  ■ 16 行版だけが大きくずれる計算は見つからなかった（小さいグラフでは再現しない可能性。plain の present を見る）")
    print(f"\n  （{(time.time() - t0) / 60:.1f} 分）-> {out_path}")


if __name__ == "__main__":
    main()
