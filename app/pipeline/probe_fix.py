# -*- coding: utf-8 -*-
"""NPU のずれの直し方を、層 0 だけの小さいグラフで試す（2026-10-01、切り分けの段 3）。

    python -m app.pipeline.probe_fix                     # 既定の 9 版を作る（変換 1 版 1〜2 分）→ 端末で比べる
    python -m app.pipeline.probe_fix build               # 作るだけ
    python -m app.pipeline.probe_fix device              # 端末で比べるだけ
    python -m app.pipeline.probe_fix --variants t16_fx_base,t16_fx_normmm

**EdgeSLM_Android の venv（$PY2）で動かす**。前提: `probe_l0 build` で作った小さいグラフ（l0p_t16_plain・l0p_t1_plain）がある。

## ここまでで分かった 2 つの原因（probe_l0 の taps・probe_rows、2026-10-01）

| 原因 | 中身 | 効く版 |
|---|---|---|
| ① **RmsNorm にまとめた正規化がずれる** | QAIRT は「x² → 平均 → +eps → √ → 1/x → ×」の並びを HTP の RmsNorm 1 つにまとめる。まとめると画像の行の層 0 の K が 32〜39% ずれ、ばらしたままなら 4%。注意の後の正規化も入力 10% → 出力 45〜49% | 1 トークン版・16 トークン版 |
| ② **[16 行 × 256] の tanh が 3 行目以降で壊れる** | 層ごとの埋め込み（PLE）の gate の GELU の tanh。0・1 行目は差 10% 前後、2〜15 行目は 90〜360%（1 トークン版は 11%） | 16 トークン版だけ |

## 試す直し方（どれも計算の中身は同じ。書き方だけ変えて、NPU の問題のある部品を避ける）

| 名前 | 書き換え | ねらい |
|---|---|---|
| `normmm` | 正規化の「平均（ReduceMean）」を「1/N を並べた重みとの行列の掛け算（MatMul）」に置き換える | RmsNorm の並びにならないので、まとめられない（行列の掛け算は NPU で正確） |
| `mulsq` | 正規化の「x²（Pow）」を「x × x（Mul）」に置き換える | 同上（もっと軽い書き換え。まとめを避けられるかを確かめる） |
| `tanhflat` | PLE の tanh の前後で形を [1, 16×256] の 1 行に変え、tanh を 1 行で計算する | 1 行の tanh は正常（1 トークン版）なので、壊れる形を避ける |
| `tanhsig` | PLE の tanh を「2・sigmoid(2x) − 1」に置き換える（tanh と同じ値） | tanh の部品そのものを使わない |

試す版: 16 トークン版は base（書き換えなし）・normmm・mulsq・tanhflat・tanhsig・normmm＋tanhflat・normmm＋tanhsig、
1 トークン版は base・normmm。出力は層 0・1 の K/V と、途中の 3 か所（注意の後の正規化・PLE の GELU・層 0 の出力）。

比べる相手は**書き換え前のグラフを PC の CPU で動かした値**（＝元の量子化済みグラフ。これが HF に近い）。
書き換えたグラフを CPU で動かした値とも比べ、書き換えで計算が変わっていないことを確かめる。

結果: `results/P5_t16/probe_fix.json`
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from app.pipeline import device_vit as D
from app.pipeline import vlm_common as C
from app.pipeline.diag_t16 import TABLES, Tables
from app.pipeline.probe_l0 import IO, compare, convert, make_feeds, paths, rewrap
from app.pipeline.probe_t16 import Cpu, Device

RES = C.RESULTS / "P5_t16"
FIXES = ("normmm", "mulsq", "tanhflat", "tanhsig")
DEFAULT = ["t16_fx_base", "t16_fx_normmm", "t16_fx_mulsq", "t16_fx_tanhflat", "t16_fx_tanhsig",
           "t16_fx_normmm_tanhflat", "t16_fx_normmm_tanhsig", "t1_fx_base", "t1_fx_normmm"]
# 途中で見る 3 か所（演算の名前。16 トークン版・1 トークン版で同じ）
WATCH = {"node_mul_17": "注意の後の正規化", "node_mul_34": "PLE の GELU", "node_mul_38": "層 0 の出力"}
PRESENTS = ["present_k_0", "present_v_0", "present_k_1", "present_v_1"]


def parse(variant: str) -> tuple[str, list[str]]:
    kind, rest = variant.split("_fx_")
    fx = [] if rest == "base" else rest.split("_")
    bad = [f for f in fx if f not in FIXES]
    if kind not in ("t16", "t1") or bad:
        raise SystemExit(f"版の名前が読めない: {variant}（例 t16_fx_normmm_tanhflat）")
    return kind, fx


# ---------------------------------------------------------------------------
# QDQ のグラフの小さな道具
# ---------------------------------------------------------------------------
class Graph:
    def __init__(self, m, batch: bool = False):
        self.m, self.g = m, m.graph
        self.n_new = 0
        # batch=True: 置き換えをためておき、commit() で 1 回だけ並べ直す（全体のグラフは演算が 2 万を超えるので、1 つずつ並べ直すと遅い）
        self.batch = batch
        self.pending: dict[str, list] = {}

    def refresh(self):
        self.prod = {o: n for n in self.g.node for o in n.output}
        self.cons: dict[str, list] = {}
        for n in self.g.node:
            for x in n.input:
                self.cons.setdefault(x, []).append(n)
        self.inits = {t.name: t for t in self.g.initializer}

    def name(self, base: str) -> str:
        self.n_new += 1
        return f"fx{self.n_new}_{base}"

    def const(self, base: str, arr: np.ndarray) -> str:
        from onnx import numpy_helper
        nm = self.name(base)
        self.g.initializer.append(numpy_helper.from_array(np.asarray(arr), nm))
        return nm

    def value(self, nm: str) -> np.ndarray:
        from onnx import numpy_helper
        return numpy_helper.to_array(self.inits[nm])

    def enc_of_dq(self, dq_out: str) -> tuple[str, str]:
        """DQ の出力名 → (scale, zero_point) の初期値の名前"""
        n = self.prod[dq_out]
        assert n.op_type == "DequantizeLinear", (dq_out, n.op_type)
        return n.input[1], n.input[2]

    def q_after(self, t: str):
        """テンソル t を量子化する Q（t → Q）"""
        qs = [n for n in self.cons.get(t, []) if n.op_type == "QuantizeLinear"]
        assert len(qs) == 1, (t, len(qs))
        return qs[0]

    def tap_of(self, node_name: str) -> str:
        """演算の出力 → Q → DQ の DQ の出力名"""
        n = next(x for x in self.g.node if x.name == node_name)
        q = self.q_after(n.output[0])
        dq = [x for x in self.cons.get(q.output[0], []) if x.op_type == "DequantizeLinear"]
        return dq[0].output[0]

    def qdq(self, t: str, scale: str, zp: str, base: str) -> tuple[list, str]:
        """t → Q → DQ の 2 つの演算と、DQ の出力名"""
        from onnx import helper
        q_out, dq_out = self.name(base + "_q"), self.name(base + "_dq")
        return [helper.make_node("QuantizeLinear", [t, scale, zp], [q_out], name=self.name(base + "_Q")),
                helper.make_node("DequantizeLinear", [q_out, scale, zp], [dq_out], name=self.name(base + "_DQ"))], dq_out

    def replace_node(self, old, new_nodes: list):
        """old を new_nodes に置き換える（並びは old の位置に）"""
        if self.batch:
            self.pending[old.name] = new_nodes
            return
        nodes = list(self.g.node)
        i = next(k for k, n in enumerate(nodes) if n.name == old.name)
        del self.g.node[:]
        self.g.node.extend(nodes[:i] + new_nodes + nodes[i + 1:])

    def commit(self):
        """ためた置き換えを 1 回で反映する"""
        if not self.pending:
            return
        seen: dict[str, int] = {}
        for n in self.g.node:
            if n.name in self.pending:
                seen[n.name] = seen.get(n.name, 0) + 1
        dup = [k for k, v in seen.items() if v != 1]
        if dup or len(seen) != len(self.pending):
            raise SystemExit(f"置き換える演算の名前が一意でない・見つからない: {dup[:5]}")
        out = []
        for n in self.g.node:
            out.extend(self.pending.get(n.name, [n]))
        del self.g.node[:]
        self.g.node.extend(out)
        self.pending = {}


def fix_normmm(G: Graph) -> int:
    """ReduceMean(x², axes=[-1], keepdims=1) → MatMul(x², W)。W は [N,1] で全部 1/N（int8 で 127、scale 1/(127N)）"""
    from onnx import helper
    G.refresh()
    k = 0
    for n in [x for x in G.g.node if x.op_type == "ReduceMean"]:
        src = n.input[0]
        axes = G.value(n.input[1]) if len(n.input) > 1 else None
        keep = next((a.i for a in n.attribute if a.name == "keepdims"), 1)
        if axes is None or list(axes.reshape(-1)) != [-1] or keep != 1:
            continue
        N = SHAPES[src][-1]
        wq = G.const("meanw_q", np.full((N, 1), 127, np.int8))
        ws = G.const("meanw_scale", np.array(1.0 / (127.0 * N), np.float32))
        wz = G.const("meanw_zp", np.array(0, np.int8))
        wdq = G.name("meanw_dq")
        new = [helper.make_node("DequantizeLinear", [wq, ws, wz], [wdq], name=G.name("meanw_DQ")),
               helper.make_node("MatMul", [src, wdq], [n.output[0]], name=G.name("mean_mm"))]
        G.replace_node(n, new)
        k += 1
    G.commit()
    return k


def fix_mulsq(G: Graph) -> int:
    """Pow(x, 2) → Mul(x, x)（正規化の x² だけ。指数が 2 の Pow）"""
    from onnx import helper
    G.refresh()
    k = 0
    for n in [x for x in G.g.node if x.op_type == "Pow"]:
        e = n.input[1]
        dq = G.prod.get(e)
        val = None
        if dq is not None and dq.op_type == "DequantizeLinear" and dq.input[0] in G.inits:
            q, s, z = (G.value(x).astype(np.float64) for x in dq.input)
            val = float(((q - z) * s).reshape(-1)[0])
        if val is None or abs(val - 2.0) > 1e-3:
            continue
        G.replace_node(n, [helper.make_node("Mul", [n.input[0], n.input[0]], [n.output[0]], name=G.name("sq_mul"))])
        k += 1
    G.commit()
    return k


def ple_tanh_nodes(G: Graph) -> list:
    """PLE の gate の GELU の tanh（入力の最後の次元が 256）"""
    G.refresh()
    return [n for n in G.g.node if n.op_type == "Tanh" and SHAPES.get(n.input[0], [0])[-1] == 256]


def fix_tanhflat(G: Graph) -> int:
    """x → Reshape[1,1,T×256] → Q/DQ → Tanh → Q/DQ → Reshape[元の形] → （元の tanh の出力の名前）"""
    from onnx import helper
    k = 0
    for n in ple_tanh_nodes(G):
        x, y = n.input[0], n.output[0]
        shp = SHAPES[x]
        flat = G.const("flat_shape", np.array([1, 1, int(np.prod(shp))], np.int64))
        back = G.const("back_shape", np.array(shp, np.int64))
        sx, zx = G.enc_of_dq(x)
        qy = G.q_after(y)
        sy, zy = qy.input[1], qy.input[2]
        xr = G.name("tanh_in_flat")
        nodes = [helper.make_node("Reshape", [x, flat], [xr], name=G.name("flat_R"))]
        qd, xr_dq = G.qdq(xr, sx, zx, "tanh_in_flat")
        nodes += qd
        yr = G.name("tanh_out_flat")
        nodes.append(helper.make_node("Tanh", [xr_dq], [yr], name=G.name("tanh_flat")))
        qd, yr_dq = G.qdq(yr, sy, zy, "tanh_out_flat")
        nodes += qd
        nodes.append(helper.make_node("Reshape", [yr_dq, back], [y], name=G.name("back_R")))
        G.replace_node(n, nodes)
        k += 1
    G.commit()
    return k


def fix_tanhsig(G: Graph) -> int:
    """tanh(x) = 2·sigmoid(2x) − 1。新しいテンソルの目盛り: 2x は x の 2 倍、sigmoid は [0,1]、2·sigmoid は [0,2]（16bit）"""
    from onnx import helper
    k = 0
    for n in ple_tanh_nodes(G):
        x, y = n.input[0], n.output[0]
        sx, zx = G.enc_of_dq(x)
        sxv = float(G.value(sx))
        u16 = lambda v: np.array(v, np.uint16)
        c2q, c2s, c2z = G.const("c2_q", u16(65535)), G.const("c2_s", np.array(2.0 / 65535, np.float32)), G.const("c2_z", u16(0))
        m1q, m1s, m1z = G.const("m1_q", u16(0)), G.const("m1_s", np.array(1.0 / 65535, np.float32)), G.const("m1_z", u16(65535))
        c2, m1 = G.name("c2_dq"), G.name("m1_dq")
        nodes = [helper.make_node("DequantizeLinear", [c2q, c2s, c2z], [c2], name=G.name("c2_DQ")),
                 helper.make_node("DequantizeLinear", [m1q, m1s, m1z], [m1], name=G.name("m1_DQ"))]
        a = G.name("x2")
        nodes.append(helper.make_node("Mul", [x, c2], [a], name=G.name("x2_mul")))
        qd, a_dq = G.qdq(a, G.const("x2_s", np.array(2.0 * sxv, np.float32)), zx, "x2")
        nodes += qd
        s = G.name("sig")
        nodes.append(helper.make_node("Sigmoid", [a_dq], [s], name=G.name("sigmoid")))
        qd, s_dq = G.qdq(s, G.const("sig_s", np.array(1.0 / 65535, np.float32)), G.const("sig_z", u16(0)), "sig")
        nodes += qd
        b2 = G.name("sig2")
        nodes.append(helper.make_node("Mul", [s_dq, c2], [b2], name=G.name("sig2_mul")))
        qd, b_dq = G.qdq(b2, G.const("sig2_s", np.array(2.0 / 65535, np.float32)), G.const("sig2_z", u16(0)), "sig2")
        nodes += qd
        nodes.append(helper.make_node("Add", [b_dq, m1], [y], name=G.name("tanh_from_sig")))
        G.replace_node(n, nodes)
        k += 1
    G.commit()
    return k


SHAPES: dict = {}


def build_variant(variant: str) -> dict:
    """書き換えた小さいグラフを作る（元: probe_l0 の l0p_{t16|t1}_plain。出力に途中の 3 か所を足す）"""
    import onnx
    import onnxruntime as ort
    from onnx import TensorProto, helper
    global SHAPES
    kind, fx = parse(variant)
    src = paths(f"{kind}_plain")["final"]
    if not src.exists():
        raise SystemExit(f"{src} が無い。先に probe_l0 build --variants {kind}_plain")
    m = onnx.load(str(src))
    G = Graph(m)
    G.refresh()
    # 形（書き換えに使う）: 元のグラフを ORT で 1 回回し、全部の途中の値の形を取る
    tmp_out = [o for n in G.g.node for o in n.output]
    probe = onnx.load(str(src))
    del probe.graph.output[:]
    probe.graph.output.extend([helper.make_tensor_value_info(o, TensorProto.FLOAT, None) for o in tmp_out
                               if G.prod[o].op_type not in ("QuantizeLinear",)])
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    s = ort.InferenceSession(probe.SerializeToString(), so, providers=["CPUExecutionProvider"])
    feed = {v.name: np.zeros([d.dim_value for d in v.type.tensor_type.shape.dim], np.float32) for v in G.g.input}
    names = [o.name for o in probe.graph.output]
    SHAPES = {k_: list(v.shape) for k_, v in zip(names, s.run(names, feed))}
    del s
    # 途中の 3 か所を出力に足す
    watch = {nn: G.tap_of(nn) for nn in WATCH}
    counts = {}
    for f in fx:
        counts[f] = {"normmm": fix_normmm, "mulsq": fix_mulsq, "tanhflat": fix_tanhflat, "tanhsig": fix_tanhsig}[f](G)
        if counts[f] == 0:
            raise SystemExit(f"{variant}: 書き換え {f} を当てる所が無い")
    outs = PRESENTS + list(watch.values())
    del G.g.output[:]
    G.g.output.extend([helper.make_tensor_value_info(o, TensorProto.FLOAT, None) for o in outs])
    s = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
    shp = {o: list(r.shape) for o, r in zip(outs, s.run(outs, feed))}
    del G.g.output[:]
    G.g.output.extend([helper.make_tensor_value_info(o, TensorProto.FLOAT, shp[o]) for o in outs])
    P = paths(variant)
    P["dir"].mkdir(parents=True, exist_ok=True)
    onnx.save(m, str(P["final"]))
    info = {"src": str(src), "fixes": fx, "counts": counts, "inputs": [v.name for v in G.g.input],
            "taps": [{"tensor": t, "node": nn, "op": "", "label": WATCH[nn]} for nn, t in watch.items()], "shapes": shp}
    P["taps"].write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  {variant}: 書き換え {counts or 'なし'} → {P['final'].name}", flush=True)
    return info


def dlc_counts(variant: str) -> dict:
    P = paths(variant)
    dlc = P["build"] / f"{P['name']}.dlc"
    if not dlc.exists():
        return {}
    b = dlc.read_bytes()
    return {k: b.count(k.encode()) for k in ("RmsNorm", "ReduceMean", "Tanh", "Sigmoid", "FullyConnected", "MatMul")}


# ---------------------------------------------------------------------------
# 比べる
# ---------------------------------------------------------------------------
def summarize(variant: str, rows: list, feeds: dict, taps: list) -> dict:
    kind, _ = parse(variant)
    tname = {t["tensor"]: t["node"] for t in taps}
    md = lambda xs: round(float(np.median(xs)), 5) if xs else None
    out = {}
    for name in PRESENTS + list(tname):
        rs = [r for r in rows if r["out"] == name and "rel" in r]
        key = tname.get(name, name)
        if kind == "t16":
            img = {m["chunk"]: set(m["image_rows"]) for m in feeds["t16_meta"]}
            txt = [x for r in rs for i, x in enumerate(r.get("rows", [])) if i not in img[r["b"]]]
            imv = [x for r in rs for i, x in enumerate(r.get("rows", [])) if i in img[r["b"]]]
            t01 = [x for r in rs for i, x in enumerate(r.get("rows", [])) if i not in img[r["b"]] and i < 2]
            t2p = [x for r in rs for i, x in enumerate(r.get("rows", [])) if i not in img[r["b"]] and i >= 2]
            out[key] = {"text": md(txt), "image": md(imv), "text_rows01": md(t01), "text_rows2p": md(t2p)}
        else:
            out[key] = {"text": md([r["rel"] for r in rs if not feeds["t1_meta"][r["b"]]["image"]]),
                        "image": md([r["rel"] for r in rs if feeds["t1_meta"][r["b"]]["image"]])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="all", choices=["all", "build", "device"])
    ap.add_argument("--variants", default=",".join(DEFAULT))
    ap.add_argument("--item", default="syn_nav_S")
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    C.utf8_stdout()
    variants = [v for v in a.variants.split(",") if v]
    for v in variants:
        parse(v)
    RES.mkdir(parents=True, exist_ok=True)
    out_path = RES / "probe_fix.json"
    rec = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    t0 = time.time()
    if a.cmd in ("all", "build"):
        for v in variants:
            print(f"\n== {v}", flush=True)
            P = paths(v)
            if not (P["final"].exists() and not a.redo):
                build_variant(v)
            try:
                rec.setdefault("build", {})[v] = convert(v, a.redo)
            except SystemExit as e:
                print(f"  ★{e}", flush=True)
                rec.setdefault("build", {})[v] = {"error": str(e)}
            rec["build"][v]["dlc"] = dlc_counts(v)
            print(f"  変換後の演算: {rec['build'][v]['dlc']}", flush=True)
            C.write_json_atomic(out_path, rec)
    if a.cmd in ("all", "device"):
        tabs = Tables(TABLES)
        feeds = make_feeds(a.item, tabs)
        D.ensure_app()
        D.adb("shell", f"am force-stop {D.PKG}")
        time.sleep(2.0)
        print(f"  端末の空きメモリ {D.wait_mem()}MiB", flush=True)
        IO.mkdir(parents=True, exist_ok=True)
        ref = {}                                             # 書き換え前のグラフ（＋途中の 3 か所）を CPU で → 比べる相手
        for kind in ("t16", "t1"):
            base = f"{kind}_fx_base"
            if any(v.startswith(kind) for v in variants):
                if not paths(base)["final"].exists():
                    build_variant(base)
                cpu = Cpu(paths(base)["final"])
                ref[kind] = [cpu.run(fd) for fd in (feeds["t16"] if kind == "t16" else feeds["t1"])]
        dev_res = rec.setdefault("device", {})
        for v in variants:
            kind, fx = parse(v)
            P = paths(v)
            if not P["epctx"].exists() and not (P["build"].exists() and list(P["build"].glob("ctx_*/*.bin"))):
                print(f"  {v}: 変換したものが無いので飛ばす", flush=True)
                continue
            fl = feeds["t16"] if kind == "t16" else feeds["t1"]
            print(f"\n== {v}（{len(fl)} 回ぶん）", flush=True)
            rw = rewrap(v)
            own = [Cpu(P["final"]).run(fd) for fd in fl]          # 書き換えたグラフを CPU で（書き換えで計算が変わっていないか）
            equiv = max(float(np.linalg.norm(np.asarray(o[k]) - np.asarray(r[k])) / max(np.linalg.norm(np.asarray(r[k])), 1e-12))
                        for o, r in zip(own, ref[kind]) for k in PRESENTS)
            binf = sorted(P["build"].glob("ctx_*/*.bin"))[0]
            rel_model = f"models/l0probe/{P['epctx'].name}"
            D.push_many([(P["epctx"], rel_model), (binf, f"models/l0probe/{binf.name}")], f"{v} のモデル")
            on, _ = Device(rel_model, v).run(fl, [f"b{i}" for i in range(len(fl))], keep=False)
            taps = json.loads(P["taps"].read_text(encoding="utf-8"))["taps"]
            rows = compare(v, on, ref[kind], {})
            sm = summarize(v, rows, feeds, taps)
            dev_res[v] = {"cpu_equiv_max": round(equiv, 6), "missing": rw["missing"], "summary": sm,
                          "ort_run_ms": [o.get("_ort_run_ms") for o in on], "compare": rows}
            f = lambda x: "  —  " if x is None else f"{x * 100:5.1f}%"
            print(f"    書き換えで CPU の計算が変わった量 {equiv * 100:.3f}%（0 に近いはず）", flush=True)
            for k in PRESENTS + list(WATCH):
                x = sm.get(k)
                if x:
                    extra = f"  （文字 0・1 行目 {f(x['text_rows01'])}、2 行目以降 {f(x['text_rows2p'])}）" if kind == "t16" and k == "node_mul_34" else ""
                    print(f"    {WATCH.get(k, k):14s} 文字 {f(x['text'])}  画像 {f(x['image'])}{extra}", flush=True)
            C.write_json_atomic(out_path, rec)
        D.adb("shell", f"am force-stop {D.PKG}")
        # まとめの表
        f = lambda x: "  —  " if x is None else f"{x * 100:5.1f}%"
        print("\n  ■ NPU と元のグラフ（CPU）の差（中央値）")
        print("    版                         K0 文字/画像     K1 文字/画像     V1 画像   注意後の正規化 文字   PLE GELU 2行目以降   RmsNorm")
        for v in variants:
            r = dev_res.get(v)
            if not r:
                continue
            sm = r["summary"]
            g = lambda k, s: sm.get(k, {}).get(s)
            print(f"    {v:26s} {f(g('present_k_0','text'))}/{f(g('present_k_0','image'))}  {f(g('present_k_1','text'))}/{f(g('present_k_1','image'))}"
                  f"  {f(g('present_v_1','image'))}      {f(g('node_mul_17','text'))}           {f(g('node_mul_34','text_rows2p'))}"
                  f"        {rec.get('build', {}).get(v, {}).get('dlc', {}).get('RmsNorm')}")
    C.write_json_atomic(out_path, rec)
    print(f"\n  （{(time.time() - t0) / 60:.1f} 分）-> {out_path}")


if __name__ == "__main__":
    main()
