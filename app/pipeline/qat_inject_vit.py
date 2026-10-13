# -*- coding: utf-8 -*-
"""Phase 2-2: 画像エンコーダの QAT の値を、量子化後の QDQ へ**そのまま**書き込む（重み＋活性の 8bit の丸め）（2026-09-28）。

    python -m app.pipeline.qat_inject_vit --qdq <..._qdq_ort.onnx> --prepped <..._prepped.onnx> --out <出力> [--no-srq]

EdgeSLM_AndroidGemma4 の `qat_inject.py`（md5 42b99bb8…）を複製し、**重みの一覧（qat_linears）だけを画像エンコーダ用に**
書き換えた。書き込みの手順（inject）は同じ（引き継ぎ一覧を参照）。

## やること

1. **重み**: p2b_ops の prepare（spec の `weight_bits: 4`）が ORT に作らせた「INT4・出力チャネルごと」の DQ を見つけ、
   QAT の整数とスケールに置き換える。対応づけは**値で**（prepped の float の重み [in, out] と、QAT から戻した重み [out, in] の転置が
   完全一致するもの）。画像エンコーダの QAT は全部 8bit なので、INT4 から **INT8 に型を替える**
   - 16 層 × 7 本（q/k/v/o・gate/up/down）= 112 本
2. **QAT でない重み**（パッチの埋め込み 768→768・射影層 768→1536）: 値から INT8・出力チャネルごと・対称で作り直す（from_float）。
   活性の丸めは掛けない（QAT の目盛りが無い）
3. **活性の 8bit の丸め（SRQ）**: 112 本の MatMul について
   - 出力の Q/DQ を **uint8（零点 128）・QAT の出力スケール**に書き換える
   - 入力の手前に **uint8（零点 128）・QAT の入力スケール**の Q/DQ を 1 組はさむ（行列積ごとに別の組）
   HTP の uint8（零点 128）は、QAT の int8 対称（−128〜127）と同じ格子。`--no-srq` なら入れない（活性は 16bit のまま）

最後に読み直し、QAT の重みの「整数×スケール」が書き出し前の値と 1 ビットも違わないことを確かめる。

## SRQ あり・なしの 2 版を作る理由

2-1 で、画像エンコーダは SRQ に敏感（計算の順序だけで特徴の cos が 0.84 まで揺れる）が、SRQ なしの特徴でも答えの品質は
同じと分かった（results/P2_20260928_vit_export.md 6 節）。そこで
- **SRQ なしの版**: 揺れが無いので、量子化と NPU の算術の誤差だけを HF（SRQ なし）と比べて測れる
- **SRQ ありの版**: 学習時の形に近い。行列積の入力が 8bit になるので NPU で速い可能性がある
の両方を作り、変換のしやすさ・速さ・答えの品質で端末の版を選ぶ。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from app.pipeline import vlm_common as C


def pack_int4(vals: np.ndarray) -> bytes:
    """ONNX の INT4（符号付き）: 2 つを 1 バイトに、下位 4 ビットが先。（Gemma4 の qat_inject と同じ）"""
    v = (vals.astype(np.int8).reshape(-1) & 0x0F).astype(np.uint8)
    if v.size % 2:
        v = np.concatenate([v, np.zeros(1, np.uint8)])
    return (v[0::2] | (v[1::2] << 4)).astype(np.uint8).tobytes()


# ★出力チャネルごとのスケールの下限（最大のチャネルのスケールに対する比）。2026-09-29 に端末で見つけた（C-14）:
#   パッチの埋め込み（768→768）には、重みがほぼ 0 のチャネルが 222 本ある（max|w| 2e-8〜5e-4、ふつうは 0.03 前後）。
#   max|w|/127 をそのままスケールにすると 4e-6 未満になり、**そのチャネルだけ端末の NPU で値が壊れる**（出力の範囲の端に
#   張り付き、符号もでたらめ）。PC の QDQ（ORT）では壊れないので、端末に持って行くまで分からない。
#   境目は s ≈ 4.0e-6（最大のチャネルの 1/659）で、壊れたのは全部それより小さいチャネル、正常なのは全部それより大きいチャネルだった。
#   NPU は「入力のスケール × 重みのスケール ÷ 出力のスケール」を固定小数点の倍率で持つので、小さすぎると表せないとみられる。
#   下限を最大の 1/64 にする（境目に対して 10 倍の余裕）。そのチャネルの重みは小さいので、整数が 0〜十数になるだけで値はほぼ変わらない
SCALE_FLOOR_RATIO = 1.0 / 64


def _from_float(w: np.ndarray) -> dict:
    """QAT でない float の重み [out, in] を INT8・出力チャネルごと・対称にする（Gemma4 の PLE の射影と同じやり方）。

    スケールには下限（最大のチャネルの SCALE_FLOOR_RATIO 倍）を付ける（C-14。上の注記）。
    """
    raw = np.maximum(np.abs(w).max(axis=1), 1e-12) / 127.0
    floor = raw.max() * SCALE_FLOOR_RATIO
    s = np.maximum(raw, floor)
    q = np.clip(np.rint(w / s[:, None]), -127, 127).astype(np.int8)
    # ★突き合わせは**元の値**で行う（グラフには丸める前の値が入っている）。書き込むのは丸めた整数とスケール
    return {"codes": q, "scale": s.astype(np.float32), "bits": 8, "w": w, "in_act": None, "out_act": None,
            "from_float": True, "n_floored": int((raw < floor).sum())}


def vit_linears() -> dict:
    """名前 → {codes [out,in] int8, scale [out], bits, w [out,in] float32, in_act, out_act[, from_float]}。

    vit_graph.QatFile と同じ読み方（配線の確認に使ったのと同じ値）。
    """
    from app.pipeline.vit_graph import PFX, QatFile
    qf = QatFile()
    out = {}
    for i in range(C.VIT_LAYERS):
        p = f"{PFX}encoder.layers.{i}."
        for n in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                  "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"):
            d = qf.linear(p + n + ".linear")
            out[p + n] = {"codes": d["codes"].numpy(), "scale": d["scale"].numpy().astype(np.float32),
                          "bits": 8, "w": d["w"].numpy(), "in_act": d["in_act"], "out_act": d["out_act"]}
    out[PFX + "patch_embedder.input_proj"] = _from_float(qf.vec(PFX + "patch_embedder.input_proj.weight").numpy())
    out["model.embed_vision.embedding_projection"] = _from_float(
        qf.vec("model.embed_vision.embedding_projection.weight").numpy())
    return out


def _u8_qdq(scale: float):
    return np.float32(scale), np.uint8(128)          # QAT の int8 対称（−128〜127）と同じ格子


def inject(qdq: Path, prepped: Path, lin: dict, out: Path, srq: bool = True) -> dict:
    """（Gemma4 の qat_inject.inject と同じ手順。重みの一覧 lin を引数で受け取るようにした）"""
    by_shape: dict = {}
    for nm, d in lin.items():
        by_shape.setdefault(tuple(d["w"].T.shape), []).append(nm)
    pm = onnx.load(str(prepped))
    p_inits = {t.name: t for t in pm.graph.initializer}
    qm = onnx.load(str(qdq))
    g = qm.graph
    q_inits = {t.name: t for t in g.initializer}
    cons: dict[str, list] = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)

    mapping, written, retyped, srq_in, srq_out = {}, 0, 0, 0, 0
    new_nodes: list[tuple[int, list]] = []          # (MatMul の位置, 前に入れるノード)
    wt_cache: dict = {}                              # 同じ形の候補が多い（768×768 が 65 本）ので転置を使い回す
    for nd in list(g.node):
        if nd.op_type != "DequantizeLinear" or nd.input[0] not in q_inits:
            continue
        x = q_inits[nd.input[0]]
        if x.data_type != TensorProto.INT4:
            continue
        base = nd.input[0][: -len("_quantized")] if nd.input[0].endswith("_quantized") else None
        if base is None or base not in p_inits:
            raise SystemExit(f"prepped に元の重みが無い: {nd.input[0]}")
        wf = numpy_helper.to_array(p_inits[base])                       # [in, out]
        same = [nm for nm in by_shape.get(tuple(wf.shape), []) if nm not in {m["qat"] for m in mapping.values()}]
        cands = []
        for nm in same:
            if nm not in wt_cache:
                wt_cache[nm] = np.ascontiguousarray(lin[nm]["w"].T)
            if np.array_equal(wt_cache[nm], wf):
                cands.append(nm)
        if len(cands) != 1:
            raise SystemExit(f"{base}: QAT の重みと一意に対応しない（候補 {cands}、形 {list(wf.shape)}）")
        nm = cands[0]
        d = lin[nm]
        codes = d["codes"].T.copy()                                    # [in, out]
        x.ClearField("raw_data"); x.ClearField("int32_data")
        if d["bits"] == 8:
            x.data_type = TensorProto.INT8
            x.raw_data = codes.astype(np.int8).tobytes()
            retyped += 1
        else:
            x.raw_data = pack_int4(codes)
        sc = q_inits[nd.input[1]]
        sc.CopyFrom(numpy_helper.from_array(d["scale"], sc.name))
        if len(nd.input) > 2 and nd.input[2] in q_inits:
            zp = q_inits[nd.input[2]]
            n_zp = int(np.prod(zp.dims)) if zp.dims else 1
            zp.ClearField("raw_data"); zp.ClearField("int32_data")
            if d["bits"] == 8:
                zp.data_type = TensorProto.INT8
                zp.raw_data = np.zeros(n_zp, np.int8).tobytes()
            else:
                zp.raw_data = pack_int4(np.zeros(n_zp, np.int8))
        mapping[base] = {"qat": nm, "bits": d["bits"], "from_float": bool(d.get("from_float"))}
        written += 1

        # ---- 活性の 8bit の丸め（SRQ）。QAT でない重み（in_act が無い）には掛けない ----
        if not srq or not d.get("in_act"):
            continue
        mm = next((c for c in cons.get(nd.output[0], []) if c.op_type == "MatMul"), None)
        if mm is None:
            raise SystemExit(f"{base}: 重みを使う MatMul が見つからない")
        oq = next((c for c in cons.get(mm.output[0], []) if c.op_type == "QuantizeLinear"), None)
        if (d.get("out_act") and oq is not None and len(oq.input) > 2
                and oq.input[1] in q_inits and oq.input[2] in q_inits):
            s_o, z_o = _u8_qdq(d["out_act"])
            q_inits[oq.input[1]].CopyFrom(numpy_helper.from_array(s_o.reshape(()), oq.input[1]))
            q_inits[oq.input[2]].CopyFrom(numpy_helper.from_array(z_o.reshape(()), oq.input[2]))
            srq_out += 1
        src = mm.input[0]
        s_i, z_i = _u8_qdq(d["in_act"])
        sn, zn = f"{base}_srq_in_scale", f"{base}_srq_in_zp"
        g.initializer.extend([numpy_helper.from_array(s_i.reshape(()), sn), numpy_helper.from_array(z_i.reshape(()), zn)])
        qn = helper.make_node("QuantizeLinear", [src, sn, zn], [f"{base}_srq_q"], f"{base}_srq_Q")
        dn = helper.make_node("DequantizeLinear", [f"{base}_srq_q", sn, zn], [f"{base}_srq_dq"], f"{base}_srq_DQ")
        mm.input[0] = f"{base}_srq_dq"
        new_nodes.append((list(g.node).index(mm), [qn, dn]))
        srq_in += 1

    if new_nodes:
        nodes = list(g.node)
        for idx, extra in sorted(new_nodes, key=lambda t: -t[0]):
            nodes[idx:idx] = extra
        del g.node[:]
        g.node.extend(nodes)
    onnx.save(qm, str(out))

    # ---- 検査: 読み直して、QAT の重みの整数×スケールが prepped の float と完全一致するか ----
    chk = onnx.load(str(out))
    c_inits = {t.name: t for t in chk.graph.initializer}
    bad, approx = [], {}
    for base, mp in mapping.items():
        qv = numpy_helper.to_array(c_inits[base + "_quantized"])
        sv = numpy_helper.to_array(c_inits[base + "_scale"]).astype(np.float32)
        wf = numpy_helper.to_array(p_inits[base])
        deq = qv.astype(np.float32) * sv[None, :]
        if mp["from_float"]:
            # 書き込んだ整数・スケールが作ったとおりか（厳密に）＋ 元の float との誤差（参考。8bit の丸めの分）
            d = lin[mp["qat"]]
            if not (np.array_equal(qv, d["codes"].T) and np.array_equal(sv, d["scale"])):
                bad.append(base)
            r = float(np.linalg.norm(deq - wf) / max(np.linalg.norm(wf), 1e-12))
            approx[mp["qat"]] = r
            if r > 5e-2:                                   # 出力チャネルごとの int8 なら通常 1% 前後。5% を超えたら異常
                bad.append(base)
        elif not np.array_equal(deq, wf):
            bad.append(base)
    onnx.checker.check_model(chk) if chk.ByteSize() < 2**31 else None
    remaining_int4 = sum(1 for t in chk.graph.initializer if t.data_type == TensorProto.INT4)
    info = {"qdq_in": str(qdq), "qdq_out": str(out), "srq": srq, "written": written, "retyped_int8": retyped,
            "srq_inputs": srq_in, "srq_outputs": srq_out,
            "unmatched_qat": sorted(set(lin) - {m["qat"] for m in mapping.values()}), "verify_failed": bad,
            "approx_from_float": approx, "remaining_int4": remaining_int4, "mapping": mapping,
            "scale_floored": {mp["qat"]: lin[mp["qat"]].get("n_floored", 0) for mp in mapping.values() if mp["from_float"]}}
    out.with_suffix(".inject.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"書き込み {written} 本（うち QAT でない重み {sum(m['from_float'] for m in mapping.values())} 本）/ "
          f"INT8 に型を替えた {retyped} 本 / SRQ 入力 {srq_in}・出力 {srq_out} / 検査の不一致 {len(bad)} 本 / "
          f"使わなかった重み {len(info['unmatched_qat'])} 本 / 残った INT4 {remaining_int4} 本 -> {out}")
    for k, v in approx.items():
        print(f"  QAT でない重みを 8bit にした誤差（相対 L2）: {k} {v:.2e}"
              f"（スケールを下限に上げたチャネル {info['scale_floored'].get(k, 0)} 本。C-14）")
    if bad or info["unmatched_qat"] or remaining_int4:
        raise SystemExit(f"不一致 {bad[:5]} / 未対応 {info['unmatched_qat'][:5]} / INT4 が残った {remaining_int4}")
    n_qat = sum(1 for d in lin.values() if d.get("in_act"))
    if srq and not (srq_in == srq_out == n_qat):
        # 一部の行列積にだけ SRQ が入った状態で先へ進まない（出力の Q/DQ が見つからない形があると起きる）
        raise SystemExit(f"SRQ を入れた数が合わない: 入力 {srq_in}・出力 {srq_out}（QAT の行列積 {n_qat} 本）")
    return info


def scale_ratio_report(path: Path) -> dict:
    """重みの DQ（INT8・出力チャネルごと）のスケールについて「最小 ÷ 最大」を重みごとに出し、いちばん小さいものを返す（C-14 の見張り）。

    本当の条件は「入力のスケール × 重みのスケール ÷ 出力のスケール」が小さすぎないことだが、同じ行列の中では入力・出力の
    スケールは 1 つなので、チャネルどうしの比で「1 本だけ極端に小さい」ものは見つけられる。SCALE_FLOOR_RATIO 未満なら注意。
    """
    m = onnx.load(str(path))
    inits = {t.name: t for t in m.graph.initializer}
    worst, n = None, 0
    for nd in m.graph.node:
        if nd.op_type != "DequantizeLinear" or nd.input[0] not in inits or inits[nd.input[0]].data_type != TensorProto.INT8:
            continue
        sc = numpy_helper.to_array(inits[nd.input[1]]).astype(np.float64).reshape(-1)
        if sc.size < 2:
            continue
        n += 1
        r = float(sc.min() / sc.max())
        if worst is None or r < worst[1]:
            worst = (nd.input[0], r)
    return {"n_weights": n, "min_ratio": worst[1] if worst else None, "min_ratio_tensor": worst[0] if worst else None,
            "ok": bool(worst is None or worst[1] >= SCALE_FLOOR_RATIO * 0.999)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdq", type=Path, required=True)
    ap.add_argument("--prepped", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-srq", action="store_true", help="活性の 8bit の丸めを入れない（16bit のまま）")
    a = ap.parse_args()
    C.utf8_stdout()
    inject(a.qdq, a.prepped, vit_linears(), a.out, srq=not a.no_srq)


if __name__ == "__main__":
    main()
