# -*- coding: utf-8 -*-
"""QAT の値を、量子化後の QDQ へ**そのまま**書き込む（重み＋活性の 8bit の丸め）。

    python -m app.pipeline.qat_inject --qdq <qdq.onnx> --prepped <prepped.onnx> --layers 0-4 --ple-proj --out <出力>

やること:
1. **重み**: p2b_ops の prepare（`weight_bits: 4`）が ORT に決め直させたスケールを、QAT の整数とスケールに置き換える。
   対応づけは**値で**（prepped の float の重み [in, out] と、QAT から戻した重み [out, in] の転置が完全一致するもの）。
   4bit・2bit は INT4 の器へ（2bit の −2〜1 もそのまま入る）。8bit（PLE のゲートと射影）は INT4 から **INT8 に型を替える**
2. **PLE の射影**（`per_layer_model_projection`。QAT では量子化されていない bf16）: 値から INT8・出力チャネルごと・対称で作り直す
3. **活性の 8bit の丸め（SRQ、既定で有効）**: QAT は「行列積の入力と出力を、層ごとに決まったスケールで −128〜127 に丸める」ことを
   込みで学習している（P0B: 外すと 23問が 20 → 18 に落ち、崩れた答えが出る）。そこで各 MatMul について
   - 出力の Q/DQ を **uint8（零点 128）・QAT の出力スケール**に書き換える
   - 入力の手前に **uint8（零点 128）・QAT の入力スケール**の Q/DQ を1組はさむ（同じテンソルを複数の行列積が使うため、
     行列積ごとに別の組を入れる）
   HTP の uint8（零点 128）は、QAT の int8 対称（−128〜127）と**同じ格子**を表す

最後に読み直し、重みの「整数×スケール」が書き出し前の値と1ビットも違わないことを確かめる。

（2026-10-18 追記）
- どの段で使うか: export_chunk export → p2b_ops prepare（ORT の静的量子化で qdq と prepped を作る）→ **qat_inject**
  → downgrade_actact_rhs（C-8）→ build_context.sh（StudyMemo（EdgeSLM_AndroidGemma4）05 の流れ図・02「QAT の丸めを端末で再現する」）
- 入力: --qdq（量子化後の QDQ）・--prepped（量子化の前処理を済ませた float のグラフ。重みの値で対応づけるのに使う）・
  --layers（かたまりに含めた層）・重みは gemma4_graph.QatFile（QAT のファイル）から
- 出力: --out の QDQ と、同じ名前の .inject.json（書き込んだ本数・SRQ の本数・対応表・検査の結果）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from app.pipeline import gemma4_graph as M

# PLE の射影（入口のかたまりにだけある。QAT では量子化されていない）の重みの名前
PLE_PROJ = f"{M.PFX}per_layer_model_projection"


def pack_int4(vals: np.ndarray) -> bytes:
    """ONNX の INT4（符号付き）: 2つを1バイトに、下位4ビットが先。

    vals は −8〜7 の整数（2bit の −2〜1 もそのまま入る）。& 0x0F で 2 の補数の下位 4bit を取り、奇数個なら 0 を 1 つ足す。
    """
    v = (vals.astype(np.int8).reshape(-1) & 0x0F).astype(np.uint8)
    if v.size % 2:
        v = np.concatenate([v, np.zeros(1, np.uint8)])
    return (v[0::2] | (v[1::2] << 4)).astype(np.uint8).tobytes()


def qat_linears(layers: list[int], ple_proj: bool = False, lm_head: bool = False) -> dict:
    """名前 → {codes [out,in] int8, scale [out], bits, w [out,in] float32, in_act, out_act}。

    layers の各層の行列（q/o/gate/up/down/PLE の門と射影、KV を自分で作る層なら k/v も）を QatFile から読む。
    lm_head=True で出力層、ple_proj=True で PLE の射影（作り直し）も加える。fin は各行列の入力の幅（bit を決めるのに使う）。
    """
    qf = M.QatFile()
    H = qf.cfg["hidden_size"]
    out = {}
    for i in layers:
        info = qf.layer_info(i)
        p = f"{M.PFX}layers.{i}."
        names = [("self_attn.q_proj", H), ("self_attn.o_proj", qf.cfg["num_attention_heads"] * info["hd"]),
                 ("mlp.gate_proj", H), ("mlp.up_proj", H), ("mlp.down_proj", info["inter"]),
                 ("per_layer_input_gate", H), ("per_layer_projection", qf.cfg["hidden_size_per_layer_input"])]
        if not info["shared"]:
            names += [("self_attn.k_proj", H), ("self_attn.v_proj", H)]
        for n, fin in names:
            d = qf.linear(p + n, fin)
            out[p + n] = {"codes": d["codes"].numpy(), "scale": d["scale"].numpy().astype(np.float32),
                          "bits": d["bits"], "w": d["w"].numpy(), "in_act": d["in_act"], "out_act": d["out_act"]}
    if lm_head:
        # 出力層（QAT は 2bit・行ごと）。重みの値だけをここで書き込む。
        # ★2026-10-18 訂正: 旧「SRQ の入出力スケールもあるので、行列積の入出力も 8bit にそろえる」。
        #   実際は出力層の SRQ のスケールは 0（＝較正していない印）で、下の「活性の 8bit の丸め（SRQ）」の段で
        #   外している（EdgeSLM_AndroidGemma4 の work/ops/g4head_L256/g4head_L256_a16w8_rhs8_qdq.inject.json でも SRQ 入力 0・出力 0）。入出力の丸めは付けない。
        d = qf.linear("lm_head", H)
        out["lm_head"] = {"codes": d["codes"].numpy(), "scale": d["scale"].numpy().astype(np.float32),
                          "bits": d["bits"], "w": d["w"].numpy(), "in_act": d["in_act"], "out_act": d["out_act"]}
    if ple_proj:
        # QAT では量子化されていない（bf16）。INT8・出力チャネルごと・対称で作り直す。活性の丸めは掛けない
        w = qf.vec(PLE_PROJ + ".weight").numpy()                        # [8960, 1536]
        s = np.maximum(np.abs(w).max(axis=1), 1e-12) / 127.0
        q = np.clip(np.rint(w / s[:, None]), -127, 127).astype(np.int8)
        # ★突き合わせは**元の値**で行う（グラフには丸める前の値が入っている）。書き込むのは丸めた整数とスケール。
        #   したがってこの1本だけは「整数×スケール＝元の値」にならない（検査は誤差で見る）
        out[PLE_PROJ] = {"codes": q, "scale": s.astype(np.float32), "bits": 8, "w": w,
                         "in_act": None, "out_act": None, "from_float": True}
    return out


def _u8_qdq(scale: float):
    """SRQ の Q/DQ に入れる（スケール, 零点）。uint8・零点 128 なら q−128 が −128〜127 になり、QAT の int8 対称と同じ値の格子になる。"""
    return np.float32(scale), np.uint8(128)          # QAT の int8 対称（−128〜127）と同じ格子


def inject(qdq: Path, prepped: Path, layers: list[int], out: Path, ple_proj: bool = False, srq: bool = True,
           lm_head: bool = False) -> dict:
    """量子化後の QDQ の重みを QAT の値に書き換え、SRQ の Q/DQ を入れて out に保存し、読み直して確かめる。

    段: ① QAT の行列を読み、形ごとに引ける表（by_shape）を作る → ② QDQ の INT4 の重み（DequantizeLinear の入力）を 1 本ずつ、
        prepped の float の重みと値で突き合わせて QAT の行列を決め、整数・スケール・零点を書き換える
        → ③ SRQ: その重みを使う MatMul の出力の Q/DQ を QAT の出力スケールに、入力の手前に Q/DQ を 1 組足す
        → ④ 足したノードを MatMul の直前に並べて保存 → ⑤ 読み直して検査し、.inject.json に記録
    戻り値: 記録の dict。対応しない重み・検査の不一致があれば止める。
    """
    # ① QAT の行列を読み、重みの形 [in, out] → 名前の list を作る（形が一意なら値の照合を軽くできる）
    lin = qat_linears(layers, ple_proj, lm_head)
    by_shape: dict = {}
    for nm, d in lin.items():
        by_shape.setdefault(tuple(d["w"].T.shape), []).append(nm)
    pm = onnx.load(str(prepped))
    p_inits = {t.name: t for t in pm.graph.initializer}
    qm = onnx.load(str(qdq))
    g = qm.graph
    q_inits = {t.name: t for t in g.initializer}
    # prod: テンソル名 → それを作るノード、cons: テンソル名 → それを使うノードの list（グラフをたどるための表）
    prod = {o: n for n in g.node for o in n.output}
    cons: dict[str, list] = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)

    mapping, written, retyped, srq_in, srq_out = {}, 0, 0, 0, 0
    new_nodes: list[tuple[int, list]] = []          # (MatMul の位置, 前に入れるノード)
    # ② 重みの DQ（入力が初期化子で INT4 のもの）を探す。名前は「<元の重み>_quantized」なので、元の名前で prepped の値を引く
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
        same = by_shape.get(tuple(wf.shape), [])
        if len(same) == 1:
            # ★形が一意なら全要素を突き合わせない。出力層は [1536, 262144]（1.6GB）あり、
            #   転置して丸ごと比べると PC のメモリが足りなくなる。まばらに 4096 点だけ確かめる
            nm0 = same[0]
            rng = np.random.default_rng(0)
            r = rng.integers(0, wf.shape[0], 4096); c = rng.integers(0, wf.shape[1], 4096)
            cands = same if np.array_equal(wf[r, c], lin[nm0]["w"][c, r]) else []
        else:
            cands = [nm for nm in same if np.array_equal(lin[nm]["w"].T, wf)]
        if len(cands) != 1:
            raise SystemExit(f"{base}: QAT の重みと一意に対応しない（候補 {cands}）")
        nm = cands[0]
        d = lin[nm]
        # ONNX の MatMul の重みは [in, out]（x @ W）。QAT は [out, in] なので転置して書く。スケールは出力チャネルごと（[out]）
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
        # 零点は対称なので 0。型は重みと同じにする（8bit なら INT8、それ以外は INT4）
        if len(nd.input) > 2 and nd.input[2] in q_inits:
            zp = q_inits[nd.input[2]]
            n_zp = int(np.prod(zp.dims)) if zp.dims else 1
            zp.ClearField("raw_data"); zp.ClearField("int32_data")
            if d["bits"] == 8:
                zp.data_type = TensorProto.INT8
                zp.raw_data = np.zeros(n_zp, np.int8).tobytes()
            else:
                zp.raw_data = pack_int4(np.zeros(n_zp, np.int8))
        mapping[base] = {"qat": nm, "bits": d["bits"]}
        written += 1

        # ③ ここから SRQ（重み 1 本ごとに、その重みを使う MatMul の入出力へ）
        # ---- 活性の 8bit の丸め（SRQ）----
        # ★スケールが 0 の本は「較正していない」印で、QAT 側も丸めない（transformers の `apply_srq` が何もしない）。
        #   出力層（`lm_head`）がこれにあたる。0 を入れるとゼロ除算になるので、ここで外す。
        if not srq or not d.get("in_act"):
            continue
        mm = next((c for c in cons.get(nd.output[0], []) if c.op_type == "MatMul"), None)
        if mm is None:
            raise SystemExit(f"{base}: 重みを使う MatMul が見つからない")
        # 出力側: MatMul の出力を量子化している Q/DQ のスケールと型を置き換える
        oq = next((c for c in cons.get(mm.output[0], []) if c.op_type == "QuantizeLinear"), None)
        if (d.get("out_act") and oq is not None and len(oq.input) > 2
                and oq.input[1] in q_inits and oq.input[2] in q_inits):
            s_o, z_o = _u8_qdq(d["out_act"])
            q_inits[oq.input[1]].CopyFrom(numpy_helper.from_array(s_o.reshape(()), oq.input[1]))
            q_inits[oq.input[2]].CopyFrom(numpy_helper.from_array(z_o.reshape(()), oq.input[2]))
            srq_out += 1
        # 入力側: MatMul の手前に uint8 の Q/DQ を1組はさむ（同じテンソルを他の行列積も使うので行列積ごとに別の組）
        src = mm.input[0]
        s_i, z_i = _u8_qdq(d["in_act"])
        sn, zn = f"{base}_srq_in_scale", f"{base}_srq_in_zp"
        g.initializer.extend([numpy_helper.from_array(s_i.reshape(()), sn), numpy_helper.from_array(z_i.reshape(()), zn)])
        qn = helper.make_node("QuantizeLinear", [src, sn, zn], [f"{base}_srq_q"], f"{base}_srq_Q")
        dn = helper.make_node("DequantizeLinear", [f"{base}_srq_q", sn, zn], [f"{base}_srq_dq"], f"{base}_srq_DQ")
        mm.input[0] = f"{base}_srq_dq"
        new_nodes.append((list(g.node).index(mm), [qn, dn]))
        srq_in += 1

    # 入れるノードを、対応する MatMul の直前に置く（トポロジ順を保つ）
    # ④ 後ろの位置から入れていくと、前の位置の番号がずれない
    if new_nodes:
        nodes = list(g.node)
        for idx, extra in sorted(new_nodes, key=lambda t: -t[0]):
            nodes[idx:idx] = extra
        del g.node[:]
        g.node.extend(nodes)
    onnx.save(qm, str(out))

    # ---- 検査: 読み直して、整数×スケールが prepped の float と完全一致するか ----
    chk = onnx.load(str(out))
    c_inits = {t.name: t for t in chk.graph.initializer}
    bad, approx = [], {}
    for base, mp in mapping.items():
        qv = numpy_helper.to_array(c_inits[base + "_quantized"])
        sv = numpy_helper.to_array(c_inits[base + "_scale"]).astype(np.float32)
        wf = numpy_helper.to_array(p_inits[base])
        # ⑤ 書いた整数×スケール（qv×sv）が prepped の float（wf）と完全に一致するか。QAT の値なら一致するはず
        if qv.size > 50_000_000 and not lin[mp["qat"]].get("from_float"):
            # ★出力層は [1536, 262144]。丸ごと float に戻すと 1.6GB×2 になるので、列を分けて確かめる
            ng = False
            for c0 in range(0, qv.shape[1], 16384):
                seg = qv[:, c0:c0 + 16384].astype(np.float32) * sv[None, c0:c0 + 16384]
                if not np.array_equal(seg, wf[:, c0:c0 + 16384]):
                    ng = True
                    break
            if ng:
                bad.append(base)
            continue
        deq = qv.astype(np.float32) * sv[None, :]
        if lin[mp["qat"]].get("from_float"):
            # 元が量子化されていない重み（PLE の射影）。ここで 8bit にしたので、誤差で見る
            r = float(np.linalg.norm(deq - wf) / max(np.linalg.norm(wf), 1e-12))
            approx[mp["qat"]] = r
            if r > 1e-2:
                bad.append(base)
        elif not np.array_equal(deq, wf):
            bad.append(base)
    # ONNX の形式の検査（2GB を超えるモデルは checker が読めないので省く）
    onnx.checker.check_model(chk) if chk.ByteSize() < 2**31 else None
    info = {"qdq_in": str(qdq), "qdq_out": str(out), "written": written, "retyped_int8": retyped,
            "srq_inputs": srq_in, "srq_outputs": srq_out,
            "unmatched_qat": sorted(set(lin) - {m["qat"] for m in mapping.values()}), "verify_failed": bad,
            "approx_from_float": approx,
            "bits": {b: sum(1 for m in mapping.values() if m["bits"] == b) for b in (2, 4, 8)}, "mapping": mapping}
    out.with_suffix(".inject.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"書き込み {written} 本（2bit {info['bits'][2]}・4bit {info['bits'][4]}・8bit {info['bits'][8]}）/ "
          f"INT8 に型を替えた {retyped} 本 / SRQ 入力 {srq_in}・出力 {srq_out} / 検査の不一致 {len(bad)} 本 / "
          f"使わなかった QAT の重み {len(info['unmatched_qat'])} 本 -> {out}")
    if bad or info["unmatched_qat"]:
        raise SystemExit(f"不一致 {bad[:5]} / 未対応 {info['unmatched_qat'][:5]}")
    return info


def main():
    """引数を読み、--layers（「a-b」・カンマ区切り・空か none）を層の list にして inject を呼ぶ。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdq", type=Path, required=True)
    ap.add_argument("--prepped", type=Path, required=True)
    ap.add_argument("--layers", required=True, help="含めた層（a-b またはカンマ区切り）")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ple-proj", action="store_true", help="PLE の射影（量子化されていない bf16）も 8bit で書き込む")
    ap.add_argument("--lm-head", action="store_true", help="出力層（2bit）も書き込む")
    ap.add_argument("--no-srq", action="store_true", help="活性の 8bit の丸めを入れない（16bit のまま）")
    a = ap.parse_args()
    if a.layers in ("", "none"):
        layers = []
    elif "-" in a.layers:
        lo, hi = (int(x) for x in a.layers.split("-"))
        layers = list(range(lo, hi + 1))
    else:
        layers = [int(x) for x in a.layers.split(",")]
    inject(a.qdq, a.prepped, layers, a.out, ple_proj=a.ple_proj, srq=not a.no_srq, lm_head=a.lm_head)


if __name__ == "__main__":
    main()
