# -*- coding: utf-8 -*-
"""Phase 2: 入口でないかたまりの「量子化されていない入力」に Q/DQ を1組はさむ（C-1 対策）。

    python -m app.pipeline.fix_float_inputs --qdq <in.onnx> --out <out.onnx> --ref <A の qdq> --ref <B の qdq>

なぜ要るか: ORT は Gather / Transpose / Reshape を量子化の対象にしない。そのため

- `ple_all`（→ Gather）
- `shared_k_13` / `shared_v_13` / `shared_k_14` / `shared_v_14`（→ Transpose・Reshape）

は**グラフ入力から float のまま計算ノードへ**入る。HTP はこれを許さない（C-1）。
かたまりの中で作られている場合（かたまりAの `ple_all`、かたまりBの層13・14 の K/V）は
`DQ → 計算 → Q` になっていて、もともと C-1 を通っている。

どうするか: **その値を作る側（前のかたまり）の出力と同じ目盛り**で、入力の直後に Q/DQ を入れる。
同じ目盛りなので、繋いだ後は「前の DQ →（この）Q」で往復するだけで値は1ビットも動かない。

名前の対応: `ple_all` → `ple_all`、`shared_k_13` → `present_k_13`（`shared_` を `present_` に読み替える）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import helper, numpy_helper


def ref_name(inp: str) -> str:
    return "present_" + inp[len("shared_"):] if inp.startswith("shared_") else inp


def collect_scales(refs: list[Path]) -> dict[str, tuple]:
    """前のかたまりの QDQ から「出力の名前 → (scale, zero point)」を集める。"""
    got: dict[str, tuple] = {}
    for r in refs:
        m = onnx.load(str(r), load_external_data=False)
        prod = {o: n for n in m.graph.node for o in n.output}
        inits = {t.name: t for t in m.graph.initializer}
        for v in m.graph.output:
            n = prod.get(v.name)
            if n is None or n.op_type != "DequantizeLinear" or n.input[1] not in inits:
                continue
            got.setdefault(v.name, (numpy_helper.to_array(inits[n.input[1]]),
                                    numpy_helper.to_array(inits[n.input[2]])))
    return got


def float_inputs(model: onnx.ModelProto) -> list[str]:
    """量子化を経ずに計算ノードへ入っているグラフ入力。"""
    g = model.graph
    cons: dict[str, list] = {}
    for n in g.node:
        for i in n.input:
            cons.setdefault(i, []).append(n)
    bad = []
    for v in g.input:
        us = cons.get(v.name, [])
        if us and not all(u.op_type == "QuantizeLinear" for u in us):
            bad.append(v.name)
    return bad


def fix(qdq: Path, refs: list[Path], out: Path) -> dict:
    scales = collect_scales(refs)
    m = onnx.load(str(qdq))
    g = m.graph
    todo = float_inputs(m)
    done = []
    for name in todo:
        rn = ref_name(name)
        if rn not in scales:
            raise SystemExit(f"{name}: 目盛りの出どころ（{rn}）が参照のかたまりに無い")
        s, z = scales[rn]
        sn, zn = f"{name}_in_scale", f"{name}_in_zp"
        g.initializer.extend([numpy_helper.from_array(s.reshape(()), sn),
                              numpy_helper.from_array(z.reshape(()), zn)])
        q = helper.make_node("QuantizeLinear", [name, sn, zn], [f"{name}_in_q"], f"{name}_in_Q")
        d = helper.make_node("DequantizeLinear", [f"{name}_in_q", sn, zn], [f"{name}_in_dq"], f"{name}_in_DQ")
        users = 0
        for n in g.node:
            for i, x in enumerate(n.input):
                if x == name:
                    n.input[i] = f"{name}_in_dq"
                    users += 1
        g.node.insert(0, d)
        g.node.insert(0, q)
        done.append({"input": name, "from": rn, "scale": float(s), "zero_point": int(z), "users": users})
        print(f"  {name}: Q/DQ をはさんだ（{rn} と同じ目盛り scale {float(s):.3e} zp {int(z)}、使う側 {users} 箇所）")
    out.parent.mkdir(parents=True, exist_ok=True)
    big = m.ByteSize() > 1_800_000_000
    onnx.save(m, str(out), save_as_external_data=big, all_tensors_to_one_file=True,
              location=out.name + ".data" if big else None)
    print(f"-> {out.name}（{len(done)} 本）")
    return {"fixed": done, "out": str(out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdq", type=Path, required=True)
    ap.add_argument("--ref", type=Path, action="append", required=True, help="前のかたまりの QDQ（複数可）")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    fix(a.qdq, a.ref, a.out)


if __name__ == "__main__":
    main()
