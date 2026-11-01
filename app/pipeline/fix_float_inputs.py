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

（2026-10-18 追記: 学習用の説明）
■ どの段で使うか
  Gemma 4 を「かたまり」（A〜E・H）に分けて作る流れ（build_full.py）で、かたまりごとの
  prepare（量子化）→ inject の後に掛ける（StudyMemo 03「35層をかたまりで作って繋ぐ」・05 の表、
  結果 results/P2_20260920_full.md 2 章）。
■ 入力と出力
  --qdq : 直したいかたまりの QDQ。--ref : その入力を作る側（前のかたまり）の QDQ（複数可）。
  --out : Q/DQ をはさんだ QDQ（1.8GB を超えるときは .data を 1 つ伴う）。
■ 注意（2026-10-18 追記。コードからの読み取り）
  float_inputs() は入力の型を見ていないので、int の入力（索引など）が Q 以外へ入る場合も拾い、
  参照に同名の出力が無ければ止まる。いま扱うかたまりで問題になっていないのは、そういう入力が
  無いか、参照側に同名の出力があるためと推定（確かめていない）。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import helper, numpy_helper


def ref_name(inp: str) -> str:
    """入力名を「それを作る側（前のかたまり）の出力名」に読み替える。shared_* → present_*、他はそのまま。"""
    return "present_" + inp[len("shared_"):] if inp.startswith("shared_") else inp


def collect_scales(refs: list[Path]) -> dict[str, tuple]:
    """前のかたまりの QDQ から「出力の名前 → (scale, zero point)」を集める。

    引数: refs（前のかたまりの QDQ のパス。重みは読まない）。
    戻り値: {出力名: (scale の配列, zero point の配列)}。出力が DequantizeLinear から出ているもの
    （＝量子化された値として出ているもの）だけ。同じ名前が複数の参照にあれば先のものを使う。
    """
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
    """量子化を経ずに計算ノードへ入っているグラフ入力。

    戻り値: 入力名のリスト。使う側が 1 つでも QuantizeLinear 以外なら入れる（使われない入力は除く）。
    verify_artifacts.check_c1_no_float_segments() のグラフ入力の検査と同じ考え方（ただし型は見ない）。
    """
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
    """qdq の「量子化されていない入力」の直後に、作る側と同じ目盛りの Q/DQ を 1 組ずつはさむ。

    引数: qdq（直すかたまり）、refs（作る側のかたまりの QDQ）、out（書き出し先）。
    戻り値: {"fixed": [{input, from, scale, zero_point, users（付け替えた使い先の数）}, ...], "out": パス}
    同じ目盛りなので、繋いだ後は「前の DQ → この Q」で往復するだけで値は変わらない。
    """
    # ① 作る側の目盛りを集め、直す対象の入力を拾う
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
        # ② 目盛り（per-tensor なので値は 1 個。形を () のスカラーにする）を初期化子として足し、
        #   入力 → Q → DQ のノードを作る
        sn, zn = f"{name}_in_scale", f"{name}_in_zp"
        g.initializer.extend([numpy_helper.from_array(s.reshape(()), sn),
                              numpy_helper.from_array(z.reshape(()), zn)])
        q = helper.make_node("QuantizeLinear", [name, sn, zn], [f"{name}_in_q"], f"{name}_in_Q")
        d = helper.make_node("DequantizeLinear", [f"{name}_in_q", sn, zn], [f"{name}_in_dq"], f"{name}_in_DQ")
        # ③ その入力を使っていたノードの入力を、DQ の出力へ付け替える
        users = 0
        for n in g.node:
            for i, x in enumerate(n.input):
                if x == name:
                    n.input[i] = f"{name}_in_dq"
                    users += 1
        # ONNX はノードを使う順（トポロジカル順）に並べる決まりなので、Q・DQ を先頭へ入れる（Q が先）
        g.node.insert(0, d)
        g.node.insert(0, q)
        done.append({"input": name, "from": rn, "scale": float(s), "zero_point": int(z), "users": users})
        print(f"  {name}: Q/DQ をはさんだ（{rn} と同じ目盛り scale {float(s):.3e} zp {int(z)}、使う側 {users} 箇所）")
    # ④ 保存。1.8GB を超えるなら重みを 1 つの .data にまとめる外部データ形式にする（protobuf の 2GB 上限）
    out.parent.mkdir(parents=True, exist_ok=True)
    big = m.ByteSize() > 1_800_000_000
    onnx.save(m, str(out), save_as_external_data=big, all_tensors_to_one_file=True,
              location=out.name + ".data" if big else None)
    print(f"-> {out.name}（{len(done)} 本）")
    return {"fixed": done, "out": str(out)}


def main():
    """コマンドの入口。--qdq / --ref（複数可）/ --out を受けて fix() を呼ぶ。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--qdq", type=Path, required=True)
    ap.add_argument("--ref", type=Path, action="append", required=True, help="前のかたまりの QDQ（複数可）")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    fix(a.qdq, a.ref, a.out)


if __name__ == "__main__":
    main()
