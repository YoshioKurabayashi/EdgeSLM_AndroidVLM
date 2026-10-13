# -*- coding: utf-8 -*-
"""Phase 2: かたまりごとに量子化した QDQ を、1つのグラフに繋ぐ。

    python -m app.pipeline.merge_chunks --chunks g4c0_8h_L256,g4c9_17_L256,... --out work/ops/g4full_L256/g4full_final_qdq.onnx

なぜ分けて作るか: 35層を一度に浮動小数で広げると PC のメモリが 9GB を超える（量子化ではさらに要る）。
かたまりごとなら 2〜3GB で済む。**端末に載せるのは1つのグラフ**（セッションも1つ）なので、量子化の後に繋ぐ。

繋ぎ方:
- `hidden_out` → 次の `hidden_in`、`ple_all` → 次の `ple_all`（入口のかたまりだけが `ple_all` を作る）
- 共通の入力（cos_s / sin_s / cos_g / sin_g / attn_mask）は、後ろのかたまり側を前のかたまりの入力に付け替えて1本にする
- 名前の衝突は接頭辞で避ける（`onnx.compose.add_prefix`）
- 境目は「前の DQ → 次の Q」。スケールはかたまりごとの較正で少し違うが、量子化し直すだけで害はない
"""
from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from onnx import compose

SHARED_INPUTS = ["cos_s", "sin_s", "cos_g", "sin_g", "attn_mask"]


def rewire_shared(model: onnx.ModelProto, prefix: str) -> list[str]:
    """接頭辞つきの共通入力を、接頭辞なしの入力へ付け替えて、グラフ入力から消す。"""
    g = model.graph
    names = {f"{prefix}{nm}": nm for nm in SHARED_INPUTS}
    have = {v.name for v in g.input}
    done = []
    for pre, plain in names.items():
        if pre not in have or plain not in have:
            continue
        for n in g.node:
            for i, inp in enumerate(n.input):
                if inp == pre:
                    n.input[i] = plain
        for k, v in enumerate(list(g.input)):
            if v.name == pre:
                del g.input[k]
                break
        done.append(pre)
    return done


def strip_prefix_io(model: onnx.ModelProto) -> list[tuple[str, str]]:
    """繋いだ後に残る接頭辞（`c1_past_k_9` など）を、グラフの入出力だけ外して分かりやすい名前に戻す。

    端末のジョブはテンソル名でファイルを指定するので、名前は素直な方がよい。
    重複が出る名前は変えない（安全側）。
    """
    import re
    g = model.graph
    # ★「グラフ入出力の名前」だけ見ると足りない。繋ぎに使われた中の枝（かたまりAの `hidden_out` など）も
    #   名前が残っているので、**グラフ内の全部の名前**と突き合わせる（重なると ORT が読み込みで落ちる）
    taken = {v.name for v in list(g.input) + list(g.output)}
    taken |= {t.name for t in g.initializer}
    for n in g.node:
        taken |= set(n.input) | set(n.output)
    renames = []
    for coll in (g.input, g.output):
        for v in coll:
            m = re.match(r"^c\d+_(.+)$", v.name)
            if not m:
                continue
            plain = m.group(1)
            if plain in taken:
                continue
            taken.discard(v.name)
            taken.add(plain)
            renames.append((v.name, plain))
    ren = dict(renames)
    for n in g.node:
        for i, inp in enumerate(n.input):
            if inp in ren:
                n.input[i] = ren[inp]
        for i, o in enumerate(n.output):
            if o in ren:
                n.output[i] = ren[o]
    for coll in (g.input, g.output, g.value_info):
        for v in coll:
            if v.name in ren:
                v.name = ren[v.name]
    return renames


def drop_outputs(model: onnx.ModelProto, names: list[str]) -> list[str]:
    """繋ぎのために残していた出力（`ple_all`）を、最後にグラフ出力から外す。

    中の枝としては残る（後ろのかたまりが使っている）ので、計算は変わらない。外さないと端末が毎回書き出してしまう。
    """
    g = model.graph
    gone = []
    for k, v in reversed(list(enumerate(g.output))):
        if v.name in names:
            del g.output[k]
            gone.append(v.name)
    return gone


def merge(chunks: list[Path], out: Path, drop: list[str] | None = None) -> dict:
    merged = onnx.load(str(chunks[0]))
    info = {"chunks": [str(chunks[0])], "rewired": [], "io_maps": []}
    for k, p in enumerate(chunks[1:], start=1):
        nxt = onnx.load(str(p))
        pre = f"c{k}_"
        nxt = compose.add_prefix(nxt, prefix=pre)
        outs = {v.name for v in merged.graph.output}
        ins = {v.name for v in nxt.graph.input}
        io_map, keep_out = [], []
        # ★2つめ以降を繋いだ後は、出力の名前に接頭辞が付く（`c1_hidden_out`）。末尾で探す
        if f"{pre}hidden_in" in ins:
            cand = [o for o in outs if o == "hidden_out" or o.endswith("_hidden_out")]
            if len(cand) != 1:
                raise SystemExit(f"{p.name}: 繋ぎ元の hidden_out が一意でない（{cand}）")
            io_map.append((cand[0], f"{pre}hidden_in"))
        if "ple_all" in outs and f"{pre}ple_all" in ins:
            io_map.append(("ple_all", f"{pre}ple_all"))
            keep_out.append("ple_all")      # ★C・D・E も同じ `ple_all` を使う。繋いだ後も出力として残す
        # ★層13・14 の K/V は「後ろのかたまりへ渡す」と同時に「次のステップのためのグラフ出力」でもある。
        #   `shared_k_13` などへ繋ぎつつ、**出力として残す**（残さないと端末がキャッシュを更新できない）
        #   すでに繋いだ後は名前に接頭辞が付く（`c1_present_k_13`）ので、末尾で見る
        for o in sorted(outs):
            for i in (13, 14):
                for kv in ("k", "v"):
                    if o.endswith(f"present_{kv}_{i}") and f"{pre}shared_{kv}_{i}" in ins:
                        io_map.append((o, f"{pre}shared_{kv}_{i}"))
                        keep_out.append(o)
        if not io_map:
            raise SystemExit(f"{p.name}: 繋ぎ先が見つからない（出力 {sorted(outs)[:4]}… / 入力 {sorted(ins)[:4]}…）")
        # 既定では「繋いだ出力」はグラフ出力から外れるので、残したいものを明示する
        keep = [o.name for o in merged.graph.output
                if o.name not in {a for a, _ in io_map} or o.name in keep_out]
        keep += [o.name for o in nxt.graph.output]
        merged = compose.merge_models(merged, nxt, io_map=io_map, outputs=keep)
        done = rewire_shared(merged, pre)
        info["chunks"].append(str(p))
        info["io_maps"].append(io_map)
        info["rewired"].append(done)
        print(f"  + {p.name}: 繋ぎ {io_map}、共通入力を付け替え {len(done)} 本")
    gone = drop_outputs(merged, drop or [])
    if gone:
        print(f"  繋ぎのために残していた出力を外した: {gone}")
    info["dropped_outputs"] = gone
    ren = strip_prefix_io(merged)
    print(f"  入出力の名前から接頭辞を外した: {len(ren)} 本")
    info["renamed_io"] = ren
    out.parent.mkdir(parents=True, exist_ok=True)
    big = merged.ByteSize() > 1_800_000_000
    onnx.save(merged, str(out), save_as_external_data=big, all_tensors_to_one_file=True,
              location=out.name + ".data" if big else None)
    g = merged.graph
    print(f"入力 {len(g.input)} 本 / 出力 {len(g.output)} 本 / ノード {len(g.node)}")
    print("  入力:", [v.name for v in g.input])
    print("  出力:", [v.name for v in g.output])
    print(f"-> {out}")
    info.update({"inputs": [v.name for v in g.input], "outputs": [v.name for v in g.output],
                 "nodes": len(g.node), "out": str(out)})
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True, help="繋ぐ順に、QDQ のファイル（カンマ区切り）")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--drop", default="ple_all", help="最後にグラフ出力から外すもの（繋ぎ用に残していた分）")
    a = ap.parse_args()
    merge([Path(x) for x in a.chunks.split(",")], a.out, [x for x in a.drop.split(",") if x])


if __name__ == "__main__":
    main()
