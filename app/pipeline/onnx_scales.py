# -*- coding: utf-8 -*-
"""量子化済み ONNX から、QuantizeLinear の目盛り（scale・zero point）を読む（2026-09-27）。

    python -m app.pipeline.onnx_scales                       # 端末の Gemma 4 のかたまり A の入口の範囲を表示
    python -m app.pipeline.onnx_scales --onnx <path> --names inputs_embeds_QuantizeLinear,ple_tok_QuantizeLinear

## なぜ要るか

端末の文章グラフは、入口の `inputs_embeds` を 16bit の整数に量子化してから計算する。
量子化の目盛りは**文章だけで較正**して決まっているので、表せる値の範囲は

    lo = (0 − zero_point) × scale,   hi = (65535 − zero_point) × scale     （uint16 の場合）

に限られる。画像の特徴がこの範囲を超えると、端末では切り詰められる（PC の HF には起きない）。
Phase 1 では、画像の特徴がこの範囲に収まるかを PC で確かめる（check_wiring.py）。

## なぜ onnx パッケージを使わないか

借りている venv（EdgeSLM_MultiHead_Multi）に onnx が入っていないため。ONNX のファイルは protobuf なので、
必要な欄（ノードの名前・入力・種類、初期値の名前・型・中身）だけを直接たどって読む。
大きな重みは読まない（64 バイト以下の初期値＝scale と zero point だけを取り出す）。

protobuf の欄番号（onnx.proto より）:
    ModelProto.graph = 7
    GraphProto.node = 1, initializer = 5, input = 11
    NodeProto.input = 1, output = 2, name = 3, op_type = 4
    TensorProto.dims = 1, data_type = 2, float_data = 4, int32_data = 5, name = 8, raw_data = 9

## 使う段・入力と出力

- Phase 1（StudyMemo 4.2 節「入口の範囲を ONNX から読む」）。check_wiring.py が範囲を借りて「収まるか」「切り詰めたらどうなるか」を測る
- 入力: 量子化済みの ONNX（既定は端末の Gemma 4 のかたまり A、`vlm_common.DEVICE_CHUNK_A_QDQ`）と、見たい QuantizeLinear の名前
- 出力: 名前ごとに scale・zero point・整数の型・表せる範囲 [lo, hi]（画面に表示。read_quant_points は辞書で返す）
- 入口そのもの（inputs_embeds）の範囲は約 [−7.03, 3.52]（StudyMemo 12.2 節の演習 ex03 で確かめられる）
"""
from __future__ import annotations

import argparse
import struct

import numpy as np

from app.pipeline import vlm_common as C

# TensorProto.data_type → (numpy の型, 整数の最小, 最大)
DTYPES = {1: ("<f4", None, None), 2: ("u1", 0, 255), 3: ("i1", -128, 127),
          4: ("<u2", 0, 65535), 5: ("<i2", -32768, 32767), 6: ("<i4", None, None), 7: ("<i8", None, None)}

# 端末の文章グラフの入口で、画像の特徴が通る量子化の点
ENTRY_NAMES = [
    "inputs_embeds_QuantizeLinear",             # 入口そのもの（画像の特徴がそのまま入る）
    "inputs_embeds_prescaled_QuantizeLinear",   # inputs_embeds × 1536^-0.5（PLE の射影の前）
    "ple_tok_QuantizeLinear",                   # PLE の表の行（画像の位置は pad の行）
    "ple_all_QuantizeLinear",                   # PLE の射影と表の行を足したもの（全層ぶん）
]


# protobuf の可変長整数（varint）を 1 つ読む。b[i] から読み、(値, 次の位置) を返す。
# 1 バイトの下 7 ビットが値の一部で、最上位ビットが 1 なら「まだ続く」。下位の 7 ビットから順に並ぶ。
def _varint(b: bytes, i: int) -> tuple[int, int]:
    r = s = 0
    while True:
        c = b[i]
        i += 1
        r |= (c & 0x7F) << s
        s += 7
        if c < 0x80:
            return r, i


def _fields(b: bytes, start: int = 0, end: int | None = None):
    """protobuf の 1 階層ぶんの欄を順に返す: (欄番号, wire type, 値 or (開始, 終了))。"""
    i, end = start, (len(b) if end is None else end)
    # 欄の頭（キー）は varint で「欄番号 << 3 | wire type」。wire type で値の長さの決め方が変わる:
    #   0 = varint、1 = 8 バイト固定、5 = 4 バイト固定、2 = 長さ付き（文字列・入れ子のメッセージ・詰めた配列）
    # 2 の値は中身を切り出さず (開始, 終了) の位置だけ返す（大きな重みを写さないため）。
    while i < end:
        k, i = _varint(b, i)
        f, t = k >> 3, k & 7
        if t == 0:
            v, i = _varint(b, i)
            yield f, t, v
        elif t == 1:
            yield f, t, b[i:i + 8]
            i += 8
        elif t == 5:
            yield f, t, b[i:i + 4]
            i += 4
        elif t == 2:
            n, i = _varint(b, i)
            yield f, t, (i, i + n)
            i += n
        else:
            raise ValueError(f"未対応の wire type {t}")


def read_quant_points(path, names: list[str]) -> dict[str, dict]:
    """指定した名前の QuantizeLinear ノードについて、scale・zero point・表せる範囲を返す。

    引数: path = 量子化済み ONNX のパス、names = QuantizeLinear ノードの名前の並び。
    戻り値: {名前: {"scale": float, "zero_point": int, "qtype": numpy の型の文字, "lo": float, "hi": float}}。
    グラフに無い名前は None。lo・hi は「整数の最小・最大を戻した値」（モジュールの説明の式）。
    ONNX を丸ごと読み込まず、ノードの名前・入力と、64 バイト以下の初期値だけを拾う（onnx パッケージが無い venv で動かすため）。
    """
    # ① ファイル全体をバイト列で読み、ModelProto の欄 7（graph）の位置を探す
    b = open(path, "rb").read()
    graph = [v for f, t, v in _fields(b) if f == 7][0]
    # ② graph の中を 1 周して、名前の合うノード（入力の名前つき）と、小さな初期値（scale・zero point の候補）を集める
    nodes, small = {}, {}
    for f, t, v in _fields(b, *graph):
        if f == 1:                                            # NodeProto
            d = {"in": [], "op": "", "name": ""}
            for f2, t2, v2 in _fields(b, *v):
                # 欄 1 = 入力の名前（複数）、3 = ノードの名前、4 = 種類（op_type）。どれも文字列（wire type 2）
                if t2 != 2 or f2 not in (1, 3, 4):
                    continue
                s = b[v2[0]:v2[1]].decode("utf-8")
                if f2 == 1:
                    d["in"].append(s)
                elif f2 == 3:
                    d["name"] = s
                else:
                    d["op"] = s
            if d["name"] in names:
                nodes[d["name"]] = d
        elif f == 5:                                          # TensorProto（初期値）
            name, dt, raw, fd, i32 = None, None, None, [], []
            for f2, t2, v2 in _fields(b, *v):
                # 初期値の値は raw_data（欄 9）・float_data（欄 4）・int32_data（欄 5）のどれかに入る。書き手によって違うので 3 通りとも読む
                if f2 == 8:
                    name = b[v2[0]:v2[1]].decode("utf-8")
                elif f2 == 2:
                    dt = v2
                elif f2 == 9:
                    raw = v2
                elif f2 == 4:
                    fd += list(struct.unpack(f"<{(v2[1] - v2[0]) // 4}f", b[v2[0]:v2[1]])) if t2 == 2 \
                        else [struct.unpack("<f", v2)[0]]
                elif f2 == 5:
                    if t2 == 2:
                        j = v2[0]
                        while j < v2[1]:
                            x, j = _varint(b, j)
                            i32.append(x)
                    else:
                        i32.append(v2)
            # 64 バイト以下だけを残す（scale・zero point はふつう 1 要素。大きな重みは読まない）
            if raw is not None and raw[1] - raw[0] <= 64:
                small[name] = (dt, np.frombuffer(b[raw[0]:raw[1]], dtype=DTYPES.get(dt, ("u1",))[0]))
            elif fd:
                small[name] = (dt, np.array(fd, dtype=np.float32))
            elif i32:
                small[name] = (dt, np.array(i32))
    # ③ QuantizeLinear の入力は (x, scale, zero_point) の順。2 番目・3 番目の名前で初期値を引き、
    #    zero point の型（uint16 なら 0〜65535）から表せる範囲を計算する
    out = {}
    for nm in names:
        if nm not in nodes:
            out[nm] = None
            continue
        n = nodes[nm]
        _, sc = small[n["in"][1]]
        zdt, zp = small[n["in"][2]]
        qmin, qmax = DTYPES[zdt][1], DTYPES[zdt][2]
        s, z = float(sc.ravel()[0]), int(zp.ravel()[0])
        out[nm] = {"scale": s, "zero_point": z, "qtype": DTYPES[zdt][0],
                   "lo": round((qmin - z) * s, 6), "hi": round((qmax - z) * s, 6)}
    return out


# 端末の文章グラフ（既定）の入口の量子化の点を読んで表示する。--onnx・--names で別のファイル・名前も読める。
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=str(C.DEVICE_CHUNK_A_QDQ))
    ap.add_argument("--names", default=",".join(ENTRY_NAMES))
    a = ap.parse_args()
    res = read_quant_points(a.onnx, a.names.split(","))
    for k, v in res.items():
        print(f"{k:42s} " + ("（無い）" if v is None else
                             f"scale={v['scale']:.6g} zp={v['zero_point']} {v['qtype']}  範囲 [{v['lo']}, {v['hi']}]"))


if __name__ == "__main__":
    main()
