# -*- coding: utf-8 -*-
"""qairt-converter を、**INT4 の重みを読めるように1行だけ直して**起動する（SDK のファイルは変えない）。

## 何が壊れているか（QAIRT 2.48.0.260626、2026-09-19 に踏んだ）

ONNX の DequantizeLinear の変換（`onnx/data_translations.py` の
`OnnxDequantizeLinearTranslation.extract_parameters`）が、入力の階数を

    input_rank = len(x_data.shape)

で数えている。ところが **INT4 / UINT4 の初期化子は、numpy に 4bit の型が無いので TensorProto のまま**
重みの表に入る（`onnx/util.py` の `fetch_constant_tensor` の注記どおり）。TensorProto には `.shape` が無く、

    AttributeError: shape
    ERROR - Node val_117_DequantizeLinear: shape

で止まる。同じ関数の後半は TensorProto を `decode_tensorproto_raw_data(w).reshape(w.dims)` で正しく扱っており、
**直っていないのは階数を数えるこの1行だけ**（階数は負の axis を正に直すのにしか使わない）。

## 直し方

起動時にこの関数のソースを取り出し、その1行を「`dims` があればそれを数える」に置き換えて差し替える。
置き換える文字列が見つからなければ（SDK の版が変わった等）**止める**。

    <QAIRT 用 Python> scripts/qairt_converter_int4.py <qairt-converter と同じ引数>
"""
import inspect
import os
import runpy
import sys
import textwrap

from qti.aisw.converters.onnx import data_translations as dt

_cls = dt.OnnxDequantizeLinearTranslation
_src = textwrap.dedent(inspect.getsource(_cls.extract_parameters))
_OLD = "input_rank = len(x_data.shape)"
_NEW = "input_rank = len(x_data.dims) if hasattr(x_data, 'dims') else len(x_data.shape)"
if _src.count(_OLD) != 1:
    raise SystemExit(f"qairt_converter_int4: 置き換える行が見つからない（SDK の版が違う？）: {_OLD}")
_ns = {}
exec(compile(_src.replace(_OLD, _NEW), dt.__file__, "exec"), dt.__dict__, _ns)
_cls.extract_parameters = _ns["extract_parameters"]
print("qairt_converter_int4: DequantizeLinear の INT4 の階数の数え方を直した", flush=True)

_conv = os.path.join(os.environ["QAIRT_U_WIN"], "bin", "x86_64-windows-msvc", "qairt-converter")
sys.argv = [_conv] + sys.argv[1:]
runpy.run_path(_conv, run_name="__main__")
