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

## どこで使うか（2026-10-18 追記）

重みを 4bit（INT4）にしたモデルの S5（DLC 変換）。scripts/build_context.sh が環境変数
`QAIRT_INT4_PATCH=1` を見て、qairt-converter の代わりにこのラッパーを起動する
（その際 `QAIRT_U_WIN` に QAIRT の場所を Windows の書き方で渡す）。最初に踏んだのは
EdgeSLM_AndroidScale（S-8）で、Gemma 4 の変換でも使う（EdgeSLM_AndroidGemma4 の StudyMemo 03）。

  入力 : qairt-converter と同じ引数（--input_network <qdq.onnx> --output_path <dlc> など）
  出力 : qairt-converter と同じ（DLC）。最初に「直した」と 1 行表示する
  前提 : QAIRT の Python 環境（qairtenv、3.10）で動かす（qti.aisw… を import するため）
"""
import inspect
import os
import runpy
import sys
import textwrap

from qti.aisw.converters.onnx import data_translations as dt

# ① 直す関数のソースを文字列として取り出す（字下げを外して、トップレベルの関数として読めるようにする）
_cls = dt.OnnxDequantizeLinearTranslation
_src = textwrap.dedent(inspect.getsource(_cls.extract_parameters))
_OLD = "input_rank = len(x_data.shape)"
_NEW = "input_rank = len(x_data.dims) if hasattr(x_data, 'dims') else len(x_data.shape)"
if _src.count(_OLD) != 1:
    raise SystemExit(f"qairt_converter_int4: 置き換える行が見つからない（SDK の版が違う？）: {_OLD}")
# ② 1 行だけ置き換えたソースを、元のモジュールの名前空間（dt.__dict__）でコンパイルし直し、
#   できた関数をクラスに差し替える。SDK のファイルそのものは書き換えない（このプロセスの中だけ）
_ns = {}
exec(compile(_src.replace(_OLD, _NEW), dt.__file__, "exec"), dt.__dict__, _ns)
_cls.extract_parameters = _ns["extract_parameters"]
print("qairt_converter_int4: DequantizeLinear の INT4 の階数の数え方を直した", flush=True)

# ③ 本物の qairt-converter（拡張子の無い Python のスクリプト）を、同じプロセスの中で __main__ として
#   実行する。sys.argv を差し替えて、受け取った引数をそのまま渡す（差し替えた関数が使われる）
_conv = os.path.join(os.environ["QAIRT_U_WIN"], "bin", "x86_64-windows-msvc", "qairt-converter")
sys.argv = [_conv] + sys.argv[1:]
runpy.run_path(_conv, run_name="__main__")
