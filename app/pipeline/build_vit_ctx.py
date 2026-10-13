# -*- coding: utf-8 -*-
"""Phase 2-3: 量子化した画像エンコーダ（final_qdq）を QAIRT で v68 の context binary にし、端末用の ONNX で包む（2026-09-28）。

    python -m app.pipeline.build_vit_ctx                          # 70 トークン、SRQ なし → あり の順に 2 版
    python -m app.pipeline.build_vit_ctx --variants nosrq         # 片方だけ
    python -m app.pipeline.build_vit_ctx --redo                   # 済んでいてもやり直す

**EdgeSLM_Android の venv（$PY2）で動かす**（onnx が要る）。変換そのもの（qairt-converter など）は、
複製した `scripts/build_context.sh` を **Git Bash から**呼び、その中で QAIRT 専用の Python 3.10（D:\\QAIRT\\qairtenv）が動く。
EdgeSLM_AndroidGemma4 の `bench_prefill.build` / `build_full` と同じ呼び方（そこから手順を写した。コードは新しく書いた）。

## 流れ（1 版あたり）

| 段 | 道具 | 出力（work/ops/vit_{tag}/） |
|---|---|---|
| S5 変換 | `build_context.sh` の qairt-converter（`--preserve_io_datatype`。C-4） | `build_{版}/vit_{tag}_{版}.dlc` |
| S6 context binary | 同 qnn-context-binary-generator（`--htp_socs sm8350` だけ。C-2・C-3） | `build_{版}/ctx_vit_{tag}_{版}/vit_{tag}_{版}_ctx.bin` |
| S7 メタデータ | 同 qnn-context-binary-utility | `build_{版}/vit_{tag}_{版}_meta.json` |
| 検査 | `verify_artifacts.verify_all`（C-1・C-5 ＋ C-2 socModel=0・C-3 config なし・C-4） | 結果の JSON |
| S8 包む | `wrap_epctx.run(embed=False)`（binary を ONNX に抱えさせない。二重持ちで RSS が倍になるのを避ける） | `vit_{tag}_{版}_epctx.onnx` ＋ 同じ場所に `.bin` |

- 重みは INT8 なので、INT4 用の回避策（`QAIRT_INT4_PATCH=1`。Gemma4 の文章グラフで使った）は**既定では使わない**
  （`--int4-patch` で付けられる。INT8 の変換で converter が止まったときに試す）
- 版の順番は **SRQ なし → あり**。SRQ なしの版は端末の算術の誤差（Q10）を見るのに使い、グラフも単純（入力側の丸めの Q/DQ が 112 組少なく、出力側も 16bit のまま）なので、
  変換でつまずくならこちらで先に分かる
- 結果は `results/P2_vit_ctx/{tag}_build.json`。変換のログは `build_{版}/` の `*_convert.log`・`*_gen.log`（build_context.sh が書く）と
  `work/logs/vit_{tag}_{版}_build.log`（このスクリプトが画面に出したものの写し）
"""
from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

from app.pipeline import vlm_common as C

RES = C.RESULTS / "P2_vit_ctx"
GIT_BASH = os.environ.get("GIT_BASH", r"C:\Program Files\Git\bin\bash.exe")   # Gemma4 と同じ場所


def tag_for(budget: int, portrait: bool) -> tuple[str, tuple[int, int]]:
    grid = (C.PORTRAIT_GRIDS if portrait else C.FIXED_GRIDS)[budget]
    return f"b{budget}_{grid[0]}x{grid[1]}", grid


def paths(tag: str, variant: str) -> dict:
    d = C.OPS / f"vit_{tag}"
    name = f"vit_{tag}_{variant}"                     # build_context.sh の <tag>（ファイル名の頭）
    b = d / f"build_{variant}"
    return {"dir": d, "name": name, "build": b, "final": d / f"{name}_final_qdq.onnx",
            "dlc": b / f"{name}.dlc", "ctxdir": b / f"ctx_{name}", "meta": b / f"{name}_meta.json",
            "cmdlog": b / f"{name}_commands.txt", "epctx": d / f"{name}_epctx.onnx",
            "log": C.WORK / "logs" / f"{name}_build.log"}


def run_streaming(cmd: list[str], log: Path, env: dict) -> int:
    """外のコマンドを動かし、出力を 1 行ずつ画面とログの両方に出す（変換は数十分かかるので、黙って待たせない）。"""
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "w", encoding="utf-8") as f:
        f.write(" ".join(cmd) + "\n\n")
        p = subprocess.Popen(cmd, cwd=str(C.ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, encoding="utf-8", errors="replace", env=env)
        for line in p.stdout:
            print("    " + line.rstrip(), flush=True)
            f.write(line)
            f.flush()                                   # 途中でもログを読めるように（変換は長い）
        return p.wait()


def build_one(tag: str, grid: tuple[int, int], variant: str, a) -> dict:
    from app.pipeline import verify_artifacts, wrap_epctx
    P = paths(tag, variant)
    if not P["final"].exists():
        raise SystemExit(f"{P['final']} が無い。先に quantize_vit を動かすこと")
    out = {"final_qdq": str(P["final"])}

    # ---- 済んでいれば飛ばす（包んだ ONNX が final_qdq より新しく、binary も残っている） ----
    bins = sorted(P["ctxdir"].glob("*.bin")) if P["ctxdir"].exists() else []
    done = (P["epctx"].exists() and bins and P["epctx"].stat().st_mtime > P["final"].stat().st_mtime
            and bins[0].stat().st_mtime > P["final"].stat().st_mtime)
    t0 = time.time()
    if done and not a.redo:
        print(f"  変換は済んでいる（{P['epctx'].name}）。やり直すなら --redo", flush=True)
        out["skipped"] = True
    else:
        # 古い生成物を使い回さない（前の版の .bin を拾って包んでしまうのを防ぐ）
        for old in [P["dlc"], P["meta"], P["epctx"], *bins]:
            old.unlink(missing_ok=True)
        if not Path(GIT_BASH).exists():
            raise SystemExit(f"Git Bash が無い: {GIT_BASH}（環境変数 GIT_BASH で場所を指定できる）")
        env = dict(os.environ)
        if a.int4_patch:
            env["QAIRT_INT4_PATCH"] = "1"
        else:
            env.pop("QAIRT_INT4_PATCH", None)
        print(f"  build_context.sh（S5〜S7）…　数十分かかることがある", flush=True)
        rc = run_streaming([GIT_BASH, "scripts/build_context.sh", str(P["final"]), str(P["build"]), P["name"]],
                           P["log"], env)
        out["build_sec"] = round(time.time() - t0, 1)
        bins = sorted(P["ctxdir"].glob("*.bin")) if P["ctxdir"].exists() else []
        if rc != 0 or not bins:
            out["error"] = f"build_context.sh が失敗した（終了コード {rc}）。{P['build']} の *_convert.log / *_gen.log を見ること"
            print("  ★" + out["error"], flush=True)
            return out
    out["dlc_mib"] = round(P["dlc"].stat().st_size / 2**20, 1) if P["dlc"].exists() else None
    out["ctx_bin"] = str(bins[0])
    out["ctx_mib"] = round(bins[0].stat().st_size / 2**20, 1)

    # ---- 検査: C-1・C-5（final_qdq）、C-2（メタデータの socModel=0）、C-3・C-4（実際に使ったコマンド） ----
    cmds = P["cmdlog"].read_text(encoding="utf-8", errors="replace").splitlines() if P["cmdlog"].exists() else []
    ver = verify_artifacts.verify_all(P["final"], P["meta"], cmds)
    out["verify"] = {"all_pass": bool(ver["all_pass"]),
                     "checks": [{"check": c["check"], "pass": bool(c.get("pass")), "detail": str(c.get("detail", ""))}
                                for c in ver["checks"]]}
    for c in out["verify"]["checks"]:
        print(f"  {'合格' if c['pass'] else '★不合格'}  {c['check']}: {c['detail']}", flush=True)
    if not ver["all_pass"]:
        out["error"] = "検査に通らない（端末でロードに失敗する。上の ★ を直すこと）"
        return out

    # ---- S8: EPContext の ONNX で包む（binary は別ファイル。embed=False） ----
    if not out.get("skipped"):
        info = wrap_epctx.run(bins[0], P["final"], P["epctx"], P["name"], embed=False)
    else:
        import onnx
        m = onnx.load(str(P["epctx"]), load_external_data=False)
        info = {"bytes": P["epctx"].stat().st_size,
                "inputs": [{"name": v.name, "shape": [d.dim_value for d in v.type.tensor_type.shape.dim]} for v in m.graph.input],
                "outputs": [{"name": v.name, "shape": [d.dim_value for d in v.type.tensor_type.shape.dim]} for v in m.graph.output]}
    # 端末のアプリが渡す形と一致するか（固定の格子: パッチ数 N = 縦×横、特徴 M = N / 9）
    N, M = grid[0] * grid[1], (grid[0] // C.POOL) * (grid[1] // C.POOL)
    want_in = [{"name": "pixel_values", "shape": [1, N, C.PATCH_DIM]}]
    want_out = [{"name": "image_embeds", "shape": [1, M, C.TEXT_HIDDEN]}]
    got_in = [{"name": i["name"], "shape": list(i["shape"])} for i in info["inputs"]]
    got_out = [{"name": o["name"], "shape": list(o["shape"])} for o in info["outputs"]]
    out["epctx"] = {"path": str(P["epctx"]), "bytes": info["bytes"], "inputs": got_in, "outputs": got_out,
                    "io_ok": got_in == want_in and got_out == want_out}
    print(f"  包んだ ONNX: {P['epctx'].name}（{info['bytes']} B）＋ {bins[0].name}（{out['ctx_mib']}MiB）", flush=True)
    print(f"  入出力: {got_in} → {got_out} {'（想定どおり）' if out['epctx']['io_ok'] else '★想定と違う'}", flush=True)
    if not out["epctx"]["io_ok"]:
        out["error"] = f"入出力の形が想定と違う（想定 {want_in} → {want_out}）"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=70, choices=sorted(C.FIXED_GRIDS))
    ap.add_argument("--portrait", action="store_true")
    ap.add_argument("--variants", default="nosrq,srq")
    ap.add_argument("--redo", action="store_true", help="済んでいても変換からやり直す")
    ap.add_argument("--int4-patch", action="store_true", help="QAIRT_INT4_PATCH=1 を付ける（既定では付けない。重みは INT8）")
    a = ap.parse_args()
    C.utf8_stdout()
    tag, grid = tag_for(a.budget, a.portrait)
    RES.mkdir(parents=True, exist_ok=True)
    res_path = RES / f"{tag}_build.json"
    out = {"tag": tag, "grid": list(grid), "variants": {}}
    if res_path.exists():                                   # 片方だけ作り直したとき、もう片方の記録を消さない
        import json
        try:
            out["variants"] = json.loads(res_path.read_text(encoding="utf-8")).get("variants", {})
        except Exception:
            pass
    t0 = time.time()
    failed = []
    for v in a.variants.split(","):
        print(f"\n== 版: {v}（SRQ {'あり' if v == 'srq' else 'なし'}）", flush=True)
        r = build_one(tag, grid, v, a)
        out["variants"][v] = r
        C.write_json_atomic(res_path, out)                  # 版ごとに書く（2 版目の途中で止めても 1 版目の記録は残る）
        if r.get("error"):
            failed.append(v)
    out["sec"] = round(time.time() - t0, 1)
    C.write_json_atomic(res_path, out)
    print(f"\n-> {res_path}（{out['sec']:.0f}s）")
    if failed:
        raise SystemExit(f"失敗した版: {failed}")


if __name__ == "__main__":
    main()
