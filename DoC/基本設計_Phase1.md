# 基本設計 — Phase 1（PC での確認）

| | |
|---|---|
| 版 | 0.1（2026-09-27） |
| 対象 | `app/pipeline/` の Phase 1 の道具 |
| 前提 | [要件定義書](要件定義書.md) 4章の Phase 1、[P0 棚卸し](P0_既存資産の棚卸し.md) |

---

## 1. Phase 1 で答えを出す問い

| # | 問い | 道具 | 合格の目安 |
|---|---|---|---|
| Q1 | 画像トークン数（70 / 140 / 280）で、用途ごとの品質はどう変わるか | `ref_vlm.py` | 表ができること（合否ではなく判断材料） |
| Q2 | 前処理を端末と同じ固定の格子（レターボックス）にすると品質は落ちるか | `ref_vlm.py --prep fixed` | native と同等 |
| Q3 | 「文章側のグラフは作り直さずに使える」は本当か | `check_wiring.py`（A↔B） | SRQ なしで logits の相対 L2 が 1e-4 以下、1 位のトークンが全位置で一致 |
| Q4 | 1 トークンずつ流しても同じか（端末の decode グラフの動き） | `check_wiring.py`（B↔C） | SRQ なしで 1e-4 以下。SRQ ありは差を記録（丸めの境目で転ぶことがある） |
| Q5 | 画像の特徴は、端末のグラフの入口の目盛り（文章で較正）に収まるか。収まらないと答えは変わるか | `check_wiring.py`（値域・切り詰め） | 範囲の外が 0%、または切り詰めても答えが変わらない |
| Q6（09-28 追加） | 端末と同じ流し方（プロンプトも 1 トークンずつ、SRQ あり）で、答えの品質は保たれるか | `ref_vlm.py --feed step` | HF の流し方と正誤の入れ替わりがほぼ半々で、点が同等 |

Q3〜Q6 は Phase 3（端末でつなぐ）の前提。Q5 が駄目なら、文章グラフの入口のかたまり（層 0〜8）を画像込みで較正し直す必要がある（要件定義書 R-2）。

## 2. 動かす環境

| 項目 | 内容 |
|---|---|
| 場所 | 利用者の Windows PC（RTX 5070 Ti）。クラウドのサンドボックスからはモデルも公開データも取れないため |
| Python | **EdgeSLM_MultiHead_Multi の venv を借りる**（torch 2.11.0+cu128 / torchvision 0.26 / transformers 5.14.1 / pillow / huggingface_hub）。**パッケージは足さない・変えない** |
| なぜその venv か | EdgeSLM_Android の venv は torch が CPU 版で、QAT 版の生成が 1 トークン約 4.7 秒（Gemma4 P0B の実測）。126 問 × 3 条件では 10 時間を超える。CUDA 版なら数十分の見込み |
| モデル | `EdgeSLM_AndroidGemma4/work/hf/gemma-4-E2B-it-qat-mobile-transformers`（読むだけ。環境変数 `VLM_HF_DIR` で差し替え可） |

```bash
cd D:/AI/source/n_chap5/EdgeSLM_AndroidVLM
../EdgeSLM_MultiHead_Multi/.venv/Scripts/python.exe -m app.pipeline.fetch_eval_data
../EdgeSLM_MultiHead_Multi/.venv/Scripts/python.exe -m app.pipeline.check_wiring
../EdgeSLM_MultiHead_Multi/.venv/Scripts/python.exe -m app.pipeline.ref_vlm run --limit 2      # 動作確認
../EdgeSLM_MultiHead_Multi/.venv/Scripts/python.exe -m app.pipeline.ref_vlm run                # 本番（native × 70/140/280）
../EdgeSLM_MultiHead_Multi/.venv/Scripts/python.exe -m app.pipeline.ref_vlm run --prep fixed --budgets 70
../EdgeSLM_MultiHead_Multi/.venv/Scripts/python.exe -m app.pipeline.ref_vlm report
```

## 3. ファイルの構成と役割

```
app/pipeline/
├── vlm_common.py      置き場所・モデルの数字・小さな補助（UTF-8 出力、JSON の安全な書き込み）
├── eval_defs.py       評価セットの定義（どの画像に何を聞き、何が含まれれば正解か）
├── judge.py           答えの自動判定（表記の正規化・キーワード・文字の再現率）。自己テスト付き
├── synth_images.py    OCR・画面・文書の合成画像を作る（作り直さない）
├── fetch_eval_data.py 公開データを取得し、合成画像と合わせて manifest.json を作る
├── hf_vlm.py          PC で QAT 版を画像つきで動かす部品（読み込み・前処理・生成・端末と同じ入れ方）
├── ref_vlm.py         品質の評価（Q1・Q2）と集計
├── check_wiring.py    配線の検算と値域（Q3〜Q5）
└── onnx_scales.py     量子化済み ONNX から入口の目盛りを読む（onnx パッケージ無しで）
```

依存の向き: `ref_vlm` / `check_wiring` → `hf_vlm` → `vlm_common`。`fetch_eval_data` → `eval_defs` / `synth_images` / `judge`。
transformers を import するのは `hf_vlm` の関数の中だけ（評価セットの取得や判定は torch 無しでも動く）。

## 4. データの流れ

```
eval_defs.py ──┐
               ├─ fetch_eval_data.py ─→ work/eval/images/{vg,wild,car,synth}/  ＋  work/eval/manifest.json
synth_images ──┘                                     │
                                                      ▼
                 ref_vlm.py run ─→ results/P1_ref_vlm/{prep}_{budget}.json ─→ ref_vlm.py report ─→ summary.md
                 check_wiring.py ─→ results/P1_check_wiring/wiring.json
```

manifest.json の 1 件（1 問）:

| 欄 | 中身 |
|---|---|
| id / set / tags | 問題の名前、出どころ（vg・wild・car・synth）、切り口（ocr・car・driver_state・small など） |
| image | `work/eval/` からの相対パス |
| question / answer_ref | 質問、元データの正解の文（あれば） |
| keywords / ref_text | 判定の語（グループのリスト）または正解の全文（OCR の読み上げ） |
| max_new | 生成の上限 |
| source / license | 出どころの URL など、ライセンス |

## 5. 設計の要点

### 5.1 プロンプトは端末と同じ形にする

```
<bos><|turn>system\n画像についての質問に、日本語で簡潔に答えてください。<turn|>\n
<|turn>user\n<|image>{画像トークン × N}<image|>{質問}<turn|>\n<|turn>model\n
```

- chat template が `<bos>` を書くので、トークン化では特別なトークンを足さない（`add_special_tokens=False`）。先頭が `<bos>` 1 つであることを毎回確かめる
- 画像トークン数はプロセッサに `images_kwargs={"max_soft_tokens": N}` で渡す。渡し方を間違えると黙って既定の 280 になるので、実際の数が N 以下であること（fixed ならちょうど N）を毎回確かめる

### 5.2 前処理の 2 方式

| | native（HF の既定） | fixed（端末と同じ） |
|---|---|---|
| 大きさ | 縦横比を保ち、パッチ数が N×9 以下になる最大の大きさ（48px の倍数） | 固定の格子（70: 336×480、140: 480×672、280: 672×960） |
| 余り | パディング（位置 -1。ViT の attention で無視、プーリング後に取り除く） | 黒で埋める（レターボックス）。黒い部分も画像として見る |
| 画像トークン数 | 画像ごとに変わる（N 以下） | 常に N |
| 端末で使えるか | 使えない（形が変わる） | 使える |

どちらも EXIF の向きを先に反映する。

### 5.3 「端末と同じ入れ方」（hf_vlm の device_style_*）

端末では、画像エンコーダ（プロセス A）が出した N×1536 を、文章側（プロセス B）の既存グラフへ 1 トークンずつ入れる。
PC ではこれを HF の部品で再現する:

| 位置 | inputs_embeds | ple_tok |
|---|---|---|
| 文章 | 埋め込み表を引く（√1536 倍込み） | PLE の表を引く |
| 画像 | 画像の特徴そのもの（√1536 倍しない） | pad（ID 0）の行 |

これを `language_model` だけに渡す（HF の `Gemma4Model.forward` が中でやっていることと同じ）。
位置は連番、マスクは因果マスク（E2B は `use_bidirectional_attention = null`）。

### 5.4 SRQ あり・なしの 2 本立て

SRQ（活性の 8bit の丸め）ありが「端末で目指す答え」。ただし丸めは微小な差を増幅する（格子の境目で 1 段転ぶ）ので、
**配線が正しいかは SRQ なしで判定**し、SRQ ありは差の大きさを記録する。

### 5.5 途中で止めても続きから

`ref_vlm` は 1 問ごとに結果を書く（一時ファイルに書いてから置き換える）。同じ条件で動かし直すと、済んだ問題は飛ばす。
画面への表示は書いた後に行う（表示の失敗で結果を失わない）。出力は UTF-8 にそろえる（日本語版 Windows の cp932 対策）。

### 5.6 追加（2026-09-28）: 初回の結果を受けた修正

| 対象 | 修正 | 理由 |
|---|---|---|
| check_wiring | 値域・切り詰めを SRQ あり・なしの両方で測る。`--srq {both,on,off}`。結果は `wiring_b{budget}_srq-{mode}.json`（初回の `wiring.json` は残す）。差が最大の位置が画像か文章かも記録 | 初回は SRQ ありの回だけで測り、切り詰めの効果が丸めの揺れに埋もれた |
| hf_vlm | `generate_device_style()`（端末と同じ入れ方で生成。プロンプトを 1 トークンずつ／一度に）、`generate_step()` | SRQ ありでは流し方で logits の 1 位が 77〜89% しか一致しない。生成の品質で確かめる |
| ref_vlm | `--feed step`、結果に生成したトークンの ID（`gen_ids`）を保存、report に「流し方の比較」、`rejudge` コマンド | 同上。判定を直したときにモデルを動かし直さずに付け直す |
| judge | 行頭の箇条書きの番号を消す。芯が漢字・数字 1 文字なら語尾を落とす | 「5つ…4. …」が正解「4つ」に当たった。「鳥です」のまま探して「カモメのような鳥」を落とした |

## 6. 既知の限界

- 自動判定はキーワードの一次判定。説明を求める問題（keywords=None）は判定しない。最終的な結論は答えの文を読んで出す
- `check_wiring` の切り詰めの実験は、端末の入口 4 点（inputs_embeds・prescaled・ple_tok・ple_all）の範囲だけを真似る。
  グラフの中の各演算の 16bit の目盛り（これも文章で較正）での切り詰めは真似ていない。そちらは Phase 3 で端末の値と比べる
- PC の時間は端末の見込みに使えない
- 連続フレームの評価は Phase 4（データも未準備）
