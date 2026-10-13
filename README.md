# EdgeSLM_AndroidVLM

**画像を読める LLM（VLM）を、スマホ（Xperia 5 III / Snapdragon 888 / HTP v68）の NPU で動かす検証。**
2026-09-27 開始。日本語での応答と、日本語の文字の読み取りを評価に含める。

## 何をするプロジェクトか

これまでの検証で、SD888 の NPU では**文章だけの LLM**が動くようになった（Qwen2.5-0.5B / 1.5B、Gemma 4 E2B）。
本プロジェクトでは、そこに**画像エンコーダ（ViT）と射影層**をつなぎ、画像を見て日本語で答えるところまでを NPU 中心で動かす。

対象の用途は次の4つ（すべて）:

| 用途 | 例 |
|---|---|
| 汎用 VQA・キャプション | 「この写真に何が写っている？」 |
| 車室内・車外カメラ理解 | 乗員の様子、前方の状況 |
| OCR・文書・画面理解 | 標識・看板・画面の日本語を読む |
| 連続フレーム・準リアルタイム | カメラ映像を数秒おきに説明する |

## 方針（2026-09-27 時点）

**本命は Gemma 4 E2B（QAT モバイル版）の画像部分を足すこと。** 理由は [DoC/P0_既存資産の棚卸し.md](DoC/P0_既存資産の棚卸し.md) 4章:

- 文章部分は [EdgeSLM_AndroidGemma4](../EdgeSLM_AndroidGemma4) で**全ノード NPU・日本語の会話・端末のトークナイザまで済んでいる**
- そのグラフの入口は、すでに**トークン ID ではなく埋め込み（`inputs_embeds`）**。画像の特徴をそのまま差し込める
- E2B は画像トークンにも**通常の因果マスクと1次元の位置**を使う（Qwen2.5-VL の M-RoPE や Qwen3-VL の DeepStack が無い）→ **文章側のグラフは作り直さずに使える見込み**
- 取得済みのモデル（`google/gemma-4-E2B-it-qat-mobile-transformers`）に**画像エンコーダ（16層 ViT）も入っており、重みは INT8・活性の目盛りも QAT 済み**
- 画像エンコーダを**別プロセス**に置けば NPU のメモリ上限（プロセスごと約 1,710MiB）に当たらない。233MiB ＋ 1,467MiB の同居は実測済み

## 状態

| Phase | 内容 | 状態 |
|---|---|---|
| **0** | 既存資産の棚卸し・モデル選定の根拠づくり | **完了**（[P0](DoC/P0_既存資産の棚卸し.md)） |
| **1** | PC で画像つきの日本語評価（4用途の評価セット 126 問・画像トークン数ごとの品質・「文章グラフは無改造で済む」の検算・入口の目盛りに収まるか・端末と同じ流し方で品質が保たれるか） | **完了（2026-09-28）**。記録は [P1](results/P1_20260928_pc_eval.md)（まとめは 9 節） |
| **2** | 画像エンコーダ（ViT）単体を v68 へ（解像度を固定・INT8・PC とのコサイン類似度と時間） | **2-1 完了**: 自前の配線は HF と一致（1e-5）、ONNX に書き出せた。画像エンコーダは SRQ の丸めにとても敏感（特徴の cos 0.84 まで揺れる）が、**揺れた特徴でも答えの品質は同じ**（42 → 46 問）（[記録](results/P2_20260928_vit_export.md)）。**2-2 量子化済み**: 2 版（SRQ あり・なし）とも C-1・C-8 を満たし 150MiB。SRQ なしの版は HF と cos 0.997 以上＝算術の誤差は小さい。量子化済みの特徴で答えの品質 44/123（基準 42）で合格（[記録](results/P2_20260928_vit_quant.md)）。**2-3 変換済み**: 2 版とも v68 の context binary（165〜178MiB）になり C-1〜C-5 合格。1 回に約 1GB を DDR に出し入れする見積もり（[記録](results/P2_20260929_vit_ctx.md)）。**2-4 の 1 回目**: 全部 NPU で動く（strict）・1 枚 158〜195ms・NPU 152MiB。ただし出力が壊れていた。**原因を特定**: パッチの埋め込みの、重みがほぼ 0 のチャネル（222 本）のスケールが小さすぎて NPU で表せなかった。スケールに下限を付けて作り直した（C-14）。**直した後: 端末の出力は HF と cos 0.991 以上**（PC の QDQ との差は 7.4%）、1 枚 159〜195ms（[記録](results/P2_20260929_vit_device.md) 7・8 節）。**2-5 完了**: 端末の特徴で 126 問 **45/123**（基準 42）、文章側（1,467MiB）と別プロセスで同居できた（同 9 節）。**70 トークン・横長で Phase 2 の問いは全部答えが出た**。残りは 2-6（280 トークン・縦長）。設計は [基本設計 Phase 2](DoC/基本設計_Phase2.md) |
| **3** | 端末でつなぐ（ViT プロセス → 特徴 → 既存の Gemma 4 グラフへ `inputs_embeds` として流す） | **70 トークン・横長で完了（2026-09-29）。残りはカメラで撮る手動の確認など**（[記録](results/P3_20260929_device_vlm.md) 6 節）。**端末の中だけで写真 → 日本語の答えが出た**（モデルは 2 つとも NPU）。前処理は PC と 108/108 でビット一致、画面が作るプロンプトは HF と一致。**端末だけで 126 問 48/123**（同じ画像の特徴で PC の文章側 45/123、入れ替わり 8↑・5↓で差があるとは言えない）。最初の文字まで 8.3 秒、答えは 1 トークン 69〜78ms（画面・評価の中央値）、画像の読み取り 1.17 秒。モデルを開く途中に OS（lmkd）に止められることがあり（画面の無い裏側で開こうとした 10 回中 3 回プロセスが消えた）、**画面を前に出したまま開く入口を足して 5/5 回、止められなかった**。設計は [基本設計 Phase 3](DoC/基本設計_Phase3.md) |
| 4 | 用途別の評価（日本語・速度・メモリ・発熱・連続フレーム） | 未着手 |
| **5** | 最適化（まとめ読み・画像エンコーダの常駐・小さな改善） | **まとめ読み・常駐・小さな改善まで完了（2026-09-30）**。16 トークン版の文章モデル 1 本で読むのも答えを作るのもやる（1,467.1MiB、変換 3.9 時間）。**最初の文字まで 8.3 秒 → 0.6 秒**（画面 0.8〜1.0 秒）、答えの 1 トークン 1.12 倍（87.0ms）、**画像エンコーダ 1.0〜1.2 秒 → 0.33 秒**（開いたまま。読み込み込みで 1.2 → 0.46 秒）。品質は 45/123（1 トークン版 48、差があるとは言えない）だが HF との一致は下がった（課題）。**system 文を「添付の画像を見て、…」に変えて 54/123**。答えが上限で切れたら印。設計と結果は [基本設計 Phase 5](DoC/基本設計_Phase5.md) 8 章、いまの全体は [システム設計書](DoC/システム設計書.md) **精度の切り分けも完了（2026-10-03）: 文章側は元の 16 トークン版で確定**（[記録](results/P5_20261003_t16_accuracy.md)） |

計画と合格の条件は [DoC/要件定義書.md](DoC/要件定義書.md)。

### Phase 1 で分かったこと（2026-09-28、[記録](results/P1_20260928_pc_eval.md)）

- **文章側のグラフは無改造で使える（PC で確定）。** HF の本来の経路と「端末と同じ入れ方」（画像の位置は inputs_embeds＝画像の特徴、ple_tok＝pad の行）の logits が**ビット単位で一致**
- **画像の特徴は端末の入口の目盛り（文章だけで較正）にほぼ収まる。** はみ出すのは 1 画像あたり 1〜2 個だけ → 入口の較正のやり直しは要らない見込み
- **画像トークン数で品質が大きく変わる。** 自動判定で 0.37（70）→ 0.47（140）→ 0.61（280）。画面・文書のきれいな文字は 70 で足りるが、実写の細部（メーター・標識・信号）は 280 が要る。70 では「写っていない」と言い切る誤りが出る
- **端末と同じ固定の格子（レターボックス）でも品質は落ちない**（70 で 0.35 と 0.37）
- 280 でも苦手: 実写の日本語の文字、日本の名所の名前、数え上げ

追加の確認（2026-09-28、[記録 8 節](results/P1_20260928_pc_eval.md)）:

- (a) **切り詰めの影響は無視できる**（SRQ なしで測り直し、4 問とも答えが同じ）→ 入口の較正のやり直しは不要（確定）
- (b) **端末と同じ 1 トークンずつの流し方でも品質は同じ**（70: 42 → 43 問、280: 68 → 65 問、入れ替わりは合計 7↑・9↓で偏りなし）。1 問ずつの正誤は SRQ の揺れで 1 割ほど入れ替わるので、端末と PC は全体の点で比べる
- 固定の格子（横長）は、280 で**縦長の写真にだけ弱い**（26 → 21 問、横長は 46 と 46）。縦撮りを読むなら縦長用の格子が要る

## 動かす（Phase 1）

PC（Windows・RTX 5070 Ti）で、**EdgeSLM_MultiHead_Multi の venv を借りる**（CUDA 版の torch と transformers 5.14.1 が入っている。パッケージは変えない）。

```bash
cd D:/AI/source/n_chap5/EdgeSLM_AndroidVLM
PY=../EdgeSLM_MultiHead_Multi/.venv/Scripts/python.exe     # Git Bash の場合。PowerShell なら $PY = "..\EdgeSLM_MultiHead_Multi\.venv\Scripts\python.exe"

$PY -m app.pipeline.judge                          # 判定の自己テスト（数秒）
$PY -m app.pipeline.fetch_eval_data                # 公開データの取得と評価セットの一覧（数分）
$PY -m app.pipeline.check_wiring                   # 配線の検算と値域（数分〜）
$PY -m app.pipeline.ref_vlm run --limit 2          # 各出どころ 2 問ずつで動作確認
$PY -m app.pipeline.ref_vlm run                    # 本番: native × 70/140/280（126 問 × 3）
$PY -m app.pipeline.ref_vlm run --prep fixed --budgets 70
$PY -m app.pipeline.ref_vlm report                 # 集計 → results/P1_ref_vlm/summary.md
```

途中で止めても、同じコマンドで続きから動く。

### Phase 1 の追加の確認（2026-09-28）

PowerShell の場合（Git Bash なら `& $PY` を `$PY` に読み替える）:

```powershell
cd D:\AI\source\n_chap5\EdgeSLM_AndroidVLM
$PY = "D:\AI\source\n_chap5\EdgeSLM_MultiHead_Multi\.venv\Scripts\python.exe"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

& $PY -m app.pipeline.judge                                                # 判定の自己テスト（直した分を含む）
# (a) 切り詰めの影響を、SRQ の丸めの揺れ抜きで測る（数分）→ results/P1_check_wiring/wiring_b70_srq-off.json
& $PY -m app.pipeline.check_wiring --srq off
# (b) 端末と同じ流し方（プロンプトも 1 トークンずつ）で生成して、HF の生成と比べる
& $PY -m app.pipeline.ref_vlm run --prep fixed --budgets 70 --feed step      # fixed_70 と比べる（30 分前後）
& $PY -m app.pipeline.ref_vlm run --prep fixed --budgets 70 --refill-ids     # （任意）fixed_70 に gen_ids を足す。最初のトークンの一致も見られる
& $PY -m app.pipeline.ref_vlm run --prep fixed --budgets 280                # 端末の形で 280 も（HF の流し方）
& $PY -m app.pipeline.ref_vlm run --prep fixed --budgets 280 --feed step
# 判定を直した分を JSON に書き戻してから集計（モデルは動かさない。report も集計のたびに付け直す）
& $PY -m app.pipeline.ref_vlm rejudge
& $PY -m app.pipeline.ref_vlm report
```

report の「流し方の比較」の表に、同じ条件の HF（一度に）と step（1 トークンずつ）の答えの一致数・正誤の入れ替わりが出る。

## 動かす（Phase 2-1: 画像エンコーダの書き出し）

2 つの venv を使い分ける（capture は Phase 1 の HF、書き出しは onnx の入った venv）。PowerShell の場合:

```powershell
cd D:\AI\source\n_chap5\EdgeSLM_AndroidVLM
$PY1 = "D:\AI\source\n_chap5\EdgeSLM_MultiHead_Multi\.venv\Scripts\python.exe"   # Phase 1 と同じ（CUDA・HF）
$PY2 = "D:\AI\source\n_chap5\EdgeSLM_Android\.venv\Scripts\python.exe"           # onnx・onnxruntime・onnxscript
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

& $PY1 -m app.pipeline.capture_vit                 # HF の正解と較正の入力（CPU で数分）→ work/capture/vit_b70_21x30*.npz
& $PY2 -m app.pipeline.export_vit check            # 自前の配線 ↔ HF（数分）
& $PY2 -m app.pipeline.export_vit export           # ONNX・spec・較正データ → work/ops/vit_b70_21x30/
& $PY2 -m app.pipeline.export_vit ortcheck         # ONNX（ORT CPU）↔ 自前・HF
```

結果の数値は `results/P2_vit_export/`、値の範囲は `work/capture/vit_b70_21x30_meta.json`。

2-1 の追加（2026-09-28）: 自前の配線の特徴で答えの品質を見る（各 25 分前後）

```powershell
& $PY2 -m app.pipeline.export_vit check                                      # SRQ ありの層ごとの差も出る（数分）
& $PY1 -m app.pipeline.ref_vlm run --prep fixed --budgets 70 --vision own        # 自前（SRQ あり）
& $PY1 -m app.pipeline.ref_vlm run --prep fixed --budgets 70 --vision own_nosrq  # 自前（SRQ なし）
& $PY1 -m app.pipeline.ref_vlm report                                         # 「基準との比較」の表
```

## 動かす（Phase 2-2: 画像エンコーダの量子化）

```powershell
& $PY2 -m app.pipeline.quantize_vit                    # 前処理・較正・量子化 → QAT の値 → C-8（SRQ あり・なしの 2 版、実測 4 分）
& $PY2 -m app.pipeline.eval_qdq_vit acc                # 評価用 16 枚で HF と比べる（数分）
& $PY2 -m app.pipeline.eval_qdq_vit feats --variant srq     # 全画像（108 枚）の特徴（3〜5 分）
& $PY2 -m app.pipeline.eval_qdq_vit feats --variant nosrq
& $PY1 -m app.pipeline.ref_vlm run --prep fixed --budgets 70 --vision npz:work/feats/vit_b70_21x30_srq.npz     # 答えの品質（25 分）
& $PY1 -m app.pipeline.ref_vlm run --prep fixed --budgets 70 --vision npz:work/feats/vit_b70_21x30_nosrq.npz
& $PY1 -m app.pipeline.ref_vlm report
```

生成物は `work/ops/vit_b70_21x30/vit_b70_21x30_{srq,nosrq}_final_qdq.onnx`（2-3 で QAIRT に渡す）、結果は `results/P2_vit_quant/`。

## 動かす（Phase 2-3: 画像エンコーダを v68 の context binary に）

```powershell
& $PY2 -m app.pipeline.build_vit_ctx                   # SRQ なし → ありの順に、変換 → context binary → 検査（C-1〜C-5）→ EPContext で包む（実測 1 版 46〜56 分）
& $PY2 -m app.pipeline.build_vit_ctx --variants nosrq  # 片方だけ
```

中で `scripts/build_context.sh`（Git Bash・QAIRT 2.48・`--htp_socs sm8350`）を呼ぶ。GPU を使わないので、`ref_vlm` と同時に動かしてよい。
生成物は `work/ops/vit_b70_21x30/vit_b70_21x30_{nosrq,srq}_epctx.onnx` ＋ 同じ場所の `.bin`（2-4 で端末に送る）、結果は `results/P2_vit_ctx/`。

## 動かす（Phase 2-4: 端末の NPU で動かす）

1 回だけ: VLM 検証アプリ（`android_app/`、`org.edgeslm.vlm`）をビルドして端末に入れる。Gemma4 のアプリを複製し、
applicationId・表示名だけ変えたもの（Gemma4 のアプリはそのまま端末に残る）。

```powershell
cd D:\AI\source\n_chap5\EdgeSLM_AndroidVLM\android_app
$env:JAVA_HOME = "D:\Android\jdk17"
.\gradlew.bat --offline assembleDebug                                                   # 初回は数分
& D:\Android\sdk\platform-tools\adb.exe install -r app\build\outputs\apk\debug\app-debug.apk
cd ..
```

測る（端末を USB でつないだまま。画面ロックは外しておく）:

```powershell
& $PY2 -m app.pipeline.device_vit check          # 端末とアプリの確認（数秒）
& $PY2 -m app.pipeline.device_vit acc            # Q9・Q10: 評価用 16 枚。端末 ↔ PC の QDQ ↔ HF（2 版、モデル 340MB を送るので初回 数分）
& $PY2 -m app.pipeline.device_vit bench          # Q11: 1 枚の時間（2 版 × 23 回）
& $PY2 -m app.pipeline.device_vit mem            # Q11: NPU のメモリ（dma-buf）・ロードの時間
& $PY2 -m app.pipeline.device_vit coexist        # Q11: Gemma4 の会話アプリ（文章側）がモデルを開いたまま、画像エンコーダを開けるか
& $PY2 -m app.pipeline.device_vit feats --variant srq     # 2-5: 全 108 枚を端末で → work/feats/vit_b70_21x30_srq_dev.npz
& $PY1 -m app.pipeline.ref_vlm run --prep fixed --budgets 70 --vision npz:work/feats/vit_b70_21x30_srq_dev.npz   # 2-5: 端末の特徴で 126 問（20〜25 分）
```

結果は `results/P2_vit_device/`。`coexist` は Gemma4 の会話アプリを起動するだけで、中身は変えない（最後に両方とも止める）。

切り分け（2026-09-29、端末の出力が壊れていたため）: 途中の値も出力に足した「探り用」の版を作り、端末と PC を途中ごとに比べる。

```powershell
& $PY2 -m app.pipeline.probe_vit build            # 2 層版（数分〜10 分）→ 16 層版（約 1 時間）を作る。--layers 2 で 2 層版だけ
& $PY2 -m app.pipeline.probe_vit device           # 端末で動かし、途中の値ごとに「端末 ↔ PC の QDQ」を表にする（数分）
```

直したあと（2026-09-29、C-14: 重みのスケールに下限）:

```powershell
& $PY2 -m app.pipeline.probe_vit build --layers 2          # 2 層版を作り直す（5 分）
& $PY2 -m app.pipeline.probe_vit device --layers 2         # パッチの埋め込み・層 0・層 1 が PC と合うか（1 分）
& $PY2 -m app.pipeline.quantize_vit --variants nosrq       # 本番の SRQ なしの版を量子化し直す（前処理は済みなので数分）
& $PY2 -m app.pipeline.build_vit_ctx --variants nosrq      # 変換し直す（約 1 時間）
& $PY2 -m app.pipeline.device_vit acc --variants nosrq     # Q10（数分）。合えば bench / mem / coexist / feats --variant nosrq
```

2-5（端末の特徴で答えの品質）:

```powershell
& $PY2 -m app.pipeline.eval_qdq_vit feats --variant nosrq       # PC の QDQ で全 108 枚（端末と比べる相手、3 分）
& $PY2 -m app.pipeline.device_vit feats --variant nosrq         # 端末で全 108 枚（数分）
& $PY2 -m app.pipeline.device_vit coexist                       # 文章側との同居（数分）
& $PY1 -m app.pipeline.ref_vlm run --prep fixed --budgets 70 --vision npz:work/feats/vit_b70_21x30_nosrq_dev.npz   # 端末（25 分）
& $PY1 -m app.pipeline.ref_vlm run --prep fixed --budgets 70 --vision npz:work/feats/vit_b70_21x30_nosrq.npz       # PC の QDQ（25 分）
& $PY1 -m app.pipeline.ref_vlm report
```

## 動かす（Phase 3: 端末でつなぐ。写真 → 日本語の答えを端末の中だけで）

設計は [基本設計 Phase 3](DoC/基本設計_Phase3.md)。アプリに前処理（C++、PIL の BICUBIC を再現）・画像の特徴を文章側に入れる仕組み・
画面（VlmActivity、ランチャーの「VLM デモ」）・画面を前に出したまま job を動かす入口（RunnerHoldActivity、2026-09-29 夕方に追加）を足したので、**ビルドし直して入れ直す**。

```powershell
cd D:\AI\source\n_chap5\EdgeSLM_AndroidVLM\android_app
$env:JAVA_HOME = "D:\Android\jdk17"
.\gradlew.bat --offline assembleDebug
& D:\Android\sdk\platform-tools\adb.exe install -r app\build\outputs\apk\debug\app-debug.apk
cd ..
& $PY2 -m app.pipeline.setup_vlm_demo              # 3-2: 文章側のモデル等を端末の中で写す（初回 数分、約 2.8GB）・設定を書く
& $PY2 -m app.pipeline.device_vlm prep             # 3-3: 端末の前処理 ↔ PC（PIL）がビット一致するか（数分）
& $PY1 -m app.pipeline.device_vlm uicheck          # 3-5: 画面が作るプロンプト・前処理・答え・時間（1〜2 分）
& $PY1 -m app.pipeline.device_vlm eval             # 3-4: 端末だけで 126 問（30 分前後。Phase 1 の venv）
& $PY1 -m app.pipeline.ref_vlm report              # 端末だけの結果も表に並ぶ
& $PY2 -m app.pipeline.device_vlm holdcheck        # 画面を前に出したまま 1.5GB を開くと OS に止められないか（5 回、2 分）
```

- **VLM デモの画面を開いたまま評価しない**（同じプロセスで文章側をもう 1 つ開くことになる）。`eval` は最初にアプリを止める
- `eval` は 28 問ずつの job に分け、**画面を前に出したまま**（RunnerHoldActivity）動かす。スマホに「実行中」と出る。
  モデルを開く途中で OS に止められても、30 秒待って 2 回までやり直す。**止まっても同じコマンドで続きから**（最初からは `--fresh`）
- 古いアプリ（RunnerHoldActivity の無いもの）では `--no-hold`

そのあと、端末のランチャーから「VLM デモ」を開き、「写真を撮る」→ 質問（機内モードでも動く）。

### Phase 3 で分かったこと（2026-09-29、[記録](results/P3_20260929_device_vlm.md)）

- **端末の中だけで写真 → 日本語の答えが出る。** 前処理（C++、CPU）→ 画像エンコーダ（NPU、別プロセス）→ 文章側（NPU）。モデルは 2 つとも CPU への逃げ道なし
- **端末の前処理は PC と同じ値**（108/108 でビット一致）。C++ で PIL の BICUBIC を固定小数点まで写した
- **端末だけで解いても品質は PC と同等**（48/123。同じ画像の特徴で PC の文章側 45/123。端末側のキーワードの偶然の当たり 2 問を除くと 46）。
  ただし端末の答えは長め（上限で切れる答えが 44 問、PC 18 問）で、PC では出なかった「画像が提供されていません」が 3 問あった
- **待ち時間**: 最初の文字まで 8.3 秒（大半はプロンプト約 109 トークンを 1 つずつ読む時間）、答えは 1 トークン 69〜78ms（約 13〜14.5 tok/s）、画像の読み取り 1.17 秒（NPU は 166ms、大半は別プロセスを起こして開く時間）
- **1.5GB のモデルを開く途中は、画面を前に出しておく。** 開く間は一時的に約 3GB 使い、画面の無い裏側だと OS に止められることがある（10 回中 3 回プロセスが消えた）。画面が前なら止められなかった（5/5、以前の検証で 25/25）

## 動かす（Phase 5: まとめ読み・画像エンコーダの常駐）

設計は [基本設計 Phase 5](DoC/基本設計_Phase5.md)。アプリ（C++ の複数トークンの経路・画像エンコーダを開いたまま・切れた印）を変えたので**ビルドし直して入れ直す**。

```powershell
& $PY2 -m app.pipeline.build_t16                   # 16 トークン版: 書き出し〜繋ぐ（1〜3 時間）。段ごとに再開できる。窓を閉じない・中をクリックしない
& $PY2 -m app.pipeline.check_t16                   # PC の CPU で 1 トークン版・HF と比べる（数分）
& $PY2 -m app.pipeline.build_t16 --steps convert   # 変換（約 4 時間）
& $PY2 -m app.pipeline.setup_vlm_demo --text t16 --system "添付の画像を見て、質問に日本語で簡潔に答えてください。"
& $PY2 -m app.pipeline.device_vlm holdcheck --text t16   # 開くときに止められないか・速さ（5 回）
& $PY1 -m app.pipeline.device_vlm uicheck --text t16     # 画面のプロンプト・時間
& $PY1 -m app.pipeline.device_vlm eval --text t16        # 126 問（10 分前後）。--system "…" で system 文を変えて比べる
& $PY1 -m app.pipeline.record_demo --restart --tag after # 画面の録画
```

### Phase 5 で分かったこと（2026-09-30）

- **NPU の 1 回の時間は重みを読む時間でほぼ決まる。** 16 トークン版も 1 トークン版も 1 回に約 1.55GB を DDR から読む（変換のログ）。16 トークンまとめて読んでも 1 回の時間は 1.1 倍前後
- **最初の文字まで 8.3 秒 → 0.60 秒**（評価の中央値。画面 0.78〜1.03 秒）。答えの 1 トークンは 77.7 → 87.0ms（1.12 倍）
- **品質の点数は同等（45/123 と 48/123）だが、HF と同じ答えを出す割合は下がった**（答えの 1 トークン目 97 → 78/126）。**同じグラフを PC の CPU で動かすと HF と 13/16 一致し文字も正しく読めた（端末は 5/16）→ 原因はグラフではなく端末での実行**（`diag_t16`）。**さらに、C++ の入れ方は正しく、NPU での 16 トークン版の計算が CPU と大きく違う**（`probe_t16`: Python の入れ方で NPU に通すと端末の答えを 7/7 再現、同じ入力での logits の差 中央値 62%、K/V の差は層 1 から 2 割）。ただし**層 0〜14 のずれは 1 トークン版にも同じだけある**（基準、2026-10-01）。16 トークン版だけ悪いのはプロンプトの最後の回の logits（38〜61%、1 トークン版 14〜33%）。**原因は 2 つと特定**（2026-10-01）: ① QAIRT が正規化を HTP の RmsNorm にまとめるとずれる（両方の版、画像の行で大）、② 16 トークン版の PLE の tanh（[16×256]）が 2〜15 行目で壊れる。書き方を変えて部品を避ける直し方（正規化の平均を行列の掛け算に・PLE の tanh を 1 行の形で）が小さいグラフで効いた（`probe_fix`、層 0 の出力の画像の行 150% → 20%）。**全体を書き直した版（`build_t16fix` → `g4t16f_W256`、2026-10-02）は、16 行まとめのずれ（②）は消えた（まとまりの中の 1 行目以降の logits の差 46.8 → 28.5%、1 トークン版と 1 トークン目が同じ 80 → 101/126）が、126 問は 45 → 37/123 に下がった**（「画像が提供されていません」系の答え 0 → 12 問。HF と 1 トークン目が同じ 78 → 83/126）。1 トークン 87 → 101ms。**PC の CPU どうしでも、ORT のグラフ最適化を切るだけで logits が 14% 変わる**（丸め 1 目盛りの違いが 35 層で育つ。`diag_t16fix`）＝ NPU と CPU の「差 2 割」は不具合の物差しにならず、答えで比べる。**2026-10-03 に訂正: 「① RmsNorm がずれる」は誤りで、ずれていたのは基準にした PC の CPU の方**。量子化済みグラフの正規化の途中の値（x²・平均・1/√）の目盛りは文章だけで決めたもので、画像の行でははみ出して切れる（重みのない正規化の出力の行ごとの二乗平均: NPU 1.000、CPU は画像の行で 0.56〜0.93。理想の計算との差: NPU 0.1% 未満、CPU 29〜63%）。NPU は正規化を RmsNorm 1 つにまとめて計算するので正確。平均を行列の掛け算にした書き直しは NPU を CPU の誤った値に合わせるものだったので、t16f は画像が見えなくなった（PC の CPU だけで解かせても「画像が提供されていません」と答える）。tanh だけ直した版 `g4t16t_W256` は元と同じ答え（45/123、HF 78）＝②は効いていない。**結論: 元の 16 トークン版（t16）をそのまま使う**（1 トークン版と同等: 45 と 48、新しい system 文で 54）。`probe_rows --self`（NPU どうしで「まとまりの中」と「1 行だけ」を比べる。CPU を基準にしない）では、**16 行をまとめたことによる大きな不具合は見つからなかった**（1 位の語 15/15 で同じ。深い層の K/V の差は PC の CPU どうしと同じ水準。CPU どうしでも logits は 13% 違う＝量子化したグラフでは「まとめて通す」と「1 行ずつ通す」は丸め方が違う）。NPU だけの小さな違い（0 行目がほかの行の中身で少し変わる、logits で 9%）は残るが、点・1 位の語には出ていない。**利用者の決定（2026-10-03）: 元の 16 トークン版で確定し、精度の調査を閉じる**。数字と経過のまとめは [記録](results/P5_20261003_t16_accuracy.md)
- **system 文を「添付の画像を見て、質問に日本語で簡潔に答えてください。」にすると 45 → 54/123**（入れ替わり 11↑・2↓）。「判断できません」と逃げる答えが減った。デモの設定をこの文にした
- **画像エンコーダ 1.0〜1.2 秒 → 0.33 秒**（開いたまま。読み込み・前処理込みで 1.2 → 0.46 秒）
- 「後」の録画の 1 本目で、2 枚目の写真の後に約 16 秒間タップが効かないことがあった（原因は未特定）。タップのログと押した見た目を足した版の 2 本目（`results/P3_device_vlm/demo/vlm_demo_20260930_102426_after.mp4`）では再現せず、写真 2 枚 × 質問 2 つが通った。同じ写真への 2 問目は最初の文字まで 0.22〜0.24 秒（前は 1.4 秒）

## 構成

| 場所 | 中身 |
|---|---|
| `DoC/` | **システム設計書（いまの形の全体）**・要件定義書・棚卸し（Phase 0）・基本設計（Phase 1〜5）・引き継ぎ一覧・**StudyMemo** |
| `app/pipeline/` | PC 側の道具（各ファイルの先頭に、何をするか・なぜそうするかの説明） |
| `android_app/` | 端末の VLM 検証アプリ（`org.edgeslm.vlm`。2026-09-29 に EdgeSLM_AndroidGemma4 のアプリを複製し、applicationId・表示名だけ変えた。複製元のファイルごとの md5 は `android_app/COPY_MANIFEST.md5`。**元のアプリは変えない**） |
| `results/` | 検証の記録（`P1_…` のように Phase ごと） |
| `work/` | 生成物（評価の画像・一覧など。大きい。記録ではない） |

## ドキュメント

| 文書 | 内容 |
|---|---|
| [DoC/システム設計書.md](DoC/システム設計書.md) | **いま動いている形の全体**（構成・モデル・アプリの部品・処理の流れ・job と設定の約束・グラフの入出力・ファイル配置・性能・設計の判断・制約・動かし方） |
| [DoC/要件定義書.md](DoC/要件定義書.md) | 目的・スコープ・構成・Phase の計画・目標値の案・リスク・約束 |
| [DoC/P0_既存資産の棚卸し.md](DoC/P0_既存資産の棚卸し.md) | 既存の検証で分かっていること、モデル選定の根拠 |
| [DoC/基本設計_Phase1.md](DoC/基本設計_Phase1.md) | Phase 1 の道具の設計（問い・環境・ファイルの役割・データの流れ・設計の要点・限界） |
| [DoC/基本設計_Phase2.md](DoC/基本設計_Phase2.md) | Phase 2 の設計（問い・段階・グラフの形と書き換え・重みと量子化の対応・環境・v68 の制約の当たり方・リスク） |
| [DoC/引き継ぎ一覧.md](DoC/引き継ぎ一覧.md) | 他のプロジェクトから読むだけのもの・複製したコード（md5） |
| [results/P1_20260928_pc_eval.md](results/P1_20260928_pc_eval.md) | Phase 1 の記録（配線の検算・値域・画像トークン数ごとの品質・判定の確かさ・次の一手） |
| [DoC/基本設計_Phase3.md](DoC/基本設計_Phase3.md) | Phase 3 の設計（端末でつなぐ形・作るもの・段階・リスク） |
| [results/P3_20260929_device_vlm.md](results/P3_20260929_device_vlm.md) | Phase 3 の記録（前処理の照合・画面の照合・端末だけで 126 問・OS に止められる件と対策） |
| [DoC/基本設計_Phase5.md](DoC/基本設計_Phase5.md) | Phase 5 の設計と結果（16 トークン版のグラフの約束・端末の複数トークンの経路・画像エンコーダの常駐・小さな改善・Q17〜Q22・課題） |
| [results/P5_20261003_t16_accuracy.md](results/P5_20261003_t16_accuracy.md) | **Phase 5 の精度の切り分けの記録**（16 トークン版で HF と同じ答えが減った理由。4 つの版の 126 問、正規化を理想の計算と比べた結果、NPU どうしの比べ、読み違いと訂正、道具の一覧。結論: 元の 16 トークン版で確定） |
| [DoC/StudyMemo.md](DoC/StudyMemo.md) | **学習メモ**。VLM の構造、Gemma 4 の画像の通り道、NPU に載せる設計、量子化の目盛りと画像、評価の組み立て、検算の考え方、画像エンコーダの書き直し、端末でだけ壊れる、大きいモデルを開くときに OS に止められる、**まとめ読み（prefill と decode）、写真から答えまでを端から端まで** |

## ほかのプロジェクトとの関係

| プロジェクト | 扱い |
|---|---|
| EdgeSLM_AndroidGemma4 | **参照専用（変更しない）**。モデル（`work/hf/`）と量子化済み ONNX は読むだけ。コードは複製して [引き継ぎ一覧](DoC/引き継ぎ一覧.md) に記録 |
| EdgeSLM_MultiHead_Multi | venv を借りる（読むだけ・パッケージは変えない） |
| EdgeSLM_Android / AndroidScale / AndroidDemo / AndroidHead / AndroidGemma4Head | 変更しない。知見（v68 の制約 C-1〜C-13 など）は再調査せずに使う |
