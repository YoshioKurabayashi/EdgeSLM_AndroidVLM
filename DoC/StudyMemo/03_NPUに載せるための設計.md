# 3. NPU に載せるための設計 — 形・時間・メモリ

[← 目次へ](../StudyMemo.md) | [← 前: 2. Gemma 4 の画像の通り道](02_Gemma4の画像の通り道.md) | [次: 4. 量子化の目盛りと画像の特徴 →](04_量子化の目盛りと画像の特徴.md)

> **ソースと動かし方**（2026-10-14 追記）: この章の各節を、どのファイルのどの関数がやっているか・どのコマンドで確かめられるか・どの結果ファイルに数字があるかは [12 章](12_ソースと動かし方.md) の **12.3.2**。実際のコードの抜き出しと 1 行ずつの解説は [13 章](13_コードで読む検証.md) の **13.1**。スマホも GPU も要らない演習は 12.2。各節の終わりには、その節の処理をしている実際のソースの枠がある（2026-10-17〜。`sync_src` で自動で更新）。

---

## 3.1 形を固定する — レターボックス

v68 の NPU（HTP）は**動的な形を扱えない**（EdgeSLM_Android の前提）。グラフは「入力の形」を決めて作る。
ViT の入力の形は「パッチの数×768」なので、**パッチの縦×横の数を固定**しないといけない。

HF の既定（native）は画像ごとに格子が変わる（[2.2節](02_Gemma4の画像の通り道.md)）。そこで:

| 方法 | 中身 | 良い点 | 悪い点 |
|---|---|---|---|
| 引き伸ばし | 縦横比を無視して固定の大きさにする | 画素を全部使う | 形がゆがむ（丸が楕円に） |
| 切り抜き | 真ん中だけを切り出す | ゆがまない | 端が消える（標識が端にあると見えない） |
| **レターボックス** | 縦横比を保って縮小し、余白を黒で埋める | ゆがまず、全体が見える | 黒い部分のパッチが無駄になる |

本プロジェクトは**レターボックス**を採る（`hf_vlm.prepare_image(prep="fixed")`）。車載カメラは横長で固定なので、
実運用では余白はほぼ出ない。縦長の写真（評価セットの一部）では左右に黒帯が出る。

| 画像トークン | 格子（パッチ） | 大きさ | 縦横比 |
|---:|---|---|---|
| 70 | 21×30 | 336×480 | 1:1.43 |
| 140 | 30×42 | 480×672 | 1:1.40 |
| 280 | 42×60 | 672×960 | 1:1.43 |

> ★実測（2026-09-28）: 70 トークンで fixed 0.35、native 0.37。問題ごとの入れ替わりは 7 問↑・9 問↓でほぼ半々で、**差は無い**。
> レターボックスで端末の形を固定してよい。
>
> ★追加（2026-09-28）: 280 トークンでは、**横長の画像は差なし（46 と 46）、縦長の画像だけ落ちた（26 → 21）**。
> 横長の枠（960×672）に縦長の写真を入れると左右が黒帯になり、使える画素が半分以下になるため。
> 車載カメラ（横長）には影響しないが、スマホの縦撮りを読むなら**縦長用の格子（ViT のグラフをもう 1 つ）**が要る。

<!-- src: app/pipeline/hf_vlm.py | gh, gw = grid or C.FIXED_GRIDS\[budget\] | return canvas | note=prepare_image の fixed: レターボックス -->
**ソース** `app/pipeline/hf_vlm.py` L158〜L164（prepare_image の fixed: レターボックス） — md5 353d6037・`sync_src` が自動で更新

```python
  158|     gh, gw = grid or C.FIXED_GRIDS[budget]
  159|     H, W = gh * C.PATCH, gw * C.PATCH
  160|     s = min(W / img.width, H / img.height)
  161|     nw, nh = max(1, round(img.width * s)), max(1, round(img.height * s))
  162|     canvas = Image.new("RGB", (W, H), (0, 0, 0))
  163|     canvas.paste(img.resize((nw, nh), Image.BICUBIC), ((W - nw) // 2, (H - nh) // 2))
  164|     return canvas
```
<!-- /src -->

<!-- src: app/pipeline/vlm_common.py | ^FIXED_GRIDS = \{ | ^PORTRAIT_GRIDS | note=格子の表（横長と、後から足した縦長） -->
**ソース** `app/pipeline/vlm_common.py` L61〜L69（格子の表（横長と、後から足した縦長）） — md5 81595b1f・`sync_src` が自動で更新

```python
   61| FIXED_GRIDS = {
   62|     70: (21, 30),     # 336×480 ピクセル、630 パッチ
   63|     140: (30, 42),    # 480×672 ピクセル、1,260 パッチ
   64|     280: (42, 60),    # 672×960 ピクセル、2,520 パッチ
   65| }
   66| 
   67| # 縦長の格子（2026-09-28 追加）。横長の格子に縦長の写真を入れると使える画素が半分以下になり、280 で品質が落ちた
   68| # （results/P1_20260928_pc_eval.md 8.3 節）。スマホの縦撮りを読むときはこちらのグラフを使う
   69| PORTRAIT_GRIDS = {k: (w, h) for k, (h, w) in FIXED_GRIDS.items()}
```
<!-- /src -->

<!-- src: android_app/app/src/main/cpp/runner/vlm_prep.cpp | const double s = std::min\(static_cast<double>\(W\) / w | canvas.data\(\) \+ \(static_cast<size_t>\(y \+ oy\) | note=vlm_pixels: 端末の C++ でも同じレターボックス -->
**ソース** `android_app/app/src/main/cpp/runner/vlm_prep.cpp` L195〜L214（vlm_pixels: 端末の C++ でも同じレターボックス） — md5 d4375be9・`sync_src` が自動で更新

```cpp
  195|     const double s = std::min(static_cast<double>(W) / w, static_cast<double>(H) / h);
  196|     // Python の round（偶数への丸め）。既定の丸めの向き（最も近い偶数）の nearbyint と同じ
  197|     int nw = std::max(1, static_cast<int>(std::nearbyint(w * s)));
  198|     int nh = std::max(1, static_cast<int>(std::nearbyint(h * s)));
  199|     nw = std::min(nw, W);
  200|     nh = std::min(nh, H);
  201|     std::vector<uint8_t> small;
  202|     const uint8_t* sp = rgb;
  203|     if (nw != w || nh != h) {
  204|         small.resize(static_cast<size_t>(nw) * nh * 3);
  205|         pil_bicubic_resize_rgb(rgb, w, h, small.data(), nw, nh);
  206|         sp = small.data();
  207|     }
  208|     const double t1 = now_ms();
  209|     // 黒い画像に貼る
  210|     std::vector<uint8_t> canvas(static_cast<size_t>(W) * H * 3, 0);
  211|     const int ox = (W - nw) / 2, oy = (H - nh) / 2;
  212|     for (int y = 0; y < nh; ++y)
  213|         std::copy(sp + static_cast<size_t>(y) * nw * 3, sp + static_cast<size_t>(y + 1) * nw * 3,
  214|                   canvas.data() + (static_cast<size_t>(y + oy) * W + ox) * 3);
```
<!-- /src -->

| 目印 | 何をしているか・なぜ |
|---|---|
| `s = min(W / img.width, H / img.height)` | 縦横比を保つ倍率。横と縦のうちきつい方に合わせるので、はみ出さず、もう一方に余白が出る |
| `canvas = Image.new("RGB", (W, H), (0, 0, 0))` | 余白は黒。縮めた写真を真ん中に貼る（`(W - nw) // 2`） |
| `70: (21, 30),     # 336×480 ピクセル、630 パッチ` | 上の表の格子。どれも 3 の倍数で、縦×横＝トークン数×9 |
| `PORTRAIT_GRIDS = {k: (w, h) for k, (h, w) in FIXED_GRIDS.items()}` | 縦長の格子（縦と横を入れ替えただけ）。上の★追加（280 で縦長の写真が落ちた）への対策として 2026-09-28 に足した |
| `std::nearbyint(w * s)` | 端末の C++ も同じ倍率と丸め（Python の round と同じ、偶数への丸め）。縮め方も PIL の BICUBIC と同じにしてあり、評価の写真 108 枚すべてで PC と画素がビット単位で一致した（`results/P3_device_vlm/b70_prep.json`） |

---

## 3.2 画像トークン数と待ち時間

いまの端末の文章グラフは **1 回に 1 トークン**しか食べられない（EdgeSLM_AndroidGemma4 P6）。1 回あたり約 77ms。

### 手を動かす: 最初の文字が出るまで

$$
t_{\text{最初の文字}} \approx t_{\text{ViT}} + (N_{\text{画像}} + N_{\text{文章}}) \times 77\,\text{ms}
$$

| 画像トークン | 文章（system＋質問＋区切り）| 合計 | 最初の文字まで（ViT を除く） |
|---:|---:|---:|---:|
| 70 | 39 | 109 | **約 8.4 秒** |
| 140 | 39 | 179 | 約 13.8 秒 |
| 280 | 39 | 319 | **約 24.6 秒** |

（文章の 39 トークンは Phase 1 の評価の平均。質問が長ければ増える）

**画像トークン数がそのまま待ち時間になる。** 準リアルタイム（数秒おきに説明）には 70 でも苦しい。

### ★実測: トークン数と品質（2026-09-28、PC・自動判定）

| 画像トークン | 全体 | 車 | 合成の画面・文書 | 「写っていない」と言い切る誤り |
|---:|---:|---:|---:|---:|
| 70 | 0.37 | 8/21 | 13/15 | 6〜7 件 |
| 140 | 0.47 | 12/21 | 12/15 | 2 件 |
| 280 | **0.61** | **15/21** | 13/15 | 0 件 |

（★2026-10-17 補足: この表の値は HF の既定の前処理（native）のもの。固定の格子（fixed）は当時 70 だけを測り、全体 0.35 だった（[P1 の記録](../../results/P1_20260928_pc_eval.md) 4.1・6 節）。「写っていない」の 6〜7 件は native/70 が 6 件・fixed/70 が 7 件）

**画面・文書のきれいな文字は 70 で足りるが、実写の細部（メーターの数字・標識・信号の色）は 280 が要る。**
つまり「待ち時間を取るか、品質を取るか」は用途で分けるしかない。280 を使うならまとめ食いが必須になる。

### まとめ食いの効き目

P6 で「1 回に 16 トークン食べるグラフ」は、1 回の時間が **1.08 倍**にしかならないと実測した
（1 回の時間の大半は重みを読むことで、トークンを増やしても計算はあまり増えない）。

$$
t \approx \left\lceil \frac{N}{16} \right\rceil \times 77 \times 1.08\,\text{ms} + \text{グラフの切り替え}\ 2.3\,\text{秒}
$$

| 合計トークン | 1 トークンずつ | まとめ食い 16（切り替え込み） |
|---:|---:|---:|
| 109 | 8.4 秒 | 約 2.9 秒 |
| 319 | 24.6 秒 | 約 4.0 秒 |

ただし、まとめ食い用と生成用の 2 つの大きなグラフ（各 1.5GB）は、RAM 8GB の端末では**同時に常駐できない**（P6 2.6節）。
切り替え（約 2.3 秒）が毎回かかる。→ Phase 5 の課題。

> ★実測（Phase 5、2026-09-30）: **16 トークン版の 1 本で読むのも答えを作るのもやる**形にした（切り替えなし）。
> プロンプト約 107 トークンを 7 回で読み、最初の文字まで**評価の中央値 0.60 秒・画面 0.8〜1.0 秒**（上の見込みの 2.9 秒より速い。
> 切り替えの 2.3 秒が無いため）。答えの 1 トークンは 1.1 倍前後（77.7 → 87.0ms）。詳しくは [10章](10_まとめ読み.md)。

<!-- src: android_app/app/src/main/cpp/runner/job.cpp | ^        if \(Tm <= 1\)$ | const int64_t tid = | note=1 トークン版: プロンプトも 1 回に 1 トークンずつ -->
**ソース** `android_app/app/src/main/cpp/runner/job.cpp` L1522〜L1529（1 トークン版: プロンプトも 1 回に 1 トークンずつ） — md5 bba78f1a・`sync_src` が自動で更新

```cpp
 1522|         if (Tm <= 1)
 1523|         for (size_t t = start_step; t < total_steps; ++t) {
 1524|             if (cancel_flag && *cancel_flag) { gerr = "cancelled"; break; }
 1525|             const double s0 = now_ms();
 1526| 
 1527|             // このステップで食わせる語: 文の途中なら文の t 番目、生成に入ったら直前に選んだ語。画像の位置なら表を引かず
 1528|             // 特徴の 1 行を入れ、PLE は pad（image_pad_id）の行を引く。文字の表の行は 2bit（q2row）か 16bit の値
 1529|             const int64_t tid = (t < prompt.size()) ? prompt[t] : last;
```
<!-- /src -->

<!-- src: android_app/app/src/main/cpp/runner/job.cpp | // ---- プロンプトを Tm 個ずつ読む | prefill_ms \+= wall; continue; | note=16 トークン版（Phase 5）: プロンプトを Tm 個ずつまとめて読む -->
**ソース** `android_app/app/src/main/cpp/runner/job.cpp` L1481〜L1493（16 トークン版（Phase 5）: プロンプトを Tm 個ずつまとめて読む） — md5 bba78f1a・`sync_src` が自動で更新

```cpp
 1481|             // ---- プロンプトを Tm 個ずつ読む（最後の回の答えが 1 つ目のトークン）----
 1482|             size_t t = start_step;
 1483|             int64_t best = 0;
 1484|             while (gerr.empty() && t < prompt.size()) {
 1485|                 if (cancel_flag && *cancel_flag) { gerr = "cancelled"; break; }
 1486|                 const size_t n = std::min<size_t>(static_cast<size_t>(Tm), prompt.size() - t);
 1487|                 std::vector<int64_t> toks(prompt.begin() + static_cast<long>(t), prompt.begin() + static_cast<long>(t + n));
 1488|                 std::vector<bool> imgs(n);
 1489|                 for (size_t k = 0; k < n; ++k) imgs[k] = img_rows > 0 && toks[k] == img_tok;
 1490|                 double ms = 0.0, wall = 0.0;
 1491|                 if (!run_chunk(toks, imgs, &best, &ms, &wall)) break;
 1492|                 t += n;
 1493|                 if (t < prompt.size()) { prefill_ms += wall; continue; }
```
<!-- /src -->

| 目印 | 何をしているか・なぜ |
|---|---|
| `for (size_t t = start_step; t < total_steps; ++t)` | 1 トークン版は、プロンプトも答えも 1 回に 1 トークン。最初の文字まで（N_画像＋N_文章）回まわる＝上の式の「×77ms」 |
| `const size_t n = std::min<size_t>(static_cast<size_t>(Tm), prompt.size() - t);` | 16 トークン版は 1 回に最大 Tm（16）個を読む。回数は ⌈N/16⌉（107 個なら 7 回） |
| `imgs[k] = img_rows > 0 && toks[k] == img_tok;` | まとめて読む 16 個の中にも画像の位置が混ざる。その行には特徴の 1 行を入れる（[1.1 節](01_VLMの構造.md)と同じ入れ方） |
| `if (t < prompt.size()) { prefill_ms += wall; continue; }` | 最後の回より前の時間は「読む時間」として数える。最後の回の答えが 1 つ目のトークン |

---

## 3.3 メモリとプロセスの分け方

NPU が 1 プロセスで持てるのは dma-buf で**約 1,710MiB**（AndroidScale P0A）。文章側（Gemma 4）が 1,467〜1,485MiB を使う。

| 置き方 | NPU の量 | 判断 |
|---|---|---|
| 同じプロセスに ViT を足す | 1,485 ＋ 約 150〜250 ＝ 1,635〜1,735MiB | 上限すれすれ。**避ける** |
| **ViT を別プロセスに** | プロセス A: 約 150〜250 / プロセス B: 1,485 | 上限はプロセスごと。**233MiB ＋ 1,467MiB の同居は実測で成功**（P6 2.6節） |

### 手を動かす: ViT の大きさ

$$
\text{1 層} = \underbrace{4 \times 768^2}_{\text{q,k,v,o}} + \underbrace{3 \times 768 \times 3072}_{\text{gate,up,down}} = 2{,}359{,}296 + 7{,}077{,}888 = 9{,}437{,}184
$$

$$
16 \text{ 層} \times 9{,}437{,}184 \approx 1.51 \text{ 億} \Rightarrow \text{INT8 で約 144MiB}
$$

safetensors のヘッダで数えた値（144.7MiB）と一致する。これに活性の作業領域が加わる。

> ★2026-10-17 追記: 照合した。safetensors のヘッダで画像エンコーダの 16 層ぶんを数え直すと 144.7MiB（うち INT8 の重みが 144.0MiB）。
> Phase 2 の実測では、量子化した画像エンコーダ（70 トークン）は NPU（dma-buf）**152MiB** で、文章側（1,467MiB）と別のプロセスで**同居できた**
> （[P2 の記録](../../results/P2_20260929_vit_device.md) 1・9 節）。

<!-- src: android_app/app/src/main/AndroidManifest.xml | android:name="\.RunnerService2" | android:foregroundServiceType | note=画像エンコーダを動かすサービスは別プロセス（:probe） -->
**ソース** `android_app/app/src/main/AndroidManifest.xml` L195〜L198（画像エンコーダを動かすサービスは別プロセス（:probe）） — md5 f7acccf1・`sync_src` が自動で更新

```xml
  195|             android:name=".RunnerService2"
  196|             android:exported="true"
  197|             android:process=":probe"
  198|             android:foregroundServiceType="dataSync" />
```
<!-- /src -->

<!-- src: android_app/app/src/main/java/org/edgeslm/demo/VlmActivity.kt | private fun runVit\(base: File\): JSONObject \{ | startForegroundService\(svc\) | note=runVit: 画像エンコーダの job を別プロセスに渡す -->
**ソース** `android_app/app/src/main/java/org/edgeslm/demo/VlmActivity.kt` L583〜L606（runVit: 画像エンコーダの job を別プロセスに渡す） — md5 8fc090e4・`sync_src` が自動で更新

```kotlin
  583|     private fun runVit(base: File): JSONObject {
  584|         val jobId = "vlm_vit_${System.currentTimeMillis()}_${vitSeq++}"
  585|         val job = JSONObject().apply {
  586|             put("job_id", jobId)
  587|             put("task", "infer")
  588|             put("model", cfg.optString("vit_model"))
  589|             put("ep", epJson())
  590|             put("batches", JSONArray().put(JSONObject().apply {
  591|                 put("id", "cam")
  592|                 put("inputs", JSONObject().put("pixel_values", "inputs/vlm/cam_pix.bin"))
  593|             }))
  594|             put("output", JSONObject().put("dir", "out/vlm"))
  595|             put("keep_session", true)                  // Phase 5: 開いたままにする（次の写真は動かすだけ）
  596|         }
  597|         File(base, "jobs").mkdirs()
  598|         File(base, "jobs/vlm_vit.json").writeText(job.toString())
  599|         val resultFile = File(base, "out/last_result.json")
  600|         // 前の結果を消してから頼み、job_id が一致する結果が出るまで 40ms ごとに見る（最長 VIT_TIMEOUT_MS）
  601|         resultFile.delete()
  602|         val svc = Intent(this, RunnerService2::class.java).apply {
  603|             action = RunnerActivity.ACTION_RUN
  604|             putExtra(RunnerActivity.EXTRA_JOB, "jobs/vlm_vit.json")
  605|         }
  606|         startForegroundService(svc)
```
<!-- /src -->

| 目印 | 何をしているか・なぜ |
|---|---|
| `android:process=":probe"` | このサービスは、文章側を開いている画面のプロセスとは別のプロセスで動く。NPU の上限はプロセスごとなので、画像エンコーダの分を文章側の上限に足さずに済む（上の表の「ViT を別プロセスに」） |
| `put("task", "infer")` | 写真ごとに画像エンコーダを 1 回動かす job。入力は C++ の前処理が書いた `pixel_values` のファイル |
| `put("keep_session", true)` | Phase 5 からは開いたままにする（次の写真は動かすだけ。[10.9 節](10_まとめ読み.md)） |
| `Intent(this, RunnerService2::class.java)` | job を別プロセスのサービスに渡す。結果は `out/last_result.json` を `job_id` で見分けて待つ |

---

## 3.4 v68 の制約が ViT に与える影響

| 制約 | ViT では | 見込み |
|---|---|---|
| **C-8**: 活性×活性の MatMul は `u16×u16` 不可（右辺を 8bit） | attention の QK^T と、スコア×V に当たる | Gemma 4 は活性がもともと 8bit（QAT）なので、文章側と同じやり方で通る見込み |
| **C-9**: マスクの −3.4e38 は量子化できない | 固定の格子ならパディングが無く、マスクが要らない | 当たらない |
| **C-10**: `Expand` は読めない | KV ヘッド＝Q ヘッド（12/12）で、GQA の複製が無い | 当たりにくい（書き出しで出ないかは確認） |
| **C-1**: 量子化されない float の入力は不可 | パッチの画素・位置の入力で踏みやすい | Gemma 4 の `fix_float_inputs` を流用 |
| attention の大きさ | 1 層あたりのスコアは 12 ヘッド × 630×630 ＝ 476 万（70 トークン）、280 トークンなら 7,620 万 | 70 トークンから始める。VTCM は 4MB なので、大きいとタイル分割の負担が増える |

<!-- src: app/pipeline/quantize_vit.py | print\("\[3/4\] 活性×活性の行列積の右辺を 8bit に（C-8）" | c14 = Q.scale_ratio_report | note=量子化の後: C-8 の書き換えと C-1・C-5・C-14 の検査 -->
**ソース** `app/pipeline/quantize_vit.py` L116〜L128（量子化の後: C-8 の書き換えと C-1・C-5・C-14 の検査） — md5 6ba7effc・`sync_src` が自動で更新

```python
  116|         print("[3/4] 活性×活性の行列積の右辺を 8bit に（C-8）", flush=True)
  117|         dg = downgrade_actact_rhs.run(P["qdq"](v), P["prepped"], P["final"](v))
  118|         # 想定の右辺の数: 1 層の注意に活性×活性の行列積が 2 つ（Q×Kᵀ と 確率×V）あり、16 層で 32
  119|         exp_rhs = C.VIT_LAYERS * 2
  120|         print(f"  右辺の数 {dg['n_rhs']}（想定 {exp_rhs}）・8bit に落とした {len(dg['done'])}・"
  121|               f"対象外（既に 8bit など） {len(dg['skipped'])}", flush=True)
  122|         if dg["n_rhs"] != exp_rhs:
  123|             print(f"  ★活性×活性の行列積の右辺の数が想定と違う（{dg['n_rhs']} / {exp_rhs}）。prepped の形を確かめること", flush=True)
  124|         print("[4/4] 検査（C-1・C-5・C-14）", flush=True)
  125|         c1 = verify_artifacts.check_c1_no_float_segments(P["final"](v))
  126|         c5 = verify_artifacts.check_c5_no_batchnorm(P["final"](v))
  127|         # C-14（2026-09-29 に端末で発見）: 重みのスケールが同じ行列の中で極端に小さいチャネルは、端末の NPU で値が壊れる
  128|         c14 = Q.scale_ratio_report(P["final"](v))
```
<!-- /src -->

| 目印 | 何をしているか・なぜ |
|---|---|
| `downgrade_actact_rhs.run(P["qdq"](v), P["prepped"], P["final"](v))` | C-8: 活性×活性の行列積（QK^T とスコア×V）の右辺を 8bit に落とす。文章側（EdgeSLM_AndroidGemma4）と同じ道具 |
| `exp_rhs = C.VIT_LAYERS * 2` | 右辺は 1 層に 2 つ（k^T と v）なので 16 層で 32。実際に 32 個すべてを 8bit にできた（`results/P2_vit_quant/b70_21x30_quantize.json`） |
| `verify_artifacts.check_c1_no_float_segments` | C-1: 量子化されない float の区間が残っていないか。結果は合格（float の残存なし） |
| `c14 = Q.scale_ratio_report` | C-14（端末で後から見つかった小さすぎる重みのスケール。[8 章](08_端末でだけ壊れる.md)）の検査も、後からここに足した |

---

## 3.5 まとめ: 端末での形（Phase 3 で作るもの）

> ★Phase 5 で変わったところ（2026-09-30）: 文章側は **16 トークン版**（プロンプトは 16 トークンずつ、past 240）、画像エンコーダは
> **開いたまま**（写真ごとに開かない）。いまの形の全体は [11章](11_写真から答えまで.md) と [システム設計書](../システム設計書.md)。
> 下の図は Phase 3 の時点のもの。

```
カメラ → 336×480 にレターボックス → 630 パッチ（CPU）
   │
   ▼  プロセス A（NPU、約 150〜250MiB）
 ViT 16 層 → 3×3 平均 → ×√768 → 射影層 → 70 × 1536
   │
   ▼  プロセス B（NPU、既存の Gemma 4 グラフ 1,485MiB、無改造）
 [system の 30 トークン][<|image>][画像 70 トークン][<image|>][質問] を 1 トークンずつ
   画像の位置: inputs_embeds ← 画像の特徴の 1 行 / ple_tok ← pad の行
   │
   ▼
 日本語の答え
```

<!-- src: なし | 章のまとめの図（Phase 3 の時点）。それぞれのソースは 3.1〜3.3 節と 1.1・2.5 節の枠、いまの形は 11 章 -->
> **ソース**: なし — 章のまとめの図（Phase 3 の時点）。それぞれのソースは 3.1〜3.3 節と 1.1・2.5 節の枠、いまの形は 11 章
<!-- /src -->
