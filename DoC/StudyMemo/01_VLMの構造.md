# 1. VLM の構造 — 画像が「言葉と同じ形」になるまで

[← 目次へ](../StudyMemo.md) | [次: 2. Gemma 4 の画像の通り道 →](02_Gemma4の画像の通り道.md)

> **全体の構成図**（2026-10-23 追加）: この章の「画像エンコーダ → 射影層 → LLM」が、スマホのどの部品で・どのプロセスで動いているかは [コンポーネント構成図](../コンポーネント構成図.html)（ブラウザで開く）。

> **ソースと動かし方**（2026-10-14 追記）: この章の各節を、どのファイルのどの関数がやっているか・どのコマンドで確かめられるか・どの結果ファイルに数字があるかは [12 章](12_ソースと動かし方.md) の **12.3.1**。実際のコードの抜き出しと 1 行ずつの解説は [13 章](13_コードで読む検証.md) の **13.2**。スマホも GPU も要らない演習は 12.2。各節の終わりには、その節の処理をしている実際のソースの枠がある（2026-10-17〜。`sync_src` で自動で更新）。

---

## 1.1 LLM は「ベクトルの列」しか見ていない

文章だけの LLM の入口を思い出す。

```
「エアコンを22度に」 ─(トークナイザ)→ ID の列 [1234, 567, ...] ─(埋め込みの表を引く)→ ベクトルの列 [T, 1536]
                                                                                        ↓
                                                                            Transformer の層（35 層）
```

層が受け取るのは**ベクトルの列 `[T, 1536]`** であって、ID ではない。表を引くのは入口の 1 回だけ。
だから、**表を引かずに、別の方法で作った 1536 次元のベクトルを混ぜても、層から見れば区別がつかない**。

VLM はこれを利用する。

```
写真 ─(画像エンコーダ)→ ベクトルの列 [N, 768] ─(射影層)→ [N, 1536] ─┐
                                                                  ├─ 並べる → [T, 1536] → Transformer の層
「この写真に何が写っている？」 ─(表を引く)→ [M, 1536] ─────────────┘
```

画像 1 枚が **N 個の「トークン」**になり、文章のトークンと同じ列に並ぶ。これが**画像トークン**（soft token）。

> **ポイント**: LLM 側から見ると、画像トークンは「語彙の表を引かずに外から渡された埋め込み」にすぎない。
> 端末の文章グラフの入口が「ID」ではなく「埋め込み（inputs_embeds）」になっていれば、画像を差し込む余地がある。
> EdgeSLM_AndroidGemma4 のグラフはすでにそうなっている（埋め込みの表は CPU で引いている）。

<!-- src: app/pipeline/hf_vlm.py | def device_style_embeds\(model, input_ids, feats\): | return emb, ple_tok, img_mask | note=PC: 端末と同じ入れ方を HF の部品で作る -->
**ソース** `app/pipeline/hf_vlm.py` L298〜L320（PC: 端末と同じ入れ方を HF の部品で作る） — md5 353d6037・`sync_src` が自動で更新

```python
  298| def device_style_embeds(model, input_ids, feats):
  299|     """端末の文章グラフに渡す 2 つの入力を、HF の部品で作る。
  300| 
  301|     inputs_embeds [1,T,1536]:
  302|         文章の位置 → 埋め込み表を引く（HF の embed_tokens。√1536 倍まで含む）
  303|         画像の位置 → image_features() の出力をそのまま入れる（√1536 倍はしない。HF の masked_scatter と同じ）
  304|     ple_tok [1,T,35,256]:
  305|         PLE の表を引く。画像の位置は ID を pad（0）に置き換えてから引く（HF の Gemma4Model.forward と同じ）
  306| 
  307|     端末の入口の量子化の範囲で切り詰める実験は check_wiring.py の clip_inputs / entry_clip で行う（全位置に掛ける）。
  308|     """
  309|     import torch
  310|     # img_mask: 画像トークンの位置。ids_pad: 画像の位置を pad（0）に替えた ID（埋め込みと PLE はこれで引く）
  311|     lm = model.model.language_model
  312|     ids = input_ids[0]
  313|     img_mask = ids == C.IMAGE_TOKEN_ID
  314|     assert int(img_mask.sum()) == feats.shape[0], "画像トークンの数と特徴の数が合わない"
  315|     ids_pad = torch.where(img_mask, torch.full_like(ids, C.PAD_ID), ids)[None]
  316|     with torch.no_grad():
  317|         emb = lm.embed_tokens(ids_pad).clone()
  318|         emb[0, img_mask] = feats.to(emb.dtype)
  319|         ple_tok = lm.get_per_layer_inputs(ids_pad, None)
  320|     return emb, ple_tok, img_mask
```
<!-- /src -->

<!-- src: android_app/app/src/main/cpp/runner/job.cpp | const bool is_img = img_rows > 0 && t < prompt.size\(\) && tid == img_tok; | \+\+img_fed; | note=端末の C++（1 トークンずつ読む経路）: 画像の位置は表を引かない -->
**ソース** `android_app/app/src/main/cpp/runner/job.cpp` L1530〜L1536（端末の C++（1 トークンずつ読む経路）: 画像の位置は表を引かない） — md5 bba78f1a・`sync_src` が自動で更新

```cpp
 1530|             const bool is_img = img_rows > 0 && t < prompt.size() && tid == img_tok;   // 画像の位置（Phase 3）
 1531|             const int64_t tab_id = is_img ? img_pad : tid;                             // PLE はこの ID の行を引く
 1532|             if (is_img) {
 1533|                 std::memcpy(inputs[emb_idx].data(), img_raw.data() + static_cast<size_t>(img_next) * H * 4,
 1534|                             static_cast<size_t>(H) * 4);
 1535|                 ++img_next;
 1536|                 ++img_fed;
```
<!-- /src -->

| 目印 | 何をしているか・なぜ |
|---|---|
| `emb = lm.embed_tokens(ids_pad).clone()` | 文章の位置は、いつもどおり埋め込みの表を引く（ここまでは文章だけの LLM と同じ入口） |
| `emb[0, img_mask] = feats.to(emb.dtype)` | 画像の位置の行だけを、画像エンコーダ＋射影層の出力（1536 次元）で上書きする。層が受け取るのは `[T, 1536]` の列だけなので、これで画像が「言葉と同じ形」になる |
| `assert int(img_mask.sum()) == feats.shape[0]` | 画像トークン（ID 258880）の数と特徴の行数が合わなければ止める。1 枚＝N 行（70 トークンなら 70 行） |
| `std::memcpy(inputs[emb_idx].data(), img_raw.data()` | 端末も同じ。画像の位置では表を引かず、特徴のファイルの次の 1 行を文章グラフの入口 `inputs_embeds` にそのまま写す。グラフが ID ではなく埋め込みを受け取るので、作り直さずに画像を差し込める |
| `const int64_t tab_id = is_img ? img_pad : tid;` | PLE（層ごとの埋め込み）だけは、画像の位置でも表を引く。そのとき使う ID は pad（0）（[2.5 節](02_Gemma4の画像の通り道.md)） |

---

## 1.2 部品は 3 つ

| 部品 | 役割 | 大きさの例（Gemma 4 E2B） |
|---|---|---|
| **画像エンコーダ（ViT）** | 画像をパッチに切り、各パッチの「意味」をベクトルにする | 16 層・約 1.5 億パラメータ |
| **射影層** | ViT の次元（768）を LLM の次元（1536）へ写す。「画像の言葉」を「LLM の言葉」に訳す | 768×1536 ≒ 118 万パラメータ |
| **LLM** | 画像トークンと文章トークンを一緒に読んで、答えを書く | 35 層 |

射影層は小さいが、**ViT と LLM の「通訳」**なので、どちらかを別のものに替えると働かなくなる。

<!-- src: app/pipeline/vlm_common.py | ^TEXT_HIDDEN = 1536 | ^VIT_INTER = 3072 | omit=^SUPPORTED_SOFT_TOKENS..^# 画像エンコーダ（vision_config） | note=部品の大きさ（config.json から） -->
**ソース** `app/pipeline/vlm_common.py` L46〜L75（部品の大きさ（config.json から）） — md5 81595b1f・`sync_src` が自動で更新

```python
   46| TEXT_HIDDEN = 1536            # 文章側の hidden（inputs_embeds の次元）
   47| NUM_LAYERS = 35               # 文章側の層の数
   48| PLE_DIM = 256                 # PLE（層ごとの埋め込み）の次元
   49| PAD_ID = 0                    # text_config.pad_token_id。画像トークンの PLE はこの行を使う（HF の実装どおり）
   50| IMAGE_TOKEN_ID = 258880       # 画像の中身が入る位置の目印（<|image|>）
   51| BOI_TOKEN_ID = 255999         # 画像の始まり
   52| EOI_TOKEN_ID = 258882         # 画像の終わり
   53| 
   54| VISION_HIDDEN = 768           # 画像エンコーダ（ViT）の hidden
   55| PATCH = 16                    # 1 パッチ = 16×16 ピクセル
   56| POOL = 3                      # 3×3 のパッチを平均して 1 つの画像トークンにする
   57| SUPPORTED_SOFT_TOKENS = (70, 140, 280, 560, 1120)   # HF の画像前処理が受け付ける「画像トークン数の上限」
     | …（中略: 13 行。L58〜L70）…
   71| # 画像エンコーダ（vision_config）の数字。Phase 2 の自前の配線（vit_graph.py）が使う
   72| VIT_LAYERS = 16
   73| VIT_HEADS = 12                # KV も 12（GQA ではない）
   74| VIT_HEAD_DIM = 64
   75| VIT_INTER = 3072              # MLP の中間
```
<!-- /src -->

<!-- src: app/pipeline/vit_graph.py | self.layers = torch.nn.ModuleList\(\[VitLayer | self.proj = Lin\( | note=VitGraph.__init__: 画像エンコーダ 16 層と射影層 -->
**ソース** `app/pipeline/vit_graph.py` L255〜L260（VitGraph.__init__: 画像エンコーダ 16 層と射影層） — md5 e32d0644・`sync_src` が自動で更新

```python
  255|         self.layers = torch.nn.ModuleList([VitLayer(qf, i) for i in range(C.VIT_LAYERS)])
  256|         # 射影層（Gemma4MultimodalEmbedder）: RMSNorm（重みなし、eps は vision の rms_norm_eps）→ Linear（float）
  257|         # ③ ×√768 を省くときは eps を 1/768 にする（RMSNorm(c·x; ε) = RMSNorm(x; ε/c²)、c² = 768。StudyMemo 7.4 節）
  258|         eps = qf.eps / C.VISION_HIDDEN if fold_pool_scale else qf.eps
  259|         self.proj_norm = RMSNorm(torch.ones(C.VISION_HIDDEN), eps)
  260|         self.proj = Lin({"w": qf.vec("model.embed_vision.embedding_projection.weight")})
```
<!-- /src -->

| 目印 | 何をしているか・なぜ |
|---|---|
| `TEXT_HIDDEN = 1536`・`NUM_LAYERS = 35` | LLM（文章側）の次元と層の数。画像トークンもこの 1536 次元にそろえて渡す |
| `VISION_HIDDEN = 768`・`VIT_LAYERS = 16`・`VIT_INTER = 3072` | 画像エンコーダの次元・層の数・MLP の中間。1 層の重みは q・k・v・o（768×768×4）と MLP（768×3072×3）で約 944 万、16 層で約 1.5 億（[3.3 節](03_NPUに載せるための設計.md)の計算） |
| `VitLayer(qf, i) for i in range(C.VIT_LAYERS)` | 画像エンコーダの 16 層を並べる（1 層の中身は [2.3 節](02_Gemma4の画像の通り道.md)） |
| `model.embed_vision.embedding_projection.weight` | 射影層の重み。safetensors では float32 の [1536, 768]＝約 118 万個（2026-10-17 にファイルのヘッダで確認）。QAT の整数ではない |
| `self.proj_norm = RMSNorm(torch.ones(C.VISION_HIDDEN), eps)` | 射影層の最初の RMSNorm は係数なし（全部 1）。大きさをそろえてから 768→1536 に写す |

---

## 1.3 「射影層を足すだけ」は、いつ成り立つか

当初の計画は「手元で動いている LLM に、射影層を足して VLM にする」だった。
これが成り立つには 2 つの条件がある。

### 条件 1: LLM の重みがそのままであること

多くの VLM は、学習のときに **LLM の重みも更新している**（画像の話ができるように微調整する）。
そのため、

- 既存の LLM（たとえば Qwen2.5-1.5B）＋ 別の VLM（たとえば Qwen2-VL-2B）の射影層

を組み合わせても、正しく動かない。射影層は「微調整後の LLM」の言葉に合わせて学習されているから。

選べる道は 2 つ:

| 道 | 中身 | 必要なもの |
|---|---|---|
| (a) VLM を丸ごと移植 | ViT・射影層・LLM すべてをその VLM の重みで作る | LLM 部分も変換し直す（既存の手順を流用） |
| (b) 射影層を新しく学習 | 既存の LLM は凍結し、射影層だけを学習する（LLaVA の最初の段階と同じ） | 学習データ（日本語の画像・文章の対）と GPU |

**Gemma 4 E2B は (a) の特別な場合**: 文章部分（LLM）はもう端末で動いていて、画像部分（ViT・射影層）は同じモデルのファイルに入っている。
「LLM を変換し直す」手間が既に済んでいる。

### 条件 2: 画像トークンに特別な扱いが無いこと

画像トークンを「普通のトークンと同じように」並べるだけで済むかは、モデルの作りによる。

| 構造 | 画像トークンの位置 | マスク | LLM 側のグラフ |
|---|---|---|---|
| LLaVA 系（ViT → MLP → 普通の LLM） | 普通の連番 | 因果 | **変えなくてよい** |
| Qwen2.5-VL | **3 次元の位置**（時間・縦・横。M-RoPE） | 因果 | RoPE の計算と位置の入力を変える |
| Qwen3-VL | 3 次元の位置（交互に並べた M-RoPE） | 因果 | さらに **DeepStack**（ViT の途中の層の特徴を LLM の途中の層に足す）で、層の中身が変わる |
| **Gemma 4 E2B** | **普通の連番** | **因果**（`use_bidirectional_attention = null`） | **変えなくてよい見込み** |
| Gemma 4 の大きいモデル | 普通の連番 | 画像の中だけ双方向 | マスクを変える |

**Gemma 4 E2B は LLaVA 系と同じく「並べるだけ」でよい。** 違いは PLE（層ごとの埋め込み）があることだけで、
画像トークンの PLE は「pad（ID 0）の行を引く」と決まっている（[2章](02_Gemma4の画像の通り道.md)）。

> **注意**: これは transformers の実装を読んで出した結論。Phase 1 で、HF の本来の経路と「自分で並べた経路」の logits が
> 一致することを PC で確かめてから確定する（[6章](06_検算の考え方.md)）。
>
> ★2026-10-17 追記: Phase 1 で確かめ済み。HF の本来の経路（A）と自分で並べた経路（B）の logits は、4 問とも SRQ あり・なしの両方で**差 0**だった（`results/P1_check_wiring/wiring.json`・`wiring_b70_srq-off.json`、[P1 の記録](../../results/P1_20260928_pc_eval.md) 2 節）。取得済みモデルの `config.json` でも `text_config.use_bidirectional_attention` は null（2026-10-17 に確認）。

<!-- src: app/pipeline/check_wiring.py | ref = model\(\*\*inputs\)\.logits\[0\]\.float\(\) | "A_vs_B": compare\(ref, bat, span\) | note=run_one: 3 つの経路を同じ入力で流して比べる -->
**ソース** `app/pipeline/check_wiring.py` L167〜L180（run_one: 3 つの経路を同じ入力で流して比べる） — md5 b0a5758f・`sync_src` が自動で更新

```python
  167|         ref = model(**inputs).logits[0].float()                                  # A
  168|     # ② 画像の特徴を取り出し、端末と同じ入れ方の inputs_embeds・ple_tok を作る（画像の位置は特徴そのもの、PLE は pad の行）
  169|     feats = H.image_features(model, inputs)
  170|     emb, ple, img_mask = H.device_style_embeds(model, inputs["input_ids"], feats)
  171|     bat, _ = H.device_style_step(model, emb, ple, None)                         # B（一度に）
  172|     bat = bat.float()
  173|     steps, pkv = [], None
  174|     for t in range(T):                                                          # C（1 トークンずつ）
  175|         lg, pkv = H.device_style_step(model, emb[:, t:t + 1], ple[:, t:t + 1], pkv)
  176|         steps.append(lg[-1].float())
  177|     stp = torch.stack(steps)
  178|     # ③ 3 通りを 2 つずつ比べる（A↔B は入れ方、B↔C は 1 トークンずつの持ち越し、A↔C は全体）
  179|     r = {"id": it["id"], "T": T, "n_image_tokens": n_img, "image_span": list(span), "sec": round(time.time() - t0, 1),
  180|          "A_vs_B": compare(ref, bat, span), "B_vs_C": compare(bat, stp, span), "A_vs_C": compare(ref, stp, span)}
```
<!-- /src -->

| 目印 | 何をしているか・なぜ |
|---|---|
| `ref = model(**inputs).logits[0].float()` | A: HF の本来の経路。`pixel_values` を渡し、画像の差し込みは HF の中に任せる |
| `H.device_style_embeds(model, inputs["input_ids"], feats)` | B の入力を自分で並べる（1.1 節の枠）。画像の位置は特徴、PLE は pad の行。マスクと位置は文章と同じ扱い＝条件 2 の読みそのもの |
| `bat, _ = H.device_style_step(model, emb, ple, None)` | B: 文章側（language_model）だけに一度に流す。A と差 0 なら、「並べるだけでよい」という読みが正しい |
| `for t in range(T):` | C: 端末と同じく 1 トークンずつ、KV キャッシュを持ち越して流す（[6 章](06_検算の考え方.md)） |
| `"A_vs_B": compare(ref, bat, span)` | logits を位置ごとに比べる。結果は 4 問とも相対 L2 の最大が 0（上の追記） |

---

## 1.4 画像トークンの数が、そのまま「長さ」になる

画像 1 枚が N 個のトークンになるので、**プロンプトが N だけ長くなる**。

| 画像トークン | 画像以外（system＋質問） | プロンプトの長さ |
|---:|---:|---:|
| 70 | 39 | 109 |
| 280 | 39 | 319 |

（★2026-10-17 訂正: 旧「画像以外 約 30、プロンプトの長さ 約 100・約 310」。これは Phase 1 の前の見積もり。実測（`results/P1_ref_vlm/summary.md` の時間の表。固定の格子で 126 問の平均。70 トークンは 109、280 トークンは 319）に合わせた。画像以外の 39 は、system・質問・画像の始まりと終わりの印・ターンの区切りを合わせた数で、[3.2 節](03_NPUに載せるための設計.md)の表と同じ）

長さは、端末では**待ち時間**と**KV キャッシュの大きさ**に直結する（[3章](03_NPUに載せるための設計.md)）。
Gemma 4 は 1 枚あたりのトークン数を 70〜1120 から選べる（多いほど細かく見える）。
**どこまで減らしても用途に足りるか**が、Phase 1 で測る一番の問い。

<!-- src: app/pipeline/vlm_common.py | ^POOL = 3 | ^SUPPORTED_SOFT_TOKENS | note=1 枚あたりの画像トークン数の候補 -->
**ソース** `app/pipeline/vlm_common.py` L56〜L57（1 枚あたりの画像トークン数の候補） — md5 81595b1f・`sync_src` が自動で更新

```python
   56| POOL = 3                      # 3×3 のパッチを平均して 1 つの画像トークンにする
   57| SUPPORTED_SOFT_TOKENS = (70, 140, 280, 560, 1120)   # HF の画像前処理が受け付ける「画像トークン数の上限」
```
<!-- /src -->

<!-- src: app/pipeline/hf_vlm.py | def build_inputs\(proc, img | assert n_img == budget | note=プロンプトを作り、画像トークンの数を確かめる -->
**ソース** `app/pipeline/hf_vlm.py` L201〜L231（プロンプトを作り、画像トークンの数を確かめる） — md5 353d6037・`sync_src` が自動で更新

```python
  201| def build_inputs(proc, img: Image.Image, question: str, budget: int, system: str | None, device: str,
  202|                  expect_exact: bool = False):
  203|     """chat template で文を作り、画像トークン数を指定してプロセッサに通す。
  204| 
  205|     文の形（Gemma 4 の chat template）:
  206|         <bos><|turn>system\\n{system}<turn|>\\n<|turn>user\\n<|image|>{質問}<turn|>\\n<|turn>model\\n
  207|     プロセッサが <|image|> を <|image>（始まり）＋ 画像トークン×N ＋ <image|>（終わり）に置き換える。
  208| 
  209|     ★ chat template が <bos> を既に書くので、トークン化では特別なトークンを足さない（add_special_tokens=False）。
  210|       足すと <bos><bos> で始まる、端末では使わない形のプロンプトになる。先頭が <bos> 1 つであることを確かめる。
  211|     expect_exact=True（fixed の前処理）のときは、画像トークン数がちょうど budget であることも確かめる。
  212|     """
  213|     global _dropped_shown
  214|     # ① 会話の形を作り、chat template で文字列にする（画像は {"type": "image"} の印だけ。中身は images で渡す）
  215|     msgs = []
  216|     if system:
  217|         msgs.append({"role": "system", "content": system})        # 文字列で渡す（リストだと末尾に空白が付く）
  218|     msgs.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]})
  219|     text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
  220|     # ② プロセッサに文と画像を通す。max_soft_tokens＝画像トークン数の上限（70/140/280）
  221|     inputs = proc(text=[text], images=[[img]], return_tensors="pt", add_special_tokens=False,
  222|                   images_kwargs={"max_soft_tokens": budget})
  223|     # ③ 先頭の <bos> が 1 つか・画像トークン数が上限以内か（fixed ならちょうどか）を確かめる
  224|     ids = inputs["input_ids"][0]
  225|     bos = proc.tokenizer.bos_token_id
  226|     assert int(ids[0]) == bos and int(ids[1]) != bos, f"先頭の <bos> が 1 つではない: {ids[:3].tolist()}"
  227|     n_img = int((ids == C.IMAGE_TOKEN_ID).sum())
  228|     # 指定した上限を超えていないか（kwargs の渡し方を間違えると黙って既定の 280 になるので、ここで必ず確かめる）
  229|     assert 0 < n_img <= budget, f"画像トークン数 {n_img} が上限 {budget} と合わない"
  230|     if expect_exact:
  231|         assert n_img == budget, f"固定の格子なのに画像トークン数が {n_img}（期待 {budget}）。プロセッサが大きさを変えた？"
```
<!-- /src -->

| 目印 | 何をしているか・なぜ |
|---|---|
| `SUPPORTED_SOFT_TOKENS = (70, 140, 280, 560, 1120)` | HF の前処理が受け付ける「1 枚あたりの画像トークン数の上限」。多いほど細かく見えるが、そのぶんプロンプトが長くなる |
| `POOL = 3` | 3×3 のパッチを 1 トークンにまとめる。だからパッチ数＝トークン数×9（70 トークン＝630 パッチ） |
| `画像トークン×N` | プロセッサが `<\|image\|>` を「始まり＋画像トークン N 個＋終わり」に置き換える。プロンプトの長さ＝N＋文章（上の表の 39） |
| `images_kwargs={"max_soft_tokens": budget}` | 画像トークン数（70/140/280）をここで指定する |
| `assert 0 < n_img <= budget` | 渡し方を間違えると、黙って既定の 280 になる（`config.json` の `vision_soft_tokens_per_image`）。長さが約 3 倍になるので、数を必ず確かめる |
