# 2. Gemma 4 E2B の画像の通り道 — 336×480 の写真が 70 個のトークンになるまで

[← 目次へ](../StudyMemo.md) | [← 前: 1. VLM の構造](01_VLMの構造.md) | [次: 3. NPU に載せるための設計 →](03_NPUに載せるための設計.md)

> **ソースと動かし方**（2026-10-14 追記）: この章の各節を、どのファイルのどの関数がやっているか・どのコマンドで確かめられるか・どの結果ファイルに数字があるかは [12 章](12_ソースと動かし方.md) の **12.3.1**。実際のコードの抜き出しと 1 行ずつの解説は [13 章](13_コードで読む検証.md) の **13.1・13.4**。スマホも GPU も要らない演習は 12.2。各節の終わりには、その節の処理をしている実際のソースの枠がある（2026-10-17〜。`sync_src` で自動で更新）。transformers の中（HF のモデル定義）の話は、このプロジェクトがそれを写した・書き直した所（`vit_graph.py`・`gemma4_graph.py`・`hf_vlm.py`）を載せる。

出どころ: transformers 5.14.1 の `models/gemma4/`（`image_processing_gemma4.py`・`modeling_gemma4.py`）と、
取得済みモデルの `config.json`・`processor_config.json`。行番号は transformers 5.14.1 のもの。

---

## 2.1 全体の流れ（形の変化）

70 トークン・固定の格子（端末と同じ）で、1 枚の写真を追う。

| # | 処理 | 形 | 場所 |
|---|---|---|---|
| 0 | 写真（例: 1600×1067） | 画素 | — |
| 1 | 336×480 に収める（縦横比を保って縮小、余白は黒） | 336×480×3 | 前処理（CPU） |
| 2 | 0〜255 → 0〜1（÷255） | 同上 | 前処理 |
| 3 | 16×16 のパッチに切る: 336/16＝**21**、480/16＝**30** | **630 × 768**（1 パッチ＝16×16×3＝768 値） | 前処理 |
| 4 | 0〜1 → −1〜1（`2×(x−0.5)`） | 630 × 768 | ViT の入口 |
| 5 | パッチの埋め込み: 全結合 768→768 ＋ x の位置の表 ＋ y の位置の表 | 630 × 768 | ViT |
| 6 | ViT 16 層（双方向の attention・2 次元 RoPE・q/k/v に RMSNorm） | 630 × 768 | ViT |
| 7 | 3×3 のパッチを平均: 21/3＝7、30/3＝10 → **70** | **70 × 768** | プーリング |
| 8 | ×√768（≒27.7） | 70 × 768 | プーリング |
| 9 | RMSNorm（係数なし）→ 全結合 768→1536 | **70 × 1536** | 射影層（`embed_vision`） |
| 10 | 文章のトークンと並べて LLM へ | (70＋文章) × 1536 | 文章側 |

### 手を動かす: 数を確かめる

- パッチの数: $21 \times 30 = 630$。画像トークン数の上限は $630 / 9 = 70$ ✓
- 1 パッチの値の数: $16 \times 16 \times 3 = 768$（たまたま ViT の hidden と同じ）
- 280 トークンなら: 42×60 パッチ＝2,520、画像は 672×960px
- 固定の格子の辺は **48px（＝16px×3）の倍数**でないといけない（パッチで割り切れ、さらにプーリングの 3 で割り切れる）

<!-- src: app/pipeline/hf_vlm.py | x = np.asarray\(img_fixed.convert\("RGB"\), dtype=np.uint8\) | return np.ascontiguousarray | note=pixels_fixed_np: 手順 2・3（÷255 とパッチへの並べ替え） -->
**ソース** `app/pipeline/hf_vlm.py` L189〜L193（pixels_fixed_np: 手順 2・3（÷255 とパッチへの並べ替え）） — md5 353d6037・`sync_src` が自動で更新

```python
  189|     x = np.asarray(img_fixed.convert("RGB"), dtype=np.uint8)            # [H,W,3]
  190|     assert x.shape == (gh * C.PATCH, gw * C.PATCH, 3), f"格子の大きさでない: {x.shape}（prepare_image の fixed を通すこと）"
  191|     x = x.astype(np.float32) * np.float32(cfg["rescale_factor"])        # torch の「uint8 × Python の float」と同じく float32 で掛ける
  192|     x = x.reshape(gh, C.PATCH, gw, C.PATCH, 3).transpose(0, 2, 1, 3, 4)  # (縦, py, 横, px, c) → (縦, 横, py, px, c)
  193|     return np.ascontiguousarray(x.reshape(gh * gw, C.PATCH * C.PATCH * 3))
```
<!-- /src -->

<!-- src: app/pipeline/vit_graph.py | def run\(self, pixel_values: torch.Tensor, keep: bool = False\) | return out$ | note=VitGraph.run: 手順 4〜9 -->
**ソース** `app/pipeline/vit_graph.py` L274〜L294（VitGraph.run: 手順 4〜9） — md5 e32d0644・`sync_src` が自動で更新

```python
  274|     def run(self, pixel_values: torch.Tensor, keep: bool = False) -> dict:
  275|         """keep=True なら途中の値（パッチの埋め込み・各層の出力・プーリングの後）も返す（HF との突き合わせ用）。"""
  276|         # ① 入力の 0〜1 を −1〜1 に直し、パッチの埋め込み（768→768）に位置の表を足す
  277|         out = {}
  278|         x = 2.0 * (pixel_values - 0.5)
  279|         h = self.patch_proj(x) + self.pos_emb
  280|         if keep:
  281|             out["patch_embed"] = h
  282|         # ② 16 層を順に通す（cos/sin は全層で同じバッファ）
  283|         for i, ly in enumerate(self.layers):
  284|             h = ly(h, self.cos, self.sin)
  285|             if keep:
  286|                 out[f"layer_{i}"] = h
  287|         # ③ 3×3 のプーリング → 射影層（重みなしの RMSNorm と 768→1536 の行列）で文章側の大きさの特徴にする
  288|         p = self.pool(h)
  289|         if not self.fold_pool_scale:
  290|             p = p * self.pool_scale
  291|         if keep:
  292|             out["pooled"] = p if not self.fold_pool_scale else p * self.pool_scale   # HF の pooler の出力と同じ尺度で返す
  293|         out["image_embeds"] = self.proj(self.proj_norm(p))
  294|         return out
```
<!-- /src -->

| 目印 | 何をしているか・なぜ |
|---|---|
| `x.astype(np.float32) * np.float32(cfg["rescale_factor"])` | 手順 2。rescale_factor は 1/255（`processor_config.json`）。平均・標準偏差での正規化は無い |
| `x.reshape(gh, C.PATCH, gw, C.PATCH, 3).transpose(0, 2, 1, 3, 4)` | 手順 3。336×480×3 を（縦 21・横 30・16・16・3）に並べ替えて 630×768 にする。パッチの番号は「左上から右へ、行ごとに下へ」 |
| `x = 2.0 * (pixel_values - 0.5)` | 手順 4。0〜1 を −1〜1 に。HF の ViT の入口と同じ |
| `h = self.patch_proj(x) + self.pos_emb` | 手順 5。全結合 768→768 に、x と y の位置の表を足した定数を足す（格子が固定なので先に計算してある。[7 章](07_画像エンコーダを書き直す.md)） |
| `p = self.pool(h)` | 手順 7。3×3 の平均で 630 → 70（2.4 節の枠） |
| `out["image_embeds"] = self.proj(self.proj_norm(p))` | 手順 8・9。×√768 は既定（`fold_pool_scale=True`）では掛けず、RMSNorm の eps に吸収している（2.4 節）。出力は 70×1536 |

---

## 2.2 HF の既定（native）は形が変わる

HF のプロセッサは、画像ごとに次のように大きさを決める（`get_aspect_ratio_preserving_size`）。

1. 画素数がちょうど「上限のパッチ数×256」になる倍率を求める
2. 縦・横をそれぞれ 48px の倍数に切り下げる
3. 足りないパッチは**パディング**（位置 −1）で埋め、上限のパッチ数にそろえる

たとえば縦長の写真（1200×1800）を 70 トークンで処理すると、

$$
\text{倍率} = \sqrt{\frac{630 \times 256}{1200 \times 1800}} \approx 0.2733 \quad\Rightarrow\quad 328 \times 492 \Rightarrow 48 \text{の倍数に切り下げて } 288 \times 480
$$

→ 18×30＝540 パッチ（実物）＋ 90 パッチ（パディング）＝630。画像トークンは 540/9＝**60**。

**同じ「70 トークン」でも、画像の縦横比によってパッチの並び（18×30 か 21×30 か…）と実際のトークン数が変わる。**
PC なら問題ないが、端末では形の違うグラフを何本も持てない。→ [3章](03_NPUに載せるための設計.md)で固定の格子にする。

<!-- src: app/pipeline/hf_vlm.py | def prepare_image\(img: Image.Image, prep: str | ^\s+return img$ | note=prepare_image の native: 大きさは HF に任せる -->
**ソース** `app/pipeline/hf_vlm.py` L130〜L154（prepare_image の native: 大きさは HF に任せる） — md5 353d6037・`sync_src` が自動で更新

```python
  130| def prepare_image(img: Image.Image, prep: str, budget: int, grid: tuple[int, int] | None = None) -> Image.Image:
  131|     """画像の前処理の方式を選ぶ。
  132| 
  133|     native: そのまま返す。HF のプロセッサが「縦横比を保ち、パッチ数が budget×9 以下になる最大の大きさ」に拡大・縮小し、
  134|             足りない分をパディング（位置 -1）で埋める。画像ごとにパッチの並びが変わる＝**形が固定されない**
  135|     fixed:  端末と同じ固定の格子（vlm_common.FIXED_GRIDS、例: 70 → 21×30 パッチ = 336×480px）に収める。
  136|             縦横比を保って拡大・縮小し、余白は黒で埋める（レターボックス）。プロセッサから見ると「ちょうどの大きさ」なので
  137|             大きさの変更もパディングも起きず、パッチ数は常に budget×9 になる
  138| 
  139|     grid=(gh, gw) を渡すと、fixed のときその格子に収める（既定は FIXED_GRIDS[budget]。縦長の格子 PORTRAIT_GRIDS 用）。
  140| 
  141|     どちらも、先に EXIF の向きを反映する（スマホの縦写真は、画素は横のまま「回転して表示せよ」という印だけ付いていることがある。
  142|     HF はファイルから読むときだけこれを反映し、PIL の画像を渡したときは反映しない）。透明部分は白で埋める。
  143|     """
  144|     # ① EXIF の向きを反映し、透明部分を白で塗ってから RGB にする
  145|     from PIL import ImageOps
  146|     img = ImageOps.exif_transpose(img)
  147|     if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
  148|         rgba = img.convert("RGBA")
  149|         bg = Image.new("RGB", rgba.size, (255, 255, 255))
  150|         bg.paste(rgba, mask=rgba.split()[-1])
  151|         img = bg
  152|     img = img.convert("RGB")
  153|     if prep == "native":
  154|         return img
```
<!-- /src -->

<!-- src: app/pipeline/hf_vlm.py | def n_valid_patches\(inputs: dict\) -> int: | return int\( | note=パディングを除いた実物のパッチ数 -->
**ソース** `app/pipeline/hf_vlm.py` L240〜L242（パディングを除いた実物のパッチ数） — md5 353d6037・`sync_src` が自動で更新

```python
  240| def n_valid_patches(inputs: dict) -> int:
  241|     """パディングを除いた、実際の画像のパッチ数（位置が -1 のものはパディング）。"""
  242|     return int((inputs["image_position_ids"][0, :, 0] >= 0).sum())
```
<!-- /src -->

| 目印 | 何をしているか・なぜ |
|---|---|
| `native: そのまま返す` | native では大きさを決めるのは HF のプロセッサ（`get_aspect_ratio_preserving_size`。transformers の中）。画像ごとに格子が変わる |
| `img = ImageOps.exif_transpose(img)` | スマホの縦写真は、画素は横のまま「回転して表示」という印だけが付いていることがある。先に向きを直さないと、縦長の写真が横長として扱われる |
| `if prep == "native":` | native は向きと色だけを整えて返す。fixed（[3.1 節](03_NPUに載せるための設計.md)）との分かれ目 |
| `inputs["image_position_ids"][0, :, 0] >= 0` | パディングのパッチは位置が −1。上の例なら 630 のうち 540 だけが実物で、画像トークンは 60 |

---

## 2.3 ViT の中身で押さえておくこと

| 項目 | 中身 | 端末で気にすること |
|---|---|---|
| 位置 | 2 次元の位置の表（x・y を足す）＋ 2 次元 RoPE（head_dim 64 のうち半分を x、半分を y） | 格子を固定すれば、位置の表の引き当ても cos/sin も**定数**になる |
| attention | 双方向。パディングがあるときだけマスクを使う | 固定の格子ならパディングが無いので**マスク不要**（C-9 の心配が無い） |
| ヘッド | 12 ヘッド、KV も 12（GQA ではない） | GQA の複製（`Expand`）が出ない＝C-10 を踏みにくい |
| 正規化 | 各層に RMSNorm が 4 つ、q・k・v にも RMSNorm | 文章側と同じ部品 |
| 重み | **INT8・出力チャネルごと**、Linear の入出力に**活性の目盛り（QAT）** | 文章側と同じ形。端末でも同じ手順が使える見込み |

`scaling = 1.0`（attention のスコアを $1/\sqrt{d}$ で割らない）なのは、q と k を RMSNorm しているため。

<!-- src: app/pipeline/vit_graph.py | def forward\(self, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor\) | return h$ | note=VitLayer.forward: 1 層の中身（HF の Gemma4VisionEncoderLayer と同じ順） -->
**ソース** `app/pipeline/vit_graph.py` L202〜L223（VitLayer.forward: 1 層の中身（HF の Gemma4VisionEncoderLayer と同じ順）） — md5 e32d0644・`sync_src` が自動で更新

```python
  202|     def forward(self, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
  203|         # h = [1,N,768]、cos/sin = [1,N,1,64]（全層で同じ表）。戻り値は同じ形の h
  204|         # ① 注意: 正規化 → Q/K/V → Q・K だけ 2 次元 RoPE で位置を入れる
  205|         N = h.shape[1]
  206|         res = h
  207|         a = self.input_layernorm(h)
  208|         q = self.q_norm(self.q_proj(a).view(1, N, self.nh, self.hd))
  209|         k = self.k_norm(self.k_proj(a).view(1, N, self.nh, self.hd))
  210|         v = self.v_norm(self.v_proj(a).view(1, N, self.nh, self.hd))
  211|         q = q * cos + rotate_half_2d(q) * sin                               # cos/sin [1,N,1,64]
  212|         k = k * cos + rotate_half_2d(k) * sin
  213|         # ② 全部のパッチどうしで注意を取る（画像の中は双方向。パディングが無いのでマスクも要らない）
  214|         q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)   # [1,12,N,64]
  215|         s = torch.matmul(q, k.transpose(2, 3))                              # [1,12,N,N]。スケール 1.0、マスクなし
  216|         ctx = torch.matmul(torch.softmax(s, dim=-1), v)                      # [1,12,N,64]
  217|         ctx = ctx.transpose(1, 2).reshape(1, N, self.nh * self.hd)
  218|         h = res + self.post_attention_layernorm(self.o_proj(ctx))
  219|         # ③ MLP（gate と up の積を down で戻す）。前後に正規化、残差で足す
  220|         res = h
  221|         b = self.pre_feedforward_layernorm(h)
  222|         h = res + self.post_feedforward_layernorm(self.down_proj(gelu_tanh(self.gate_proj(b)) * self.up_proj(b)))
  223|         return h
```
<!-- /src -->

<!-- src: app/pipeline/vit_graph.py | def linear\(self, base: str\) -> dict: | "out_act": float | note=QatFile.linear: QAT の Linear（INT8・出力チャネルごと・活性の目盛り） -->
**ソース** `app/pipeline/vit_graph.py` L107〜L114（QatFile.linear: QAT の Linear（INT8・出力チャネルごと・活性の目盛り）） — md5 e32d0644・`sync_src` が自動で更新

```python
  107|     def linear(self, base: str) -> dict:
  108|         """QAT の Linear（int8・出力チャネルごと）。整数・スケール・活性の目盛りと、float32 に戻した重み [out, in]。"""
  109|         w = self.t(base + ".weight")
  110|         assert w.dtype == torch.int8, f"{base}: int8 のはず（{w.dtype}）"
  111|         s = self.t(base + ".weight_scale").float().reshape(-1)
  112|         return {"name": base, "codes": w, "scale": s, "bits": 8, "w": w.float() * s[:, None],
  113|                 "in_act": float(self.t(base + ".input_activation_scale")),
  114|                 "out_act": float(self.t(base + ".output_activation_scale"))}
```
<!-- /src -->

| 目印 | 何をしているか・なぜ |
|---|---|
| `self.post_feedforward_layernorm(` | 各層の 4 つの RMSNorm（input・post_attention・pre_feedforward・post_feedforward）。文章側と同じ部品 |
| `q = self.q_norm(self.q_proj(a).view(1, N, self.nh, self.hd))` | q・k・v にも RMSNorm（v_norm は係数なし）。12 ヘッド×64 に分けてから正規化する |
| `q = q * cos + rotate_half_2d(q) * sin` | 2 次元 RoPE。cos/sin は [N, 64] の定数で、前半 32 を x、後半 32 を y に使う（`rope2d_tables`） |
| `s = torch.matmul(q, k.transpose(2, 3))` | スケール 1.0（$1/\sqrt{64}$ で割らない）・マスクなし（双方向。固定の格子ではパディングが無い）。KV も 12 ヘッドなので複製（Expand）が要らない |
| `assert w.dtype == torch.int8` | 重みは INT8。`weight_scale` は出力チャネルごとで、`s[:, None]` で行ごとに掛けて float に戻す |
| `"in_act": float(self.t(base + ".input_activation_scale"))` | Linear の入出力の活性の目盛り（QAT で学習したもの）。文章側と同じ形 |

---

## 2.4 プーリングと √768

プーリングは、パッチの位置 (x, y) から「どの 3×3 のかたまりに属するか」を計算して平均する（`_avg_pool_by_positions`）。

$$
\text{かたまりの番号} = \lfloor x/3 \rfloor + \frac{\text{横のパッチ数}}{3} \times \lfloor y/3 \rfloor
$$

21×30 なら横 10 かたまり × 縦 7 かたまり＝70。**画像トークンの並びは「左上から右へ、行ごとに下へ」**。

平均した後に **×√768** する。コメントによれば、この掛け算で値が fp16 の範囲（最大 65504）を超えうるので float32 で計算している。
つまり**画像の特徴は値が大きくなりうる**。ただしすぐ後の射影層の最初が RMSNorm なので、大きさはそこで正規化される。

<!-- src: app/pipeline/vit_graph.py | def pool\(self, h: torch.Tensor\) -> torch.Tensor: | return t.reshape\(1, self.M, D\) | note=VitGraph.pool: HF の平均の取り方を reshape と平均に書き直したもの -->
**ソース** `app/pipeline/vit_graph.py` L263〜L268（VitGraph.pool: HF の平均の取り方を reshape と平均に書き直したもの） — md5 e32d0644・`sync_src` が自動で更新

```python
  263|     def pool(self, h: torch.Tensor) -> torch.Tensor:
  264|         """3×3 のパッチを平均して [1,M,768] にする。パッチ n = y·gw + x、出力 m = ky·(gw/3) + kx（HF と同じ並び）。"""
  265|         gh, gw, P, D = self.gh, self.gw, C.POOL, C.VISION_HIDDEN
  266|         t = h.reshape(gh, gw // P, P, D).mean(dim=2)                            # 横の 3 つを平均 → [gh, gw/3, D]
  267|         t = t.reshape(gh // P, P, gw // P, D).mean(dim=1)                       # 縦の 3 つを平均 → [gh/3, gw/3, D]
  268|         return t.reshape(1, self.M, D)
```
<!-- /src -->

<!-- src: app/pipeline/vit_graph.py | # 射影層（Gemma4MultimodalEmbedder） | self.pool_scale = | note=×√768 を射影層の RMSNorm の eps に吸収する -->
**ソース** `app/pipeline/vit_graph.py` L256〜L261（×√768 を射影層の RMSNorm の eps に吸収する） — md5 e32d0644・`sync_src` が自動で更新

```python
  256|         # 射影層（Gemma4MultimodalEmbedder）: RMSNorm（重みなし、eps は vision の rms_norm_eps）→ Linear（float）
  257|         # ③ ×√768 を省くときは eps を 1/768 にする（RMSNorm(c·x; ε) = RMSNorm(x; ε/c²)、c² = 768。StudyMemo 7.4 節）
  258|         eps = qf.eps / C.VISION_HIDDEN if fold_pool_scale else qf.eps
  259|         self.proj_norm = RMSNorm(torch.ones(C.VISION_HIDDEN), eps)
  260|         self.proj = Lin({"w": qf.vec("model.embed_vision.embedding_projection.weight")})
  261|         self.pool_scale = C.VISION_HIDDEN ** 0.5
```
<!-- /src -->

| 目印 | 何をしているか・なぜ |
|---|---|
| `t = h.reshape(gh, gw // P, P, D).mean(dim=2)` | 横に並ぶ 3 パッチを平均。HF は位置から「かたまりの番号」を計算し、one_hot の行列で平均する。格子が固定なら reshape と平均 2 回で同じ値になる（[7.3 節](07_画像エンコーダを書き直す.md)） |
| `t = t.reshape(gh // P, P, gw // P, D).mean(dim=1)` | 縦の 3 つを平均して 7×10＝70 |
| `出力 m = ky·(gw/3) + kx` | 上の式の「かたまりの番号」。並びは左上から右へ、行ごとに下へ（HF と同じ） |
| `eps = qf.eps / C.VISION_HIDDEN if fold_pool_scale else qf.eps` | ×√768 を掛けず、RMSNorm の eps を 1/768 にする。RMSNorm(c·x; ε) = RMSNorm(x; ε/c²) で c²＝768。本文の「値が大きくなりうる」所を作らずに済む |
| `self.pool_scale = C.VISION_HIDDEN ** 0.5` | √768 ≒ 27.7。HF の途中の値と比べるとき（`keep=True`）だけ掛ける |

---

## 2.5 文章側への入れ方（ここが一番大事）

`Gemma4Model.forward`（2318〜2425 行あたり）がやっていること:

| 手順 | 中身 |
|---|---|
| 1 | 画像トークンの ID（258880）を **pad（0）に置き換えて**から、埋め込みの表を引く（×√1536 込み） |
| 2 | PLE の表も、**pad に置き換えた ID** で引く |
| 3 | 画像トークンの位置の埋め込みを、射影層の出力で**置き換える**（masked_scatter）。**√1536 倍はしない** |
| 4 | 文章側（language_model）へ `inputs_embeds` と PLE を渡す。PLE の射影は文章側の中で計算される |
| 5 | マスク: E2B は `use_bidirectional_attention = null` → **普通の因果マスク** |
| 6 | 位置: **普通の連番**（画像の中でも 1 つずつ増える） |

端末の文章グラフの入力に当てはめると:

| グラフの入力 | 文章の位置 | 画像の位置 |
|---|---|---|
| `inputs_embeds` [1,1,1536] | 表を引く（端末の CPU で、いまと同じ） | **画像エンコーダの出力の 1 行** |
| `ple_tok` [1,1,35,256] | PLE の表を引く（いまと同じ） | **pad（ID 0）の行**（定数） |
| cos/sin・マスク・KV | いまと同じ | いまと同じ |

**グラフそのものは変えなくてよい。** 変わるのは「CPU 側で何を入れるか」だけ。

### PLE とは（少しだけ）

Gemma 3n / 4 の小型版は、各層に「そのトークン専用の小さな埋め込み」（256 次元）を足す。作り方は:

$$
\text{PLE} = \frac{1}{\sqrt{2}}\left(\mathrm{RMSNorm}\left(\frac{W \cdot \text{inputs\_embeds}}{\sqrt{1536}}\right) + \text{ple\_tok}\right)
$$

- 前半（射影）は inputs_embeds から作る → 画像の位置では**画像の特徴から**作られる
- 後半（ple_tok）は表を引く → 画像の位置では **pad の行**

つまり画像の特徴は、残差の本流だけでなく **PLE の射影の枝にも入る**。
[4章](04_量子化の目盛りと画像の特徴.md)で、この枝の量子化の目盛り（文章で較正）に収まるかを見る。

<!-- src: android_app/app/src/main/cpp/runner/job.cpp | // ★画像の特徴（EdgeSLM_AndroidVLM Phase 3、2026-09-29）。 | const int64_t img_pad = job.value | note=端末の C++: 画像の位置の入れ方（手順 1〜3 の端末版） -->
**ソース** `android_app/app/src/main/cpp/runner/job.cpp` L1242〜L1253（端末の C++: 画像の位置の入れ方（手順 1〜3 の端末版）） — md5 bba78f1a・`sync_src` が自動で更新

```cpp
 1242|         // ★画像の特徴（EdgeSLM_AndroidVLM Phase 3、2026-09-29）。
 1243|         //   job に `image_features`（float32 [M][H] のファイル。画像エンコーダの出力 image_embeds）があれば、
 1244|         //   プロンプトの画像トークン（`image_token_id`、既定 258880）の位置に、前から順に 1 行ずつ入れる。
 1245|         //   - inputs_embeds: 表を引かず、特徴をそのまま入れる（√H 倍しない。HF の masked_scatter と同じ）
 1246|         //   - ple_tok: その位置だけ `image_pad_id`（既定 0 = pad）の行を引く（HF の Gemma4Model.forward と同じ）
 1247|         //   Phase 1 で、この入れ方が HF の本来の経路と logits までビット一致することを確かめている。
 1248|         //   数えるのは**これから食わせる位置**（start_step 以降）だけ。前のターンのキャッシュを使い回して画像の位置を
 1249|         //   飛ばしたときは 0 個になり、特徴は使わない（同じ写真への 2 問目以降）。
 1250|         std::vector<uint8_t> img_raw;
 1251|         int64_t img_rows = 0, img_next = 0, img_fed = 0;
 1252|         const int64_t img_tok = job.value("image_token_id", static_cast<int64_t>(258880));
 1253|         const int64_t img_pad = job.value("image_pad_id", static_cast<int64_t>(0));
```
<!-- /src -->

<!-- src: app/pipeline/gemma4_graph.py | proj = F.linear\(x0, self.ple_proj\) \* self.ple_in_scale | ple_all = \(self.ple_norm\(proj\) | note=DecodeChunk.forward: PLE の作り方（HF の project_per_layer_inputs を書き直したもの） -->
**ソース** `app/pipeline/gemma4_graph.py` L402〜L404（DecodeChunk.forward: PLE の作り方（HF の project_per_layer_inputs を書き直したもの）） — md5 802b3df9・`sync_src` が自動で更新

```python
  402|             proj = F.linear(x0, self.ple_proj) * self.ple_in_scale
  403|             proj = proj.reshape(1, x0.shape[1], self.n_layers, self.ple_dim)
  404|             ple_all = (self.ple_norm(proj) + ple_in) * self.ple_combine
```
<!-- /src -->

| 目印 | 何をしているか・なぜ |
|---|---|
| `img_tok = job.value("image_token_id", static_cast<int64_t>(258880))` | 手順 1 の目印の ID。プロンプトの中でこの ID の位置が画像の位置 |
| `√H 倍しない。HF の masked_scatter と同じ` | 手順 3。文章の位置は表の値（×√1536 込み）だが、画像の位置は射影層の出力をそのまま入れる。PC 側の同じ処理は [1.1 節](01_VLMの構造.md)の枠（`device_style_embeds`） |
| `img_pad = job.value("image_pad_id", static_cast<int64_t>(0))` | 手順 2。画像の位置の PLE は pad（ID 0）の行を引く。どの写真でも同じ行（定数） |
| `proj = F.linear(x0, self.ple_proj) * self.ple_in_scale` | PLE の前半（射影）。`ple_in_scale` は 1536^−0.5（`__init__` で `hidden_size ** -0.5`）。x0 は inputs_embeds なので、画像の位置では画像の特徴から作られる |
| `ple_all = (self.ple_norm(proj) + ple_in) * self.ple_combine` | RMSNorm してから表の行（ple_tok。画像の位置は pad の行）を足し、`ple_combine`＝2^−0.5（1/√2）を掛ける。上の式そのもの |
