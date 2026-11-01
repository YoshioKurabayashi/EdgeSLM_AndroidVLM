<<OUT DoC/StudyMemo/13_コードで読む検証.md>>
# 13. コードで読む検証 ―― 道具・コマンド・実際のソース

> **このファイルは生成物**です（`app/study/make_code_tour.py` が `app/study/tour/13_code.md` と実際のソースから作る。2026-10-14）。
> ソースの抜き出し・行番号・結果の値は作った時点のものそのものです（最後の 13.9 にソースの md5）。
> 作り直し: `& $PY2 -B -m app.study.make_code_tour`（`--check` で、抜き出しの目印が全部ソースに見つかるかだけを確かめる）。

12 章は「どのファイルの何行目を見ればよいか」の地図でした。この章はその先の**中身**です。検証ごとに
**何を確かめたか → 道具とコマンド → 流れ → 実際のソース（行番号つきの抜き出しと、どの行が何をしているか・なぜの表）→ 結果**の順で読みます。
表の「目印」は抜き出しの中のその行の書き出しです。

| 節 | やったこと | 主なソース |
|---|---|---|
| 13.1 | **写真 → 画像エンコーダの入力**（レターボックスとパッチ。PC と端末の C++） | `hf_vlm.py`・`vlm_prep.cpp` |
| 13.2 | **文章側は無改造で使えるか**（3 つの経路 A・B・C） | `hf_vlm.py`・`check_wiring.py` |
| 13.3 | **入口の目盛り**を ONNX から読み、切り詰めを真似る | `onnx_scales.py`・`check_wiring.py` |
| 13.4 | **画像エンコーダを NPU 向けに書き直す** | `vit_graph.py` |
| 13.5 | **端末でだけ壊れる（C-14）**: 小さすぎる重みのスケール | `qat_inject_vit.py` |
| 13.6 | **端末でつなぐ**: 画像の特徴を文章側に入れる・開く間は画面を前に | `job.cpp`・`RunnerHoldActivity.kt` |
| 13.7 | **まとめ読み（16 トークン版）** | `g4t_graph.py`・`job.cpp` |
| 13.8 | 自分で確かめを 1 つ足すときの手順 | — |

---

## 13.1 写真 → 画像エンコーダの入力（`hf_vlm.py`・`vlm_prep.cpp`）

### 何を確かめたか

NPU のグラフは形が固定なので、写真の大きさ・縦横比によらず**いつも同じ格子**（70 トークン → 21×30 パッチ ＝ 336×480 ピクセル）に収めます（3.1 節）。
PC（PIL）で作る入力と、端末の C++ で作る入力が**ビット単位で同じ**であることも確かめました。

```powershell
& $PY2 -m app.pipeline.device_vlm prep         # 端末の前処理 ↔ PC（PIL）がビット一致するか（数分）
& $PY2 -B -m app.study.ex01_letterbox_patches  # 演習: レターボックスとパッチの並び
```

### レターボックス（`prepare_image` の fixed）

<<CODE app/pipeline/hf_vlm.py | from PIL import ImageOps | ^    return canvas$ | from=^def prepare_image>>

| 目印 | 何をしているか・なぜ |
|---|---|
| `ImageOps.exif_transpose(img)` | スマホの縦写真は「画素は横のまま、回転して表示せよ」という印だけのことがある。先に向きを直す（HF は PIL の画像を渡すと直さない） |
| `bg.paste(rgba, mask=…)` | 透明な部分は白で埋める |
| `gh, gw = grid or C.FIXED_GRIDS[budget]` | 格子（パッチの縦×横）。縦長の格子を使うときは `grid` で渡す |
| `s = min(W / img.width, H / img.height)` | 縦横比を保って、格子に収まる最大の倍率 |
| `canvas = Image.new("RGB", (W, H), (0, 0, 0))`・`paste(… ((W - nw) // 2, (H - nh) // 2))` | 黒い画像の真ん中に貼る（レターボックス）。プロセッサから見ると「ちょうどの大きさ」なので、それ以上大きさを変えない |

### パッチに並べる（`pixels_fixed_np`）

<<CODE app/pipeline/hf_vlm.py | x = np.asarray\(img_fixed.convert\("RGB"\), dtype=np.uint8\) | return np.ascontiguousarray\(x.reshape\(gh \* gw, C.PATCH \* C.PATCH \* 3\)\)>>

`(縦, py, 横, px, c)` を `(縦, 横, py, px, c)` に並べ替えて `[N, 768]`。パッチ n ＝ y·横の数 ＋ x（HF の `convert_image_to_patches` と同じ）。
×1/255 は float32 で掛ける（torch の「uint8 × Python の float」と同じ型にして、値をビット単位でそろえる）。

### 同じことを端末の C++ で（`vlm_pixels`）

<<CODE android_app/app/src/main/cpp/runner/vlm_prep.cpp | ^bool vlm_pixels | ^    const double t2 = now_ms\(\); | omit=// 黒い画像に貼る..    // パッチ化と ×1/255>>

| 目印 | 何をしているか・なぜ |
|---|---|
| `std::nearbyint(w * s)` | Python の `round`（偶数への丸め）と同じ丸め方。1 ピクセル違うと全部の値がずれる |
| `pil_bicubic_resize_rgb(…)` | Pillow の Resample.c と同じ 3 次の縮め方（係数・22bit の整数の重み・丸めまで同じ）を C++ で書いた |
| （中略した所） | 黒い画像の真ん中に貼る |
| `static_cast<float>(0.00392156862745098)` | `processor_config.json` の rescale_factor（1/255）。float で掛ける（numpy の float32 の掛け算と同じ） |
| 4 重のループ | パッチ (py, px) の中を (yy, xx, c) の順に並べる（PC の reshape・transpose と同じ並び） |

結果: 端末 ↔ PC（PIL）でビット一致 <<VAL results/P3_device_vlm/b70_prep.json | "%d/%d 枚" % (d["n_exact"], d["n"])>>（`results/P3_device_vlm/b70_prep.json`）。

---

## 13.2 文章側は無改造で使えるか（3 つの経路）

### 何を確かめたか

端末の文章グラフ（EdgeSLM_AndroidGemma4 で作った、文章だけの Gemma 4）を**作り直さずに**使えるか（6 章）。端末では画像の特徴を自分で
入力の埋め込みに差し込むので、その入れ方が HF の本来の経路と同じ計算になるかを、PC で 3 通りに比べました。

```powershell
& $PY1 -m app.pipeline.check_wiring --srq off   # SRQ（QAT の 8bit の丸め）なしで、4 問・70 トークン
```

### 端末と同じ入れ方（`device_style_embeds`）

<<CODE app/pipeline/hf_vlm.py | img_mask = ids == C.IMAGE_TOKEN_ID | return emb, ple_tok, img_mask | from=^def device_style_embeds>>

| 目印 | 何をしているか・なぜ |
|---|---|
| `img_mask = ids == C.IMAGE_TOKEN_ID` | プロンプトの中の画像トークン（258880）の位置 |
| `ids_pad = torch.where(img_mask, … C.PAD_ID …)` | 画像の位置の ID を pad（0）に置き換えてから表を引く（HF の `Gemma4Model.forward` と同じ） |
| `emb[0, img_mask] = feats` | 画像の位置には画像の特徴を**そのまま**入れる（文章の埋め込みと違い √1536 倍しない。HF の masked_scatter と同じ） |
| `ple_tok = lm.get_per_layer_inputs(ids_pad, None)` | PLE（層ごとの埋め込み）は、画像の位置は pad の行になる |

### 3 つの経路を比べる（`run_one`）

<<CODE app/pipeline/check_wiring.py | ref = model\(\*\*inputs\).logits\[0\].float\(\) | "A_vs_B": compare\(ref, bat, span\) | from=^def run_one>>

| 目印 | 経路 | 意味 |
|---|---|---|
| `ref = model(**inputs).logits` | **A** | HF の本来の経路（`pixel_values` を渡し、HF の中で特徴を差し込む） |
| `bat, _ = H.device_style_step(model, emb, ple, None)` | **B** | 端末と同じ入れ方（自分で作った `inputs_embeds`・`ple_tok`）で、文章側だけに**一度に**入れる |
| `for t in range(T): … emb[:, t:t + 1] …` | **C** | B と同じ入力を、端末の decode と同じく **1 トークンずつ**（KV キャッシュを持ち越す） |

A↔B が一致すれば「画像の位置の入れ方・PLE・マスク・位置」の理解が正しい。B↔C の差は行列の形の違いによる足し算の順序の差だけです。
**SRQ を外して比べる**のは、丸めありだと正しい配線でも微小な差が格子の境目で転んで大きく見えるため（6.3 節）。

結果（SRQ なし、4 問。`results/P1_check_wiring/wiring_b70_srq-off.json`）:

<<JSON results/P1_check_wiring/wiring_b70_srq-off.json | [{"問題": r["id"], "A↔B 相対L2 最大": r["A_vs_B"]["rel_l2_max"], "B↔C 相対L2 最大": round(r["B_vs_C"]["rel_l2_max"], 7), "B↔C 1 位の一致": r["B_vs_C"]["top1_agree"]} for r in d["runs"]]>>

A↔B は 0（ビット一致）。**文章側のグラフは本当に無改造で使える**（一番大事なこと 8）。

---

## 13.3 入口の目盛りを ONNX から読み、切り詰めを真似る（`onnx_scales.py`・`check_wiring.py`）

端末の文章グラフの入口の量子化は**文章だけで較正**してあるので、表せる範囲は `lo = (0 − zp) × scale`・`hi = (65535 − zp) × scale` に限られます（4 章）。
借りている venv に onnx が無いので、`read_quant_points` は ONNX（protobuf）の欄を**直接たどって**、小さな初期値（scale・zero point）だけを読みます。

<<CODE app/pipeline/onnx_scales.py | ^    out = \{\}$ | ^    return out$ | from=^def read_quant_points>>

範囲の外は端末では切り詰められます。PC の HF では起きないので、**わざと切り詰めて**影響を測ります。

<<CODE app/pipeline/check_wiring.py | ^def clip_inputs | ^    return emb, ple$>>

<<CODE app/pipeline/check_wiring.py | def clipped\(inputs_embeds, per_layer_inputs=None\): | lm.project_per_layer_inputs = orig | from=^def entry_clip>>

| 目印 | 何をしているか・なぜ |
|---|---|
| `clip_inputs`: `emb.clamp(e["lo"], e["hi"])` | 入口の `inputs_embeds`・`ple_tok` を範囲で切る（端末も全部の位置に同じ目盛り） |
| `entry_clip` の `clipped` | PLE の射影（文章側の中で計算される）を一時的に包んで、その入力・出力も切る |
| `lm.project_per_layer_inputs = clipped` → `finally: … = orig` | **一時的に**差し替えて、必ず元に戻す（`with` の中だけ効く） |

演習 ex03 で、本物の `read_quant_points` を動かして範囲 [−7.03, 3.52] を読めます。画像の特徴はほぼこの範囲に収まり、切り詰めても答えは変わりませんでした（4.3 節）。

---

## 13.4 画像エンコーダを NPU 向けに書き直す（`vit_graph.py`）

### 何を確かめたか

HF の画像エンコーダのコードをそのまま書き出さず、**同じ計算を、格子を固定して素の演算で**書き直しました（7 章）。位置・cos/sin を定数に、
プーリングを reshape と平均に、×√768 を RMSNorm の eps に吸収。どれも数学的に同じ形です。

```powershell
& $PY1 -m app.pipeline.capture_vit         # HF の正解と較正の入力を取っておく
& $PY2 -m app.pipeline.export_vit check    # 自前の配線 ↔ HF
& $PY2 -m app.pipeline.export_vit export   # ONNX に書き出す
```

### 定数にするもの（`VitGraph.__init__`）

<<CODE app/pipeline/vit_graph.py | pos = grid_positions\(gh, gw\) | self.pool_scale = C.VISION_HIDDEN \*\* 0.5 | from=^class VitGraph>>

| 目印 | 何をしているか・なぜ |
|---|---|
| `self.register_buffer("pos_emb", (table[0][pos[:, 0]] + table[1][pos[:, 1]])[None])` | 位置の表の x 行と y 行を足した定数 [1,N,768]。格子が固定なので位置も固定。**Gather をグラフに入れない** |
| `cos, sin = rope2d_tables(pos)` → `register_buffer` | 2 次元 RoPE の cos/sin も定数の表に（毎回計算しない。演習 ex02 の 3） |
| `eps = qf.eps / C.VISION_HIDDEN if fold_pool_scale else qf.eps` | プーリングの後の ×√768 を掛けず、射影層の RMSNorm の eps を eps/768 に（`RMSNorm(c·x; ε) = RMSNorm(x; ε/c²)`。演習 ex02 の 1） |

### プーリングと流れ（`pool`・`run`）

<<CODE app/pipeline/vit_graph.py | def pool\(self, h: torch.Tensor\) -> torch.Tensor: | return t.reshape\(1, self.M, D\)>>

<<CODE app/pipeline/vit_graph.py | x = 2.0 \* \(pixel_values - 0.5\) | out\["image_embeds"\] = self.proj\(self.proj_norm\(p\)\)>>

| 目印 | 何をしているか・なぜ |
|---|---|
| `h.reshape(gh, gw // P, P, D).mean(dim=2)` → `reshape(gh // P, P, gw // P, D).mean(dim=1)` | 横の 3 つ → 縦の 3 つを平均（HF は one_hot の行列 630×70 を掛ける。定数行列を持たない・テンソルは 4 次元まで） |
| `x = 2.0 * (pixel_values - 0.5)` | 0〜1 の画素を −1〜1 に |
| `h = self.patch_proj(x) + self.pos_emb` | パッチの埋め込み＋位置（定数） |
| `if not self.fold_pool_scale: p = p * self.pool_scale` | 吸収しない版（確かめ用）だけ ×√768 |
| `self.proj(self.proj_norm(p))` | 射影層（重みなしの RMSNorm → 768→1536）。これが文章側に渡す画像の特徴 [1,70,1536] |

### 1 層（`VitLayer.forward`）

<<CODE app/pipeline/vit_graph.py | def forward\(self, h: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor\) -> torch.Tensor: | return h$ | from=^class VitLayer>>

`s = q @ kᵀ` にスケールが無いのは q・k をそれぞれ正規化しているため（Gemma 4 の決まり）。マスクも無し（画像の中は双方向・パディング無し）。
`k^T` と `v` は行列積の**右辺だけ**に使うテンソルになるようにしてあり、C-8（右辺だけ 8bit）を当てられます。

---

## 13.5 端末でだけ壊れる（C-14）: 小さすぎる重みのスケール（`qat_inject_vit.py`）

### 何が起きたか

量子化した画像エンコーダを端末の NPU で動かすと、**エラーなく動くのに値がでたらめ**でした（8 章）。途中の値を出力に足した探り用のグラフ（`probe_vit.py`）で
「端末 ↔ 同じグラフの PC の QDQ」を比べ、パッチの埋め込みの、重みがほぼ 0 のチャネルだけが壊れていると分かりました。

```powershell
& $PY2 -m app.pipeline.probe_vit build --layers 2     # 2 層版の探り用グラフ
& $PY2 -m app.pipeline.probe_vit device --layers 2    # 途中の値ごとに 端末 ↔ PC の QDQ
& $PY2 -B -m app.study.ex04_c14_scale_floor           # 演習
```

<<CODE app/pipeline/qat_inject_vit.py | ★出力チャネルごとのスケールの下限 | ^def _from_float | omit=^SCALE_FLOOR_RATIO = 1.0 / 64..^def _from_float>>

<<CODE app/pipeline/qat_inject_vit.py | ^def _from_float | "from_float": True, "n_floored": int\(\(raw < floor\).sum\(\)\)\}>>

| 目印 | 何をしているか・なぜ |
|---|---|
| コメントの「境目は s ≈ 4.0e-6（最大のチャネルの 1/659）」 | 壊れたのは全部それより小さいスケールのチャネル、正常なのは全部大きいチャネルだった（実測で境目が決まった） |
| `raw = max|w| / 127` | ふつうの INT8・出力チャネルごと・対称のスケール |
| `floor = raw.max() * SCALE_FLOOR_RATIO`・`s = np.maximum(raw, floor)` | 下限を最大の 1/64 に（境目に対して約 10 倍の余裕）。小さい重みが少し粗くなるだけで、値はほぼ変わらない |
| `"n_floored": …` | 持ち上げたチャネルの数を記録する |

<<CODE app/pipeline/qat_inject_vit.py | ^def scale_ratio_report | "ok": bool\(worst is None or worst\[1\] >= SCALE_FLOOR_RATIO \* 0.999\)\}>>

見張り: 同じ行列の中では入力・出力のスケールは 1 つなので、重みのスケールの「最小 ÷ 最大」が極端に小さいものを探せば足ります（`quantize_vit.py` の検査で毎回出す）。
**PC の QDQ は丸めは真似るが、NPU の計算の限界は真似ない**（一番大事なこと 14）。

---

## 13.6 端末でつなぐ（`job.cpp`・`RunnerHoldActivity.kt`）

### 画像の特徴を文章側に入れる（`job.cpp`）

<<CODE android_app/app/src/main/cpp/runner/job.cpp | ★画像の特徴（EdgeSLM_AndroidVLM Phase 3、2026-09-29） | img_rows = n_img_prompt; | omit=if \(emb_idx < 0 \|\| H <= 0\) \{..                if \(!read_all>>

<<CODE android_app/app/src/main/cpp/runner/job.cpp | auto fill_row = \[&\]\(int64_t r, int64_t tid, bool is_img, int64_t pos\) -> bool \{ | +10>>

| 目印 | 何をしているか・なぜ |
|---|---|
| `for (size_t t = start_step; …) n_img_prompt += (prompt[t] == img_tok)` | **これから食わせる位置**の画像トークンだけ数える（同じ写真への 2 問目以降はキャッシュを使い回すので 0 個になり、特徴は使わない） |
| `img_raw.size() / row != n_img_prompt` → エラー | 特徴の行の数とプロンプトの画像トークンの数が合わなければ止める |
| `fill_row`: `tab_id = is_img ? img_pad : tid` | 画像の位置は PLE を pad の行で引く（13.2 節の `device_style_embeds` と同じ） |
| `std::memcpy(e, img_raw.data() + img_next * H * 4, …)` | 画像の位置の埋め込みは、特徴を前から順に 1 行ずつそのまま入れる |

### 開く間は画面を前に出す（`RunnerHoldActivity.kt`）

1.5GB の文章側のモデルを開く間、端末は一時的に約 3GB 使います。画面の無い裏側（Service だけ）だと OS（lmkd）に止められることがありました（9 章）。

<<CODE android_app/app/src/main/java/org/edgeslm/demo/RunnerHoldActivity.kt | override fun onCreate\(savedInstanceState: Bundle\?\) \{ | setTurnScreenOn\(true\)>>

<<CODE android_app/app/src/main/java/org/edgeslm/demo/RunnerHoldActivity.kt | if \(!restored\) \{ | startForegroundService\(svc\)>>

| 目印 | 何をしているか・なぜ |
|---|---|
| `FLAG_KEEP_SCREEN_ON` | 画面を消さない。消えると画面が裏に回り、優先度（oom_score_adj）が下がる |
| `setShowWhenLocked(true)` | スワイプのロックの上にも出す（ロックの裏だと「前にある」扱いにならない） |
| `startForegroundService(svc)` | RunnerActivity と同じく、同じプロセスの RunnerService に job を渡す。画面はそのまま前に居続ける |

PC 側は `device_vit.run_job(job, hold=True)` がこの入口を使います（VLMSpec の StudyMemo 6.1 節に抜き出しと解説）。結果: 画面を前に出すと <<VAL results/P3_device_vlm/b70_holdcheck_hold.json | "%d/%d" % (d["n_ok"], d["n"])>> で止められなかった（`results/P3_device_vlm/b70_holdcheck_hold.json`）。

---

## 13.7 まとめ読み（16 トークン版）（`g4t_graph.py`・`job.cpp`）

### 何を確かめたか

1 トークン版は 1 回 約 77ms × プロンプトの長さ（107 トークン）で、最初の文字まで 8 秒かかりました。NPU の 1 回の時間は重みを読む時間でほぼ決まるので、
**1 回で 16 トークン読む**グラフを作ると、16 トークン入れても 1 回の時間は 1.1 倍（10 章）。グラフの約束（窓・マスク・空の行・行の選択）は次の 2 か所にあります。

```powershell
& $PY2 -m app.pipeline.build_t16      # 16 トークン版: 書き出し〜繋ぐ（1〜3 時間）
& $PY2 -m app.pipeline.check_t16      # PC の CPU で 1 トークン版・HF と比べる
& $PY2 -B -m app.study.ex05_t16_window   # 演習: 約束を小さな注意で再現
```

### 1 層の窓（`layer_forward_t`）

<<CODE app/pipeline/g4t_graph.py | q = ly.q_proj\(x\).view\(1, T, ly.nh, ly.hd\) | ctx = torch.matmul\(p, pv.reshape\(1, Lw, ly.hd\)\) | from=^def layer_forward_t>>

| 目印 | 何をしているか・なぜ |
|---|---|
| `new_k, new_v = k.transpose(1, 2), …` | この回の T 行の K・V（出力するのはこれ。端末が本物の n 行だけを past に足す） |
| `pk = torch.cat((past_k, new_k), dim=2)` | 窓 = past P 個 ＋ 新しい T 個 = W 個（1 トークン版と同じ長さ）。★past を Concat の最初に入れる順（Slice に直接入る float の入力は量子化されない） |
| `s = … + attn_mask` | マスクは端末が作る（past の中身のある所＋この回の因果。空の行は 0 行目と同じ） |
| `ctx = torch.matmul(p, pv.reshape(1, Lw, ly.hd))` | 確率 × V の右辺は形を変えた V（C-8 で右辺だけ 8bit にできるように） |

### 行の選択（`DecodeChunkT.forward` の最後）

<<CODE app/pipeline/g4t_graph.py | if self.select_row:$ | return \(h, ple_all, \*presents\) if self.with_ple_proj else \(h, \*presents\) | from=^    def forward\(self, x0>>

`h = hidden^T @ logit_sel`（[1,H,T] @ [1,T,1]）。**行列積の右辺を logit_sel にする**のが要点で、C-8 で 8bit になるのは 0 と 1 だけの logit_sel（正確に表せる）。
逆（sel @ hidden）にすると hidden が 8bit になり、出力層の入力が荒れます。

### 端末: 16 個ずつ読み、1 個ずつ作る（`job.cpp`）

<<CODE android_app/app/src/main/cpp/runner/job.cpp | ---- プロンプトを Tm 個ずつ読む | if \(std::find\(stop_ids.begin\(\), stop_ids.end\(\), best\) != stop_ids.end\(\)\) stopped_by_eos = true;\n?$ | from=auto run_chunk = >>

| 目印 | 何をしているか・なぜ |
|---|---|
| `n = std::min<size_t>(Tm, prompt.size() - t)` | 1 回に入れる本物の行（最大 16） |
| `imgs[k] = img_rows > 0 && toks[k] == img_tok` | その行が画像の位置か（13.6 節の `fill_row` に渡す） |
| `if (t < prompt.size()) { prefill_ms += wall; continue; }` | まだプロンプトが残っていれば、時間を「プロンプトを読む時間」に |
| 最後の回: `last = best; newly.push_back(best);` | 最後の回の答えが答えの 1 語目 |

答えを作る段は `run_chunk({last}, {false}, …)`（0 行目だけ本物）を止める語が出るか上限まで繰り返します。
16 トークン版の結果（点数は同じ・HF との一致は少し下がった、その原因の切り分け 7 段）は 10.7 節と `results/P5_20261003_t16_accuracy.md`。
**基準にする側（PC の CPU）が正しいかを先に確かめる**、が最後に分かったこと（一番大事なこと 18）。

---

## 13.8 自分で確かめを 1 つ足すとき

1. **何と何を比べるかを先に決める**: 同じ入力・同じグラフで、違うのは 1 か所だけ（13.2 節の A・B・C、13.5 節の端末 ↔ 同じグラフの PC の QDQ）
2. **丸めを外した版で配線を確かめる**: SRQ・量子化なしで一致してから、丸めありで「揺れ」と比べる（6.3・7.9 節）
3. **途中の値を出力に足す**: 端末でしか起きないことは、途中の値を出力に足した版で層ごとに比べる（`probe_vit.py`、8.2 節）
4. **答えで比べる**: 差の割合だけでなく、126 問の点数と、端末で新しく出た誤りの種類を見る（9.6・10.7 節）
5. **記録**: `results/` に数字、`DoC/StudyMemo/` に学んだこと。このテンプレート（`app/study/tour/13_code.md`）に抜き出しを足して作り直す

---

## 13.9 この章で抜き出したソースの md5

<<MD5_TABLE>>
