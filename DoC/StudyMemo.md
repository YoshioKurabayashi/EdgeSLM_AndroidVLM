# StudyMemo — 目次

| | |
|---|---|
| 目的 | **「画像を読める LLM（VLM）を NPU に載せる」とき、何がどこで起きていて、なぜその設計にするのかを自分で説明できる**状態になること |
| 題材 | EdgeSLM_AndroidVLM の Phase 0〜5（2026-09-27〜30）。Gemma 4 E2B（QAT 版）に画像部分を足して、Xperia 5 III（SD888 / HTP v68）で動かす検証 |
| 前提 | 文章だけの LLM を NPU に載せる話（量子化・QDQ・C-1〜C-13 など）は [EdgeSLM_Android の StudyMemo](../../EdgeSLM_Android/DoC/StudyMemo.md) にある。ここでは**画像が加わって新しく出てくること**だけを扱う |
| 関連 | [システム設計書](システム設計書.md)（いまの形の全体）/ [要件定義書](要件定義書.md)（計画）/ [P0 棚卸し](P0_既存資産の棚卸し.md)（モデル選定の根拠）/ [基本設計 Phase 1〜5](基本設計_Phase5.md)（各段階の設計と結果） |
| 全体の構成図 | [コンポーネント構成図.html](コンポーネント構成図.html)（2026-10-23。写真 → 前処理 → 画像エンコーダ → 文章側への差し込みの流れと、PC の評価・変換の道具・受け渡し・端末の VLM 検証アプリ・調べる道具を 1 枚に。各部品に対応する章の番号つき。ブラウザで開く） |

**上から順に読むと積み上がる**が、知りたい所だけ読んでも通じるようにしてある。
実測で分かったことは、Phase が進むごとに各章へ追記する（「★実測」の印を付ける）。Phase 1 の実測は 2026-09-28 に追記した（3〜6 章）。Phase 5 の実測は 2026-09-30 に 10・11 章と 3 章に追記した。

**VLM の構造を学ぶなら**: 1 章（部品と「並べるだけ」の条件）→ 2 章（Gemma 4 で写真が 70 トークンになるまで）→ 10 章（prefill と decode、KV キャッシュ）→ 11 章（いまのスマホの VLM を端から端まで、実測つき）の順が早い。

---

## 読む順番

| # | 文書 | 何が分かるか |
|---|---|---|
| **0** | [用語集](StudyMemo/00_用語集.md) | VLM で新しく出てくる語（パッチ・画像トークン・射影層・プーリング・2 次元 RoPE・PLE・レターボックス など）をその場で引く |
| **1** | [VLM の構造](StudyMemo/01_VLMの構造.md) | **画像が「言葉と同じ形」になるまで。** ViT → 射影層 → LLM。3 種類の構造（LLaVA 系 / Qwen2.5-VL / Qwen3-VL）の違いと、**「射影層を足すだけ」が成り立つ条件** |
| **2** | [Gemma 4 E2B の画像の通り道](StudyMemo/02_Gemma4の画像の通り道.md) | **336×480 の写真が 70 個のトークンになるまでを手で追う。** 形の変化、プーリング、√768、射影層、そして文章側への入れ方（PLE は pad の行・マスクは因果） |
| **3** | [NPU に載せるための設計](StudyMemo/03_NPUに載せるための設計.md) | **形を固定する（レターボックス）・画像トークン数と待ち時間・メモリとプロセスの分け方・v68 の制約が ViT に与える影響** |
| **4** | [量子化の目盛りと画像の特徴](StudyMemo/04_量子化の目盛りと画像の特徴.md) | **文章だけで較正したグラフに、画像の特徴を入れると何が起きうるか。** 入口の範囲 [−7.03, 3.52] を ONNX から手で計算する。PC で再現できる丸め（SRQ）とできない丸め（入口の QDQ） |
| **5** | [評価の組み立て](StudyMemo/05_評価の組み立て.md) | **公開データの選び方とライセンス、自動判定の罠**（否定の語尾・数字のくっつき・「日本」の「本」）、合成画像で解像度の影響を切り分ける |
| **6** | [検算の考え方](StudyMemo/06_検算の考え方.md) | **「文章側は無改造で使える」をどう確かめるか。** 3 通りの経路（A/B/C）で、配線の誤りと数値の誤差を分けて見る。SRQ なしで判定する理由 |
| **7** | [画像エンコーダを書き直す](StudyMemo/07_画像エンコーダを書き直す.md) | **Phase 2。HF のコードをそのまま書き出さない理由。** 位置・cos/sin を定数に、プーリングを reshape と平均に、×√768 を RMSNorm の eps に吸収する（数学的に同じ形への変形）。**★SRQ への敏感さ**（7.8）、**量子化の誤差を揺れと比べる**（7.9）、**変換ログの DDR の見積もり**（7.10） |
| **8** | [端末でだけ壊れる](StudyMemo/08_端末でだけ壊れる.md) | **Phase 2-4。NPU でエラーなく動くのに値がでたらめ。** 途中の値を出力に足した版で切り分ける、壊れ方の形（範囲の端に張り付く・符号・チャネル）を読む、小さすぎる重みのスケールが NPU の整数の計算で表せない（C-14） |
| **9** | [端末でつなぐ](StudyMemo/09_端末でつなぐ.md) | **Phase 3。大きいモデルを開くときに OS に止められる。** Android の lmkd と優先度（oom_score_adj）、開く間だけ約 3GB 使う理由、画面を前に出す対策、止まったときに理由が残る道具。端末だけで解いた品質の見方（同じ特徴どうし・入れ替わり・端末で新しく出た誤り） |
| **10** | [まとめ読み](StudyMemo/10_まとめ読み.md) | **Phase 5。最初の文字 8 秒 → 0.6 秒。** prefill と decode、KV キャッシュ、NPU の 1 回の時間が「重みを読む時間」で決まること（1 回 1.55GB）、16 トークン版 1 本で両方やる理由、窓・マスク・空の行・行の選択、3 段の確かめ方。**★量子化モデルは活性の目盛りまで含めて 1 つのモデル**（点数は同じでも HF との一致が下がった）、**★system 文 1 行で 45 → 54 問**、画像エンコーダを開いたままにする |
| **11** | [写真から答えまで](StudyMemo/11_写真から答えまで.md) | **いまのスマホの VLM を 1 本の流れで追う。** 2 つのプロセスと 2 つのモデル、時間の流れ（実測）、形の変化（4032×3024 → 630 パッチ → 70 トークン → 107 トークン → 1 語）、プロンプトの実際の ID、KV キャッシュの一生、答えを止める条件、いま時間を使っている所（85% は答えを 1 つずつ作る段） |
| **12** | [ソースと動かし方](StudyMemo/12_ソースと動かし方.md) | **1〜11 章の内容を実際のソースとコマンドに紐づける**（2026-10-14）: 全体の地図（PC の道具・端末のアプリ・2 つの venv の使い分け）、**スマホも GPU も要らない演習 5 本**（`app/study/`: レターボックスとパッチ・画像エンコーダの書き直し・入口の目盛りを ONNX から読む・C-14・16 トークン版の窓とマスク）、章ごとの対応表（節 → 関数と行・コマンド・結果ファイル）、一番大事なこと 1〜20 とソース |
| **13** | [コードで読む検証](StudyMemo/13_コードで読む検証.md) | **検証ごとに「どの道具を、どのコマンドで動かし、中でどのコードが何をしているか」を、実際のソースの抜き出しと 1 行ずつの解説で**（2026-10-14）: 写真 → 入力（PC と端末の C++）、文章側は無改造で使えるか（3 つの経路）、入口の目盛りと切り詰め、画像エンコーダの書き直し、C-14、端末でつなぐ（画像の特徴の入れ方・画面を前に）、まとめ読み。**抜き出しはソースから自動で作る**（`app/study/make_code_tour.py`。ソースがずれると止まる） |

---

## 一番大事なことを 20 行で

1. **VLM は「画像を、言葉の埋め込みと同じ形の行列にして LLM に食わせる」仕組み。** LLM から見ると、画像トークンは「表を引かずに外から渡された埋め込み」にすぎない（[1章](StudyMemo/01_VLMの構造.md)）。
2. **「射影層を足すだけ」が成り立つかは、モデルが画像トークンに特別な位置やマスクを使うかで決まる。** Gemma 4 E2B は使わない（連番の位置・因果マスク）ので、文章側のグラフを作り直さずに済む見込み（[2章](StudyMemo/02_Gemma4の画像の通り道.md)）。
3. **端末では形を固定する。** 画像の縦横比に合わせてパッチの並びを変える HF の既定は使えない。固定の格子に収めて余白を黒で埋める（[3章](StudyMemo/03_NPUに載せるための設計.md)）。
4. **待ち時間は画像トークン数にほぼ比例する。** いまの文章グラフは 1 回 1 トークン（約 77ms）なので、文章の約 40 トークンを足して、280 トークンで約 25 秒、70 トークンで約 8 秒（[3章](StudyMemo/03_NPUに載せるための設計.md)）。
5. **量子化の目盛りは「較正に使ったデータ」の範囲しか表せない。** 文章だけで較正した入口に画像の特徴を入れると、範囲の外は切り詰められる。PC の HF では起きないので、**わざと切り詰めて**影響を測る（[4章](StudyMemo/04_量子化の目盛りと画像の特徴.md)）。
6. **自動判定は必ず甘さと厳しさの両方を疑う。** レビューで「否定を落とす」「数字がくっつく」の 2 つが見つかった。判定の自己テストを書き、答えの全文を残す（[5章](StudyMemo/05_評価の組み立て.md)）。
7. **配線の正しさは、丸めを外した状態で、同じ形の計算どうしで比べる。** 丸めありだと、正しい配線でも微小な差が格子の境目で転んで大きく見える（[6章](StudyMemo/06_検算の考え方.md)）。
8. **★実測: 文章側のグラフは本当に無改造で使える。** HF の本来の経路と「端末と同じ入れ方」の logits がビット単位で一致した。画像の特徴も入口の目盛りにほぼ収まった（[4章](StudyMemo/04_量子化の目盛りと画像の特徴.md)・[6章](StudyMemo/06_検算の考え方.md)）。
9. **★実測: 画像トークン 70 と 280 で品質が大きく違う**（0.37 → 0.61）。画面・文書は 70 で足り、実写の細部は 280 が要る。70 では「写っていない」と言い切る誤りが出る（[3章](StudyMemo/03_NPUに載せるための設計.md)）。
10. **★実測: 端末と同じ 1 トークンずつの流し方でも品質は同じ。ただし 1 問ずつの正誤は 1 割ほど入れ替わる**（SRQ の丸めの揺れ）。端末と PC は「全体の点」と「入れ替わりの偏り」で比べる（[6章](StudyMemo/06_検算の考え方.md)）。
11. **★実測: 画像エンコーダは SRQ の 8bit の丸めにとても敏感。** 同じ計算でも順序が違うだけで特徴の cos が 0.84 まで離れる。端末の合否を特徴の cos で決めず、算術の誤差は SRQ なしの版で、品質は答えで見る（[7章](StudyMemo/07_画像エンコーダを書き直す.md)）。
12. **★実測: 画像エンコーダの量子化（16bit の活性・8bit の重み）の誤差は小さい**（SRQ なしで cos 0.997 以上）。QAT の重みは整数とスケールを写すので誤差 0。量子化の誤差は SRQ の揺れより 1 桁以上小さい（[7章](StudyMemo/07_画像エンコーダを書き直す.md)）。
13. **★実測: 画像エンコーダは v68 に変換できた。ただし 630 トークンを一度に通すので、途中の値が NPU の高速メモリ（4MB）に入らず、1 回に約 1GB を DDR に出し入れする**（文章側の 1 トークンは 0.6MB）。一度に通すトークン数で、NPU で何が時間を食うかが変わる（[7章](StudyMemo/07_画像エンコーダを書き直す.md)）。
14. **★実測: 端末の NPU で「エラーなく動くのに値がでたらめ」が起きた。原因は、ほぼ 0 の重みのチャネルの小さすぎるスケール（NPU の整数の計算で表せない）。** PC の QDQ は丸めは真似るが NPU の計算の限界は真似ない。途中の値を出力に足した版で、同じグラフの PC の QDQ と比べて見つけた（[8章](StudyMemo/08_端末でだけ壊れる.md)）。
15. **★実測: 端末の中だけで写真 → 日本語の答えが出た。品質は PC と同等**（端末だけ 48/123、同じ特徴で PC の文章側 45/123、入れ替わりに偏りなし）。ただし端末の答えは長めで、「画像が提供されていません」と答える誤りが、PC では出なかった 3 問で出た。端末と PC は点数だけでなく、端末で新しく出た誤りの種類も見る（[9章](StudyMemo/09_端末でつなぐ.md)）。
16. **★実測: 1.5GB のモデルを開く瞬間が一番危ない。** 開く間は一時的に約 3GB 使い、画面の無い裏側だと OS（lmkd）に止められることがある（10 回中 3 回プロセスが消えた）。**開く間は画面を前に出す**（優先度 0）。今回 5/5、以前 25/25 で止められなかった（[9章](StudyMemo/09_端末でつなぐ.md)）。
17. **★実測: LLM の推論は prefill（プロンプトを読む）と decode（1 つずつ作る）に分かれ、prefill はまとめて読める。** NPU の 1 回の時間は重みを読む時間でほぼ決まる（16 トークン版も 1 トークン版も 1 回に約 1.55GB 読む）ので、16 トークンまとめても 1.1 倍。最初の文字まで 8.3 秒 → 0.6 秒（[10章](StudyMemo/10_まとめ読み.md)）。
18. **★実測: 点数が同じでも、元のモデルと同じ答えを出す割合は下がりうる。原因を分ける最初の一手は「同じグラフを PC で動かして端末と比べる」。** 16 トークン版は点数は同じ（45 と 48）でも HF と同じ答えの割合が下がった（1 トークン目 97 → 78/126）。同じ量子化済みグラフを PC の CPU で動かすと HF と 13/16 一致し文字も正しく読めた（端末は 5/16）→ 原因はグラフではなく端末での実行。2 段目で「Python の入れ方で NPU に 1 回ずつ通す」と端末の答えを 7/7 再現 → C++ は正しく、NPU の計算が CPU と違う。ただし 3 段目で 1 トークン版の基準を取ると、層 0〜14 のずれは 1 トークン版にも同じだけあり、16 トークン版だけ悪いのはプロンプトの最後の回の logits だった。**「〜だけがずれる」と言う前に基準を取る**。4 段目で原因を特定: **変換器がまとめた RmsNorm** と、**[16×256] の tanh の不具合**。5 段目で**同じ計算を別の演算で書いて部品を避ける**と小さいグラフでは直った。6 段目で全体を直すと、まとめ読みのずれは消えたが 126 問の点は下がり（45 → 37）、**部品の誤差が減ることと答えが良くなることは別**と分かった。また量子化したグラフは丸め 1 目盛りの違いで logits が 1〜2 割変わる（CPU どうしでも）ので、**差の割合ではなく答えで比べる**。7 段目で、**基準にしていた PC の CPU の方が画像の行で誤っていた**と分かった（正規化の途中の目盛りは文章だけで決めたもので、画像の行ではみ出して切れる。NPU はまとめた RmsNorm で正確）。4〜5 段目の「NPU の不具合」は読み違いで、元の 16 トークン版をそのまま使うのが正解。**基準にする側が正しいかを、どちらの側にも当てられる物差し（出力の二乗平均が 1 か、など）で先に確かめる**（[10章](StudyMemo/10_まとめ読み.md)）。
19. **★実測: system 文 1 行で 45 → 54/123。** 「添付の画像を見て」と念押しするだけで、「判断できません」と逃げる答えが減った。プロンプトもモデルの一部として同じ 126 問で比べる（[10章](StudyMemo/10_まとめ読み.md)）。
20. **★実測: いまのスマホの VLM は、写真の読み取り 0.3 秒・最初の文字 0.9 秒。時間の 85% は答えを 1 つずつ作る段。** 次に速くするなら答えを短くするか、1 回で数トークン当てにいく方法（[11章](StudyMemo/11_写真から答えまで.md)）。

## ソースとの対応（節ごと・自動で作る）

各学習の章の節の終わりに置いたソースの枠の一覧（2026-10-17〜）。`<!-- srcindex -->` の中は `& $PY2 -B -m app.study.sync_src` が書き直す。

<!-- srcindex -->
| 章 | 節 | 載せているソース |
|---|---|---|
| [1. VLM の構造 — 画像が「言葉と同じ形」になるまで](StudyMemo/01_VLMの構造.md) | 1.1 LLM は「ベクトルの列」しか見ていない | `app/pipeline/hf_vlm.py` L298〜L320（PC: 端末と同じ入れ方を HF の部品で作る）<br>`android_app/app/src/main/cpp/runner/job.cpp` L1530〜L1536（端末の C++（1 トークンずつ読む経路）: 画像の位置は表を引かない） |
|  | 1.2 部品は 3 つ | `app/pipeline/vlm_common.py` L46〜L75（部品の大きさ（config.json から））<br>`app/pipeline/vit_graph.py` L255〜L260（VitGraph.__init__: 画像エンコーダ 16 層と射影層） |
|  | 1.3 「射影層を足すだけ」は、いつ成り立つか | `app/pipeline/check_wiring.py` L167〜L180（run_one: 3 つの経路を同じ入力で流して比べる） |
|  | 1.4 画像トークンの数が、そのまま「長さ」になる | `app/pipeline/vlm_common.py` L56〜L57（1 枚あたりの画像トークン数の候補）<br>`app/pipeline/hf_vlm.py` L201〜L231（プロンプトを作り、画像トークンの数を確かめる） |
| [2. Gemma 4 E2B の画像の通り道 — 336×480 の写真が 70 個のトークンになるまで](StudyMemo/02_Gemma4の画像の通り道.md) | 2.1 全体の流れ（形の変化） | `app/pipeline/hf_vlm.py` L189〜L193（pixels_fixed_np: 手順 2・3（÷255 とパッチへの並べ替え））<br>`app/pipeline/vit_graph.py` L274〜L294（VitGraph.run: 手順 4〜9） |
|  | 2.2 HF の既定（native）は形が変わる | `app/pipeline/hf_vlm.py` L130〜L154（prepare_image の native: 大きさは HF に任せる）<br>`app/pipeline/hf_vlm.py` L240〜L242（パディングを除いた実物のパッチ数） |
|  | 2.3 ViT の中身で押さえておくこと | `app/pipeline/vit_graph.py` L202〜L223（VitLayer.forward: 1 層の中身（HF の Gemma4VisionEncoderLayer と同じ順））<br>`app/pipeline/vit_graph.py` L107〜L114（QatFile.linear: QAT の Linear（INT8・出力チャネルごと・活性の目盛り）） |
|  | 2.4 プーリングと √768 | `app/pipeline/vit_graph.py` L263〜L268（VitGraph.pool: HF の平均の取り方を reshape と平均に書き直したもの）<br>`app/pipeline/vit_graph.py` L256〜L261（×√768 を射影層の RMSNorm の eps に吸収する） |
|  | 2.5 文章側への入れ方（ここが一番大事） | `android_app/app/src/main/cpp/runner/job.cpp` L1242〜L1253（端末の C++: 画像の位置の入れ方（手順 1〜3 の端末版））<br>`app/pipeline/gemma4_graph.py` L402〜L404（DecodeChunk.forward: PLE の作り方（HF の project_per_layer_inputs を書き直したもの）） |
| [3. NPU に載せるための設計 — 形・時間・メモリ](StudyMemo/03_NPUに載せるための設計.md) | 3.1 形を固定する — レターボックス | `app/pipeline/hf_vlm.py` L158〜L164（prepare_image の fixed: レターボックス）<br>`app/pipeline/vlm_common.py` L61〜L69（格子の表（横長と、後から足した縦長））<br>`android_app/app/src/main/cpp/runner/vlm_prep.cpp` L195〜L214（vlm_pixels: 端末の C++ でも同じレターボックス） |
|  | 3.2 画像トークン数と待ち時間 | `android_app/app/src/main/cpp/runner/job.cpp` L1522〜L1529（1 トークン版: プロンプトも 1 回に 1 トークンずつ）<br>`android_app/app/src/main/cpp/runner/job.cpp` L1481〜L1493（16 トークン版（Phase 5）: プロンプトを Tm 個ずつまとめて読む） |
|  | 3.3 メモリとプロセスの分け方 | `android_app/app/src/main/AndroidManifest.xml` L195〜L198（画像エンコーダを動かすサービスは別プロセス（:probe））<br>`android_app/app/src/main/java/org/edgeslm/demo/VlmActivity.kt` L583〜L606（runVit: 画像エンコーダの job を別プロセスに渡す） |
|  | 3.4 v68 の制約が ViT に与える影響 | `app/pipeline/quantize_vit.py` L116〜L128（量子化の後: C-8 の書き換えと C-1・C-5・C-14 の検査） |
|  | 3.5 まとめ: 端末での形（Phase 3 で作るもの） | なし（章のまとめの図（Phase 3 の時点）。それぞれのソースは 3.1〜3.3 節と 1.1・2.5 節の枠、いまの形は 11 章） |
| [4. 量子化の目盛りと画像の特徴 — 文章で較正したグラフに画像を入れる](StudyMemo/04_量子化の目盛りと画像の特徴.md) | 4.1 端末のグラフは「較正したデータの範囲」しか表せない | `app/pipeline/onnx_scales.py` L50〜L55（画像の特徴が通る、端末の文章グラフの入口の量子化の点） |
|  | 4.2 手を動かす: 入口の範囲を ONNX から読む | `app/pipeline/onnx_scales.py` L161〜L168（read_quant_points の終わり: scale・zero point から表せる範囲を計算する） |
|  | 4.3 画像の特徴は収まるか（★Phase 1 で実測する） | `app/pipeline/check_wiring.py` L182〜L202（run_one: 画像の位置の値域と、範囲の外に出る割合）<br>`app/pipeline/check_wiring.py` L145〜L148（範囲の外に出る割合）<br>`android_app/app/src/main/cpp/runner/job.cpp` L1597〜L1610（端末の C++ が PLE の表（4bit）を戻す所） |
|  | 4.4 PC で再現できる丸め、できない丸め | `app/pipeline/vit_graph.py` L75〜L80（SRQ: QAT の 8bit の丸め（transformers の apply_srq と同じ式を写したもの））<br>`app/pipeline/check_wiring.py` L119〜L128（入口の範囲で切り詰める（inputs_embeds と ple_tok））<br>`app/pipeline/check_wiring.py` L104〜L112（entry_clip: PLE の枝の中で切り詰める） |
|  | 4.5 収まらなかったら（逃げ道） | なし（対処の候補を並べた表で、どれも実行していない（4.3 節の追記のとおり、入口の較正のやり直しは不要と確定したため）） |
| [5. 評価の組み立て — 公開データ・自動判定・合成画像](StudyMemo/05_評価の組み立て.md) | 5.1 4 つの用途を、どのデータで測るか | `app/pipeline/eval_defs.py` L52〜L62（出どころ・件数・ライセンスの定義）<br>`app/pipeline/fetch_eval_data.py` L98〜L102（JA-VG-VQA-500: 先頭から 40 画像、各画像の最初の 1 問）<br>`app/pipeline/fetch_eval_data.py` L125〜L128（In-the-Wild: 行の並びが定義を作ったときと同じかを照合する） |
|  | 5.2 自動判定の考え方 | `app/pipeline/judge.py` L56〜L72（_hit・judge_keywords: グループごとに「どれか 1 つが含まれるか」）<br>`app/pipeline/eval_defs.py` L194〜L197（上の例の実際の定義（car_02）） |
|  | 5.3 ★自動判定の罠（コードレビューで実際に見つかったもの、2026-09-27） | `app/pipeline/judge.py` L115〜L154（answer_to_keywords: 正解の文から判定の語を作る（罠 1・3））<br>`app/pipeline/judge.py` L48〜L53（normalize: 表記をそろえる（罠 2））<br>`app/pipeline/eval_defs.py` L97〜L104（WILD_KEYWORDS の一部（罠 3・4）） |
|  | 5.4 合成画像で「解像度の効き目」を切り分ける | `app/pipeline/eval_defs.py` L282〜L293（SYNTH_SPECS の始め: 同じ文面を大きい字と小さい字で）<br>`app/pipeline/synth_images.py` L141〜L148（build: 一度作った画像は作り直さない） |
|  | 5.5 ★実測: 直した後にも残っていた判定の誤り（2026-09-28） | `app/pipeline/judge.py` L179〜L189（自己テストに足した例（python -m app.pipeline.judge）） |
| [6. 検算の考え方 — 「文章側は無改造で使える」をどう確かめるか](StudyMemo/06_検算の考え方.md) | 6.1 確かめたいこと | `app/pipeline/hf_vlm.py` L298〜L320（2 章の結論（端末の入れ方）を HF の部品で書いた所。これを検算する） |
|  | 6.2 3 つの経路 | `app/pipeline/check_wiring.py` L151〜L180（同じ画像・質問を A・B・C の 3 通りに流す）<br>`app/pipeline/check_wiring.py` L67〜L76（比べ方（位置ごとの相対 L2 と 1 位の一致）） |
|  | 6.3 SRQ を外して判定する理由 | `app/pipeline/hf_vlm.py` L108〜L124（SRQ を外す（活性のスケールを 0 に））<br>`app/pipeline/check_wiring.py` L248〜L256（main: SRQ あり→なしの順に同じ問題を流す） |
|  | 6.4 値域と切り詰めの実験 | `app/pipeline/check_wiring.py` L119〜L128（入口の範囲で切り詰める）<br>`app/pipeline/check_wiring.py` L205〜L214（run_one の中: 切り詰めの影響（最後の位置の logits と答え）） |
|  | 6.5 ★実測（2026-09-28、[P1 の記録](../../results/P1_20260928_pc_eval.md)） | `app/pipeline/check_wiring.py` L33〜L35（docstring に残した失敗の記録）<br>`app/pipeline/hf_vlm.py` L400〜L412（ref_vlm --feed step（端末と同じ 1 トークンずつの流し方））<br>`app/pipeline/ref_vlm.py` L342〜L347（report: 流し方を変えた結果を、基準（HF の流し方）と 1 問ずつ並べる） |
| [7. 画像エンコーダを NPU 向けに書き直す — 同じ計算を、別の形で](StudyMemo/07_画像エンコーダを書き直す.md) | 7.1 なぜ HF のコードをそのまま書き出さないのか | `app/pipeline/vit_graph.py` L29〜L39（docstring: HF との違いの一覧） |
|  | 7.2 書き換え 1: 位置を定数にする | `app/pipeline/vit_graph.py` L148〜L151（パッチ n の (x, y)）<br>`app/pipeline/vit_graph.py` L246〜L254（VitGraph.__init__: 位置の表と cos/sin を定数にする） |
|  | 7.3 書き換え 2: プーリングを reshape と平均で書く | `app/pipeline/vit_graph.py` L263〜L268（VitGraph.pool） |
|  | 7.4 書き換え 3: ×√768 を消す（RMSNorm の性質を使う） | `app/pipeline/vit_graph.py` L256〜L261（VitGraph.__init__: eps を 1/768 に）<br>`app/pipeline/vit_graph.py` L288〜L293（VitGraph.run の最後） |
|  | 7.5 書き換えなかったもの | `app/pipeline/vit_graph.py` L117〜L128（RMSNorm の形）<br>`app/pipeline/vit_graph.py` L274〜L279（VitGraph.run の最初） |
|  | 7.6 確かめ方（6 章と同じ考え方） | `app/pipeline/export_vit.py` L118〜L133（check: 自前 ↔ HF（SRQ なし）、出力と途中の値） |
|  | 7.7 ★実測（2026-09-28、[記録](../../results/P2_20260928_vit_export.md)） | `app/pipeline/export_vit.py` L57〜L62（書き出した ONNX の演算の見張り）<br>`app/pipeline/export_vit.py` L134〜L138（check: ×√768 を消した形 ↔ 消さない形） |
|  | 7.8 ★SRQ（8bit の丸め）への敏感さ — 画像エンコーダは文章側よりずっと揺れる | `app/pipeline/vit_graph.py` L75〜L80（SRQ の丸め（transformers の gemma_quant と同じ））<br>`app/pipeline/export_vit.py` L140〜L157（check: SRQ ありは記録だけ）<br>`app/pipeline/hf_vlm.py` L415〜L423（ref_vlm --vision own: 自前の特徴で 126 問を解く） |
|  | 7.9 ★量子化してみる（2026-09-28、[記録](../../results/P2_20260928_vit_quant.md)）— 新しい誤差は「すでにある揺れ」と比べる | `app/pipeline/qat_inject_vit.py` L88〜L107（QAT の 112 本はそのまま、QAT でない 2 本は作り直す）<br>`app/pipeline/qat_inject_vit.py` L217〜L236（inject の最後: 書き込んだ値の検査） |
|  | 7.10 ★変換してみる（2026-09-29、[記録](../../results/P2_20260929_vit_ctx.md)）— 「1 回に何トークン通すか」で NPU の苦労が変わる | `app/pipeline/build_vit_ctx.py` L111〜L123（build_one: QAIRT で context binary に） |
| [8. 端末でだけ壊れる — 小さすぎるスケールと、途中の値での切り分け（2026-09-29）](StudyMemo/08_端末でだけ壊れる.md) | 8.1 何が見えたか | `app/pipeline/device_vit.py` L455〜L461（strict＝CPU に逃げずに開けたか）<br>`app/pipeline/device_vit.py` L516〜L525（cmd_acc: 端末の出力を PC の QDQ・HF と比べる） |
|  | 8.2 切り分けの考え方: 途中の値をグラフの出力に足す | `app/pipeline/probe_vit.py` L82〜L109（途中の値もグラフの出力にする探り用のグラフ）<br>`app/pipeline/probe_vit.py` L308〜L309（cmd_device: 最初に大きくずれる所） |
|  | 8.3 壊れ方を「形」で読む | なし（張り付きの割合・チャネルごとの集計・スケールの境目の比べは記録（results/P2_20260929_vit_device.md 7.2 節）だけで、集計のスクリプトはリポジトリに残っていない。見つけた境目は 8.4 節の枠（qat_inject_vit.py の注記）に残っている） |
|  | 8.4 なぜスケールが小さいと壊れるのか | `app/pipeline/qat_inject_vit.py` L62〜L85（C-14 の下限と _from_float） |
|  | 8.5 直し方と、見張り | `app/pipeline/qat_inject_vit.py` L261〜L282（C-14 の見張り）<br>`app/pipeline/quantize_vit.py` L127〜L130（量子化の段で毎回検査する） |
|  | 8.5.1 ★確かめた（2026-09-29） | `app/pipeline/device_vit.py` L86〜L86（目安）<br>`app/pipeline/device_vit.py` L532〜L539（cmd_acc: Q10 の判定） |
|  | 8.6 ここから言えること | なし（章のまとめ（根拠は 8.1〜8.5.1 節）） |
| [9. 端末でつなぐ — 大きいモデルを開くときに OS に止められる、端末だけの品質の見方（2026-09-29）](StudyMemo/09_端末でつなぐ.md) | 9.1 何が見えたか | `app/pipeline/device_vlm.py` L210〜L215（cmd_eval: 1 回目・2 回目の記録から入れた手当て） |
|  | 9.2 Android はどうやってアプリを止めるか | `app/pipeline/device_vlm.py` L384〜L391（cmd_holdcheck: 読み込み中の優先度を 1 秒ごとに読む） |
|  | 9.3 なぜ「開くとき」だけなのか | `android_app/app/src/main/java/org/edgeslm/demo/RunnerHoldActivity.kt` L23〜L30（なぜ要るか（クラスの説明））<br>`app/pipeline/device_vit.py` L174〜L186（開く前に空きメモリを待つ） |
|  | 9.4 対策: 開く間は画面を前に出す | `android_app/app/src/main/java/org/edgeslm/demo/RunnerHoldActivity.kt` L66〜L74（画面を消さない・ロックの上に出す）<br>`android_app/app/src/main/java/org/edgeslm/demo/RunnerHoldActivity.kt` L100〜L110（RunnerService を起こし、結果が出るまで前に居続ける） |
|  | 9.5 見張り方の教訓 | `app/pipeline/device_vit.py` L225〜L250（run_job: プロセスが消えたらすぐ止める）<br>`app/pipeline/device_vit.py` L274〜L287（消えたらすぐ logcat を 2 通り残す）<br>`app/pipeline/device_vlm.py` L235〜L242（cmd_eval: 落ちたら待ってやり直す） |
|  | 9.6 端末だけで解いた品質の見方 | `app/pipeline/device_vlm.py` L195〜L196（cmd_eval: 結果を PC の結果と同じ置き場・同じ形で書く）<br>`app/pipeline/device_vlm.py` L278〜L280（cmd_eval: 点と、画像の特徴を入れた数の確かめ） |
|  | 9.7 ここから言えること | なし（章のまとめ（根拠は 9.1〜9.6 節）） |
| [10. まとめ読み — 1 回で 16 トークンを読むと、なぜ 8 秒が 0.6 秒になるのか](StudyMemo/10_まとめ読み.md) | 10.1 LLM の推論は 2 段階ある — prefill と decode | `android_app/app/src/main/cpp/runner/job.cpp` L1523〜L1529（1 トークン版の経路: プロンプトも答えも同じ「1 回 1 トークン」の繰り返し）<br>`android_app/app/src/main/cpp/runner/job.cpp` L1743〜L1750（同じ繰り返しの中で prefill と decode を分ける所） |
|  | 10.2 16 トークン入れても、1 回の時間はほとんど増えない — NPU の時間は「重みを読む時間」 | `app/pipeline/build_t16fix.py` L239〜L246（変換のログから DDR の見積もりを読む（Phase 5 の書き直し版の変換で使った）） |
|  | 10.3 なぜ「16 トークン版の 1 本」で読むのも答えを作るのもやるのか | `android_app/app/src/main/cpp/runner/job.cpp` L1502〜L1513（16 トークン版のまま答えを作る（本物 1 行＋空 15 行）） |
|  | 10.4 グラフの約束 — 窓・マスク・空の行・行の選択 | `app/pipeline/g4t_graph.py` L73〜L86（layer_forward_t: 窓 = cat(past 240, 新 16)）<br>`android_app/app/src/main/cpp/runner/job.cpp` L1413〜L1428（run_chunk: 行ごとのマスクと logit_sel）<br>`app/pipeline/g4t_graph.py` L158〜L174（DecodeChunkT.forward: 出力は新しい行だけ（層 13・14 は窓全体）、最後に行を選ぶ） |
|  | 10.5 端末の C++ がやること — 16 個ずつ読み、1 個ずつ作る | `android_app/app/src/main/cpp/runner/job.cpp` L1481〜L1501（プロンプトを 16 個ずつ。最後の回の答えが 1 つ目）<br>`android_app/app/src/main/cpp/runner/job.cpp` L1435〜L1452（run_chunk: 回したあとの KV の更新） |
|  | 10.6 確かめ方 — 3 段で「配線」と「丸め」と「端末」を分ける | `app/pipeline/build_t16.py` L286〜L294（① 浮動小数どうしで比べる（docstring））<br>`app/pipeline/check_t16.py` L161〜L172（② 量子化済みを PC の CPU で比べる） |
|  | 10.7 ★実測: 速さは 14 倍、品質は「点数は同じ、HF との一致は少し下がる」 | `app/pipeline/diag_t16.py` L13〜L18（1 段目の問いの立て方（docstring））<br>`app/pipeline/probe_t16.py` L349〜L393（2 段目: 同じ入力の NPU と CPU を比べる）<br>`app/pipeline/probe_t16.py` L89〜L90（3 段目の基準（1 トークン版））<br>`app/pipeline/probe_l0.py` L76〜L77（4 段目: plain と taps）<br>`app/pipeline/probe_rows.py` L19〜L24（4 段目: 同じトークンを 2 通りで通す（docstring））<br>`app/pipeline/probe_fix.py` L183〜L206（5 段目: 平均を 1/N の行列の掛け算に）<br>`app/pipeline/probe_fix.py` L236〜L261（5 段目: tanh を 1 行の形で）<br>`app/pipeline/diag_t16fix.py` L172〜L201（6 段目 (2): 同じグラフを設定違いで動かして物差しを作る）<br>`app/pipeline/probe_rows.py` L118〜L154（7 段目 (5): NPU どうし・CPU どうしで比べる） |
|  | 10.8 ★実測: system 文を 1 行変えると、グラフの最適化より効いた | `app/pipeline/device_vlm.py` L450〜L451（評価の system 文を差し替える）<br>`app/pipeline/setup_vlm_demo.py` L156〜L161（デモの設定 jobs/vlm_demo.json） |
|  | 10.9 画像エンコーダを開いたままにする — 1.1 秒 → 0.33 秒 | `android_app/app/src/main/java/org/edgeslm/demo/RunnerService.kt` L159〜L185（開いたセッションを使い回す）<br>`android_app/app/src/main/java/org/edgeslm/demo/VlmActivity.kt` L555〜L577（文章側を開き終えたら「開くだけ」の job を出す） |
|  | 10.10 まとめ | なし（章のまとめ（根拠は 10.1〜10.9 節）） |
| [11. 写真から答えまでを端から端まで — いまの端末の VLM を 1 本の流れで追う](StudyMemo/11_写真から答えまで.md) | 11.1 登場人物 — 2 つのプロセスと 2 つのモデル | `android_app/app/src/main/java/org/edgeslm/demo/VlmActivity.kt` L395〜L410（openTextModel: 文章側を開いてから画像エンコーダを先に開く）<br>`android_app/app/src/main/AndroidManifest.xml` L195〜L197（画像エンコーダは別プロセス :probe） |
|  | 11.2 時間の流れ（実測、1 枚目の写真） | `android_app/app/src/main/java/org/edgeslm/demo/VlmActivity.kt` L527〜L530（readPhoto: 「vlm image:」の行）<br>`android_app/app/src/main/java/org/edgeslm/demo/VlmActivity.kt` L806〜L811（finish: 「vlm stats:」の行） |
|  | 11.3 形の変化 — 写真が 107 個のベクトルになり、1 つの語になるまで | `android_app/app/src/main/java/org/edgeslm/demo/VlmActivity.kt` L631〜L637（loadBitmap: 2 の累乗で間引いて復号）<br>`android_app/app/src/main/cpp/runner/vlm_prep.cpp` L189〜L231（縦横比を保って収め、黒に貼り、パッチに並べる）<br>`android_app/app/src/main/cpp/runner/job.cpp` L1337〜L1345（画像の位置は特徴の 1 行をそのまま） |
|  | 11.4 プロンプトの中身（実際の ID） | `android_app/app/src/main/java/org/edgeslm/demo/VlmActivity.kt` L675〜L681（プロンプトの ID を作る）<br>`android_app/app/src/main/java/org/edgeslm/demo/VlmActivity.kt` L837〜L844（特殊トークンと区切り） |
|  | 11.5 KV キャッシュの一生 | `android_app/app/src/main/java/org/edgeslm/demo/VlmActivity.kt` L814〜L816（finish: 窓からあふれたら写真から作り直す） |
|  | 11.6 答えを止める 3 つの条件 | `android_app/app/src/main/java/org/edgeslm/demo/VlmActivity.kt` L790〜L795（finish: 止まり方ごとの表示） |
|  | 11.7 いま時間を使っているのはどこか | なし（時間の内訳は 11.2 節のログ（vlm stats の image_s・first_s・total_s）から計算したもの。ログを出す所は 11.2 節の枠） |
|  | 11.8 まとめ | なし（章のまとめ（根拠は 11.1〜11.7 節）） |
<!-- /srcindex -->
