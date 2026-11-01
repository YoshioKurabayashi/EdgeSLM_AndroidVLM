// EdgeSLM Runner - VLM の画像の前処理（EdgeSLM_AndroidVLM Phase 3、2026-09-29）
//
// 写真（RGB のバイト列）→ 画像エンコーダの入力 pixel_values [N, 768] float32 を作る。
// PC の評価（hf_vlm.prepare_image の fixed ＋ hf_vlm.pixels_fixed_np）と**同じ値**になるように書いてある:
//
//   1. 縦横比を保って、固定の格子（例: 70 トークン → 21×30 パッチ = 336×480 ピクセル）に収まる大きさへ縮める。
//      倍率 s = min(W/w, H/h)、新しい大きさ = round(w·s), round(h·s)（Python の round と同じ「偶数への丸め」）
//      縮め方は **PIL の Image.resize(BICUBIC) を固定小数点の計算まで再現**（Pillow の Resample.c と同じ手順:
//      横 → 縦の 2 回、係数は 22bit の固定小数点、途中は 8bit に丸める）
//   2. 黒い画像の真ん中（左上 = ((W−w')//2, (H−h')//2)）に貼る（レターボックス）
//   3. 16×16 のパッチに分け、(縦のパッチ, 横のパッチ, パッチ内の y, パッチ内の x, 色) の順に並べ、×1/255（float32）
//
// EXIF の向きと透明部分（白で埋める）は、呼ぶ側（Kotlin の画面・PC の照合道具）で済ませてから渡す。
//
// （2026-10-18 追記）
// どこで使うか: 画面（VlmActivity）が写真から画像エンコーダの入力を作るとき（edgeslm_jni.cpp の nativeVlmPrep）と、
//   PC からの照合（RunnerService の task "vlm_prep"）。EdgeSLM_AndroidVLM/DoC/StudyMemo の 3 章（3.1 レターボックス）・11 章。
// 結果: PC の pixel_values と 108/108 枚がビット単位で一致、縮小は中央値 5.9ms（EdgeSLM_AndroidVLM/results/P3_20260929_device_vlm.md 2 節）。
//   「近い」ではなく「同じ」を目安にしたのは、前処理がずれると答えが変わったときに原因が前処理か NPU か分からなくなるため。
// 入力: RGB のバイト列（1 画素 3 バイト・行の詰め物なし）と幅・高さ。出力: pixel_values（float32、[gh·gw, patch·patch·3]）。
// 動かし方（照合）: EdgeSLM_AndroidVLM で & $PY2 -m app.pipeline.device_vlm prep（同 DoC/StudyMemo/12_ソースと動かし方.md の 12.3.2）
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "third_party/json.hpp"

namespace edgeslm {

// PIL の BICUBIC（8bit・RGB）で (sw, sh) → (dw, dh) に大きさを変える。src/dst は 1 ピクセル 3 バイト、行の詰め物なし
void pil_bicubic_resize_rgb(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh);

// 写真 → pixel_values。gh・gw はパッチの数（縦・横）、patch は 16。out は gh·gw × patch·patch·3 個の float
// info に縮めた大きさ・貼った位置・時間を入れる
// 戻り値: 成功で true。大きさが 0 以下なら false と理由（err）
bool vlm_pixels(const uint8_t* rgb, int w, int h, int gh, int gw, int patch, std::vector<float>* out,
                nlohmann::json* info, std::string* err);

// JSON の job で呼ぶ入口（画面の JNI と、PC からの照合用 task "vlm_prep" の両方）。
//   {"items": [{"rgb": "inputs/x.rgb", "width": W, "height": H, "out": "inputs/x_pix.bin"}, ...],
//    "grid": [gh, gw], "patch": 16, "runtime": {"base_dir": ...}}
//   （items の代わりに rgb / width / height / out を直接書いてもよい＝1 枚）
// 戻り値: {task:"vlm_prep", job_id, records[1 枚ごとに out と info（または error）], completed（できた枚数）,
//          ok（全部できたら true）}。grid の既定は [21, 30]（70 トークンの格子）
nlohmann::json run_vlm_prep(const nlohmann::json& job);

}  // namespace edgeslm
