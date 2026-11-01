// EdgeSLM Runner - VLM の画像の前処理（説明は vlm_prep.h）
//
// （2026-10-18 追記）中身の 3 段: pil_bicubic_resize_rgb（Pillow と同じ縮め方）→ vlm_pixels（黒に貼ってパッチに並べる）
// → run_vlm_prep（job の JSON で呼ぶ入口。ファイルを読み、作った pixel_values をファイルに書く）。
// Pillow の手順を写したのは、PC の評価と同じ値（ビット単位で一致）にするため（EdgeSLM_AndroidVLM/results/P3_20260929_device_vlm.md 2 節）
#include "runner/vlm_prep.h"

#include <android/log.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>

#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, "EdgeSLMRunner", __VA_ARGS__)

namespace edgeslm {

using nlohmann::json;

namespace {

// 今の時刻（ms）。時間を測る用途だけに使う（steady_clock）
double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

// ---- Pillow の Resample.c と同じ定数・関数 ----
constexpr int PRECISION_BITS = 32 - 8 - 2;          // 係数の固定小数点の桁（22bit）

double bicubic_filter(double x) {                    // a = −0.5 の 3 次畳み込み（Pillow と同じ式）
    const double a = -0.5;
    if (x < 0.0) x = -x;
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1;
    if (x < 2.0) return (((x - 5) * x + 8) * x - 4) * a;
    return 0.0;
}
constexpr double BICUBIC_SUPPORT = 2.0;

// 出力の 1 画素ごとに「どの入力の画素を、どの重みで足すか」を作る（Pillow の precompute_coeffs と
// normalize_coeffs_8bpc を合わせたもの）。bounds は (最初の入力画素, 個数)、kk は 22bit の整数の重み
int precompute_coeffs(int in_size, double in0, double in1, int out_size, std::vector<int>* bounds,
                      std::vector<int32_t>* kk) {
    double scale = (in1 - in0) / out_size;
    double filterscale = scale < 1.0 ? 1.0 : scale;  // 縮めるときは、縮める倍率だけ窓を広げる（ぼかして折り返しを防ぐ）
    const double support = BICUBIC_SUPPORT * filterscale;
    const int ksize = static_cast<int>(std::ceil(support)) * 2 + 1;
    bounds->assign(static_cast<size_t>(out_size) * 2, 0);
    std::vector<double> k(static_cast<size_t>(out_size) * ksize, 0.0);
    for (int xx = 0; xx < out_size; ++xx) {
        const double center = in0 + (xx + 0.5) * scale;
        const double ss = 1.0 / filterscale;
        int xmin = static_cast<int>(center - support + 0.5);   // C のキャスト（0 に向かって切り捨て）＝ Pillow と同じ
        if (xmin < 0) xmin = 0;
        int xmax = static_cast<int>(center + support + 0.5);
        if (xmax > in_size) xmax = in_size;
        xmax -= xmin;
        double ww = 0.0;
        double* kr = &k[static_cast<size_t>(xx) * ksize];
        for (int x = 0; x < xmax; ++x) {
            const double w = bicubic_filter((x + xmin - center + 0.5) * ss);
            kr[x] = w;
            ww += w;
        }
        for (int x = 0; x < xmax; ++x) {
            if (ww != 0.0) kr[x] /= ww;
        }
        (*bounds)[static_cast<size_t>(xx) * 2 + 0] = xmin;
        (*bounds)[static_cast<size_t>(xx) * 2 + 1] = xmax;
    }
    kk->assign(k.size(), 0);
    for (size_t i = 0; i < k.size(); ++i) {           // normalize_coeffs_8bpc（四捨五入して 22bit の整数に）
        (*kk)[i] = k[i] < 0 ? static_cast<int32_t>(-0.5 + k[i] * (1 << PRECISION_BITS))
                            : static_cast<int32_t>(0.5 + k[i] * (1 << PRECISION_BITS));
    }
    return ksize;
}

inline uint8_t clip8(int32_t v) {                      // Pillow の clip8: 22bit 右シフト（負は床）して 0..255 に
    const int32_t s = v >> PRECISION_BITS;
    return static_cast<uint8_t>(s < 0 ? 0 : (s > 255 ? 255 : s));
}

// ファイルを丸ごとバイト列で読む（ate で開いて末尾の位置＝大きさを先に知る）。開けなければ false と理由
bool read_file(const std::string& path, std::vector<uint8_t>* out, std::string* err) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { *err = "cannot open: " + path; return false; }
    const std::streamsize n = f.tellg();
    f.seekg(0);
    out->resize(static_cast<size_t>(n));
    if (n > 0 && !f.read(reinterpret_cast<char*>(out->data()), n)) { *err = "read failed: " + path; return false; }
    return true;
}

// n バイトをファイルに書く（上書き）。書けなければ false と理由
bool write_file(const std::string& path, const void* p, size_t n, std::string* err) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) { *err = "cannot write: " + path; return false; }
    f.write(reinterpret_cast<const char*>(p), static_cast<std::streamsize>(n));
    if (!f) { *err = "write failed: " + path; return false; }
    return true;
}

// job の相対パスを runtime.base_dir から解決する（絶対パス・空・base が空ならそのまま）
std::string resolve_path(const std::string& base, const std::string& p) {
    if (p.empty() || p[0] == '/' || base.empty()) return p;
    return base + "/" + p;
}

}  // namespace

// (sw, sh) → (dw, dh) に縮める（宣言の説明は vlm_prep.h）。
// ① 横・縦それぞれの「どの画素を、どの重みで足すか」を作る → ② 横の計算（使う行だけ）→ ③ 縦の計算。
// 足し算は 22bit の固定小数点の整数で行い、各段の終わりで 8bit に丸める（Pillow と同じ。途中を float にすると丸めが変わり、ビット単位では一致しなくなると推定）
void pil_bicubic_resize_rgb(const uint8_t* src, int sw, int sh, uint8_t* dst, int dw, int dh) {
    // Pillow の ImagingResampleInner と同じ: 横 → 縦の 2 回。大きさが変わらない向きは飛ばす
    // ① 係数の表（bh・kh は横、bv・kv は縦）
    std::vector<int> bh, bv;
    std::vector<int32_t> kh, kv;
    const int ksh = precompute_coeffs(sw, 0.0, sw, dw, &bh, &kh);
    const int ksv = precompute_coeffs(sh, 0.0, sh, dh, &bv, &kv);
    const bool need_h = dw != sw;
    const bool need_v = dh != sh;
    // 縦の計算に使う入力の行の範囲（横の計算はこの範囲だけ行う。Pillow の ybox_first / ybox_last）
    const int y_first = bv[0];
    const int y_last = bv[static_cast<size_t>(dh) * 2 - 2] + bv[static_cast<size_t>(dh) * 2 - 1];
    std::vector<uint8_t> tmp;
    const uint8_t* vin = src;
    int vin_w = sw;
    int y_off = 0;                                      // 縦の計算で、入力の行番号から引く量
    // ② 横の計算。結果は tmp（dw × rows の画像）に置く。s0〜s2 の初期値 1 << 21 は四捨五入のための 0.5
    if (need_h) {
        const int rows = y_last - y_first;
        tmp.resize(static_cast<size_t>(dw) * rows * 3);
        for (int yy = 0; yy < rows; ++yy) {
            const uint8_t* row = src + static_cast<size_t>(yy + y_first) * sw * 3;
            uint8_t* orow = tmp.data() + static_cast<size_t>(yy) * dw * 3;
            for (int xx = 0; xx < dw; ++xx) {
                const int xmin = bh[static_cast<size_t>(xx) * 2], xmax = bh[static_cast<size_t>(xx) * 2 + 1];
                const int32_t* k = &kh[static_cast<size_t>(xx) * ksh];
                int32_t s0 = 1 << (PRECISION_BITS - 1), s1 = s0, s2 = s0;
                for (int x = 0; x < xmax; ++x) {
                    const uint8_t* p = row + static_cast<size_t>(x + xmin) * 3;
                    s0 += p[0] * k[x];
                    s1 += p[1] * k[x];
                    s2 += p[2] * k[x];
                }
                orow[xx * 3 + 0] = clip8(s0);
                orow[xx * 3 + 1] = clip8(s1);
                orow[xx * 3 + 2] = clip8(s2);
            }
        }
        vin = tmp.data();
        vin_w = dw;
        y_off = y_first;
    }
    // ③ 縦の計算。入力は横の結果（tmp）か元の画像。ymin は tmp の中の行番号に直してから使う
    if (need_v) {
        for (int yy = 0; yy < dh; ++yy) {
            const int ymin = bv[static_cast<size_t>(yy) * 2] - y_off, ymax = bv[static_cast<size_t>(yy) * 2 + 1];
            const int32_t* k = &kv[static_cast<size_t>(yy) * ksv];
            uint8_t* orow = dst + static_cast<size_t>(yy) * dw * 3;
            for (int xx = 0; xx < vin_w; ++xx) {
                int32_t s0 = 1 << (PRECISION_BITS - 1), s1 = s0, s2 = s0;
                for (int y = 0; y < ymax; ++y) {
                    const uint8_t* p = vin + (static_cast<size_t>(y + ymin) * vin_w + xx) * 3;
                    s0 += p[0] * k[y];
                    s1 += p[1] * k[y];
                    s2 += p[2] * k[y];
                }
                orow[xx * 3 + 0] = clip8(s0);
                orow[xx * 3 + 1] = clip8(s1);
                orow[xx * 3 + 2] = clip8(s2);
            }
        }
    } else {
        // 縦は変わらない: 横だけの結果（または元の画像）をそのまま写す
        // （横を計算したときも、縦が同じ大きさなら使う行は 0..dh−1 の全部＝ y_first は 0）
        for (int yy = 0; yy < dh; ++yy)
            std::copy(vin + static_cast<size_t>(yy) * vin_w * 3, vin + static_cast<size_t>(yy + 1) * vin_w * 3,
                      dst + static_cast<size_t>(yy) * dw * 3);
    }
}

// 写真 → pixel_values（宣言の説明は vlm_prep.h）。
// ① 縦横比を保って格子（W × H ピクセル）に収まる大きさを決めて縮める → ② 黒い画像の真ん中に貼る
// → ③ 16×16 のパッチに分けて ×1/255。info に大きさ・貼った位置・時間を入れる
bool vlm_pixels(const uint8_t* rgb, int w, int h, int gh, int gw, int patch, std::vector<float>* out,
                json* info, std::string* err) {
    if (w <= 0 || h <= 0 || gh <= 0 || gw <= 0 || patch <= 0) { *err = "bad size"; return false; }
    const double t0 = now_ms();
    // ① 格子の大きさ（例: 21×30 パッチ × 16 ＝ 336×480 ピクセル）と倍率 s（はみ出さない方に合わせる）
    const int H = gh * patch, W = gw * patch;
    const double s = std::min(static_cast<double>(W) / w, static_cast<double>(H) / h);
    // Python の round（偶数への丸め）。既定の丸めの向き（最も近い偶数）の nearbyint と同じ
    int nw = std::max(1, static_cast<int>(std::nearbyint(w * s)));
    int nh = std::max(1, static_cast<int>(std::nearbyint(h * s)));
    nw = std::min(nw, W);
    nh = std::min(nh, H);
    std::vector<uint8_t> small;
    const uint8_t* sp = rgb;
    if (nw != w || nh != h) {
        small.resize(static_cast<size_t>(nw) * nh * 3);
        pil_bicubic_resize_rgb(rgb, w, h, small.data(), nw, nh);
        sp = small.data();
    }
    const double t1 = now_ms();
    // 黒い画像に貼る
    std::vector<uint8_t> canvas(static_cast<size_t>(W) * H * 3, 0);
    const int ox = (W - nw) / 2, oy = (H - nh) / 2;
    for (int y = 0; y < nh; ++y)
        std::copy(sp + static_cast<size_t>(y) * nw * 3, sp + static_cast<size_t>(y + 1) * nw * 3,
                  canvas.data() + (static_cast<size_t>(y + oy) * W + ox) * 3);
    // パッチ化と ×1/255（numpy の float32 の掛け算と同じ: float(画素) × float(1/255)）
    const float rescale = static_cast<float>(0.00392156862745098);   // processor_config.json の rescale_factor
    const size_t pd = static_cast<size_t>(patch) * patch * 3;
    out->assign(static_cast<size_t>(gh) * gw * pd, 0.0f);
    for (int py = 0; py < gh; ++py)
        for (int px = 0; px < gw; ++px) {
            float* o = out->data() + (static_cast<size_t>(py) * gw + px) * pd;
            for (int yy = 0; yy < patch; ++yy)
                for (int xx = 0; xx < patch; ++xx) {
                    const uint8_t* c = canvas.data() + (static_cast<size_t>(py * patch + yy) * W + px * patch + xx) * 3;
                    float* d = o + (static_cast<size_t>(yy) * patch + xx) * 3;
                    d[0] = static_cast<float>(c[0]) * rescale;
                    d[1] = static_cast<float>(c[1]) * rescale;
                    d[2] = static_cast<float>(c[2]) * rescale;
                }
        }
    const double t2 = now_ms();
    *info = {{"in", {w, h}}, {"resized", {nw, nh}}, {"offset", {ox, oy}}, {"canvas", {W, H}},
             {"resize_ms", t1 - t0}, {"patch_ms", t2 - t1}};
    return true;
}

// job の JSON で呼ぶ入口（宣言の説明は vlm_prep.h）。1 枚ごとに: RGB のファイルを読む → 大きさを確かめる
// → vlm_pixels → pixel_values を float32 のままファイルに書く。1 枚が失敗しても次へ進み、records に理由を残す
json run_vlm_prep(const json& job) {
    json out;
    out["task"] = "vlm_prep";
    out["job_id"] = job.value("job_id", std::string());
    const std::string base = job.value("runtime", json::object()).value("base_dir", std::string());
    const json grid = job.value("grid", json::array({21, 30}));
    const int gh = grid.at(0).get<int>(), gw = grid.at(1).get<int>();
    const int patch = job.value("patch", 16);
    json items = job.value("items", json::array());
    // items が無く rgb などが直接書いてあれば、1 枚の items として扱う
    if (items.empty() && job.contains("rgb")) {
        items.push_back({{"rgb", job["rgb"]}, {"width", job["width"]}, {"height", job["height"]}, {"out", job["out"]}});
    }
    json recs = json::array();
    int n_ok = 0;
    for (const auto& it : items) {
        json rec;
        std::string err;
        std::vector<uint8_t> rgb;
        const int w = it.value("width", 0), h = it.value("height", 0);
        const std::string in_path = resolve_path(base, it.value("rgb", std::string()));
        const std::string out_path = resolve_path(base, it.value("out", std::string()));
        rec["out"] = it.value("out", std::string());
        if (!read_file(in_path, &rgb, &err)) { rec["error"] = err; recs.push_back(rec); continue; }
        // ファイルの大きさが 幅 × 高さ × 3 と合わなければ、幅・高さの書き違いなので止める（読み違えた画像を作らない）
        if (rgb.size() != static_cast<size_t>(w) * h * 3) {
            rec["error"] = "rgb size mismatch: " + std::to_string(rgb.size()) + " != " + std::to_string(w * h * 3);
            recs.push_back(rec);
            continue;
        }
        std::vector<float> px;
        json info;
        if (!vlm_pixels(rgb.data(), w, h, gh, gw, patch, &px, &info, &err) ||
            !write_file(out_path, px.data(), px.size() * sizeof(float), &err)) {
            rec["error"] = err;
            recs.push_back(rec);
            continue;
        }
        rec["info"] = info;
        recs.push_back(rec);
        ++n_ok;
    }
    LOGI("vlm_prep: %d / %zu", n_ok, items.size());
    out["records"] = recs;
    out["completed"] = n_ok;
    out["ok"] = n_ok > 0 && n_ok == static_cast<int>(items.size());
    return out;
}

}  // namespace edgeslm
