// EdgeSLM Demo - 生成の途中経過を UI へ渡すための共有状態
//
// なぜ必要か:
//   デモでは「トークンが1つずつ現れる」ことが見せ場になる。ところが JNI の
//   nativeRunJob は生成が終わるまで戻らない（同期呼び出し）ので、その間の
//   途中経過を UI 側から読む経路が別に要る。
//
// なぜコールバックではなくポーリングか（基本設計③ 6.2節の方針の延長）:
//   C++ のワーカースレッドから Kotlin を呼ぶには AttachCurrentThread が要り、
//   デタッチ漏れ・例外・ライフサイクルの事故が増える。ここは
//   「C++ 側が状態を置き、Kotlin が 50ms 間隔で JSON を読む」に倒す。
//   JNI 境界を JSON 文字列に固定する既存方針とも揃う。
//
// なぜプロセス唯一（シングルトン）か:
//   デモアプリで走る生成は常に1本であり、複数同時実行の要件がない。
//   run_job() のシグネチャを変えずに済み、コピー元 EdgeSLM_Android/android_runner
//   との差分を小さく保てる。
//
// （2026-10-18 追記）どこで使うか:
//   書く側: job.cpp の generate（窓ずらし）と generate_kv（KV キャッシュ）が、生成の前に begin、1 語ごとに push、
//           終わりに finish を呼ぶ（同じ写しを使うほかのプロジェクトの task にも、これを呼ぶものがある）。
//   読む側: edgeslm_jni.cpp の nativeProgress → snapshot()。画面（ChatActivity など）が 50ms ごとに読み、
//           rev が変わったときだけ中身を使う（EdgeSLM_AndroidDemo/DoC/StudyMemo/04_デモの画面を作る.md 4.2）。
// 入力と出力: push する「語の ID・実時間・OrtRun の時間」→ snapshot の JSON
//   {running, done, rev, max_new, window, prompt_len, ids, n, error?, last_ms?, median_ms?, median_ort_ms?, tokens_per_sec?}
// 動かし方: 単独では動かない。画面で生成を始めると、この状態が動く

#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

#include "third_party/json.hpp"

namespace edgeslm {

// 生成の途中経過を 1 か所に置く入れ物。プロセスに 1 つ（instance()）。
// すべての読み書きは m_ でロックする（書くのは生成のスレッド、読むのは画面のスレッド）
class GenProgress {
public:
    // プロセスに 1 つの GenProgress を返す（最初に呼ばれたときに作る）
    static GenProgress& instance();

    // 生成の開始を記録する（以前の結果は破棄される）
    //   max_new   : 生成する語数の上限
    //   window    : 長さの目安。generate は窓の長さ W、generate_kv は attn_mask の最後の次元 L（KV のスロット数と推定）、
    //               Gemma4Head の gen 版の prefill_gen はグラフの長さ T を入れている（各 job.cpp の呼び出し）
    //   prompt_len: プロンプトのトークン数
    void begin(int max_new, int window, size_t prompt_len);

    // 1トークン確定するたびに呼ぶ。
    //   step_ms : そのトークンに要した実時間（推論＋出力取り出し＋argmax）
    //   ort_ms  : そのうち OrtRun そのものの時間
    // **2つ分けて持つ**。デモで示す「1トークン何ms」は前者でなければならず、
    // 後者だけを出すと実際より速い数字を見せることになる。
    void push(int64_t id, double step_ms, double ort_ms);

    // 生成の終了。error が空なら正常終了
    void finish(const std::string& error);

    // UI が読む用のスナップショット（ロックしてコピーを返す）
    nlohmann::json snapshot() const;

private:
    // 外から作らせない（instance() だけが作る）
    GenProgress() = default;

    // mutable: snapshot() は const だが、ロックはするため
    mutable std::mutex m_;
    bool running_ = false;
    bool done_ = false;
    std::string error_;
    int max_new_ = 0;
    int window_ = 0;
    size_t prompt_len_ = 0;
    std::vector<int64_t> ids_;      // 生成された新規トークンのみ
    std::vector<double> step_ms_;
    std::vector<double> ort_ms_;
    // UI が「前回から進んだか」を安く判定するための版番号。
    // ids_ の長さでも判定できるが、開始/終了の遷移も拾いたいので独立に持つ。
    uint64_t rev_ = 0;
};

}  // namespace edgeslm
