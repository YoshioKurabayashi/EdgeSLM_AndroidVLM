// EdgeSLM Demo - 生成の途中経過の共有状態の本体（なぜ要るか・誰が読み書きするかは progress.h）
//
// 中身は「ロックして値を書く／ロックして写しを返す」だけ。生成のスレッドと画面のスレッドが同時に触るので、
// どの関数も最初に m_ をロックする。
#include "runner/progress.h"

#include <algorithm>

namespace edgeslm {

using nlohmann::json;

// 関数の中の static は、最初に呼ばれたときに 1 回だけ作られる（C++11 以降、同時に呼ばれても 1 回だけ）。
// プロセスが終わるまで残る
GenProgress& GenProgress::instance() {
    static GenProgress g;
    return g;
}

// 新しい生成の始まり。前の結果（ids・時間・error）を捨て、running=true・done=false にする。
// rev を進めるので、画面は「新しい生成が始まった」ことに気づける
void GenProgress::begin(int max_new, int window, size_t prompt_len) {
    std::lock_guard<std::mutex> lk(m_);
    running_ = true;
    done_ = false;
    error_.clear();
    max_new_ = max_new;
    window_ = window;
    prompt_len_ = prompt_len;
    ids_.clear();
    step_ms_.clear();
    ort_ms_.clear();
    ++rev_;
}

// 1 語確定するたびに呼ぶ。id: 選んだ語の ID、step_ms: その語の実時間、ort_ms: そのうち OrtRun の時間（どちらも ms）。
// rev を 1 つ進める（画面はこれで「進んだ」と分かる）
void GenProgress::push(int64_t id, double step_ms, double ort_ms) {
    std::lock_guard<std::mutex> lk(m_);
    ids_.push_back(id);
    step_ms_.push_back(step_ms);
    ort_ms_.push_back(ort_ms);
    ++rev_;
}

namespace {
// 中央値。値を写して（引数が値渡し）並べ替え、真ん中を取る。偶数個のときは上側の真ん中（v[n/2]）。
// 空なら 0
double median_of(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
}  // namespace

// 生成の終わり。error が空なら正常終了、空でなければ理由（"cancelled" など）を残す。
// ids と時間は消さない（画面が最後の結果を読めるように）
void GenProgress::finish(const std::string& error) {
    std::lock_guard<std::mutex> lk(m_);
    running_ = false;
    done_ = true;
    error_ = error;
    ++rev_;
}

// 画面が読む用の写し（JSON）。ロックしたまま写すので、書いている途中の半端な値は見えない。
// 時間の項目（last_ms・median_ms・median_ort_ms・tokens_per_sec）は 1 語以上あるときだけ入る
json GenProgress::snapshot() const {
    std::lock_guard<std::mutex> lk(m_);
    json j;
    j["running"] = running_;
    j["done"] = done_;
    j["rev"] = rev_;
    j["max_new"] = max_new_;
    j["window"] = window_;
    j["prompt_len"] = prompt_len_;
    j["ids"] = ids_;
    j["n"] = ids_.size();
    if (!error_.empty()) j["error"] = error_;

    if (!step_ms_.empty()) {
        // 中央値を出す（平均は初回の外れ値に引きずられる。基本設計① 5章と同じ理由）。
        const double med = median_of(step_ms_);
        j["last_ms"] = step_ms_.back();
        j["median_ms"] = med;                        // 1トークンの実時間
        j["median_ort_ms"] = median_of(ort_ms_);     // うち OrtRun
        // 1 秒あたりの語数＝ 1000ms ÷ 1 語の実時間の中央値
        j["tokens_per_sec"] = (med > 0.0) ? (1000.0 / med) : 0.0;
    }
    return j;
}

}  // namespace edgeslm
