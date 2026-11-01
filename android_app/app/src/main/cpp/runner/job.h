// EdgeSLM Runner - ジョブ実行
//
// task の値で振る舞いを切り替える（基本設計③ 5.2節）。
// 新しい検証を追加するたびに APK を作り直さないための仕掛けである。
//   probe     : セッションを開いて配置情報だけ返す
//   bench     : ウォームアップ + 反復 + 統計（基本設計① 5章のプロトコル）
//   op_check  : 単体 op モデルを次々に開いて可否を判定する（v0.2 / P2b）
//   infer     : 実データを読んで推論し、**出力をファイルに書く**（v0.3 / P3）
//               実機での精度を測るため。ホスト上のシミュレーションでは
//               HTP の演算順序・丸めの違いが分からない
//   soak      : 指定秒数の連続実行（計画のみ。★2026-10-18 訂正: 旧「v0.3 で実装」。いまの is_supported_task に
//               soak は無く、渡すと「未実装」のエラーを返す）
//   generate  : 窓ずらしの生成ループ（v0.4）。prefill モデルの窓を1つずつずらす
//   generate_kv : KVキャッシュを使う decode ループ（seq=1）。
//               **generate とは別の task**にしてある——窓ずらしは検証済みの経路なので
//               そこへ手を入れずに並べる。モデルも別で、同時には開かない
//
// いま実装している task は probe / bench / op_check / infer / generate / generate_kv（job.cpp の
// is_supported_task）。それ以外は「未実装」と明示して返す（v0.3 の時点では probe / bench / op_check / infer）。
// ★2026-10-18 訂正: 旧「v0.3 では probe / bench / op_check / infer を実装し、…」。いまの一覧に合わせた
// 黙って別の動作をしないこと。
//
// （2026-10-18 追記）どこで使うか: edgeslm_jni.cpp の nativeRunJob が、開いた Session と job の JSON を渡して run_job を呼ぶ。
//   job は PC から adb push した jobs/*.json（検証用の RunnerService が読む）か、画面（Activity）が組んだ JSON。
// 入力と出力: job（task・job_id と task ごとの項目）→ 結果の JSON。RunnerService は out/<種類>_<時刻>.json と
//   out/last_<種類>.json に書き、PC が adb pull で持ち帰る。画面は戻り値をそのまま読む。
// 動かし方: adb shell am start -n <パッケージ>/.RunnerActivity -a <パッケージ>.RUN --es job jobs/<job>.json
//   （例は EdgeSLM_AndroidDemo/README.md。EdgeSLM_Android は bash ../scripts/run_runner.sh run jobs/<job>.json）

#pragma once

#include <string>

#include "runner/session.h"
#include "third_party/json.hpp"

namespace edgeslm {

// 実装済みの task か（task の名前の一覧と照らすだけ。一覧はこのファイルの冒頭と同じ）。
// run_job はこれが false の task を実行せず、ok=false・「… is not implemented …」の error を返す
bool is_supported_task(const std::string& task);

// ジョブを実行して結果 JSON を返す。session は open 済みであること。
//   job        : job.json の中身。task（既定 "probe"）・job_id と、task ごとの項目（例: bench の protocol、infer の batches、generate の prompt_ids）
//   cancel_flag: nativeCancel が立てるフラグ（null 可）。生成や反復のループが毎回見て、立っていれば途中で止める
//   戻り値     : job_id・task・describe（Session::describe）・versions・power_profile（いつも "default(unsigned_pd)"）と、
//                task ごとの結果（ok・records など）
// 時間がかかる task（generate など）でも、終わるまで戻らない（同期）
nlohmann::json run_job(Session& session, const nlohmann::json& job, const bool* cancel_flag);

}  // namespace edgeslm
