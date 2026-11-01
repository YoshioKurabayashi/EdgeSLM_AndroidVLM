// EdgeSLM Runner - memprobe（EdgeSLM_AndroidScale Phase 0-A のために追加）
//
// 1つのプロセスに NPU のモデルを順に開いて**保持**し、メモリの推移を記録する。
// 問い:「1プロセスで何 GB まで NPU のモデルを持てるか」「超えたとき、誰がどこで止めるか」。
//
// 通常の task と入口を分けている理由:
//   nativeRunJob は Session を先に開いてから呼ばれるため、**何も載せていない状態**を測れない。
//   memprobe はモデルの開閉を自分で行い、開く前の値から記録する（nativeMemProbe から呼ぶ）。
//
// 途中で SIGKILL されても記録が残るよう、
//   - 1段ごとの記録は JSON Lines で追記する（out/memprobe_<job_id>_steps.jsonl）
//   - 別スレッドが一定間隔で RSS などを CSV に追記する（out/memprobe_<job_id>_samples.csv）
// どちらも1行ごとに fflush する。プロセスが殺されても、書いた分はカーネル側に残る。
//
// （2026-10-18 追記）
// どこで使うか: EdgeSLM_AndroidScale の Phase 0-A（記録 EdgeSLM_AndroidScale/results/P0A_20260919_memory_ceiling.md）。
//   分かったこと: 1 プロセスが NPU 用に持てるバッファ（dma-buf）は約 1,710MiB（1,709.5MiB は可・1,729.5MiB は不可）。
//   超えると context の生成が error 1002 で失敗する。これとは別に、ロード中は一時的に binary の約 2 倍のメモリが要り、
//   MemFree が少ないと lmkd（メモリが足りないときにアプリを止める OS の見張り役）に止められる。
// 入力: job（下の run_memprobe の項目）。出力: 結果の JSON（RunnerService が out/ に書く）と、上の 2 つの記録ファイル。
// 動かし方: RunnerService の task "memprobe"（Kotlin 側で分岐して nativeMemProbe を呼ぶ）。PC 側は
//   EdgeSLM_AndroidScale の app/pipeline/memprobe.py（run / xproc）が job を送り、記録・exit-info・logcat を回収する

#pragma once

#include "third_party/json.hpp"

namespace edgeslm {

// job の主な項目:
//   models      : 開くモデルの相対パス（base_dir 基準）。この順に開く
//   hold        : true なら開いたセッションを保持し続ける（既定 true）。false なら1本ごとに閉じる
//   run_once    : 開いた直後にゼロ入力で1回推論する（既定 true）。遅延確保を確実に起こすため
//   settle_ms   : 各段のあと待ってから、もう一度測る（既定 1000）
//   sample_ms   : サンプリング間隔（既定 100）
//   stop_on_fail: 開けなかったら以降を試さない（既定 true）
//   hold_ms     : 全部開いたあと、閉じずに待つ時間（既定 0）。別プロセスとの同時使用の試験用
//   prereclaim_mb / prereclaim_step_mb / prereclaim_sleep_ms :
//                 各ロードの前に、この量を少しずつ確保して触り、解放する（既定 0 = しない）。
//                 ファイルキャッシュを先に回収させ、ロード時の lmkd の kill を避けるため
//   ep / runtime: 通常の job と同じ（runtime は RunnerService が注入する）
// 戻り値: {job_id, task:"memprobe", pid, versions, baseline（何も開く前）, hwm_reset_supported,
//          records[モデルごと: open_ms・open_ok・backend/error・before・after_open・after_run・settled・after_release],
//          releases（新しい順に閉じたときの値）, final, steps_file, samples_file, ok}
// メモリの値は /proc から読んだ kB（dma-buf だけはバイト）
nlohmann::json run_memprobe(const nlohmann::json& job);

// ロード前に mb だけメモリを step_mb ずつ確保して触り、解放する（ファイルキャッシュを先に回収させる）。
// Session::open も job の `prereclaim_mb` があれば呼ぶ。返り値は前後の MemFree などの記録
//   mb: 確保する合計（MB）、step_mb: 1 回に確保する量（MB。0 だと終わらない。memprobe.cpp の注記）、
//   sleep_ms: 1 回ごとに待つ時間（ゆっくり確保して、カーネルに回収の時間を与える）
// 戻り値: {before, peak, after（それぞれ snapshot）, allocated_mb, alloc_ms}
nlohmann::json prereclaim(long long mb, long long step_mb, int sleep_ms);

}  // namespace edgeslm
