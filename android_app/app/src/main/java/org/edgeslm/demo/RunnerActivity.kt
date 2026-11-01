package org.edgeslm.demo

import android.app.Activity
import android.content.Intent
import android.os.Bundle
import android.util.Log

/**
 * UI を持たない入口。
 *
 * 存在理由は2つだけ（基本設計③ 3章）:
 *  ① インストール直後の「stopped 状態」から抜ける入口を用意する
 *  ② **Android 12+ はバックグラウンドからの Foreground Service 起動を制限する**ため、
 *     フォアグラウンド文脈（＝アクティビティ）から Service を起動する
 *
 * ★2026-10-18 補足: 下の使い方の `org.edgeslm.demo/.RunnerActivity` は同僚向けのデモアプリ（org.edgeslm.demo）の書き方。
 *   複製したアプリ（org.edgeslm.gemma4・org.edgeslm.vlm など）では `-n <アプリの ID>/org.edgeslm.demo.RunnerActivity` と
 *   クラスを完全な名前で書く（短い `.RunnerActivity` は <アプリの ID>.RunnerActivity と読まれ、見つからない。namespace は
 *   org.edgeslm.demo のまま。EdgeSLM_AndroidVLM の app/pipeline/device_vit.py の注記）。アクション名 org.edgeslm.demo.RUN は変わらない。
 * 起動されたらすぐ Service を開始して finish する。画面は出ない。
 *
 * 使い方:
 *   adb shell am start -n org.edgeslm.demo/.RunnerActivity \
 *       -a org.edgeslm.demo.RUN --es job jobs/job.json
 *
 * 使う段: PC から job を投げる検証のすべて（PC の道具は adb で am start を打つ）。この画面は RunnerService を起こすだけ。
 * 入力: `--es job`（files/ からの相対パス。例 jobs/job.json）・`--es overrides`（job を部分的に上書きする JSON。省略可）。
 * 出力: なし（結果は RunnerService が files/out/ に書く）。ログのタグは EdgeSLMRunner。
 * （2026-10-18 学習用に追記）
 */
open class RunnerActivity : Activity() {

    /** ① 置き場のフォルダを作る → ② job の指定が無ければ版だけログに出して閉じる → ③ RunnerService を起こして閉じる。 */
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // ① 置き場の準備（実パスをログへ出し、サブフォルダを作る）
        // 外部ファイルディレクトリの実パスを必ずログに出す。
        // ホスト側のハーネスはこのパスに push する（基本設計③ リスク#5）。
        val base = getExternalFilesDir(null)
        Log.i(TAG, "externalFilesDir=${base?.absolutePath}")

        // ★サブディレクトリは **アプリ自身が作る**。
        // adb shell の mkdir で作ると所有者が shell になり、
        // アプリ側から入れず「job file not found」になる（2026-09-12 に実際に踏んだ）。
        // files 直下はアプリ所有かつ group=ext_data_rw なので、
        // アプリが作ったディレクトリには adb（shell）からも push できる。
        base?.let { b ->
            listOf("models", "models/ops", "jobs", "inputs", "out").forEach { sub ->
                val d = java.io.File(b, sub)
                if (!d.isDirectory && !d.mkdirs()) Log.w(TAG, "mkdir failed: ${d.absolutePath}")
            }
            Log.i(TAG, "subdirs ready: ${b.list()?.joinToString(",")}")
        }

        // ② job の指定（files/ からの相対パス）を受け取る
        val job = intent?.getStringExtra(EXTRA_JOB)
        if (job.isNullOrBlank()) {
            // ジョブ指定なしで起動された場合（ランチャーからのタップ等）は
            // パスだけ報告して終了する。これも「疎通確認」として有用。
            Log.i(TAG, "no job specified; nothing to run. versions=${NativeBridge.nativeVersions()}")
            finish()
            return
        }

        // ③ Service へ job と overrides を引き継いで起こす。startForegroundService で起こした Service は、
        //   決められた短い時間のうちに startForeground を呼ぶ決まり（RunnerService.onStartCommand の最初の行がそれ）
        val svc = serviceIntent().apply {
            action = ACTION_RUN
            putExtra(EXTRA_JOB, job)
            // job.json の一部を extras で上書きできるようにしておく（基本設計③ 5.1節）
            intent.getStringExtra(EXTRA_OVERRIDES)?.let { putExtra(EXTRA_OVERRIDES, it) }
        }
        Log.i(TAG, "starting RunnerService with job=$job")
        startForegroundService(svc)
        finish()
    }

    /** 起こす Service。既定は同じプロセスの `RunnerService`。別プロセス版（`RunnerActivity2`）が差し替える。 */
    protected open fun serviceIntent(): Intent = Intent(this, RunnerService::class.java)

    companion object {
        // TAG は logcat の絞り込み用（RunnerService と同じ名前）。ACTION_RUN・EXTRA_JOB・EXTRA_OVERRIDES は am start の -a・--es に書く名前
        const val TAG = "EdgeSLMRunner"
        const val ACTION_RUN = "org.edgeslm.demo.RUN"
        const val EXTRA_JOB = "job"
        const val EXTRA_OVERRIDES = "overrides"
    }
}
