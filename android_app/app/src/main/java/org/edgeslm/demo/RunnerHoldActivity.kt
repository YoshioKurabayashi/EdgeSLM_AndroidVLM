package org.edgeslm.demo

import android.app.Activity
import android.content.Intent
import android.graphics.Color
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.view.Gravity
import android.view.WindowManager
import android.widget.TextView
import org.json.JSONObject
import java.io.File

/**
 * **画面を前に出したまま job を動かす入口**（EdgeSLM_AndroidVLM Phase 3、2026-09-29）。
 *
 * ## なぜ要るか
 *
 * 1.5GB の文章側のモデルを開くとき、端末は一時的に約 3.1GB を使う（context binary をヒープに読み込む 1.48GB ＋
 * NPU 側の dma-buf 1.51GB。EdgeSLM_AndroidScale P2）。端末で空けられるのは 3.5〜4.0GB ほどで、余裕がほとんど無い。
 * このとき OS（lmkd）に止められるかどうかは、**読み込み中のアプリの優先度**で決まる:
 *
 * - 画面が前にある（oom_score_adj 0）: 他のアプリを 9 本開いて空きを減らしても **25/25 で止められなかった**
 *   （lmkd は他のアプリを 17〜49 本止めて場所を空ける。EdgeSLM_AndroidScale P3）
 * - 画面が無い（RunnerActivity はすぐ閉じるので、Service だけ。adj 50〜200）: ときどき止められる
 *   （2026-09-29 の device_vlm eval で 2 回。どちらもモデルを開き始めて 10 秒前後、adj 50、理由 "device is not responding"）
 *
 * RunnerActivity は `Theme.NoDisplay`（onResume までに finish しないと落ちる）なので、画面を残せない。
 * P0〜P4 の再現手順を変えないよう RunnerActivity には手を入れず、**別の入口**としてこれを足した。
 *
 * ## 何をするか
 *
 * 1. RunnerActivity と同じく RunnerService（同じプロセス）を起こして job を渡す
 * 2. 「実行中: <job>（経過秒）」を表示したまま前に居続ける。画面は消さない（FLAG_KEEP_SCREEN_ON。
 *    消えると画面が裏に回り、優先度が下がる）。スワイプのロックの上にも出す
 * 3. 結果のファイル（out/last_result.json・last_describe.json・last_error.json）が書かれたら、1 秒後に自分で閉じる
 *
 * プロセスが OS に止められたら、この画面も一緒に消える（同じプロセス）。PC の道具（device_vit.run_job）はそれで気づく。
 *
 * 使い方（PC の道具は `run_job(..., hold=True)` がこれを使う）:
 *
 *     adb shell am start -n org.edgeslm.vlm/org.edgeslm.demo.RunnerHoldActivity --es job jobs/job.json
 *
 * 使う段: 文章側の大きいモデル（約 1.5GB）を PC から job で動かすとき（EdgeSLM_AndroidVLM の Phase 3 以降の評価）。
 * 入力: `--es job`（files/ からの相対パス）・`--es overrides`（省略可）。出力: なし（結果は RunnerService が files/out/ に書く）。
 * （2026-10-18 学習用に追記）
 */
class RunnerHoldActivity : Activity() {

    private val ui = Handler(Looper.getMainLooper())
    private lateinit var label: TextView
    private var jobRel = ""
    private var jobId = ""
    private var startWallMs = 0L            // 結果のファイルの更新時刻と比べる（これより古いファイルは前の job のもの）
    private var startElapsed = 0L
    private var finishing = false

    /**
     * ① 前に居続ける画面を作る → ② job の指定を確かめる → ③ RunnerService を起こす（画面が作り直されたときは起こさない）
     * → ④ 1 秒ごとの見張り（poll）を始める。
     */
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // ① 画面を消さない・ロックの上にも出す（どちらも「前にある」優先度を保つため）
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        if (Build.VERSION.SDK_INT >= 27) {
            setShowWhenLocked(true)          // スワイプのロックの上にも出す（ロックの裏だと「前にある」扱いにならない）
            setTurnScreenOn(true)
        }
        label = TextView(this).apply {
            textSize = 16f
            setTextColor(Color.rgb(30, 30, 30))
            setBackgroundColor(Color.WHITE)
            gravity = Gravity.CENTER
            setPadding(48, 48, 48, 48)
        }
        setContentView(label)

        // ② job の指定を確かめ、job_id を読んでおく（結果のファイルが「この job のもの」かを見分けるのに使う）
        val base = getExternalFilesDir(null)
        jobRel = intent?.getStringExtra(RunnerActivity.EXTRA_JOB).orEmpty()
        if (base == null || jobRel.isBlank()) {
            label.text = "job の指定がありません（--es job jobs/xxx.json）"
            ui.postDelayed({ finish() }, 3000)
            return
        }
        // サブディレクトリはアプリ自身が作る（RunnerActivity と同じ理由: adb で作ると所有者が shell になる）
        listOf("models", "models/ops", "jobs", "inputs", "out").forEach { File(base, it).mkdirs() }
        jobId = runCatching { JSONObject(File(base, jobRel).readText()).optString("job_id") }.getOrDefault("")

        // 画面が作り直されたとき（保存された状態があるとき）は、job を二重に起こさない
        val restored = savedInstanceState != null
        startWallMs = savedInstanceState?.getLong(KEY_START_WALL) ?: System.currentTimeMillis()
        startElapsed = savedInstanceState?.getLong(KEY_START_ELAPSED) ?: SystemClock.elapsedRealtime()
        // ③ job を RunnerService へ渡す（RunnerActivity と同じ引き継ぎ方）。経過の時刻は作り直しをまたいで引き継ぐ
        if (!restored) {
            val svc = Intent(this, RunnerService::class.java).apply {
                action = RunnerActivity.ACTION_RUN
                putExtra(RunnerActivity.EXTRA_JOB, jobRel)
                intent.getStringExtra(RunnerActivity.EXTRA_OVERRIDES)?.let { putExtra(RunnerActivity.EXTRA_OVERRIDES, it) }
            }
            Log.i(TAG, "hold: starting RunnerService with job=$jobRel (screen stays in front)")
            startForegroundService(svc)
        }
        // ④ 見張りを始める
        ui.post(poll)
    }

    /** 画面が作り直されても経過の時刻を失わないよう、始めた時刻を保存する（onCreate の restored で読み戻す）。 */
    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        outState.putLong(KEY_START_WALL, startWallMs)
        outState.putLong(KEY_START_ELAPSED, startElapsed)
    }

    /** 見張りを止める（閉じた画面に表示し続けない）。 */
    override fun onDestroy() {
        ui.removeCallbacksAndMessages(null)
        super.onDestroy()
    }

    /** 1 秒ごとに経過を表示し、結果のファイルが書かれたか見る */
    private val poll = object : Runnable {
        override fun run() {
            if (isFinishing || finishing) return
            val sec = (SystemClock.elapsedRealtime() - startElapsed) / 1000
            val done = doneKind()
            if (done != null) {
                finishing = true
                label.text = "終わりました（${done}、${sec} 秒）\n$jobRel"
                Log.i(TAG, "hold: job finished ($done, ${sec}s); closing")
                ui.postDelayed({ finish() }, 1000)
                return
            }
            if (sec * 1000 > HOLD_MAX_MS) {           // 念のための上限（結果が永遠に来ない場合に画面を残し続けない）
                Log.w(TAG, "hold: no result after ${sec}s; closing")
                finish()
                return
            }
            label.text = "実行中（画面を前に出したまま動かしています）\n$jobRel\n${sec} 秒"
            ui.postDelayed(this, 1000)
        }
    }

    /** この job の結果が書かれていれば、その種類（result / describe / error）。まだなら null */
    private fun doneKind(): String? {
        val out = File(getExternalFilesDir(null) ?: return null, "out")
        for (kind in listOf("result", "describe", "error")) {
            val f = File(out, "last_$kind.json")
            // ★describe・error は job_id を見ないので、前の job が 1 秒の余裕の中で終わっていると、それを拾いうる（推定。ふつうは起きない）
            if (!f.isFile || f.lastModified() + 1000 < startWallMs) continue      // 前の job の結果（1 秒の余裕）
            if (kind == "result" && jobId.isNotEmpty()) {
                // job_id が違う結果（別の入口が同時に書いたもの）は見送る。書いている途中で読めなければ次の回に
                val id = runCatching { JSONObject(f.readText()).optString("job_id") }.getOrNull() ?: continue
                if (id.isNotEmpty() && id != jobId) continue
            }
            return kind
        }
        return null
    }

    companion object {
        private const val TAG = RunnerActivity.TAG
        private const val KEY_START_WALL = "start_wall_ms"
        private const val KEY_START_ELAPSED = "start_elapsed_ms"
        // 3 時間たっても結果が来なければ見張りをやめて閉じる（念のための上限。poll の中の注記）
        private const val HOLD_MAX_MS = 3L * 60L * 60L * 1000L       // 3 時間
    }
}
