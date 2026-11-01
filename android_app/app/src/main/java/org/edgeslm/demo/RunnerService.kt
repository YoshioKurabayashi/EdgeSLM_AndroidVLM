package org.edgeslm.demo

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.concurrent.thread

/**
 * 推論を実行する Foreground Service。UI は持たない。
 *
 * 責務は最小限に絞る（基本設計③ 6.1節）:
 *   - ジョブ仕様ファイルを読む
 *   - JNI を呼ぶ
 *   - 返ってきた JSON をファイルへ書く
 *
 * **推論と計測のロジックは持たない。** それらは C++ 側にある。
 * 理由は ①最終目標が「C++ 推論サービス」であること
 *        ②Kotlin を触ると APK の再ビルドが必要になること。
 *
 * **端末の状態監視（温度・クロック・dma-buf）も持たない。**
 * それはホスト側のハーネスが adb 経由で並行して行う（基本設計① 8章）。
 * 責務を分けることで、計測項目を増やしてもこのアプリを作り直さずに済む。
 *
 * 使う段: PC から job を投げる検証（RunnerActivity が起こす）。チャットなどの画面は自分で NativeBridge を呼ぶが、VLM の画面（VlmActivity）は画像エンコーダをこの経路（別プロセスの RunnerService2）で動かす。
 * 入力: files/ からの相対パスの job（JSON）と overrides（JSON）。
 * 出力: files/out/result_<日時>.json と last_result.json（開けなかったときは describe_…、例外のときは error_…）。
 * 流れ: onStartCommand（前面サービスにする・WakeLock を取る）→ 作業スレッドで runJob → stopSelfSafely。
 * ★裏で動く経路なので、メモリが足りないと lmkd に止められやすい。デモの数字は画面で測る（EdgeSLM_AndroidDemo の StudyMemo 4.7）。
 * （2026-10-18 学習用に追記）
 */
// open: 別のプロセスで動く版（RunnerService2。Probe2.kt）が継承できるように
open class RunnerService : Service() {

    @Volatile private var handle: Long = 0L
    private var worker: Thread? = null
    private var wakeLock: PowerManager.WakeLock? = null

    /** 他の部品から結び付けて使う形（bind）は使わない。startForegroundService で起こすだけ。 */
    override fun onBind(intent: Intent?): IBinder? = null

    /**
     * Service の入口（Intent ごとに呼ばれる）。前面サービスにして WakeLock を取り、job は別のスレッドで動かす。
     * START_NOT_STICKY: OS に止められても自動では作り直さない（同じ job が勝手に流れ直さない）。
     */
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        // ① まず前面サービスにする（startForegroundService で起こされたので、すぐに呼ぶ決まり）。通知が出る
        startForeground(NOTIFICATION_ID, buildNotification("running"))

        // ② job の場所と上書きを Intent から受け取る（RunnerActivity が引き継いだもの）
        val jobRelPath = intent?.getStringExtra(RunnerActivity.EXTRA_JOB)
        val overrides = intent?.getStringExtra(RunnerActivity.EXTRA_OVERRIDES)
        if (jobRelPath.isNullOrBlank()) {
            Log.e(TAG, "no job path given")
            stopSelfSafely()
            return START_NOT_STICKY
        }

        // 画面を消したまま連続実行したい（発熱要因を減らすため）ので PARTIAL_WAKE_LOCK。
        // 取得に失敗しても**ジョブは続行する**。WakeLock は「あると望ましい」ものであって、
        // これが原因で検証全体が止まるのは割に合わないため。
        try {
            val pm = getSystemService(POWER_SERVICE) as PowerManager
            wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "$TAG:job").apply {
                setReferenceCounted(false)
                acquire(WAKELOCK_TIMEOUT_MS)
            }
        } catch (t: Throwable) {
            Log.w(TAG, "wake lock unavailable; continuing without it: ${t.message}")
            wakeLock = null
        }

        // ③ job は作業スレッドで動かす（推論は数秒〜数分かかるので、Service の呼び出し元のスレッドを止めない）
        worker = thread(name = "edgeslm-job") { runJob(jobRelPath, overrides) }
        return START_NOT_STICKY
    }

    /**
     * job を 1 つ動かして結果を書く（作業スレッドで動く）。
     * ① job を読む → ② task で分ける（tokenize・memprobe はモデルを開かない）→ ③ 実行時のパスを注入 →
     * ④ モデルを開く（nativeCreate）→ ⑤ job を動かす（nativeRunJob）→ 結果を書く → ⑥ 必ず閉じて Service を止める。
     */
    private fun runJob(jobRelPath: String, overrides: String?) {
        val base = getExternalFilesDir(null)
        if (base == null) {
            Log.e(TAG, "external files dir unavailable")
            stopSelfSafely(); return
        }
        Log.i(TAG, "externalFilesDir=${base.absolutePath}")
        Log.i(TAG, "versions=${NativeBridge.nativeVersions()}")

        // ① job のファイルを読む（files/ からの相対パス）
        val jobFile = File(base, jobRelPath)
        if (!jobFile.isFile) {
            Log.e(TAG, "job file not found: ${jobFile.absolutePath}")
            stopSelfSafely(); return
        }

        try {
            var jobJson = jobFile.readText()

            // ★task == "tokenize" は**モデルを開かずに** Kotlin 側で完結させる。
            //   トークナイズは CPU の辞書引きで NPU を使わないため、772MB のセッションを
            //   開く意味がない（開くと数秒かかり、検証の回転が落ちる）。
            //   ネイティブ側は model 必須の設計なので、ここで分岐する方が素直である。
            // ② task を先に覗く（JSON として読めなければ空。その場合は C++ の側で扱われる）
            val task = try {
                org.json.JSONObject(jobJson).optString("task")
            } catch (t: Throwable) {
                ""
            }
            if (task == "tokenize") {
                val r = TokenizerCheck.run(base, org.json.JSONObject(jobJson))
                Log.i(TAG, "result=${r.toString().take(2000)}")
                writeResult(base, "result", r.toString())
                stopSelfSafely(); return
            }

            // ネイティブ側が相対パスを解決できるよう、基準ディレクトリと
            // ネイティブライブラリディレクトリを注入する。
            // 後者は **FastRPC が libQnnHtp.so の実ファイルパスを要求する**ため必須（C-6）。
            val injected = JsonInject.addRuntimePaths(
                jobJson,
                baseDir = base.absolutePath,
                nativeLibDir = applicationInfo.nativeLibraryDir,
                overridesJson = overrides
            )

            Log.i(TAG, "job=${jobFile.name} nativeLibDir=${applicationInfo.nativeLibraryDir}")

            // ★task == "memprobe" は nativeCreate を通さない（EdgeSLM_AndroidScale Phase 0-A）。
            //   nativeCreate は job の model を先に開くので、「何も載せていない状態」を測れなくなる。
            //   モデルの開閉は C++ 側（runner/memprobe.cpp）が自分で行う。
            // ★task == "vlm_prep"（EdgeSLM_AndroidVLM Phase 3）もモデルを開かない。画像の前処理だけを C++ で行い、
            //   PC の照合道具（device_vlm prep）が PIL の結果と突き合わせる
            if (task == "vlm_prep") {
                val r = NativeBridge.nativeVlmPrep(injected)
                Log.i(TAG, "result=${r.take(2000)}")
                writeResult(base, "result", r)
                return
            }

            if (task == "memprobe") {
                val r = NativeBridge.nativeMemProbe(injected)
                Log.i(TAG, "result=${r.take(2000)}")
                writeResult(base, "result", r)
                return
            }

            // ④⑤ モデルを開いて job を動かす（keep_session なら開いたままにして、次の job で使い回す）
            // ★`keep_session: true`（EdgeSLM_AndroidVLM Phase 5、2026-09-29）: モデルを**開いたままにする**。
            //   画像エンコーダ（152MiB）を写真ごとに開き直すと 1 回 0.7 秒前後かかる（NPU の計算は 0.17 秒）。
            //   開いたハンドルはプロセスごとの変数（companion）に持つので、Service が止まってもプロセスが生きている間は残る。
            //   同じモデルなら使い回し、違うモデルなら閉じて開き直す。プロセスが OS に止められたら、次の job で開き直す。
            //   ★job は 1 つずつ動かす（同じハンドルを 2 つのスレッドが同時に使わないように。下の synchronized）
            // ★`task: "open"`: 開くだけ（keep_session と組で使う。デモ画面が写真の前に画像エンコーダを開いておく）
            val jo = try { org.json.JSONObject(jobJson) } catch (t: Throwable) { org.json.JSONObject() }
            val keep = jo.optBoolean("keep_session", false)
            val modelKey = jo.optString("model")
            synchronized(LOCK) {
                val t0 = android.os.SystemClock.elapsedRealtime()
                var reused = false
                // 同じモデル（job の "model" の文字列で比べる）を開いたまま持っていれば使い回す。違うモデルなら前のを閉じてから開く
                //   （1 プロセスで NPU に持てるのは約 1,710MiB なので、2 つ目を開く前に閉じる）
                //   ★keep_session を付けない job は、開いたままのセッションを閉じずに別に開く（推定: 大きいモデル同士だと上限を超えうる）
                if (keep && keptHandle != 0L && keptModel == modelKey) {
                    handle = keptHandle
                    reused = true
                } else {
                    if (keep && keptHandle != 0L) {                  // 違うモデル: 前のを閉じる
                        NativeBridge.nativeDestroy(keptHandle)
                        keptHandle = 0L
                        keptModel = ""
                    }
                    handle = NativeBridge.nativeCreate(injected)
                }
                val openMs = android.os.SystemClock.elapsedRealtime() - t0
                val describe = NativeBridge.nativeDescribe(handle)
                Log.i(TAG, "describe=${describe.take(600)} keep=$keep reused=$reused open_ms=$openMs")

                if (handle == 0L) {
                    writeResult(base, "describe", describe)
                    stopSelfSafely(); return
                }
                if (keep) {                                          // 閉じずに持っておく（finally で壊さない）
                    keptHandle = handle
                    keptModel = modelKey
                }

                val result = if (task == "open") {
                    org.json.JSONObject().apply {
                        put("job_id", jo.optString("job_id"))
                        put("ok", true)
                        put("task", "open")
                        put("describe", org.json.JSONObject(describe))
                    }.toString()
                } else {
                    NativeBridge.nativeRunJob(handle, injected)
                }
                // ⑤ ここで結果を書く（open なら describe だけ）
                // 開いたままにしたか・使い回したか・開くのにかかった時間を結果に足す
                val withKeep = try {
                    org.json.JSONObject(result).apply {
                        put("session_kept", keep)
                        put("session_reused", reused)
                        put("session_open_ms", openMs)
                    }.toString()
                } catch (t: Throwable) { result }
                Log.i(TAG, "result=${withKeep.take(2000)}")
                writeResult(base, "result", withKeep)
                if (keep) handle = 0L                                  // finally で壊さない
            }
        } catch (t: Throwable) {
            Log.e(TAG, "job failed", t)
            writeResult(base, "error", """{"error":"${t.javaClass.simpleName}: ${t.message}"}""")
        } finally {
            // ⑥ 必ず後始末をして Service を止める
            // 開いたままにしたもの（keep_session）は壊さない。途中で例外が出ても、次の job で使い回すか開き直す
            if (handle != 0L && handle != keptHandle) {
                NativeBridge.nativeDestroy(handle)
            }
            handle = 0L
            stopSelfSafely()
        }
    }

    /** 結果はファイルとログの両方に出す。ホストは pull でもログ回収でも取れる。 */
    private fun writeResult(base: File, kind: String, json: String) {
        try {
            val outDir = File(base, "out").apply { mkdirs() }
            val stamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            File(outDir, "${kind}_$stamp.json").writeText(json)
            // 直近の結果は固定名でも残す（ハーネスが取りに行きやすいように）
            File(outDir, "last_$kind.json").writeText(json)
        } catch (t: Throwable) {
            Log.e(TAG, "failed to write $kind", t)
        }
    }

    /** WakeLock を放し（失敗は無視）、前面サービスをやめて通知を消し、Service を止める。 */
    private fun stopSelfSafely() {
        try { wakeLock?.let { if (it.isHeld) it.release() } } catch (_: Throwable) {}
        wakeLock = null
        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    /** Service が止められるとき、動いている job に中断を求める（handle が 0 なら何もしない）。 */
    override fun onDestroy() {
        NativeBridge.nativeCancel(handle)
        super.onDestroy()
    }

    /** 前面サービスに要る通知を作る。重要度は低め（音を出さない）。 */
    private fun buildNotification(text: String): Notification {
        val nm = getSystemService(NotificationManager::class.java)
        val ch = NotificationChannel(CHANNEL_ID, "EdgeSLM Runner", NotificationManager.IMPORTANCE_LOW)
        nm.createNotificationChannel(ch)
        return Notification.Builder(this, CHANNEL_ID)
            .setContentTitle("EdgeSLM Runner")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.stat_notify_sync)
            .setOngoing(true)
            .build()
    }

    companion object {
        const val TAG = "EdgeSLMRunner"
        // 開いたままにしたセッション（keep_session。プロセスごとに 1 つ）
        private val LOCK = Any()
        @Volatile private var keptHandle: Long = 0L
        @Volatile private var keptModel: String = ""
        private const val CHANNEL_ID = "edgeslm_runner"
        private const val NOTIFICATION_ID = 1
        // 長い job（会話・評価のまとめ実行など）でも切れないよう余裕を持たせる
        // （★2026-10-18 訂正: 旧「soak（最大5分）でも切れないよう余裕を持たせる」。soak は計画だけで実装されていない（job.cpp の is_supported_task に無い））
        // 30 分。時間を付けて取っておくと、放し忘れても電池を食い続けない
        private const val WAKELOCK_TIMEOUT_MS = 30L * 60L * 1000L
    }
}
