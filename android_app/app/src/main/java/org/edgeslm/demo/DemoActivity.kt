package org.edgeslm.demo

import android.app.Activity
import android.content.Intent
import android.graphics.Color
import android.graphics.Typeface
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.style.ForegroundColorSpan
import android.text.style.StyleSpan
import android.util.Log
import android.util.TypedValue
import android.view.Gravity
import android.view.View
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.widget.Button
import android.widget.EditText
import android.widget.HorizontalScrollView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import kotlin.concurrent.thread

/**
 * デモ用の画面。
 *
 * 見せたいことは1つだけ——**この端末だけで文章が生成されている**。
 * したがって画面には「入力」「生成中のテキスト」「1トークンあたりの時間」しか置かない。
 *
 * 構成上の判断:
 *  - **レイアウトはコードで組む**。XML も Compose も入れない。依存を1つも増やさないことで、
 *    QNN ランタイムと ORT の版が噛み合っている現在の構成（FR-12）に手を触れずに済む
 *  - **セッションはこの画面が持ち、開いたまま使い回す**。772MB のロードは起動時の1回きり
 *    （実測 3.4 秒）。生成のたびに開き直すと、その待ち時間がデモを壊す
 *  - 逐次表示は `NativeBridge.nativeProgress()` の 50ms ポーリング。
 *    JNI コールバックを使わない理由は cpp/runner/progress.h の注記を参照
 *
 * 既存の検証経路（`am start --es job ...` → RunnerService）はこの画面とは独立に残してある。
 *
 * 方式: **窓ずらし**（毎回モデル全体を計算し直す。task=generate）。ランチャーの「EdgeSLM Demo」。
 *   既定のモデルは qwen24ext_epctx.onnx（窓16 固定の最初の版）。demo_modes.json で窓64 などに切り替える（README「画面とモード」）。
 *   速さは 33〜42 ms/token（README の表）。KV の画面（KvDemoActivity）とは下のボタンで行き来する。
 * 使う段: EdgeSLM_AndroidDemo の D1（端末の中で生成を回す。StudyMemo 1 章）と、画面の見せ方（StudyMemo 4 章）。
 * 入力: 画面に打った文。files/ の tokenizer/・models/ops/・jobs/demo_override.json・jobs/demo_modes.json。
 * 出力: 画面の表示（プロンプトは灰色、生成した分は黒の太字）と logcat（タグ EdgeSLMDemo）。
 * 動かし方: adb shell am start -n org.edgeslm.demo/.DemoActivity（またはランチャーのアイコン）。
 * （2026-10-18 学習用に追記）
 */
class DemoActivity : Activity() {

    private lateinit var statusView: TextView
    private lateinit var promptEdit: EditText
    private lateinit var outputView: TextView
    private lateinit var statsView: TextView
    private lateinit var generateButton: Button
    private lateinit var stopButton: Button
    private lateinit var outputScroll: ScrollView

    // handle: 開いているセッション（0 なら未だ開いていない）。tokenizer: 読み込み済みの辞書。generating: 生成中か。
    //   どれも作業スレッドが書き、画面のスレッドが読むので @Volatile
    @Volatile private var handle: Long = 0L
    @Volatile private var tokenizer: Qwen2Tokenizer? = null
    @Volatile private var generating = false

    // モード（窓ずらし / KVキャッシュ decode）。
    // `files/jobs/demo_modes.json` があるときだけ画面に出る。無ければ従来どおり。
    private var modes: List<Pair<String, JSONObject>> = emptyList()
    @Volatile private var modeIndex = 0
    private var modeRow: LinearLayout? = null

    // promptText: 灰色で描くプロンプト。lastRev: 最後に読んだ途中経過の版（rev が変わったときだけ描き直す）
    private var promptText: String = ""
    private var lastRev: Long = -1L
    private val ui = Handler(Looper.getMainLooper())

    /** 画面を作り、辞書とモデルの読み込みを別のスレッドで始める（読み込みは数秒かかるので画面のスレッドを止めない）。 */
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        modes = loadModes()
        setContentView(buildUi())
        setBusy(true, "起動中…")
        thread(name = "edgeslm-init") { initSession() }
    }

    /** 画面を閉じるときは、生成を止めてからセッションを閉じる（NPU のメモリを返す）。 */
    //   ★生成の途中で閉じると、nativeCancel は印を立てるだけなので、作業スレッドがまだ使っているセッションを
    //     nativeDestroy で壊しうる（推定。switchTo は生成中なら何もしないが、onDestroy には同じ守りが無い）
    override fun onDestroy() {
        if (handle != 0L) {
            NativeBridge.nativeCancel(handle)
            NativeBridge.nativeDestroy(handle)
            handle = 0L
        }
        super.onDestroy()
    }

    // ------------------------------------------------------------------ 画面

    /**
     * 画面をコードで組む（XML を使わない）。上から: 題名・状態の行・モード行（あれば）・入力欄・定型プロンプト・
     * Generate／Stop・KV 版へのボタン・出力・速さの行。
     */
    private fun buildUi(): View {
        val dp = resources.displayMetrics.density
        // dp → 画面の画素。端末の画素密度に合わせて大きさをそろえる
        fun px(v: Int) = (v * dp).toInt()

        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(px(16), px(16), px(16), px(16))
            setBackgroundColor(Color.WHITE)
        }

        root.addView(TextView(this).apply {
            text = "Qwen2.5-0.5B — この端末の NPU で動いています"
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 17f)
            setTypeface(typeface, Typeface.BOLD)
            setTextColor(Color.BLACK)
        })

        statusView = TextView(this).apply {
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
            setTextColor(Color.parseColor("#666666"))
            setPadding(0, px(2), 0, px(10))
        }
        root.addView(statusView)

        // モード行。**modes が空なら何も足さない**ので、設定ファイルが無い端末では
        // 画面も動作も従来と同じになる。
        if (modes.size > 1) {
            val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
            modes.forEachIndexed { i, m ->
                row.addView(Button(this).apply {
                    text = m.first
                    setTextSize(TypedValue.COMPLEX_UNIT_SP, 11f)
                    setOnClickListener { switchMode(i) }
                }, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
            }
            modeRow = row
            root.addView(row, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))
        }

        promptEdit = EditText(this).apply {
            hint = "続きを書かせたい文章を入力"
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 15f)
            setTextColor(Color.BLACK)
            minLines = 2
            maxLines = 4
        }
        root.addView(promptEdit, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))

        // プリセット（デモ当日に打ち間違えないための保険）
        val presets = HorizontalScrollView(this).apply { isHorizontalScrollBarEnabled = false }
        val presetRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        for (p in PRESETS) {
            presetRow.addView(Button(this).apply {
                text = p.first
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 11f)
                setOnClickListener { promptEdit.setText(p.second) }
            })
        }
        presets.addView(presetRow)
        root.addView(presets)

        val buttonRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL }
        generateButton = Button(this).apply {
            text = "Generate"
            setOnClickListener { onGenerate() }
        }
        stopButton = Button(this).apply {
            text = "Stop"
            isEnabled = false
            setOnClickListener { NativeBridge.nativeCancel(handle) }
        }
        buttonRow.addView(generateButton, LinearLayout.LayoutParams(0, WRAP_CONTENT, 2f))
        buttonRow.addView(stopButton, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
        root.addView(buttonRow, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))

        // デモ中に画面上で方式を切り替えるためのボタン。
        // 切り替えるとモデルを開き直すので数秒かかる（先に解放してから開くため。理由は switchTo の注記）。
        root.addView(Button(this).apply {
            text = "▶ KVキャッシュ版へ切り替え"
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
            setOnClickListener { switchTo(KvDemoActivity::class.java) }
        }, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT))

        outputView = TextView(this).apply {
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 17f)
            setTextColor(Color.BLACK)
            setLineSpacing(0f, 1.25f)
            setPadding(px(14), px(14), px(14), px(14))
            setTextIsSelectable(true)
        }
        // 背景は ScrollView 側に置く。TextView に置くと**中身の高さしか塗られず**、
        // 生成前に細い帯が浮いているだけの見た目になる。
        outputScroll = ScrollView(this).apply {
            setBackgroundColor(Color.parseColor("#F4F4F6"))
            addView(outputView)
        }
        root.addView(outputScroll, LinearLayout.LayoutParams(MATCH_PARENT, 0, 1f).apply {
            topMargin = px(10)
        })

        statsView = TextView(this).apply {
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 13f)
            setTextColor(Color.parseColor("#444444"))
            gravity = Gravity.END
            setPadding(0, px(8), 0, 0)
            setTypeface(Typeface.MONOSPACE)
        }
        root.addView(statsView)

        return root
    }

    /**
     * ボタンの押せる・押せないと状態の行をまとめて変える。
     * Generate は「暇・モデルが開いている・辞書がある」ときだけ、Stop は生成中だけ押せる。status が null なら状態の行は変えない。
     */
    private fun setBusy(busy: Boolean, status: String?) {
        generateButton.isEnabled = !busy && handle != 0L && tokenizer != null
        stopButton.isEnabled = busy && generating
        status?.let { statusView.text = it }
    }

    /**
     * もう一方の画面へ切り替える。
     *
     * ★**セッションを先に解放してから** `startActivity` する。
     *   2つ同時に持たない（2026-09-19 実測: 772MB×2 は保持できるが、ロード中の一時的なメモリ要求で lmkd に kill されうる。1プロセスの NPU 側上限も約1,710MiB で、Qwen2.5 KV512 と Qwen3 を同時に持つと超える。EdgeSLM_AndroidScale P0A）。
     *   `startActivity` → `finish()` の順にすると、新しい画面の onCreate が
     *   こちらの onDestroy より先に走りうるので、順序を自分で決める。
     */
    private fun switchTo(cls: Class<*>) {
        if (generating) return
        generateButton.isEnabled = false
        statusView.text = "切り替え中…（モデルを開き直します）"
        if (handle != 0L) {
            NativeBridge.nativeCancel(handle)
            NativeBridge.nativeDestroy(handle)
            handle = 0L
        }
        startActivity(Intent(this, cls))
        finish()
    }


    // -------------------------------------------------------------- 初期化

    /** トークナイザの読み込みと 772MB モデルのロード。どちらも起動時の1回きり。 */
    private fun initSession() {
        val base = getExternalFilesDir(null)
        if (base == null) {
            ui.post { setBusy(true, "外部ストレージが使えません") }
            return
        }
        listOf("models", "models/ops", "jobs", "inputs", "out", "tokenizer").forEach {
            File(base, it).mkdirs()
        }

        try {
            ui.post { setBusy(true, "トークナイザを読み込み中…") }
            // ① 辞書を読む（files/tokenizer/）
            val t0 = System.currentTimeMillis()
            val tok = Qwen2Tokenizer.load(File(base, "tokenizer"))
            val tokMs = System.currentTimeMillis() - t0

            // ② job の土台に実行時のパスを入れてモデルを開く（772MB。strict で開けたかが「全ノード NPU」の表示になる）
            ui.post { setBusy(true, "モデルを読み込み中…（772MB。初回のみ数秒）") }
            val config = JsonInject.addRuntimePaths(
                baseJobJson(),
                baseDir = base.absolutePath,
                nativeLibDir = applicationInfo.nativeLibraryDir,
                overridesJson = null
            )
            val t1 = System.currentTimeMillis()
            val h = NativeBridge.nativeCreate(config)
            val loadMs = System.currentTimeMillis() - t1
            val describe = JSONObject(NativeBridge.nativeDescribe(h))
            Log.i(TAG, "describe=$describe")

            if (h == 0L) {
                ui.post { setBusy(true, "モデルを開けませんでした: ${describe.optString("error")}") }
                return
            }
            handle = h
            tokenizer = tok

            // ③ どこで動いているか（backend）と、CPU への逃がしを禁止して開けたか（strict）を状態の行に出す。配置は推測しない
            val backend = describe.optString("backend", "?")
            val strict = describe.optBoolean("strict_no_cpu_fallback", false)
            ui.post {
                statusView.text = "$backend" +
                    (if (strict) " · 全ノード NPU" else " · 一部 CPU") +
                    " · 起動 ${loadMs / 1000.0}s（辞書 ${tokMs}ms）"
                setBusy(false, null)
            }
        } catch (t: Throwable) {
            Log.e(TAG, "init failed", t)
            ui.post { setBusy(true, "初期化に失敗: ${t.javaClass.simpleName}: ${t.message}") }
        }
    }

    /** セッションを開くための最小の設定（モデルと EP の指定）。 */
    /**
     * 生成ジョブの土台。
     *
     * `files/jobs/demo_override.json` があれば**その中身で上書きする**。
     * モデルや窓幅を変えるたびに APK を作り直さずに済むようにするため
     * （基本設計③ 5.1節の「再ビルドを最小化する」方針）。
     *
     * 窓64（マスク入力版）へ切り替える例:
     *
     *     {
     *       "model": "models/ops/qwen_pm64_epctx.onnx",
     *       "dynamic_mask": true,
     *       "const_inputs": { "cos": "inputs/cos_w64.bin", "sin": "inputs/sin_w64.bin" },
     *       "max_new": 24
     *     }
     *
     * `dynamic_mask` を付けると C++ 側が**毎ステップ attn_mask を作る**。
     * 窓を広げるとパディングを塞ぐかどうかが効いてくるため必要になる
     * （fp32 実測: W=64 でマスク無しだと 10プロンプト中 3件しか正解と一致しない）。
     */
    // 手順: ① 既定の job（task=generate・既定のモデル・QNN の EP）→ ② demo_override.json を浅く重ねる → ③ 選んだモードを重ねる
    private fun baseJobJson(): String {
        val job = JSONObject().apply {
            put("job_id", "demo_ui")
            put("task", "generate")
            put("model", MODEL_REL)
            put("ep", JSONObject().apply {
                put("provider", "QNN")
                put("backend", "libQnnHtp.so")
                put("options", JSONObject().apply {
                    // QNN の EP の設定: 入出力の量子化・逆量子化を CPU 側に残さない（"0"）・NPU の優先度を高く
                    put("offload_graph_io_quantization", "0")
                    put("qnn_context_priority", "high")
                })
                put("session", JSONObject().apply {
                    // ORT のセッションの設定: CPU のスレッドは 1 本（計算は NPU なので CPU は待つだけ）・ログは警告以上（3）
                    put("intra_op_num_threads", 1)
                    put("log_severity_level", 3)
                })
            })
        }
        // 上書きは**浅いマージ**。ep のような入れ子ごと差し替えたい場合は
        // その階層をまるごと書く（部分マージは挙動が読みにくくなるため採らない）。
        try {
            val f = File(getExternalFilesDir(null), OVERRIDE_REL)
            if (f.isFile) {
                val ov = JSONObject(f.readText())
                for (k in ov.keys()) job.put(k, ov.get(k))
                Log.i(TAG, "job override applied from $OVERRIDE_REL: ${ov.keys().asSequence().toList()}")
            }
        } catch (t: Throwable) {
            // 上書きが壊れていても**既定で動かす**。デモが起動しないのは避ける。
            Log.w(TAG, "job override ignored: ${t.javaClass.simpleName}: ${t.message}")
        }
        // 選んでいるモードを最後に重ねる（モードが無ければ何もしない）
        modes.getOrNull(modeIndex)?.second?.let { m ->
            for (k in m.keys()) job.put(k, m.get(k))
        }
        return job.toString()
    }

    /**
     * `files/jobs/demo_modes.json` を読む。形は
     *
     *     [ {"label":"窓ずらし 64", "job":{...}},
     *       {"label":"KVキャッシュ decode", "job":{...}} ]
     *
     * `job` の中身は `baseJobJson()` に浅くマージされる（`model` / `task` /
     * `dynamic_mask` / `const_inputs` / `max_new` など）。
     *
     * **無ければ空を返す。** その場合モード行は出ず、従来どおりの単一モードで動く。
     */
    private fun loadModes(): List<Pair<String, JSONObject>> {
        return try {
            val f = File(getExternalFilesDir(null), MODES_REL)
            if (!f.isFile) return emptyList()
            val arr = JSONArray(f.readText())
            (0 until arr.length()).map { i ->
                val o = arr.getJSONObject(i)
                o.optString("label", "mode$i") to o.getJSONObject("job")
            }
        } catch (t: Throwable) {
            Log.w(TAG, "demo_modes.json ignored: ${t.javaClass.simpleName}: ${t.message}")
            emptyList()
        }
    }

    /**
     * モードを切り替える。**セッションは1つだけ開く**（2026-09-19 実測: 772MB×2 は保持できるが、ロード中の一時的なメモリ要求で lmkd に kill されうる。1プロセスの NPU 側上限も約1,710MiB で、Qwen2.5 KV512 と Qwen3 を同時に持つと超える。EdgeSLM_AndroidScale P0A）。
     * 失敗したら元のモードへ戻す。
     */
    private fun switchMode(index: Int) {
        if (generating || index == modeIndex || index !in modes.indices) return
        // 先に新しいモードの番号にしておく（baseJobJson が modeIndex を読むため）。開けなかったら prev に戻す
        val prev = modeIndex
        modeIndex = index
        setBusy(true, "${modes[index].first} に切り替え中…")
        thread(name = "edgeslm-switch") {
            try {
                if (handle != 0L) {
                    NativeBridge.nativeDestroy(handle)
                    handle = 0L
                }
                val config = JsonInject.addRuntimePaths(
                    baseJobJson(),
                    baseDir = getExternalFilesDir(null)!!.absolutePath,
                    nativeLibDir = applicationInfo.nativeLibraryDir,
                    overridesJson = null
                )
                val t0 = System.currentTimeMillis()
                val h = NativeBridge.nativeCreate(config)
                val loadMs = System.currentTimeMillis() - t0
                if (h == 0L) {
                    modeIndex = prev
                    ui.post { setBusy(true, "切り替えに失敗（元のモードへ戻します）") }
                    return@thread
                }
                handle = h
                val describe = JSONObject(NativeBridge.nativeDescribe(h))
                val backend = describe.optString("backend", "?")
                val strict = describe.optBoolean("strict_no_cpu_fallback", false)
                ui.post {
                    statusView.text = "${modes[index].first} · $backend" +
                        (if (strict) " · 全ノード NPU" else " · 一部 CPU") +
                        " · 切替 ${loadMs / 1000.0}s"
                    setBusy(false, null)
                }
            } catch (t: Throwable) {
                modeIndex = prev
                Log.e(TAG, "switchMode failed", t)
                ui.post { setBusy(true, "切り替えに失敗: ${t.javaClass.simpleName}") }
            }
        }
    }

    // ---------------------------------------------------------------- 生成

    /**
     * Generate を押したとき。① 打った文を ID 列にする → ② job を組む → ③ 途中経過の見張りを始め、別のスレッドで生成する
     * → ④ 終わったら確定した統計を出す。
     */
    private fun onGenerate() {
        val tok = tokenizer ?: return
        if (handle == 0L || generating) return

        val text = promptEdit.text.toString()
        if (text.isBlank()) return

        // ① 端末のトークナイザで ID 列にする（PC は使わない）
        val ids = tok.encode(text)
        if (ids.isEmpty()) return

        promptText = text
        //   ★-1 から始めるので、ジョブが始まる前の一瞬に前の回の語を読みうる（チャット画面では送る直前の版を覚えて直した。
        //     EdgeSLM_AndroidDemo の StudyMemo 4.2。この画面では表示が次の版ですぐ上書きされるので、目に見える害は小さい。推定）
        lastRev = -1L
        generating = true
        setBusy(true, null)
        renderOutput(emptyList())
        statsView.text = "…"

        // ② 土台の job に、プロンプトの ID 列・作る語の上限・詰め物の ID・止める ID を足す
        val job = JSONObject(baseJobJson()).apply {
            put("max_new", MAX_NEW)
            put("pad_id", PAD_ID)
            put("prompt_ids", JSONArray().also { a -> ids.forEach { a.put(it) } })
            put("stop_ids", JSONArray().also { a -> STOP_IDS.forEach { a.put(it) } })
        }
        val jobJson = JsonInject.addRuntimePaths(
            job.toString(),
            baseDir = getExternalFilesDir(null)!!.absolutePath,
            nativeLibDir = applicationInfo.nativeLibraryDir,
            overridesJson = null
        )

        // ③ 見張り（50ms ごと）を始めてから、生成を別のスレッドで動かす（nativeRunJob は終わるまで戻らない）
        ui.post(pollProgress)
        thread(name = "edgeslm-generate") {
            val result = NativeBridge.nativeRunJob(handle, jobJson)
            Log.i(TAG, "generate result=${result.take(600)}")
            ui.post {
                generating = false
                setBusy(false, null)
                showFinalStats(result)
            }
        }
    }

    /**
     * 50ms ごとに C++ 側の進捗を読む。
     * `rev` が変わったときだけ描き直す（ロックを取るのは C++ 側の snapshot だけ）。
     */
    private val pollProgress = object : Runnable {
        override fun run() {
            try {
                val p = JSONObject(NativeBridge.nativeProgress())
                val rev = p.optLong("rev", -1L)
                if (rev != lastRev) {
                    lastRev = rev
                    val arr = p.optJSONArray("ids")
                    val ids = ArrayList<Int>(arr?.length() ?: 0)
                    if (arr != null) for (i in 0 until arr.length()) ids.add(arr.getInt(i))
                    renderOutput(ids)
                    if (p.has("median_ms")) {
                        val med = p.optDouble("median_ms")
                        statsView.text = String.format(
                            "%.1f ms/token · %.1f tok/s · %d tokens",
                            med, 1000.0 / med, ids.size
                        )
                    }
                }
            } catch (t: Throwable) {
                Log.w(TAG, "progress read failed: ${t.message}")
            }
            if (generating) ui.postDelayed(this, POLL_MS)
        }
    }

    /** プロンプトを灰色、生成された分を黒の太字で描く（どこからが生成かを一目で分かるように）。 */
    private fun renderOutput(newIds: List<Int>) {
        val tok = tokenizer ?: return
        // decodeStreaming は末尾の切れた UTF-8 を持ち越すので、日本語でも途中で文字化けを見せない
        val generated = if (newIds.isEmpty()) "" else tok.decodeStreaming(newIds)
        val sb = SpannableStringBuilder()
        sb.append(promptText)
        sb.setSpan(ForegroundColorSpan(Color.parseColor("#8A8A8E")), 0, promptText.length,
            Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
        val start = sb.length
        sb.append(generated)
        sb.setSpan(StyleSpan(Typeface.BOLD), start, sb.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
        outputView.text = sb
        outputScroll.post { outputScroll.fullScroll(View.FOCUS_DOWN) }
    }

    /** 生成完了後は、途中経過ではなく確定した統計を出す。 */
    private fun showFinalStats(resultJson: String) {
        try {
            val r = JSONObject(resultJson)
            if (!r.optBoolean("ok", false)) {
                val err = r.optString("error", "unknown")
                statsView.text = if (err == "cancelled") "中断しました" else "失敗: $err"
                return
            }
            // 速さは 1 語ごとの実時間の中央値から出す（平均は最初の外れ値に引きずられる。StudyMemo 1.4）。OrtRun は NPU の計算の分
            val step = r.optJSONObject("step_ms_stats")
            val ort = r.optJSONObject("ort_run_ms")
            val n = r.optJSONArray("new_ids")?.length() ?: 0
            val med = step?.optDouble("median") ?: 0.0
            val total = r.optDouble("total_ms", 0.0)
            statsView.text = String.format(
                "%.1f ms/token · %.1f tok/s · %d tokens · %.2fs（内 OrtRun %.1f ms）",
                med, if (med > 0) 1000.0 / med else 0.0, n, total / 1000.0,
                ort?.optDouble("median") ?: 0.0
            )
        } catch (t: Throwable) {
            statsView.text = "結果を読めませんでした"
        }
    }

    companion object {
        // MODEL_REL: 既定のモデル（窓16 固定の最初の版。README「画面の中のモード切り替え」）
        private const val TAG = "EdgeSLMDemo"
        private const val MODEL_REL = "models/ops/qwen24ext_epctx.onnx"
        // これがあれば上記の既定を上書きする（モデル差し替えに再ビルドを要らなくするため）
        private const val OVERRIDE_REL = "jobs/demo_override.json"
        // これがあれば画面にモード切り替えが出る。無ければ従来どおり単一モード
        private const val MODES_REL = "jobs/demo_modes.json"
        // POLL_MS: 途中経過を読む間隔。MAX_NEW: 1 回に作る語の上限。
        // PAD_ID 151643 は <|endoftext|>（窓の空きを詰める語）、STOP_IDS は <|im_end|>（151645）と <|endoftext|>（Qwen2.5 の追加トークン）
        private const val POLL_MS = 50L
        private const val MAX_NEW = 24
        private const val PAD_ID = 151643
        private val STOP_IDS = intArrayOf(151645, 151643)

        /** デモ用の定型プロンプト。窓16トークンで素直に続く形を選んである。 */
        private val PRESETS = listOf(
            "首都" to "The capital city of Japan is Tokyo. The capital city of France is",
            "日の出" to "The sun rises in the east and sets in the",
            "言語" to "Python is a programming language. Java is a programming",
            "日本語" to "日本の首都は東京です。フランスの首都は"
        )
    }
}
