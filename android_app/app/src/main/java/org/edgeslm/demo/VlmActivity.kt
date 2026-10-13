package org.edgeslm.demo

import android.app.Activity
import android.content.ContentValues
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.graphics.drawable.StateListDrawable
import android.media.ExifInterface
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.MediaStore
import android.util.Log
import android.util.TypedValue
import android.view.Gravity
import android.view.MotionEvent
import android.view.View
import android.view.ViewGroup.LayoutParams.MATCH_PARENT
import android.view.ViewGroup.LayoutParams.WRAP_CONTENT
import android.view.WindowManager
import android.view.inputmethod.EditorInfo
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import kotlin.concurrent.thread

/**
 * **写真を撮って、日本語で質問すると、スマホの NPU だけで答える画面**（EdgeSLM_AndroidVLM Phase 3、2026-09-29）。
 *
 * ## 流れ（DoC/基本設計_Phase3.md の 2 章）
 *
 * 1. 「写真を撮る」→ カメラアプリに撮ってもらう（MediaStore に保存させる。このアプリはカメラの権限を持たない）
 * 2. 撮れたらすぐ、裏で画像を読み取る（質問を選んでいる間に済ませる）:
 *    - 読み込み・EXIF の向き（Kotlin）→ RGB のバイト列 → **前処理（C++、nativeVlmPrep）**:
 *      PIL の BICUBIC と同じ縮め方で 480×336 の黒い余白に貼り、16×16 のパッチにして pixel_values [630,768]
 *    - **画像エンコーダ**: 別プロセス（`:probe` の RunnerService2）に `infer` の job を渡す（152MiB を開いて 1 回動かす）
 *      → 特徴 [70,1536] のファイル
 * 3. 質問する → **文章側**（この画面が開いたままにしている 1,467MiB の Gemma 4）に `generate_kv` を渡す。
 *    プロンプトの画像トークン 70 個の位置に特徴を入れる（C++ 側。`image_features`）。1 トークンずつ表示する
 * 4. 答えの下に小さく 1 行: 画像の読み取り時間・最初の文字まで・トークン/秒・全部 NPU か
 *
 * 同じ写真への 2 問目以降は、差分だけ送って前のキャッシュを使い回す（画像を読み直さない）。窓（256）があふれたら、写真から作り直す。
 *
 * ## 設定: `files/jobs/vlm_demo.json`（PC の setup_vlm_demo.py が書く）
 *
 * ## テスト用のインテント引数（画面の操作の代わり。PC の device_vlm uicheck が使う）
 *
 *     am start -n org.edgeslm.vlm/org.edgeslm.demo.VlmActivity --es image inputs/vlm/test.jpg --es question "何が写っていますか？"
 *
 * image は files からの相対パス（カメラの代わりにこのファイルを写真として使う）、question を渡すと読み取り後に自動で質問する。
 * 日本語の質問は adb の引数で化けることがあるので、`--es question_file inputs/vlm/q.txt`（UTF-8 のファイル）でも渡せる。
 */
class VlmActivity : Activity() {

    // ---- 画面の部品 ----
    private lateinit var statusView: TextView
    private lateinit var photoView: ImageView
    private lateinit var photoNote: TextView
    private lateinit var cameraButton: Button
    private lateinit var questionEdit: EditText
    private lateinit var askButton: Button
    private lateinit var stopButton: Button
    private lateinit var answerView: TextView
    private lateinit var statsView: TextView
    private lateinit var scroll: ScrollView
    private val suggestionButtons = ArrayList<Button>()

    // ---- 設定 ----
    private var cfg = JSONObject()
    private var systemPrompt = DEFAULT_SYSTEM
    private var grid = intArrayOf(21, 30)
    private var nImageTokens = 70
    private var maxNew = 96
    private var stopIds = listOf(106, 1, 50)
    private var suggestions = DEFAULT_SUGGESTIONS

    // ---- 状態 ----
    @Volatile private var handle = 0L                    // 文章側のセッション
    @Volatile private var tokenizer: Gemma4Tokenizer? = null
    @Volatile private var loading = false
    @Volatile private var generating = false
    @Volatile private var reading = false                // 写真を読み取っている（前処理・画像エンコーダ）
    private var readyStrict = false
    private var photoUri: Uri? = null                    // カメラに渡した保存先
    private var featuresRel: String? = null              // 画像エンコーダの出力（files からの相対パス）
    private var vitStrict = false
    private var imageSec = 0.0                           // 読み取り全体（読み込み・前処理・画像エンコーダ）の秒
    private var vitNpuMs = 0.0                           // 画像エンコーダの NPU の時間（ort_run_ms）
    private var chatStarted = false                      // この写真について 1 問目を送ったか
    private var vitSeq = 0
    private val ui = Handler(Looper.getMainLooper())
    private var lastRev = -1L
    private var askAt = 0L
    private var firstTokenAt = 0L
    @Volatile private var genThread: Thread? = null
    private var autoQuestion: String? = null

    // ================================================================= 起動

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        loadConfig()
        setContentView(buildUi())
        setBusy()
        openTextModel()
        // カメラから戻る間に画面が作り直されたとき（メモリが足りず OS に止められたときなど）は、保存先を覚えておいて読み直す
        savedInstanceState?.getString(KEY_PHOTO)?.let { photoUri = Uri.parse(it) }
        autoQuestion = intent?.getStringExtra("question")
            ?: intent?.getStringExtra("question_file")?.let { rel ->       // adb で日本語を渡すと文字化けしうるので、ファイルでも受ける
                runCatching { File(getExternalFilesDir(null), rel).readText().trim() }.getOrNull()
            }
        val testImage = intent?.getStringExtra("image")
        when {
            testImage != null -> readPhoto(Uri.fromFile(File(getExternalFilesDir(null), testImage)))
            savedInstanceState?.getBoolean(KEY_HAS_PHOTO) == true && photoUri != null -> readPhoto(photoUri!!)
        }
    }

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        photoUri?.let { outState.putString(KEY_PHOTO, it.toString()) }
        outState.putBoolean(KEY_HAS_PHOTO, featuresRel != null)
    }

    override fun onDestroy() {
        super.onDestroy()
        ui.removeCallbacksAndMessages(null)
        val h = handle
        handle = 0L
        if (h != 0L) {
            NativeBridge.nativeCancel(h)
            val g = genThread
            thread(name = "edgeslm-vlm-close") {
                runCatching { g?.join(5000) }
                NativeBridge.nativeDestroy(h)
            }
        }
    }

    /** `files/jobs/vlm_demo.json` を読む（無ければ画面にそう出す。文章側のモデルの場所が分からないと動けない） */
    private fun loadConfig() {
        val f = File(getExternalFilesDir(null), CONFIG_REL)
        if (!f.isFile) return
        try {
            cfg = JSONObject(f.readText())
            systemPrompt = cfg.optString("system_prompt", DEFAULT_SYSTEM)
            cfg.optJSONArray("grid")?.let { grid = intArrayOf(it.getInt(0), it.getInt(1)) }
            nImageTokens = cfg.optInt("n_image_tokens", (grid[0] / 3) * (grid[1] / 3))
            maxNew = cfg.optInt("max_new", 96)
            cfg.optJSONArray("stop_ids")?.let { a -> stopIds = (0 until a.length()).map { a.getInt(it) } }
            cfg.optJSONArray("suggestions")?.let { a -> suggestions = (0 until a.length()).map { a.getString(it) } }
        } catch (t: Throwable) {
            Log.w(TAG, "config parse failed: ${t.message}")
        }
    }

    // ================================================================= 画面

    private fun dp(v: Float): Int =
        TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v, resources.displayMetrics).toInt()

    /** 押している間だけ色を変える角丸の背景（2026-09-30: 候補ボタンは押しても見た目が変わらず、押せたかが分からなかった） */
    private fun pressable(normal: Int, pressed: Int, radiusDp: Float): StateListDrawable = StateListDrawable().apply {
        addState(intArrayOf(android.R.attr.state_pressed), rounded(pressed, radiusDp))
        addState(intArrayOf(), rounded(normal, radiusDp))
    }

    /**
     * ★2026-09-30: タップが画面に届いているかを記録する（「後」の動画で、2 枚目の写真の後 16 秒ほどタップが効かなかった件の切り分け。
     * 画面に届いているのにボタンが動かないのか、そもそも届いていないのかを logcat で分けられるようにする）
     */
    override fun dispatchTouchEvent(ev: MotionEvent): Boolean {
        if (ev.actionMasked == MotionEvent.ACTION_DOWN) Log.i(TAG, "vlm touch: down x=${ev.x.toInt()} y=${ev.y.toInt()}")
        val handled = super.dispatchTouchEvent(ev)
        if (ev.actionMasked == MotionEvent.ACTION_UP) Log.i(TAG, "vlm touch: up handled=$handled")
        return handled
    }

    override fun onWindowFocusChanged(hasFocus: Boolean) {
        super.onWindowFocusChanged(hasFocus)
        Log.i(TAG, "vlm focus: $hasFocus")
    }

    private fun rounded(color: Int, radiusDp: Float): GradientDrawable =
        GradientDrawable().apply { setColor(color); cornerRadius = dp(radiusDp).toFloat() }

    private fun buildUi(): View {
        scroll = ScrollView(this).apply { setBackgroundColor(BG); isFillViewport = true }
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(dp(16f), dp(14f), dp(16f), dp(16f))
        }
        root.addView(TextView(this).apply {
            text = "画像で質問"
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 21f)
            setTypeface(typeface, Typeface.BOLD)
            setTextColor(INK)
        })
        root.addView(TextView(this).apply {
            text = "写真を撮って質問すると、インターネットを使わず、スマホの NPU だけで答えます"
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 12.5f)
            setTextColor(SUB)
        })
        statusView = TextView(this).apply {
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 12.5f)
            setTextColor(SUB)
            setPadding(0, dp(6f), 0, dp(8f))
        }
        root.addView(statusView)

        // 写真
        photoView = ImageView(this).apply {
            scaleType = ImageView.ScaleType.FIT_CENTER
            background = rounded(Color.parseColor("#E4E7EC"), 12f)
            adjustViewBounds = false
        }
        root.addView(photoView, LinearLayout.LayoutParams(MATCH_PARENT, dp(230f)))
        photoNote = TextView(this).apply {
            text = "まだ写真がありません"
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 12f)
            setTextColor(SUB)
            gravity = Gravity.CENTER_HORIZONTAL
            setPadding(0, dp(4f), 0, dp(4f))
        }
        root.addView(photoNote)
        cameraButton = Button(this).apply {
            text = "写真を撮る"
            isAllCaps = false
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 17f)
            setTextColor(Color.WHITE)
            background = pressable(ACCENT, Color.parseColor("#1A4FC4"), 10f)
            setOnClickListener { Log.i(TAG, "vlm click: camera"); takePhoto() }
        }
        root.addView(cameraButton, LinearLayout.LayoutParams(MATCH_PARENT, dp(52f)).apply { topMargin = dp(4f) })

        // 質問の候補
        val sugRow = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; setPadding(0, dp(12f), 0, dp(4f)) }
        for (s in suggestions) {
            val b = Button(this).apply {
                text = s
                isAllCaps = false
                setTextSize(TypedValue.COMPLEX_UNIT_SP, 14f)
                setTextColor(INK)
                background = pressable(Color.WHITE, Color.parseColor("#DCE6F8"), 18f)   // 押している間は色を変える（押せたかが分かるように）
                setOnClickListener { Log.i(TAG, "vlm click: suggestion '$s'"); ask(s) }
            }
            suggestionButtons.add(b)
            sugRow.addView(b, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT).apply { bottomMargin = dp(6f) })
        }
        root.addView(sugRow)

        // 入力欄
        val inRow = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
        questionEdit = EditText(this).apply {
            hint = "質問を入力（例: 何が写っていますか？）"
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 15f)
            imeOptions = EditorInfo.IME_ACTION_SEND
            setSingleLine(true)
            setOnEditorActionListener { _, actionId, _ ->
                if (actionId == EditorInfo.IME_ACTION_SEND) { ask(text.toString()); true } else false
            }
        }
        inRow.addView(questionEdit, LinearLayout.LayoutParams(0, WRAP_CONTENT, 1f))
        askButton = Button(this).apply {
            text = "質問する"
            isAllCaps = false
            setOnClickListener { Log.i(TAG, "vlm click: ask len=${questionEdit.text.length}"); ask(questionEdit.text.toString()) }
        }
        inRow.addView(askButton, LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT))
        stopButton = Button(this).apply {
            text = "停止"
            isAllCaps = false
            isEnabled = false
            setOnClickListener { if (handle != 0L) NativeBridge.nativeCancel(handle) }
        }
        inRow.addView(stopButton, LinearLayout.LayoutParams(WRAP_CONTENT, WRAP_CONTENT))
        root.addView(inRow)

        // 答え
        answerView = TextView(this).apply {
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 17f)
            setTextColor(INK)
            setTextIsSelectable(true)
            background = rounded(Color.WHITE, 12f)
            setPadding(dp(14f), dp(12f), dp(14f), dp(12f))
            text = "写真を撮ってから、質問を選ぶか入力してください"
        }
        root.addView(answerView, LinearLayout.LayoutParams(MATCH_PARENT, WRAP_CONTENT).apply { topMargin = dp(10f) })
        statsView = TextView(this).apply {
            setTextSize(TypedValue.COMPLEX_UNIT_SP, 11.5f)
            setTextColor(SUB)
            setPadding(dp(4f), dp(4f), dp(4f), 0)
        }
        root.addView(statsView)
        scroll.addView(root, MATCH_PARENT, WRAP_CONTENT)
        return scroll
    }

    /** ボタンを押せるかどうかと、上の 1 行の状態 */
    private fun setBusy() {
        val canAsk = handle != 0L && !loading && !generating && !reading && featuresRel != null
        cameraButton.isEnabled = !generating && !reading
        cameraButton.alpha = if (cameraButton.isEnabled) 1f else 0.5f
        askButton.isEnabled = canAsk
        suggestionButtons.forEach { it.isEnabled = canAsk; it.alpha = if (canAsk) 1f else 0.5f }   // 押せないときは薄く（見た目で分かるように）
        stopButton.isEnabled = generating
        statusView.text = when {
            cfg.length() == 0 -> "設定（files/$CONFIG_REL）がありません。PC で setup_vlm_demo を動かしてください"
            loading -> "文章のモデルを読み込み中…（数秒）"
            handle == 0L -> "文章のモデルを開けませんでした（上の設定とモデルの置き場を確かめてください）"
            reading -> "写真を読み取っています…"
            generating -> "答えを作っています…"
            else -> "準備できました（${if (readyStrict) "全部 NPU" else "一部 CPU"}）"
        }
    }

    // ================================================================= 文章側のモデル

    private fun jobBase(): JSONObject {
        val job = JSONObject().apply {
            put("job_id", "vlm_ui")
            put("task", "generate_kv")
            put("mask_neg", -60.0)
            put("ep", epJson())
        }
        val text = cfg.optJSONObject("text") ?: JSONObject()
        for (k in text.keys()) job.put(k, text.get(k))
        return job
    }

    private fun epJson() = JSONObject().apply {
        put("provider", "QNN")
        put("backend", "libQnnHtp.so")
        put("options", JSONObject().apply {
            put("offload_graph_io_quantization", "0")
            put("qnn_context_priority", "high")
        })
        put("session", JSONObject().apply {
            put("intra_op_num_threads", 1)
            put("log_severity_level", 3)
        })
    }

    private fun withPaths(job: JSONObject): String = JsonInject.addRuntimePaths(
        job.toString(), baseDir = getExternalFilesDir(null)!!.absolutePath,
        nativeLibDir = applicationInfo.nativeLibraryDir, overridesJson = null)

    private fun openTextModel() {
        if (cfg.length() == 0) { setBusy(); return }
        loading = true
        setBusy()
        thread(name = "edgeslm-vlm-open") {
            try {
                if (tokenizer == null) tokenizer = Gemma4Tokenizer.load(File(getExternalFilesDir(null), "tokenizer"))
                val t0 = System.currentTimeMillis()
                val h = NativeBridge.nativeCreate(withPaths(jobBase()))
                val loadMs = System.currentTimeMillis() - t0
                val d = JSONObject(NativeBridge.nativeDescribe(h))
                readyStrict = d.optBoolean("strict_no_cpu_fallback", false)
                Log.i(TAG, "vlm text ready: ok=${h != 0L} load_ms=$loadMs strict=$readyStrict error=${d.optString("error")}")
                ui.post {
                    handle = h
                    loading = false
                    setBusy()
                    maybeAutoAsk()
                }
                // ★Phase 5: 文章側を開き終えたら、画像エンコーダを別プロセスで先に開いておく（開いたまま＝keep_session）。
                //   写真を撮ったときは「動かすだけ」になり、読み取りが 1.1 秒 → 0.3 秒前後になる見込み。
                //   文章側を開き終えてから開くのは、開くときのメモリの山（文章側は一時的に約 3GB）を重ねないため
                if (h != 0L) warmVit()
            } catch (t: Throwable) {
                Log.e(TAG, "vlm text open failed", t)
                ui.post { loading = false; setBusy() }
            }
        }
    }

    // ================================================================= 写真

    private fun takePhoto() {
        if (generating || reading) return
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, "vlm_${System.currentTimeMillis()}.jpg")
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            if (Build.VERSION.SDK_INT >= 29) put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/VLMDemo")
        }
        val uri = contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
        if (uri == null) {
            answerView.text = "写真の保存先を作れませんでした"
            return
        }
        photoUri = uri
        val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE).apply {
            putExtra(MediaStore.EXTRA_OUTPUT, uri)
            addFlags(Intent.FLAG_GRANT_WRITE_URI_PERMISSION)
        }
        try {
            startActivityForResult(intent, REQ_CAMERA)
        } catch (t: Throwable) {
            answerView.text = "カメラアプリを開けませんでした: ${t.message}"
        }
    }

    @Deprecated("startActivityForResult の結果（androidx を使わないため従来の API）")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != REQ_CAMERA) return
        val uri = photoUri ?: return
        if (resultCode == RESULT_OK) {
            readPhoto(uri)
        } else {
            runCatching { contentResolver.delete(uri, null, null) }        // 撮らずに戻った: 空の保存先を消す
        }
    }

    /**
     * 写真を読み取る（裏のスレッド）: 読み込み・EXIF の向き → RGB → 前処理（C++）→ 画像エンコーダ（別プロセス）。
     * 終わったら特徴のファイルを覚えて、質問できる状態にする。
     */
    private fun readPhoto(uri: Uri) {
        reading = true
        featuresRel = null
        chatStarted = false
        answerView.text = "写真を読み取っています…"
        statsView.text = ""
        setBusy()
        thread(name = "edgeslm-vlm-read") {
            val t0 = System.currentTimeMillis()
            try {
                val bmp = loadBitmap(uri) ?: throw IllegalStateException("写真を読めませんでした")
                ui.post { photoView.setImageBitmap(bmp); photoNote.text = "${bmp.width}×${bmp.height}" }
                val base = getExternalFilesDir(null)!!
                File(base, "inputs/vlm").mkdirs()
                File(base, "out/vlm").mkdirs()
                // RGB のバイト列（行の詰め物なし）
                val w = bmp.width
                val h = bmp.height
                val px = IntArray(w * h)
                bmp.getPixels(px, 0, w, 0, 0, w, h)
                val rgb = ByteArray(w * h * 3)
                for (i in px.indices) {
                    val c = px[i]
                    rgb[3 * i] = (c shr 16).toByte()
                    rgb[3 * i + 1] = (c shr 8).toByte()
                    rgb[3 * i + 2] = c.toByte()
                }
                File(base, "inputs/vlm/cam.rgb").writeBytes(rgb)
                val t1 = System.currentTimeMillis()
                // 前処理（C++。PIL の BICUBIC と同じ）
                val prepJob = JSONObject().apply {
                    put("rgb", "inputs/vlm/cam.rgb"); put("width", w); put("height", h)
                    put("out", "inputs/vlm/cam_pix.bin")
                    put("grid", JSONArray().put(grid[0]).put(grid[1]))
                    put("runtime", JSONObject().put("base_dir", base.absolutePath))
                }
                val pr = JSONObject(NativeBridge.nativeVlmPrep(prepJob.toString()))
                if (!pr.optBoolean("ok", false)) throw IllegalStateException("前処理に失敗: $pr")
                val t2 = System.currentTimeMillis()
                // ★文章側のモデルを開いている途中なら、開き終わるまで待ってから画像エンコーダを動かす（2026-09-29）。
                //   モデルを開く間は一時的に約 3.1GB 使う（EdgeSLM_AndroidScale P2）。そこへ画像エンコーダ（別プロセス、
                //   約 0.3GB）の読み込みを重ねると、裏の :probe 側が OS に止められやすい。普段は起動直後に開き終わっているので
                //   待たないが、カメラの間に OS に止められて画面が作り直されたときは「読み直し」と「写真の読み取り」が重なる
                if (loading) {
                    ui.post { answerView.text = "文章のモデルを開き終わるのを待っています…" }
                    val until = System.currentTimeMillis() + OPEN_WAIT_MS
                    while (loading && System.currentTimeMillis() < until) Thread.sleep(50)
                    Log.i(TAG, "vlm read: waited for text model ${System.currentTimeMillis() - t2}ms (loading=$loading)")
                }
                val t2v = System.currentTimeMillis()
                // 画像エンコーダ（別プロセスの RunnerService2 に job を渡して、結果のファイルを待つ）
                val r = runVit(base)
                val rec = r.optJSONArray("records")?.optJSONObject(0)
                val outRel = rec?.optJSONArray("outputs")?.optJSONObject(0)?.optString("path")
                if (!r.optBoolean("ok", false) || outRel.isNullOrEmpty()) throw IllegalStateException("画像エンコーダに失敗: ${r.toString().take(300)}")
                vitNpuMs = rec?.optDouble("ort_run_ms", 0.0) ?: 0.0
                vitStrict = r.optJSONObject("describe")?.optBoolean("strict_no_cpu_fallback", false) ?: false
                val t3 = System.currentTimeMillis()
                imageSec = (t3 - t0 - (t2v - t2)) / 1000.0      // 文章側のモデルを待った分は含めない（読み取りそのものの時間）
                Log.i(TAG, String.format(java.util.Locale.ROOT,
                    "vlm image: %dx%d load_ms=%d prep_ms=%d wait_text_ms=%d vit_wall_ms=%d vit_npu_ms=%.1f strict=%b out=%s",
                    w, h, t1 - t0, t2 - t1, t2v - t2, t3 - t2v, vitNpuMs, vitStrict, outRel))
                ui.post {
                    featuresRel = outRel
                    reading = false
                    answerView.text = "写真を読み取りました。質問を選ぶか入力してください"
                    statsView.text = String.format(java.util.Locale.ROOT, "画像の読み取り %.1f 秒（NPU %.0fms）", imageSec, vitNpuMs)
                    setBusy()
                    maybeAutoAsk()
                }
            } catch (t: Throwable) {
                Log.e(TAG, "vlm read failed", t)
                ui.post {
                    reading = false
                    answerView.text = "（${t.message}）"
                    setBusy()
                }
            }
        }
    }

    /**
     * 画像エンコーダを別プロセス（:probe の RunnerService2）で**開くだけ**（Phase 5）。開いたままにする（keep_session）。
     * 結果は待たない（写真の読み取りの job と同じ結果のファイルを使うので、job_id で見分ける。RunnerService は job を 1 つずつ動かす）。
     * 失敗しても困らない（写真のときに開き直すだけ）。
     */
    private fun warmVit() {
        val base = getExternalFilesDir(null) ?: return
        val model = cfg.optString("vit_model")
        if (model.isEmpty()) return
        val job = JSONObject().apply {
            put("job_id", "vlm_vit_warm_${System.currentTimeMillis()}")
            put("task", "open")
            put("model", model)
            put("ep", epJson())
            put("keep_session", true)
        }
        try {
            File(base, "jobs").mkdirs()
            File(base, "jobs/vlm_vit_warm.json").writeText(job.toString())
            startForegroundService(Intent(this, RunnerService2::class.java).apply {
                action = RunnerActivity.ACTION_RUN
                putExtra(RunnerActivity.EXTRA_JOB, "jobs/vlm_vit_warm.json")
            })
            Log.i(TAG, "vlm vit warm: requested")
        } catch (t: Throwable) {
            Log.w(TAG, "vlm vit warm failed: ${t.message}")
        }
    }

    /** 画像エンコーダの job を別プロセスで動かし、結果（out/last_result.json）を待つ */
    private fun runVit(base: File): JSONObject {
        val jobId = "vlm_vit_${System.currentTimeMillis()}_${vitSeq++}"
        val job = JSONObject().apply {
            put("job_id", jobId)
            put("task", "infer")
            put("model", cfg.optString("vit_model"))
            put("ep", epJson())
            put("batches", JSONArray().put(JSONObject().apply {
                put("id", "cam")
                put("inputs", JSONObject().put("pixel_values", "inputs/vlm/cam_pix.bin"))
            }))
            put("output", JSONObject().put("dir", "out/vlm"))
            put("keep_session", true)                  // Phase 5: 開いたままにする（次の写真は動かすだけ）
        }
        File(base, "jobs").mkdirs()
        File(base, "jobs/vlm_vit.json").writeText(job.toString())
        val resultFile = File(base, "out/last_result.json")
        resultFile.delete()
        val svc = Intent(this, RunnerService2::class.java).apply {
            action = RunnerActivity.ACTION_RUN
            putExtra(RunnerActivity.EXTRA_JOB, "jobs/vlm_vit.json")
        }
        startForegroundService(svc)
        val deadline = System.currentTimeMillis() + VIT_TIMEOUT_MS
        while (System.currentTimeMillis() < deadline) {
            Thread.sleep(40)
            if (!resultFile.isFile) continue
            val txt = runCatching { resultFile.readText() }.getOrDefault("")
            val j = runCatching { JSONObject(txt) }.getOrNull() ?: continue          // 書いている途中
            if (j.optString("job_id") == jobId) return j
        }
        throw IllegalStateException("画像エンコーダの結果が ${VIT_TIMEOUT_MS / 1000} 秒で返りませんでした")
    }

    /**
     * 写真を Bitmap にする。EXIF の向きを反映し、透明部分は白で埋める（PC の hf_vlm.prepare_image と同じ扱い）。
     * 大きい写真（12MP など）は、読むときに 2 の累乗で縮める（短い辺が 672 以上になる範囲で。その後 C++ で BICUBIC）
     */
    private fun loadBitmap(uri: Uri): Bitmap? {
        val orientation = runCatching {
            contentResolver.openInputStream(uri)?.use { ExifInterface(it).getAttributeInt(
                ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_NORMAL) }
        }.getOrNull() ?: ExifInterface.ORIENTATION_NORMAL
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }
        if (bounds.outWidth <= 0 || bounds.outHeight <= 0) return null
        var sample = 1
        while (minOf(bounds.outWidth, bounds.outHeight) / (sample * 2) >= MIN_SHORT_SIDE) sample *= 2
        val opts = BitmapFactory.Options().apply {
            inSampleSize = sample
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        var bmp = contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, opts) } ?: return null
        val m = Matrix()
        when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> m.postRotate(90f)
            ExifInterface.ORIENTATION_ROTATE_180 -> m.postRotate(180f)
            ExifInterface.ORIENTATION_ROTATE_270 -> m.postRotate(270f)
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> m.postScale(-1f, 1f)
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> m.postScale(1f, -1f)
            ExifInterface.ORIENTATION_TRANSPOSE -> { m.postRotate(90f); m.postScale(-1f, 1f) }
            ExifInterface.ORIENTATION_TRANSVERSE -> { m.postRotate(270f); m.postScale(-1f, 1f) }
        }
        if (!m.isIdentity) bmp = Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, m, false)
        if (bmp.hasAlpha()) {
            val white = Bitmap.createBitmap(bmp.width, bmp.height, Bitmap.Config.ARGB_8888)
            Canvas(white).apply { drawColor(Color.WHITE); drawBitmap(bmp, 0f, 0f, null) }
            bmp = white
        }
        Log.i(TAG, "vlm bitmap: file=${bounds.outWidth}x${bounds.outHeight} sample=$sample exif=$orientation -> ${bmp.width}x${bmp.height}")
        return bmp
    }

    // ================================================================= 質問する

    private fun maybeAutoAsk() {
        val q = autoQuestion ?: return
        if (handle == 0L || loading || reading || featuresRel == null) return
        autoQuestion = null
        ui.postDelayed({ ask(q) }, 300)
    }

    /**
     * プロンプトの ID を作る（PC の hf_vlm.build_inputs と同じ並び）。
     * 1 問目: `<bos><|turn>system\n{system}<turn|>\n<|turn>user\n` ＋ `<|image>` ＋ 画像トークン × 70 ＋ `<image|>` ＋ `{質問}<turn|>\n<|turn>model\n`
     * 2 問目以降（同じ写真）: `<turn|>\n<|turn>user\n{質問}<turn|>\n<|turn>model\n`（差分だけ。前のキャッシュを使い回す）
     * 特殊トークンで区切られた所で分けてトークン化するので、全体を一度にトークン化したのと同じ ID になる。
     */
    private fun promptIds(tok: Gemma4Tokenizer, q: String, first: Boolean): IntArray {
        if (!first) return tok.encode(String.format(GEMMA_NEXT, q))
        val head = tok.encode(String.format(GEMMA_HEAD, systemPrompt))
        val tail = tok.encode(String.format(GEMMA_TAIL, q))
        val img = IntArray(nImageTokens) { IMAGE_TOKEN_ID }
        return head + intArrayOf(BOI_TOKEN_ID) + img + intArrayOf(EOI_TOKEN_ID) + tail
    }

    private fun ask(raw: String) {
        val q = raw.trim()
        // ★2026-09-30: 「押したのに何も起きない」を無くす。受け付けない理由を logcat に残し、入力が空なら画面にも出す
        val why = when {
            tokenizer == null -> "tokenizer not loaded"
            featuresRel == null -> "no photo"
            q.isEmpty() -> "empty question"
            generating -> "generating"
            loading -> "loading"
            reading -> "reading"
            handle == 0L -> "no text model"
            else -> null
        }
        if (why != null) {
            Log.i(TAG, "vlm ask ignored: $why")
            if (why == "empty question") statsView.text = "質問を入力するか、上の候補を押してください"
            return
        }
        val tok = tokenizer!!
        val feats = featuresRel!!
        val first = !chatStarted
        val ids = promptIds(tok, q, first)
        Log.i(TAG, "vlm prompt first=$first n=${ids.size} ids=${ids.joinToString(",")}")
        questionEdit.setText("")
        answerView.text = "「$q」\n考えています…"
        statsView.text = ""
        val job = jobBase().apply {
            put("max_new", maxNew)
            put("prompt_ids", JSONArray().also { a -> ids.forEach { a.put(it) } })
            put("stop_ids", JSONArray().also { a -> stopIds.forEach { a.put(it) } })
            put("image_features", feats)
            put("reuse_cache", true)
            put("reset_cache", first)
            put("prompt_is_delta", !first)
        }
        chatStarted = true
        generating = true
        setBusy()
        lastRev = runCatching { JSONObject(NativeBridge.nativeProgress()).optLong("rev", -1L) }.getOrDefault(-1L)
        askAt = System.currentTimeMillis()
        firstTokenAt = 0L
        val qShown = q
        ui.post(poll(qShown))
        val jobJson = withPaths(job)
        val h = handle
        genThread = thread(name = "edgeslm-vlm-generate") {
            val result = NativeBridge.nativeRunJob(h, jobJson)
            ui.post { finish(qShown, first, result) }
        }
    }

    /** 途中経過（50ms ごと）を読んで、答えを 1 トークンずつ表示する */
    private fun poll(q: String): Runnable = object : Runnable {
        override fun run() {
            if (!generating) return
            try {
                val p = JSONObject(NativeBridge.nativeProgress())
                val rev = p.optLong("rev", -1L)
                if (rev != lastRev) {
                    lastRev = rev
                    val arr = p.optJSONArray("ids")
                    val ids = ArrayList<Int>(arr?.length() ?: 0)
                    if (arr != null) for (i in 0 until arr.length()) ids.add(arr.getInt(i))
                    if (ids.isNotEmpty()) {
                        if (firstTokenAt == 0L) firstTokenAt = System.currentTimeMillis()
                        answerView.text = "「$q」\n" + clean(tokenizer?.decodeStreaming(ids) ?: "")
                    }
                }
            } catch (t: Throwable) {
                Log.w(TAG, "progress read failed: ${t.message}")
            }
            if (generating) ui.postDelayed(this, POLL_MS)
        }
    }

    private fun clean(s: String): String {
        var t = s
        for (x in STRIP) t = t.replace(x, "")
        return t.trim()
    }

    private fun finish(q: String, first: Boolean, result: String) {
        generating = false
        val endAt = System.currentTimeMillis()
        try {
            val r = JSONObject(result)
            if (!r.optBoolean("ok", false) && r.optString("error") != "cancelled") {
                answerView.text = "「$q」\n（エラー: ${r.optString("error", "unknown")}）"
                chatStarted = false
                setBusy()
                return
            }
            val arr = r.optJSONArray("new_ids")
            val ids = IntArray(arr?.length() ?: 0) { arr!!.getInt(it) }
            val cancelled = r.optString("error") == "cancelled"
            val ans = clean(tokenizer?.decode(ids) ?: "")
            // ★Phase 5: 上限（max_new）で止まり、終わりの印（EOS）で止まっていなければ、途中までだと分かるようにする
            val truncated = !cancelled && !r.optBoolean("stopped_by_eos", true) && ids.size >= maxNew
            answerView.text = "「$q」\n" + ans.ifEmpty { if (cancelled) "（停止しました）" else "（応答なし）" } +
                (if (truncated) TRUNC_MARK else "")
            val med = r.optJSONObject("step_ms_stats")?.optDouble("median") ?: 0.0
            val tps = if (med > 0) 1000.0 / med else 0.0
            val firstS = if (firstTokenAt > 0) (firstTokenAt - askAt) / 1000.0 else Double.NaN
            val npu = if (readyStrict && vitStrict) "全部 NPU" else "一部 CPU"
            statsView.text = String.format(java.util.Locale.ROOT,
                "%s最初の文字まで %.1f 秒 · %.1f トークン/秒 · %s",
                if (first) String.format(java.util.Locale.ROOT, "画像 %.1f 秒（NPU %.0fms）· ", imageSec, vitNpuMs) else "",
                firstS, tps, npu)
            Log.i(TAG, "vlm answer: ${ans.replace("\r", "").replace("\n", "⏎")}")
            val mt = r.optJSONObject("multi_token")                 // Phase 5: まとめ読みのモデル（無ければ 1 トークン版）
            Log.i(TAG, String.format(java.util.Locale.ROOT,
                "vlm stats: first=%b image_s=%.2f vit_npu_ms=%.1f first_s=%.2f total_s=%.2f median_ms=%.1f new=%d " +
                    "image_tokens_fed=%d abs_pos=%d cache_len=%d strict_text=%b strict_vit=%b Tm=%d prefill_runs=%d truncated=%b",
                first, imageSec, vitNpuMs, firstS, (endAt - askAt) / 1000.0, med, ids.size,
                r.optInt("image_tokens_fed", -1), r.optInt("abs_pos", -1), r.optInt("cache_len", -1), readyStrict, vitStrict,
                mt?.optInt("Tm", 1) ?: 1, mt?.optInt("prefill_runs", -1) ?: -1, truncated))
            // 窓（キャッシュの長さ）からあふれたら、次の質問は写真から作り直す（画像の位置が押し出されるため）。
            // まとめ読みのモデルは覚えていられるのが past の長さ（240）
            val absPos = r.optInt("abs_pos", 0)
            val cl = mt?.optInt("past_len", 0)?.takeIf { it > 0 } ?: r.optInt("cache_len", 0)
            if (cl in 1 until absPos || cancelled) chatStarted = false
        } catch (t: Throwable) {
            Log.w(TAG, "result parse failed: ${t.message}")
            chatStarted = false
        }
        setBusy()
        scroll.post { scroll.fullScroll(View.FOCUS_DOWN) }
    }

    companion object {
        private const val TAG = "EdgeSLMVlm"
        private const val CONFIG_REL = "jobs/vlm_demo.json"
        private const val KEY_PHOTO = "photo_uri"
        private const val KEY_HAS_PHOTO = "has_photo"
        private const val REQ_CAMERA = 41
        private const val POLL_MS = 50L
        private const val VIT_TIMEOUT_MS = 30_000L
        private const val OPEN_WAIT_MS = 60_000L          // 文章側のモデルが開き終わるのを待つ上限（普段は 2.7 秒前後）
        private const val TRUNC_MARK = " …（長いので途中まで）"       // Phase 5: 上限で切れた答えの印
        private const val MIN_SHORT_SIDE = 672           // 読むときに縮めても、短い辺はこれ以上残す（格子の短い辺 336 の 2 倍）
        private const val IMAGE_TOKEN_ID = 258880        // 画像の中身が入る位置（vlm_common.IMAGE_TOKEN_ID）
        private const val BOI_TOKEN_ID = 255999          // <|image>（画像の始まり）
        private const val EOI_TOKEN_ID = 258882          // <image|>（画像の終わり）
        private const val DEFAULT_SYSTEM = "画像についての質問に、日本語で簡潔に答えてください。"
        private const val GEMMA_HEAD = "<bos><|turn>system\n%1\$s<turn|>\n<|turn>user\n"
        private const val GEMMA_TAIL = "%1\$s<turn|>\n<|turn>model\n"
        private const val GEMMA_NEXT = "<turn|>\n<|turn>user\n%1\$s<turn|>\n<|turn>model\n"
        private val STRIP = listOf("<turn|>", "<|turn>", "<eos>")
        private val DEFAULT_SUGGESTIONS = listOf("この画像を説明してください。", "何が写っていますか？", "書かれている文字を読んでください。")
        private val BG = Color.parseColor("#F7F8FA")
        private val INK = Color.parseColor("#1F2430")
        private val SUB = Color.parseColor("#667085")
        private val ACCENT = Color.parseColor("#2563EB")
    }
}
