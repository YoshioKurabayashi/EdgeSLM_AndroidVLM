package org.edgeslm.demo

import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * 端末側トークナイザが HuggingFace と同じ ID 列を出すかを**端末上で**確かめる。
 *
 * ## なぜ端末で測るのか
 *
 * ホスト側で「同じアルゴリズムを Python に移植して突き合わせる」ことはできるが、
 * それは**移植が正しい前提**での確認にすぎない。実際に動くのは Android 上の
 * Kotlin + ICU 正規表現 + `java.text.Normalizer` であって、そこで測らないと
 * 「ホストで通ることと端末で動くことは別」という本プロジェクトで何度も踏んだ型に
 * またはまる（C-7 / C-10 / I/O 型 / err 1002）。
 *
 * 特に ICU 正規表現は素の OpenJDK と別物である（`(?U)` が構文エラーになる等）。
 * **ここを端末で通しておくと、その差が効いていないことの根拠になる。**
 *
 * ## 使い方（adb から）
 *
 *     files/tokenizer/golden.jsonl  ... host/make_tokenizer_assets.py が生成
 *     files/jobs/tokenize.json      ... {"job_id":"tok","task":"tokenize"}
 *
 *     am start -n org.edgeslm.demo/.RunnerActivity -a org.edgeslm.demo.RUN \
 *              --es job jobs/tokenize.json
 *
 * 結果は `files/out/last_result.json` に出る。**モデルは開かない**
 * （トークナイズに NPU は要らないため）。
 *
 * 結果: Qwen2.5 の辞書で encode 31/31・decode の往復 31/31（EdgeSLM_AndroidDemo の results/D2_20260913_tokenizer_verify.md）。
 * Gemma 4 の辞書（kind=gemma4）は 61/61・往復 61/61（EdgeSLM_AndroidGemma4 の results/P4_20260920_device_only.md 3 節）。
 * ★2026-10-18 補足: 上の使い方の `org.edgeslm.demo/.RunnerActivity` はデモアプリの書き方。複製したアプリでは
 *   `-n org.edgeslm.gemma4/org.edgeslm.demo.RunnerActivity` のようにクラスを完全な名前で書く（P4 の 3 節の例）。
 * 入力: job の tokenizer_dir（既定 tokenizer）・golden（既定 golden.jsonl）・texts（その場でトークナイズする文の配列。省略可）。
 * 出力: "ok"・"total"・"pass"・"fail"・"decode_total"・"decode_pass"・"failures"（と texts があれば "encoded"）を持つ JSONObject。
 *   RunnerService が files/out/ に書く。
 * （2026-10-18 学習用に追記）
 */
object TokenizerCheck {

    const val TAG = "EdgeSLMRunner"

    /**
     * golden.jsonl の各行 `{"name":..., "text":..., "ids":[...]}` を照合する。
     *
     * 合否は**ID 列の完全一致**で判定する。部分一致や長さ一致では意味がない
     * （1トークンずれれば以降の生成が別物になる）。
     *
     * @param baseDir getExternalFilesDir(null)（files/）
     * @param job tokenize の job（task は RunnerService が見て、ここへ回してくる）
     * @return 照合の結果（上の「出力」）。辞書が読めなければ "ok":false と "error"
     * 手順: ① 辞書を読む → ② texts があればその場でトークナイズ → ③ golden と ID 列を照合 → ④ decode の往復を照合
     */
    fun run(baseDir: File, job: JSONObject): JSONObject {
        val result = JSONObject()
        result.put("job_id", job.optString("job_id", "tokenize"))
        result.put("task", "tokenize")

        // 辞書の置き場と golden（正しい ID 列。PC の HuggingFace のトークナイザで作ったもの）の場所。辞書は files/ から、golden は辞書の置き場からの相対
        val tokDir = File(baseDir, job.optString("tokenizer_dir", "tokenizer"))
        val golden = File(tokDir, job.optString("golden", "golden.jsonl"))

        // ① 辞書を読む（kind で Qwen2 と Gemma 4 を選ぶ）。読み込めなければ "ok":false で返す
        // ★`kind` でトークナイザを選ぶ。既定は Qwen2（従来のジョブをそのまま動かすため）。
        //   Gemma 4 は BPE の作りが違う（正規化・バイトフォールバック・語彙 262,144）ので別クラス。
        val kind = job.optString("kind", "qwen2")
        result.put("kind", kind)
        val encode: (String) -> IntArray
        val decode: (List<Int>) -> String
        val nfc: Boolean                                  // 復号の比較で NFC 正規化を通すか
        try {
            if (kind == "gemma4") {
                val g = Gemma4Tokenizer.load(tokDir)
                result.put("vocab_size", g.vocabSize)
                encode = { s -> g.encode(s) }
                decode = { ids -> g.decode(ids.toIntArray(), skipSpecial = false) }
                nfc = false                               // Gemma は正規化で文字を変えない（空白 → ▁ のみ）
            } else {
                val q = Qwen2Tokenizer.load(tokDir)
                encode = { s -> q.encode(s) }
                decode = { ids -> q.decode(ids) }
                nfc = true
            }
        } catch (t: Throwable) {
            result.put("ok", false)
            result.put("error", "tokenizer load failed: ${t.message}")
            return result
        }

        // ② その場でのトークナイズ（照合はしない）
        // 任意の文字列をその場でトークナイズするだけの使い方も残す（デバッグ用）
        val texts = job.optJSONArray("texts")
        if (texts != null) {
            val arr = JSONArray()
            for (i in 0 until texts.length()) {
                val s = texts.getString(i)
                val o = JSONObject()
                o.put("text", s)
                o.put("ids", JSONArray(encode(s).toList()))
                arr.put(o)
            }
            result.put("encoded", arr)
        }

        // golden が無いとき: texts を処理できていれば成功（ok=true）、どちらも無ければ失敗にする
        if (!golden.isFile) {
            result.put("ok", texts != null)
            if (texts == null) result.put("error", "golden not found: ${golden.absolutePath}")
            return result
        }

        // ③ golden の 1 行 = 1 件。ID 列が 1 つでも違えば不合格にし、failures に中身を入れる
        var total = 0
        var pass = 0
        val failures = JSONArray()
        golden.forEachLine { line ->
            if (line.isBlank()) return@forEachLine
            val o = JSONObject(line)
            val text = o.getString("text")
            val expArr = o.getJSONArray("ids")
            val expected = IntArray(expArr.length()) { expArr.getInt(it) }
            val got = encode(text)
            total += 1
            if (got.contentEquals(expected)) {
                pass += 1
            } else {
                // 失敗は**中身を出す**。件数だけでは直せない。
                val f = JSONObject()
                f.put("name", o.optString("name"))
                f.put("text", text)
                f.put("expected", JSONArray(expected.toList()))
                f.put("got", JSONArray(got.toList()))
                failures.put(f)
            }
        }

        // ④ decode の往復の照合（golden の ID 列を復号し、元の文と比べる）
        // 往復（encode → decode）も見る。decode が壊れていれば表示が壊れる。
        var roundTrip = 0
        var roundTripOk = 0
        golden.forEachLine { line ->
            if (line.isBlank()) return@forEachLine
            val o = JSONObject(line)
            val text = o.getString("text")
            val expArr = o.getJSONArray("ids")
            val ids = (0 until expArr.length()).map { expArr.getInt(it) }
            roundTrip += 1
            // NFC 正規化を通すので、比較対象も正規化後の文字列にする
            //   （Qwen2 は分ける前に NFC で文字をそろえる。Gemma 4 は nfc=false で、元の文のまま比べる）
            val want = if (nfc) java.text.Normalizer.normalize(text, java.text.Normalizer.Form.NFC) else text
            if (decode(ids) == want) roundTripOk += 1
        }

        // 合格は encode と decode の両方が全件一致のときだけ
        result.put("ok", pass == total && roundTripOk == roundTrip)
        result.put("total", total)
        result.put("pass", pass)
        result.put("fail", total - pass)
        result.put("decode_total", roundTrip)
        result.put("decode_pass", roundTripOk)
        result.put("failures", failures)
        Log.i(TAG, "tokenize check($kind): encode $pass/$total, decode $roundTripOk/$roundTrip")
        return result
    }
}
