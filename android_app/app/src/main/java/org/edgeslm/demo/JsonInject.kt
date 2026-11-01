package org.edgeslm.demo

import org.json.JSONObject

/**
 * ジョブ仕様 JSON に実行時パスを注入し、extras による上書きを適用する。
 *
 * ここは「ロジック」ではなくグルーである。ネイティブ側は端末固有のパスを知りようがないため、
 * Kotlin 側でしか取得できない2つのパスだけを渡す。
 *
 *  - base_dir       : getExternalFilesDir()。モデル・入力・結果の基準
 *  - native_lib_dir : applicationInfo.nativeLibraryDir。
 *                     **FastRPC が libQnnHtp.so の実ファイルパスを要求する**ため必須（C-6）
 *
 * 使う段: RunnerService が job を C++ の runner へ渡す直前（EdgeSLM_AndroidDemo の StudyMemo 4.1「JNI の境目は JSON の文字列だけ」）。
 * 入力: job の JSON 文字列・2 つのパス・am start の `--es overrides`（JSON。無ければ null）。
 * 出力: パスを入れ、上書きを重ねた JSON 文字列（nativeCreate・nativeRunJob にそのまま渡す）。
 * （2026-10-18 学習用に追記）
 */
object JsonInject {

    /**
     * job に `runtime.base_dir`・`runtime.native_lib_dir` を書き込み、overrides があれば重ねて返す。
     *
     * @param jobJson job の JSON（端末の files/jobs/ に置いたファイルの中身）
     * @param baseDir getExternalFilesDir(null) の実パス（/sdcard/Android/data/<アプリの ID>/files）。C++ は相対パスをここから解く
     * @param nativeLibDir .so が展開されたフォルダ。useLegacyPackaging=true のときだけ実ファイルになる（app/build.gradle の C-6）
     * @param overridesJson 上書き用の JSON（RunnerActivity の `--es overrides`）。null・空白なら何もしない
     * @return 注入・上書き後の JSON 文字列
     */
    fun addRuntimePaths(
        jobJson: String,
        baseDir: String,
        nativeLibDir: String,
        overridesJson: String?
    ): String {
        val job = JSONObject(jobJson)

        // ① job に runtime が既にあればそれを使い（中の他の項目は残す）、無ければ新しく作って 2 つのパスを入れる
        val runtime = job.optJSONObject("runtime") ?: JSONObject()
        runtime.put("base_dir", baseDir)
        runtime.put("native_lib_dir", nativeLibDir)
        job.put("runtime", runtime)

        // ② 上書きは最後に重ねる。runtime も上書きの対象なので、パスを差し替えることもできる
        if (!overridesJson.isNullOrBlank()) {
            mergeInto(job, JSONObject(overridesJson))
        }
        return job.toString()
    }

    /** 浅いマージではなくオブジェクトは再帰的に上書きする */
    // 両方がオブジェクトの鍵は中へ降りて重ね、それ以外（数値・文字列・配列）は丸ごと置き換える。
    //   例: job の {"ep":{"a":1,"b":2}} に {"ep":{"b":3}} を重ねると {"ep":{"a":1,"b":3}}。
    //   DemoActivity の demo_override.json（浅いマージ。入れ子は丸ごと差し替え）とは作りが違う点に注意。
    private fun mergeInto(target: JSONObject, patch: JSONObject) {
        for (key in patch.keys()) {
            val pv = patch.get(key)
            val tv = target.opt(key)
            if (pv is JSONObject && tv is JSONObject) {
                mergeInto(tv, pv)
            } else {
                target.put(key, pv)
            }
        }
    }
}
