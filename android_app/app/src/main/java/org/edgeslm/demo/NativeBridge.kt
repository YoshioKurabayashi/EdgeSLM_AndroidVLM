package org.edgeslm.demo

/**
 * JNI 境界。
 *
 * 設計方針（基本設計③ 6.2節）:
 *   引数・戻り値を **JSON 文字列に統一**している。パラメータを1つ増やすたびに
 *   シグネチャが変わると、そのたびに APK の再ビルドが必要になるため。
 *   JSON なら変更が C++ 側に閉じ、この Kotlin ファイルは触らずに済む。
 *
 *   ビルドは（JVM の通信制約により）ユーザー実行になるため、
 *   再ビルドを要する変更を減らすことが設計上の最優先事項である。
 *
 * 使う段: すべての画面と検証用の RunnerService が C++ の runner を呼ぶ、ただ 1 つの入口。
 *   中身は C++ の edgeslm_jni.cpp（JSON と C++ の呼び出しを変換するだけ）→ runner/session.cpp・runner/job.cpp。
 * 呼ぶ順番: nativeCreate（モデルを開く）→ nativeRunJob（何度でも）→ nativeDestroy（閉じる）。
 *   生成の間は別のスレッドから nativeProgress を読み、止めたいときは nativeCancel を呼ぶ。
 * 入力・出力: どれも JSON の文字列（ハンドルは C++ の Handle（Session と中断の印）の番地を Long にしたもの）。
 * （2026-10-18 学習用に追記）
 */
object NativeBridge {

    init {
        // libedgeslm_runner.so（app/build.gradle の CMake が作る）を、この object を最初に使ったときに 1 回だけ読み込む
        System.loadLibrary("edgeslm_runner")
    }

    /**
     * ORT / QNN のセッションを生成する。
     * @param configJson モデルパス・EP オプション・セッション設定を含む JSON
     * @return セッションハンドル。0 なら失敗（理由は describe/エラー JSON で取得）
     * 開けなかった理由は nativeDescribe(0) で読める（C++ 側が直近の理由を覚えておく）。
     * strict（CPU への逃がしを禁止）で開き、error 1002 なら少し待って開き直す（session.cpp。EdgeSLM_AndroidDemo の StudyMemo 4.3）。
     * モデルの読み込みに数秒かかる（EdgeSLM_AndroidDemo の README: Qwen2.5-0.5B の 772MB で 0.9〜3.6 秒）ので、呼ぶ側は画面のスレッドとは別のスレッドで呼ぶ。
     */
    external fun nativeCreate(configJson: String): Long

    /**
     * バックエンド名・グラフ配置・入出力情報を JSON で返す。
     * セッション生成に失敗した場合もエラー内容を JSON で返す。
     * handle=0 を渡すと、直前に開けなかった理由（"ok":false と "error":"describe: open failed: …"）を返す。
     */
    external fun nativeDescribe(handle: Long): String

    /**
     * ジョブを実行し、結果を JSON で返す。
     * task（probe / bench / op_check / infer / generate / generate_kv）は jobJson の中で指定する。
     * ★2026-10-18 訂正: 旧一覧の soak は実装されておらず（job.cpp の is_supported_task）、infer・generate_kv が抜けていた。
     */
    // 生成が終わるまで戻らない（同期の呼び出し）。始めに中断の印を下ろすので、前の nativeCancel は持ち越さない。
    // 知らない task は "ok":false と「… is not implemented」を返す（黙って別の動きをしない）。
    // tokenize・memprobe は C++ の job ではなく、RunnerService が Kotlin の側で分けて扱う。
    // このアプリ（VLM）では、vlm_prep（下の nativeVlmPrep）と open（開くだけ）も RunnerService が Kotlin の側で扱う。
    external fun nativeRunJob(handle: Long, jobJson: String): String

    /**
     * 生成の途中経過を JSON で返す（task=generate 実行中に別スレッドから読む）。
     * nativeRunJob は生成が終わるまで戻らないため、逐次表示はこちらで行う。
     * 返す JSON: running / done / ids[] / n / last_ms / median_ms / median_ort_ms /
     *            tokens_per_sec / rev（前回から進んだかの判定用）
     * ハンドルを取らないのは、途中経過がプロセスに 1 つしかないため（cpp/runner/progress.h）。
     * 画面は 50ms ごとに読み、rev が変わったときだけ中身を使う（EdgeSLM_AndroidDemo の StudyMemo 4.2）。
     */
    external fun nativeProgress(): String

    /** 実行中の job の中断要求（generate・generate_kv の生成ループ、infer・op_check などの繰り返しが止まる。★2026-10-18 訂正: 旧「soak / generate の中断要求」。soak は実装されていない） */
    // 印を立てるだけ。止まるのは、ループが次にその印を見たとき（すぐには止まらない）。handle=0 なら何もせず false。
    external fun nativeCancel(handle: Long): Boolean

    /** セッションを破棄する */
    // NPU のメモリを返す。1 プロセスで NPU に持てるのは約 1,710MiB なので、別のモデルを開く前に必ず閉じる
    //   （EdgeSLM_AndroidScale P0A。EdgeSLM_AndroidDemo の StudyMemo 4.3「閉じてから開く」）。
    external fun nativeDestroy(handle: Long)

    /**
     * 1プロセスに NPU のモデルを順に開いて保持し、メモリの推移を記録する（EdgeSLM_AndroidScale Phase 0-A）。
     * **ハンドルを取らない。** モデルを開く前の状態から測るため、開閉は C++ 側が自分で行う。
     * 途中経過は out/memprobe_<job_id>_steps.jsonl / _samples.csv に逐次書かれる。
     */
    external fun nativeMemProbe(jobJson: String): String

    /**
     * VLM の画像の前処理（EdgeSLM_AndroidVLM Phase 3）。RGB のバイト列のファイル → 画像エンコーダの入力 pixel_values。
     * PC の評価（PIL の BICUBIC で縮めて黒い余白に貼り、16×16 のパッチにして ×1/255）とビット単位で同じ値を作る。
     * **ハンドルを取らない**（モデルを使わない）。job の形は cpp/runner/vlm_prep.h
     * 使う段: 写真から答えまでの最初の段（EdgeSLM_AndroidVLM の StudyMemo 9・11 章）。画面（VlmActivity）と RunnerService の task=vlm_prep が呼ぶ。
     */
    external fun nativeVlmPrep(jobJson: String): String

    /** ORT / QNN の版を JSON で返す（FR-12 の記録用。セッション不要） */
    external fun nativeVersions(): String
}
