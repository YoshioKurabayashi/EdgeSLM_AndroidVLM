package org.edgeslm.demo

import android.content.Intent

/**
 * **別プロセスで動く2つ目の入口**（検証用、2026-09-21）。
 *
 * 目的はひとつだけ: **1.5GB の NPU グラフを2プロセスで同時に開けるか**を確かめること。
 * NPU（HTP）の上限（この端末で約1,710MiB）は**プロセス単位**とされているので、
 * 別プロセスならもう1つ開けるはず——それが本当かを実機で測る。
 *
 * 通れば「入力を食べる用（まとめ食い）」と「答えを作る用」を常駐させられ、
 * 持ち替えの 2.5秒が要らなくなる（P6 2.5節）。
 *
 * 使い方（`:probe` プロセスで動く）:
 *
 *     adb shell am start -n org.edgeslm.gemma4/org.edgeslm.demo.RunnerActivity2 \
 *         -a org.edgeslm.demo.RUN --es job jobs/probe_b.json
 *
 * 本体（RunnerActivity / RunnerService）は既定のプロセスのまま。ここは**中身を足さず**、
 * マニフェストで `android:process=":probe"` を付けるためだけの薄い継承。
 *
 * 結果（EdgeSLM_AndroidGemma4 の results/P6_20260921_latency.md 2.6。PC 側の道具は app/pipeline/probe_two_proc.py）:
 *   233MiB＋1,467MiB（合計 約1.7GB）は 2 プロセスで両方開けた＝NPU の上限はプロセスごと。
 *   1,467MiB＋1,467MiB（合計 約3GB）は裏のプロセスが lowmemorykiller に止められた。壁は NPU ではなく RAM。
 *   → この端末では「2 つ常駐させて持ち替えを無くす」はできない。この入口は記録のために残してある。
 * 書き方の注意: `-n` の前半はアプリの ID（applicationId）、後半はクラスの完全な名前（namespace は org.edgeslm.demo のまま）。
 * （2026-10-18 学習用に追記）
 */
class RunnerService2 : RunnerService()

/** `:probe` プロセスで動く入口。中身は親の RunnerActivity と同じで、起こす Service だけを RunnerService2 に替える。 */
class RunnerActivity2 : RunnerActivity() {
    /** 親は `RunnerService` を起動するので、こちらは `:probe` 側の Service を起こす。 */
    override fun serviceIntent(): Intent = Intent(this, RunnerService2::class.java)
}
