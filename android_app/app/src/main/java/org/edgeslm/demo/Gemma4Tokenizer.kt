package org.edgeslm.demo

import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Gemma 4 の BPE トークナイザ（端末側実装）。
 *
 * これがあることで「文 ⇄ トークン」も端末の中で完結する（それまでは PC がやっていた）。
 * 外部ライブラリは足していない。資材は APK に埋めず `files/tokenizer/` に置く
 * （差し替えに APK の再ビルドが要らないようにするため。Qwen2Tokenizer と同じ考え方）。
 *
 * Qwen2（byte-level BPE）とは**別物**なので別クラスにした:
 *
 * | | Qwen2 | Gemma 4 |
 * |---|---|---|
 * | 正規化 | ByteLevel の写像 | 半角スペース → `▁` だけ |
 * | 前処理 | 正規表現で細かく割る | 割らない（実質そのまま BPE） |
 * | 語彙に無い文字 | ByteLevel なので起きない | **UTF-8 の1バイトずつ `<0xNN>`**（バイトフォールバック） |
 * | 語彙 | 151,936 | **262,144**、マージ 514,906 |
 *
 * 資材（`app/pipeline/export_tokenizer.py` が書き出す。PC 側の参照実装と**同じ手順**）:
 *
 *   vocab.bin   [V:int32][offset:(V+1)×int32][UTF-8 のバイト列]  ← 語に改行が入るので行区切りは使えない
 *   merges.bin  [M:int32][左のID:int32, 右のID:int32]×M          ← 並び順がそのまま優先度
 *   specials.tsv  ID<TAB>内容（追加トークン。長い順）
 *
 * PC の HF と ID 列が一致することは `golden.jsonl` で端末上で確かめる（`TokenizerCheck`）。
 */
class Gemma4Tokenizer private constructor(
    private val pieces: Array<String>,
    private val ids: HashMap<String, Int>,
    private val ranks: HashMap<Long, Int>,          // (左ID shl 32 or 右ID) → 優先度
    private val specials: List<Pair<String, Int>>,  // 内容 → ID（長い順）
    private val byte0: Int,                         // `<0x00>` の ID
) {

    // ---------------------------------------------------------------- encode

    /** 文字列 → トークンID列。追加トークン（`<|turn>` など）は書かれていればそのまま1つの ID になる。 */
    fun encode(text: String): IntArray {
        val out = ArrayList<Int>()
        var i = 0
        while (i < text.length) {
            val hit = specials.firstOrNull { text.startsWith(it.first, i) }
            if (hit != null) {
                out.add(hit.second)
                i += hit.first.length
                continue
            }
            // 次の追加トークンまでを普通の文として扱う
            var j = text.length
            for ((content, _) in specials) {
                val k = text.indexOf(content, i)
                if (k >= 0 && k < j) j = k
            }
            bpe(text.substring(i, j).replace(' ', '▁'), out)
            i = j
        }
        return out.toIntArray()
    }

    /** 1語（スペースを `▁` に置換済み）を ID 列にして `out` へ足す。 */
    private fun bpe(word: String, out: MutableList<Int>) {
        if (word.isEmpty()) return
        val syms = ArrayList<Int>(word.length)
        var i = 0
        while (i < word.length) {
            // ★サロゲートペア（絵文字など）は2文字で1つの符号位置
            val cp = word.codePointAt(i)
            val s = String(Character.toChars(cp))
            i += Character.charCount(cp)
            val id = ids[s]
            if (id != null) {
                syms.add(id)
            } else {
                for (b in s.toByteArray(Charsets.UTF_8)) syms.add(byte0 + (b.toInt() and 0xFF))
            }
        }
        while (syms.size > 1) {
            var best = Int.MAX_VALUE
            var at = -1
            for (k in 0 until syms.size - 1) {
                val r = ranks[key(syms[k], syms[k + 1])] ?: continue
                if (r < best) { best = r; at = k }
            }
            if (at < 0) break
            val merged = ids[pieces[syms[at]] + pieces[syms[at + 1]]] ?: break
            syms[at] = merged
            syms.removeAt(at + 1)
        }
        out.addAll(syms)
    }

    // ---------------------------------------------------------------- decode

    /** トークンID列 → 文字列。`<0xNN>` はバイトに戻してから UTF-8 として読む。 */
    fun decode(ids2: IntArray, skipSpecial: Boolean = true): String {
        val buf = java.io.ByteArrayOutputStream()
        for (id in ids2) {
            if (id < 0 || id >= pieces.size) continue
            val p = pieces[id]
            if (skipSpecial && specials.any { it.second == id }) continue
            val b = byteFallback(p)
            if (b >= 0) buf.write(b)
            else buf.write(p.replace('▁', ' ').toByteArray(Charsets.UTF_8))
        }
        return String(buf.toByteArray(), Charsets.UTF_8)
    }

    /** `<0xNN>` なら 0..255、そうでなければ −1。 */
    private fun byteFallback(p: String): Int {
        if (p.length != 6 || !p.startsWith("<0x") || p[5] != '>') return -1
        return try { p.substring(3, 5).toInt(16) } catch (e: NumberFormatException) { -1 }
    }

    /** 生成の途中表示用。末尾で切れた UTF-8 の断片は捨てる（文字化けを見せない）。 */
    fun decodeStreaming(ids2: List<Int>): String {
        var s = decode(ids2.toIntArray(), skipSpecial = true)
        while (s.isNotEmpty() && s.last() == '�') s = s.dropLast(1)
        return s
    }

    val vocabSize: Int get() = pieces.size

    fun idOf(piece: String): Int = ids[piece] ?: -1

    companion object {
        private fun key(a: Int, b: Int): Long = (a.toLong() shl 32) or (b.toLong() and 0xFFFFFFFFL)

        /** `files/tokenizer/` から読み込む。読み込みは 1〜2秒（語彙 262,144・マージ 514,906）。 */
        fun load(dir: File): Gemma4Tokenizer {
            val vb = dir.resolve("vocab.bin").readBytes()
            val v = ByteBuffer.wrap(vb).order(ByteOrder.LITTLE_ENDIAN)
            val n = v.int
            val offs = IntArray(n + 1) { v.int }
            val base = 4 + 4 * (n + 1)
            val pieces = Array(n) { String(vb, base + offs[it], offs[it + 1] - offs[it], Charsets.UTF_8) }
            val ids = HashMap<String, Int>(n * 2)
            for (i in 0 until n) ids[pieces[i]] = i
            val mb = dir.resolve("merges.bin").readBytes()
            val m = ByteBuffer.wrap(mb).order(ByteOrder.LITTLE_ENDIAN)
            val cnt = m.int
            val ranks = HashMap<Long, Int>(cnt * 2)
            for (k in 0 until cnt) {
                val a = m.int
                val b = m.int
                ranks[key(a, b)] = k
            }
            val specials = ArrayList<Pair<String, Int>>()
            dir.resolve("specials.tsv").forEachLine { line ->
                if (line.isNotEmpty()) {
                    val t = line.indexOf('\t')
                    if (t > 0) specials.add(line.substring(t + 1) to line.substring(0, t).toInt())
                }
            }
            specials.sortByDescending { it.first.length }       // 長い方を優先して切り出す
            val byte0 = ids["<0x00>"] ?: error("バイトフォールバックの語 <0x00> が無い")
            return Gemma4Tokenizer(pieces, ids, ranks, specials, byte0)
        }
    }
}
