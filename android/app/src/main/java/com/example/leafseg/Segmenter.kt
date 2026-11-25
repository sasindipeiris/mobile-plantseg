package com.example.leafseg

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Rect
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.Closeable
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import kotlin.math.ceil
import kotlin.math.roundToInt

class SegResult(
    val mask: Bitmap, val top: Int, val topShare: Float, val second: Int, val secondShare: Float,
    val lesionFrac: Float, val ms: Double, val backend: String,
)

/** Input: float RGB [0,1], letterboxed top-left with black padding (as leafseg/utils.py). Output: logits at 1/4. */
class Segmenter(context: Context, asset: String, useGpu: Boolean, threads: Int = 4) : Closeable {
    private var gpu: GpuDelegate? = null
    private val interp: Interpreter
    val backend: String
    private val size: Int
    private val nchw: Boolean
    private val classes: Int
    private val oh: Int
    private val ow: Int
    private val outNchw: Boolean
    private val input: ByteBuffer
    private val output: ByteBuffer
    private val px: IntArray

    init {
        val model = context.assets.openFd(asset).use { fd ->
            FileInputStream(fd.fileDescriptor).channel.use {
                it.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
            }
        }
        var interp: Interpreter? = null
        if (useGpu) {
            try {
                gpu = GpuDelegate()
                interp = Interpreter(model, Interpreter.Options().addDelegate(gpu))
            } catch (e: Throwable) {
                gpu?.close(); gpu = null
            }
        }
        this.interp = interp ?: Interpreter(model, Interpreter.Options().setNumThreads(threads))
        backend = if (gpu != null) "GPU" else "CPU x$threads"

        val inT = this.interp.getInputTensor(0)
        require(inT.dataType() == DataType.FLOAT32) { "expected float32 input" }
        val s = inT.shape()
        nchw = s[1] == 3
        size = if (nchw) s[2] else s[1]
        val o = this.interp.getOutputTensor(0).shape()
        outNchw = o[2] == o[3]
        classes = if (outNchw) o[1] else o[3]
        oh = if (outNchw) o[2] else o[1]
        ow = if (outNchw) o[3] else o[2]
        input = ByteBuffer.allocateDirect(4 * 3 * size * size).order(ByteOrder.nativeOrder())
        output = ByteBuffer.allocateDirect(4 * classes * oh * ow).order(ByteOrder.nativeOrder())
        px = IntArray(size * size)
    }

    fun segment(src: Bitmap, runs: Int = 5): SegResult {
        val s = size.toFloat() / maxOf(src.width, src.height)
        val nw = (src.width * s).roundToInt()
        val nh = (src.height * s).roundToInt()
        val lb = Bitmap.createBitmap(size, size, Bitmap.Config.ARGB_8888)
        Canvas(lb).drawBitmap(src, null, Rect(0, 0, nw, nh), Paint(Paint.FILTER_BITMAP_FLAG))
        lb.getPixels(px, 0, size, 0, 0, size, size)
        lb.recycle()

        input.rewind()
        val fb = input.asFloatBuffer()
        if (nchw) {
            for (sh in intArrayOf(16, 8, 0)) for (p in px) fb.put(((p shr sh) and 0xFF) / 255f)
        } else {
            for (p in px) {
                fb.put(((p shr 16) and 0xFF) / 255f); fb.put(((p shr 8) and 0xFF) / 255f); fb.put((p and 0xFF) / 255f)
            }
        }

        input.rewind(); output.rewind(); interp.run(input, output)
        var total = 0L
        repeat(runs) {
            input.rewind(); output.rewind()
            val t0 = System.nanoTime()
            interp.run(input, output)
            total += System.nanoTime() - t0
        }

        val n = oh * ow
        output.rewind()
        val ob = output.asFloatBuffer()
        val best = FloatArray(n) { Float.NEGATIVE_INFINITY }
        val arg = IntArray(n)
        if (outNchw) {
            for (c in 0 until classes) for (i in 0 until n) {
                val v = ob.get(c * n + i)
                if (v > best[i]) { best[i] = v; arg[i] = c }
            }
        } else {
            for (i in 0 until n) for (c in 0 until classes) {
                val v = ob.get(i * classes + c)
                if (v > best[i]) { best[i] = v; arg[i] = c }
            }
        }

        val vw = ceil(nw * ow / size.toFloat()).toInt().coerceIn(1, ow)
        val vh = ceil(nh * oh / size.toFloat()).toInt().coerceIn(1, oh)
        val counts = IntArray(classes)
        for (y in 0 until vh) for (x in 0 until vw) counts[arg[y * ow + x]]++
        val lesion = vw * vh - counts[0]
        val order = (1 until classes).sortedByDescending { counts[it] }
        val top = if (lesion > 0) order[0] else 0
        val second = if (lesion > 0 && counts[order[1]] > 0) order[1] else 0

        val colors = IntArray(vw * vh) { i ->
            when (arg[(i / vw) * ow + i % vw]) {
                0 -> 0
                top -> TOP_COLOR
                else -> OTHER_COLOR
            }
        }
        val mask = Bitmap.createBitmap(colors, vw, vh, Bitmap.Config.ARGB_8888)
        val share = { c: Int -> if (lesion > 0) counts[c].toFloat() / lesion else 0f }
        return SegResult(mask, top, share(top), second, share(second), lesion.toFloat() / (vw * vh),
            total / 1e6 / runs, backend)
    }

    override fun close() {
        interp.close()
        gpu?.close()
    }

    companion object {
        const val TOP_COLOR = 0xFFFF3B30.toInt()
        const val OTHER_COLOR = 0xFFFFC107.toInt()
    }
}
