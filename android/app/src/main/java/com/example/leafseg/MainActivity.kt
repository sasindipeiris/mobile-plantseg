package com.example.leafseg

import android.graphics.Bitmap
import android.graphics.ImageDecoder
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.RadioButton
import android.widget.RadioGroup
import android.widget.SeekBar
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.FileProvider
import java.io.File
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {
    private val worker = Executors.newSingleThreadExecutor()   // GPU delegate: create, run, close on one thread
    private var seg: Segmenter? = null
    private var segAsset = ""
    private var bitmap: Bitmap? = null
    private lateinit var labels: List<String>
    private lateinit var photo: ImageView
    private lateinit var overlay: ImageView
    private lateinit var result: TextView
    private lateinit var info: TextView
    private lateinit var gpu: RadioButton

    private val camUri by lazy {
        FileProvider.getUriForFile(this, "$packageName.fileprovider", File(cacheDir, "photo.jpg"))
    }
    private val pick = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { it?.let(::load) }
    private val shoot = registerForActivityResult(ActivityResultContracts.TakePicture()) { ok -> if (ok) load(camUri) }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        photo = findViewById(R.id.photo)
        overlay = findViewById(R.id.overlay)
        result = findViewById(R.id.result)
        info = findViewById(R.id.info)
        gpu = findViewById(R.id.gpu)
        labels = assets.open("labels.txt").bufferedReader().readLines().filter { it.isNotBlank() }

        findViewById<Button>(R.id.gallery).setOnClickListener {
            pick.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }
        findViewById<Button>(R.id.camera).setOnClickListener { shoot.launch(camUri) }
        findViewById<RadioGroup>(R.id.backend).setOnCheckedChangeListener { _, _ -> run() }
        findViewById<SeekBar>(R.id.alpha).setOnSeekBarChangeListener(object : SeekBar.OnSeekBarChangeListener {
            override fun onProgressChanged(s: SeekBar, p: Int, user: Boolean) { overlay.alpha = p / 100f }
            override fun onStartTrackingTouch(s: SeekBar) {}
            override fun onStopTrackingTouch(s: SeekBar) {}
        })
    }

    private fun load(uri: Uri) {
        bitmap = ImageDecoder.decodeBitmap(ImageDecoder.createSource(contentResolver, uri)) { d, img, _ ->
            d.allocator = ImageDecoder.ALLOCATOR_SOFTWARE
            val s = MAX_SIDE.toFloat() / maxOf(img.size.width, img.size.height)
            if (s < 1) d.setTargetSize((img.size.width * s).toInt(), (img.size.height * s).toInt())
        }
        photo.setImageBitmap(bitmap)
        overlay.setImageDrawable(null)
        run()
    }

    private fun run() {
        val bmp = bitmap ?: return
        val suffix = if (gpu.isChecked) "_fp32.tflite" else "_int8.tflite"
        val useGpu = gpu.isChecked
        result.text = "Running…"
        worker.execute {
            try {
                val asset = assets.list("")!!.first { it.endsWith(suffix) }
                if (asset != segAsset) {
                    seg?.close()
                    seg = Segmenter(this, asset, useGpu)
                    segAsset = asset
                }
                val r = seg!!.segment(bmp)
                runOnUiThread { show(r) }
            } catch (e: Throwable) {
                runOnUiThread { result.text = "Error: ${e.message}" }
            }
        }
    }

    private fun show(r: SegResult) {
        overlay.setImageBitmap(r.mask)
        result.text = if (r.top == 0 || r.lesionFrac < MIN_LESION) "No disease detected" else buildString {
            append("${name(r.top)}  ${pct(r.topShare)}\nDiseased area: ${pct(r.lesionFrac)} of image")
            if (r.second > 0) append("\nAlso seen: ${name(r.second)} (${pct(r.secondShare)})")
        }
        info.text = "%.1f ms · %s · %s".format(r.ms, r.backend, Build.MODEL)
    }

    private fun name(c: Int) = labels.getOrElse(c) { "class $c" }.replaceFirstChar { it.uppercase() }
    private fun pct(f: Float) = "%.0f%%".format(100 * f)

    override fun onDestroy() {
        worker.execute { seg?.close() }
        worker.shutdown()
        super.onDestroy()
    }

    companion object {
        const val MAX_SIDE = 1024
        const val MIN_LESION = 0.001f
    }
}
