package com.tfexample.bostondataset

import android.content.res.AssetManager
import android.os.Bundle
import android.support.v7.app.AppCompatActivity
import kotlinx.android.synthetic.main.activity_main.*
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.io.IOException
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {

  private lateinit var interpreter: Interpreter

  private val DICAPRIO: String = "dicaprio = [3, 'Jack Dawson', 'male', 19, 0, 0, 'N/A', 5.0000]"

  private val WINSLET: String = "winslet = [1, 'Rose DeWitt Bukater', 'female', 17, 1, 2, 'N/A', 100.0000]"

  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContentView(R.layout.activity_main)
    interpreter = Interpreter(loadModelFile(assets, "converted_model.tflite"))
    predictEdt.text = "$DICAPRIO\n" +
        "$WINSLET"
    predict.setOnClickListener {
      val caprio = preditForDicaprio()
      val winslet = predictForWinslet()
      predictEdt.text = "$DICAPRIO > ${caprio.contentDeepToString()}\n" +
          "$WINSLET > ${winslet.contentDeepToString()}\n" + "Where chances of being saved are less for dicaprio and more for winslet"
    }
  }

  private fun predictForWinslet(): Array<FloatArray> {
    val input = Array(1) { FloatArray(6) }
    input.get(0).set(0, 1F)
    input.get(0).set(1, 1F)
    input.get(0).set(2, 17F)
    input.get(0).set(3, 1F)
    input.get(0).set(4, 2F)
    input.get(0).set(5, 100.0F)


    val result = Array(1) { FloatArray(2) }
    interpreter.run(input, result)
    return result
  }

  private fun preditForDicaprio(): Array<FloatArray> {
    val input = Array(1) { FloatArray(6) }
    input.get(0).set(0, 3F)
    input.get(0).set(1, 0F)
    input.get(0).set(2, 19F)
    input.get(0).set(3, 0F)
    input.get(0).set(4, 0F)
    input.get(0).set(5, 5.0F)

    val result = Array(1) { FloatArray(2) }
    interpreter.run(input, result)
    return result
  }


  @Throws(IOException::class)
  private fun loadModelFile(assetManager: AssetManager, modelPath: String): MappedByteBuffer {
    val fileDescriptor = assetManager.openFd(modelPath)
    val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
    val fileChannel = inputStream.channel
    val startOffset = fileDescriptor.startOffset
    val declaredLength = fileDescriptor.declaredLength
    return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
  }


}
