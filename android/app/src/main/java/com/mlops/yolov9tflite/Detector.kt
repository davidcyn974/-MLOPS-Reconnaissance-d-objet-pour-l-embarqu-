package com.mlops.yolov9tflite

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import com.mlops.yolov9tflite.MetaData.extractNamesFromLabelFile
import com.mlops.yolov9tflite.MetaData.extractNamesFromMetadata
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.BufferedReader
import java.io.IOException
import java.io.InputStream
import java.io.InputStreamReader

class Detector(
    private val context: Context,
    private val modelPath: String,
    private val labelPath: String?,
    private val detectorListener: DetectorListener,
    private val message: (String) -> Unit
) {

    private var interpreter: Interpreter
    private var labels = mutableListOf<String>()
    private var tensorWidth = 0
    private var tensorHeight = 0
    private var numChannel = 0
    private var numElements = 0
    private val isFineTunedModel = modelPath == Constants.MASK_MODEL_PATH || modelPath == Constants.GLASSES_FINETUNE

    private val imageProcessor = ImageProcessor.Builder()
        .add(NormalizeOp(INPUT_MEAN, INPUT_STANDARD_DEVIATION))
        .add(CastOp(INPUT_IMAGE_TYPE))
        .build()

    init {
        val options = Interpreter.Options().apply {
            this.setNumThreads(4)
        }

        val model = FileUtil.loadMappedFile(context, modelPath)
        interpreter = Interpreter(model, options)

        labels.addAll(extractNamesFromMetadata(model))
        if (labels.isEmpty()) {
            if (labelPath == null) {
                message("Model not contains metadata, provide LABELS_PATH in Constants.kt")
                labels.addAll(MetaData.TEMP_CLASSES)
            } else {
                labels.addAll(extractNamesFromLabelFile(context, labelPath))
            }
        }

        val inputShape = interpreter.getInputTensor(0)?.shape()
        val outputShape = interpreter.getOutputTensor(0)?.shape()

        if (inputShape != null) {
            tensorWidth = inputShape[1]
            tensorHeight = inputShape[2]

            // If in case input shape is in format of [1, 3, ..., ...]
            if (inputShape[1] == 3) {
                tensorWidth = inputShape[2]
                tensorHeight = inputShape[3]
            }
        }

        if (outputShape != null) {
            numChannel = outputShape[1]
            numElements = outputShape[2]
        }

        message("Model loaded with input shape: ${inputShape?.contentToString()}, output shape: ${outputShape?.contentToString()}")
        message("Labels: ${labels.joinToString()}")
    }

    fun restart() {
        interpreter.close()

        val options = Interpreter.Options().apply {
            this.setNumThreads(4)
        }

        val model = FileUtil.loadMappedFile(context, modelPath)
        interpreter = Interpreter(model, options)
    }

    fun close() {
        interpreter.close()
    }

    fun detect(frame: Bitmap) {
        if (tensorWidth == 0) return
        if (tensorHeight == 0) return
        if (numChannel == 0) return
        if (numElements == 0) return

        var inferenceTime = SystemClock.uptimeMillis()

        val resizedBitmap = Bitmap.createScaledBitmap(frame, tensorWidth, tensorHeight, false)

        val tensorImage = TensorImage(INPUT_IMAGE_TYPE)
        tensorImage.load(resizedBitmap)
        val processedImage = imageProcessor.process(tensorImage)
        val imageBuffer = processedImage.buffer

        val output = TensorBuffer.createFixedSize(intArrayOf(1, numChannel, numElements), OUTPUT_IMAGE_TYPE)
        interpreter.run(imageBuffer, output.buffer)

        val bestBoxes = bestBox(output.floatArray)
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime

        if (bestBoxes == null) {
            detectorListener.onEmptyDetect()
            return
        }

        detectorListener.onDetect(bestBoxes, inferenceTime)
    }

    private fun bestBox(array: FloatArray) : List<BoundingBox>? {
        val boundingBoxes = mutableListOf<BoundingBox>()

        if (isFineTunedModel) {
            if (modelPath == Constants.GLASSES_FINETUNE) {
                val numPredictions = numElements  // 8400 predictions
                var maxConfidence = 0f
                
                for (i in 0 until numPredictions) {
                    // Get confidence score (last channel)
                    val confidence = array[4 * numPredictions + i]
                    maxConfidence = maxOf(maxConfidence, confidence)
                    
                    if (confidence > 0.3f) {  // Lower threshold for initial detection
                        // Get box coordinates (first 4 channels)
                        val x = array[0 * numPredictions + i]
                        val y = array[1 * numPredictions + i]
                        val w = array[2 * numPredictions + i]
                        val h = array[3 * numPredictions + i]

                        // Convert to normalized coordinates
                        val x1 = (x - w/2).coerceIn(0f, INPUT_SIZE) / INPUT_SIZE
                        val y1 = (y - h/2).coerceIn(0f, INPUT_SIZE) / INPUT_SIZE
                        val x2 = (x + w/2).coerceIn(0f, INPUT_SIZE) / INPUT_SIZE
                        val y2 = (y + h/2).coerceIn(0f, INPUT_SIZE) / INPUT_SIZE
                        
                        val width = x2 - x1
                        val height = y2 - y1
                        
                        // More permissive size constraints for glasses
                        if (width > 0.05f && height > 0.02f && // Allow smaller detections
                            width < 0.9f && height < 0.5f) {   // Allow larger detections
                            boundingBoxes.add(
                                BoundingBox(
                                    x1 = x1,
                                    y1 = y1,
                                    x2 = x2,
                                    y2 = y2,
                                    cx = x / INPUT_SIZE, 
                                    cy = y / INPUT_SIZE, 
                                    w = w / INPUT_SIZE, 
                                    h = h / INPUT_SIZE,
                                    cnf = confidence,
                                    cls = 0,
                                    clsName = "glasses"
                                )
                            )
                        }
                    }
                }
                
                return if (boundingBoxes.isNotEmpty()) {
                    boundingBoxes.sortedByDescending { it.cnf }.take(1)  // Only take the highest confidence detection
                } else null
                
            } else {
                // Process mask detection model output format
                val numClasses = numChannel - 4
                var maxConf = 0f
                
                // For each prediction
                for (i in 0 until numElements) {
                    // Get confidence scores for all classes
                    val confidences = FloatArray(numClasses)
                    var sum = 0f
                    for (j in 0 until numClasses) {
                        confidences[j] = array[(4 + j) * numElements + i]
                        sum += confidences[j]
                        maxConf = maxOf(maxConf, confidences[j])
                    }
                    
                    // Calculate relative probabilities
                    val maxIdx = confidences.indexOfFirst { it == confidences.maxOrNull() }
                    val relativeConf = if (sum > 0) confidences[maxIdx] / sum else 0f
                    
                    if (relativeConf > 0.75f) {  // Adjusted threshold
                        val cx = array[0 * numElements + i] / INPUT_SIZE
                        val cy = array[1 * numElements + i] / INPUT_SIZE
                        val w = array[2 * numElements + i] / INPUT_SIZE
                        val h = array[3 * numElements + i] / INPUT_SIZE
                        
                        val x1 = (cx - w/2).coerceIn(0f, 1f)
                        val y1 = (cy - h/2).coerceIn(0f, 1f)
                        val x2 = (cx + w/2).coerceIn(0f, 1f)
                        val y2 = (cy + h/2).coerceIn(0f, 1f)
                        
                        val width = x2 - x1
                        val height = y2 - y1
                        val aspectRatio = width / height
                        
                        if (width > 0.1f && height > 0.1f && 
                            width < 0.8f && height < 0.8f && 
                            aspectRatio in 0.7f..1.5f) {
                                // Use raw confidence score instead of scaling
                                boundingBoxes.add(
                                    BoundingBox(
                                        x1 = x1,
                                        y1 = y1,
                                        x2 = x2,
                                        y2 = y2,
                                        cx = cx, cy = cy, w = w, h = h,
                                        cnf = relativeConf,
                                        cls = maxIdx,
                                        clsName = labels[maxIdx]
                                    )
                                )
                        }
                    }
                }
                
                return if (boundingBoxes.isNotEmpty()) {
                    boundingBoxes.sortedByDescending { it.cnf }.take(1)
                } else null
            }
        } else {
            // Process default YOLOv11 output format [batch, predictions, channels]
            for (c in 0 until numElements) {
                var maxConf = CONFIDENCE_THRESHOLD
                var maxIdx = -1
                var j = 4
                var arrayIdx = c + numElements * j
                while (j < numChannel) {
                    if (array[arrayIdx] > maxConf) {
                        maxConf = array[arrayIdx]
                        maxIdx = j - 4
                    }
                    j++
                    arrayIdx += numElements
                }

                if (maxConf > CONFIDENCE_THRESHOLD) {
                    val clsName = labels[maxIdx]
                    val cx = array[c]
                    val cy = array[c + numElements]
                    val w = array[c + numElements * 2]
                    val h = array[c + numElements * 3]
                    val x1 = cx - (w/2F)
                    val y1 = cy - (h/2F)
                    val x2 = cx + (w/2F)
                    val y2 = cy + (h/2F)
                    if (x1 < 0F || x1 > 1F) continue
                    if (y1 < 0F || y1 > 1F) continue
                    if (x2 < 0F || x2 > 1F) continue
                    if (y2 < 0F || y2 > 1F) continue

                    boundingBoxes.add(
                        BoundingBox(
                            x1 = x1, y1 = y1, x2 = x2, y2 = y2,
                            cx = cx, cy = cy, w = w, h = h,
                            cnf = maxConf, cls = maxIdx, clsName = clsName
                        )
                    )
                }
            }
        }

        if (boundingBoxes.isEmpty()) return null

        // For finetuned models, take up to 3 highest confidence detections
        // For default model, apply NMS to allow multiple detections
        return if (isFineTunedModel) {
            val sortedBoxes = boundingBoxes.sortedByDescending { it.cnf }
            
            // Apply NMS to top detections for finetuned models
            val finalBoxes = mutableListOf<BoundingBox>()
            val isIncluded = BooleanArray(sortedBoxes.size) { true }

            for (i in sortedBoxes.indices) {
                if (!isIncluded[i]) continue
                val boxA = sortedBoxes[i]
                finalBoxes.add(boxA)
                
                // Stop if we have enough detections
                if (finalBoxes.size >= MAX_FINETUNED_DETECTIONS) break
                
                for (j in i + 1 until sortedBoxes.size) {
                    if (!isIncluded[j]) continue
                    val boxB = sortedBoxes[j]
                    
                    // Calculate IoU
                    val intersectionX1 = maxOf(boxA.x1, boxB.x1)
                    val intersectionY1 = maxOf(boxA.y1, boxB.y1)
                    val intersectionX2 = minOf(boxA.x2, boxB.x2)
                    val intersectionY2 = minOf(boxA.y2, boxB.y2)
                    
                    if (intersectionX1 < intersectionX2 && intersectionY1 < intersectionY2) {
                        val intersectionArea = (intersectionX2 - intersectionX1) * (intersectionY2 - intersectionY1)
                        val boxAArea = (boxA.x2 - boxA.x1) * (boxA.y2 - boxA.y1)
                        val boxBArea = (boxB.x2 - boxB.x1) * (boxB.y2 - boxB.y1)
                        val iou = intersectionArea / (boxAArea + boxBArea - intersectionArea)
                        
                        if (iou > NMS_IOU_THRESHOLD) {
                            isIncluded[j] = false
                        }
                    }
                }
            }
            
            finalBoxes
        } else {
            // Apply Non-Maximum Suppression for default model
            val finalBoxes = mutableListOf<BoundingBox>()
            val sortedBoxes = boundingBoxes.sortedByDescending { it.cnf }
            val isIncluded = BooleanArray(sortedBoxes.size) { true }

            for (i in sortedBoxes.indices) {
                if (!isIncluded[i]) continue
                val boxA = sortedBoxes[i]
                finalBoxes.add(boxA)
                
                for (j in i + 1 until sortedBoxes.size) {
                    if (!isIncluded[j]) continue
                    val boxB = sortedBoxes[j]
                    
                    // Calculate IoU
                    val intersectionX1 = maxOf(boxA.x1, boxB.x1)
                    val intersectionY1 = maxOf(boxA.y1, boxB.y1)
                    val intersectionX2 = minOf(boxA.x2, boxB.x2)
                    val intersectionY2 = minOf(boxA.y2, boxB.y2)
                    
                    if (intersectionX1 < intersectionX2 && intersectionY1 < intersectionY2) {
                        val intersectionArea = (intersectionX2 - intersectionX1) * (intersectionY2 - intersectionY1)
                        val boxAArea = (boxA.x2 - boxA.x1) * (boxA.y2 - boxA.y1)
                        val boxBArea = (boxB.x2 - boxB.x1) * (boxB.y2 - boxB.y1)
                        val iou = intersectionArea / (boxAArea + boxBArea - intersectionArea)
                        
                        if (iou > NMS_IOU_THRESHOLD) {
                            isIncluded[j] = false
                        }
                    }
                }
            }
            
            finalBoxes
        }
    }

    interface DetectorListener {
        fun onEmptyDetect()
        fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long)
    }

    companion object {
        private const val INPUT_SIZE = 640f
        private const val INPUT_MEAN = 0f
        private const val INPUT_STANDARD_DEVIATION = 255f
        private const val CONFIDENCE_THRESHOLD = 0.75f
        private const val MIN_BOX_SIZE = 0.05f  // Minimum box size as fraction of image
        private const val MAX_BOX_SIZE = 0.9f   // Maximum box size as fraction of image
        private val INPUT_IMAGE_TYPE = DataType.FLOAT32
        private val OUTPUT_IMAGE_TYPE = DataType.FLOAT32

        // Thresholds
        private const val NMS_IOU_THRESHOLD = 0.3f
        private const val MAX_FINETUNED_DETECTIONS = 1
    }

    private fun nonMaxSuppression(boxes: List<BoundingBox>, iouThreshold: Float): List<BoundingBox> {
        // Sort boxes by confidence
        val sortedBoxes = boxes.sortedByDescending { it.cnf }
        val selectedBoxes = mutableListOf<BoundingBox>()

        // Keep track of which boxes are still valid
        val validBoxes = BooleanArray(sortedBoxes.size) { true }

        for (i in sortedBoxes.indices) {
            if (!validBoxes[i]) continue

            selectedBoxes.add(sortedBoxes[i])

            // Compare with rest of the boxes
            for (j in i + 1 until sortedBoxes.size) {
                if (!validBoxes[j]) continue

                // Calculate IoU between boxes
                val iou = calculateIoU(sortedBoxes[i], sortedBoxes[j])
                if (iou >= iouThreshold) {
                    validBoxes[j] = false  // Mark box as invalid if IoU is above threshold
                }
            }
        }

        return selectedBoxes
    }

    private fun calculateIoU(box1: BoundingBox, box2: BoundingBox): Float {
        // Calculate intersection area
        val xLeft = maxOf(box1.x1, box2.x1)
        val yTop = maxOf(box1.y1, box2.y1)
        val xRight = minOf(box1.x2, box2.x2)
        val yBottom = minOf(box1.y2, box2.y2)

        if (xRight < xLeft || yBottom < yTop) return 0f

        val intersection = (xRight - xLeft) * (yBottom - yTop)

        // Calculate union area
        val box1Area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
        val box2Area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
        val union = box1Area + box2Area - intersection

        return if (union <= 0f) 0f else intersection / union
    }

    private fun Float.format(digits: Int) = "%.${digits}f".format(this)
}