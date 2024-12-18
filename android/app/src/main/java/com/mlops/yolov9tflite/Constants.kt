package com.mlops.yolov9tflite

object Constants {
    const val MODEL_PATH = "v11_default/v11.tflite"
    val LABELS_PATH: String? = "v11_default/v11.tflite" // or null

    const val MASK_MODEL_PATH = "mask_finetuned/kaggle_finetuned_float32.tflite"
    val MASK_LABELS_PATH = "mask_finetuned/labels.txt"

    const val GLASSES_FINETUNE = "glasses_finetuned/glasses_finetuned_float32.tflite"
    val GLASSES_LABELS_PATH = "glasses_finetuned/labels.txt"
}
