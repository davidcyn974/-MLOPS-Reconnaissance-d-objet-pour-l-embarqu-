from ultralytics import YOLO
import tensorflow as tf

# Load the YOLO model
model = YOLO('kaggle_finetuned.pt')

# Export to TensorFlow SavedModel format first
model.export(format='saved_model', imgsz=640)

# Convert SavedModel to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model('kaggle_finetuned_saved_model')

# Configure the converter
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float16]
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

# Convert to TFLite
tflite_model = converter.convert()

# Save the model
with open('kaggle_finetuned.tflite', 'wb') as f:
    f.write(tflite_model)
