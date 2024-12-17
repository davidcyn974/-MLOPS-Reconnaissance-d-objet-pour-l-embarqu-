import tensorflow as tf
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolo_glasses.pt")

# Export to TensorFlow SavedModel format first
model.export(format="saved_model", imgsz=640)

# Convert SavedModel to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("yolo_glasses_saved_model")

# Configure the converter with more flexible settings
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_types = [tf.float32]  # Use float32 instead of float16
converter.allow_custom_ops = True  # Allow custom operations
converter.experimental_new_converter = True
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS,  # Include regular TF ops
]

# Convert the model
tflite_model = converter.convert()

# Save the model
with open("yolo_glasses.tflite", "wb") as f:
    f.write(tflite_model)
