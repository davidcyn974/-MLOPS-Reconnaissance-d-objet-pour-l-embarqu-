""" from ultralytics import YOLO

model = YOLO('yolo11n.pt')
model.export(format = 'onnx') # exports the model in '.onnx' format
 """
# # QUANTIZATION
# Preprocessing the model
# https://github.com/microsoft/onnxruntime-inference-examples/blob/main/quantization/image_classification/cpu/ReadMe.md
# !python -m onnxruntime.quantization.preprocess --input yolo11n.onnx --output models/pre.onnx

# ## DYNAMIC QUANTIZATION

""" from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = 'pre.onnx'
model_int8 = 'dynamic_quantized.onnx'

# Quantize 
quantize_dynamic(model_fp32, model_int8, weight_type=QuantType.QUInt8) """

from ultralytics import YOLO

model = YOLO('kaggle_finetuned.pt')
model.export(format = 'tflite') # exports the model in '.tflite' format
