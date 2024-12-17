# Object Detection Project with YOLOv8

This project implements an object detection system using YOLOv8, featuring model finetuning, real-time detection, model optimization, and mobile deployment.

## Project Overview

The project consists of several key components:
- Custom model finetuning for specific object detection
- Real-time object detection using webcam
- Model optimization through quantization
- Mobile deployment via Android application

## Environment Setup

**Note**: YOLOv8 is currently not compatible with Python 3.12. This project uses Python 3.10.

To set up the environment:

```shell
# Install dependencies
py -3.10 -m pip install -r requirements.txt

# For using specific Python version
py -3.10 file.py
py -list  # List installed Python versions
```

## Model Finetuning

The project includes two specialized finetuned models:

### 1. Mask Detection Model
- Dataset source: Kaggle
- Trained on YOLOv8 base model
- Specialized in detecting face masks in various conditions

### 2. Glasses Detection Model
- Dataset source: Roboflow
- Custom dataset with annotated glasses
- Enhanced detection capabilities for eyewear

## Development Tools

### Live Camera Detection

The project includes a real-time detection system (`livecam.py`) featuring:
- Webcam integration for real-time object detection
- Support for multiple models (base, quantized, mask_finetuned, glasses_finetuned)
- Configurable display options and detection parameters

### Gradio Demo Interface

A web-based demo interface is available for testing the models:
- Interactive interface for uploading images
- Real-time detection visualization
- Easy model switching and comparison

## Model Optimization

### Quantization Process
The project implements model quantization to optimize performance:
- Reduced model size while maintaining accuracy
- Improved inference speed
- Optimized for mobile deployment

### TFLite Conversion
Models are converted to TensorFlow Lite format for mobile deployment:
- Conversion from PyTorch to TFLite format
- Optimized for mobile inference
- Maintained detection accuracy

## Android Application

The project includes a complete Android application for mobile deployment:
- Real-time object detection using device camera
- Optimized TFLite model integration
- User-friendly interface
- Performance optimized for mobile devices

### Required Models Setup

Before running the Android application, you need to download the TFLite models from [Google Drive](https://drive.google.com/drive/folders/1WGwhsaUoEBdg40uTbvCUshW64cQaQAhH?usp=drive_link).

Place the downloaded files in your Android project following this structure:
```
android/
├── glasses_finetuned/
│   ├── glasses_finetuned_float32.tflite
│   └── labels.txt
├── mask_finetuned/
│   ├── kaggle_finetuned_float32.tflite
│   └── labels.txt
└── v11_default/
    ├── v11.tflite
    └── labels.txt
```

Each model serves a specific purpose:
- `glasses_finetuned`: Specialized in glasses detection
- `mask_finetuned`: Optimized for face mask detection
- `v11_default`: Base YOLOv8 model for general object detection

## Project Structure

```
├── android/                  # Android application files
├── glasses_dataset/         # Custom glasses detection dataset
├── glasses_finetuning.py    # Finetuning script for glasses detection
├── livecam.py              # Real-time webcam detection
├── quantization.py         # Model quantization script
├── yolo_to_tflite_fixed2.py # TFLite conversion utility
└── requirements.txt        # Project dependencies
```

## Usage

1. **Base Model Detection**:
```shell
yolo task=detect mode=predict model=yolov8n.pt source=image.jpg
```

2. **Live Camera Detection**:
```shell
py -3.10 livecam.py
```

3. **Model Finetuning**:
```shell
py -3.10 glasses_finetuning.py
```

## Results

Detection results are saved in the `runs/detect/predict` directory by default. The system provides:
- Bounding box visualization
- Class predictions
- Confidence scores

## Dependencies

Key dependencies are listed in `requirements.txt`. The project primarily uses:
- YOLOv8
- OpenCV
- TensorFlow
- PyTorch

Voici l'output de la commande `yolo task=detect mode=predict model=yolov8n.pt source=image.jpg` qui savegarde sa prédiction dans 
`\runs/detect/predict` :

```shell

PS C:\Users\user_\Desktop\S9\MLOPS\projet> yolo task=detect mode=predict model=yolov8n.pt source=image.jpg
Ultralytics YOLOv8.0.0  Python-3.10.7 torch-2.5.1+cpu CPU
Downloading https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt to yolov8n.pt...
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 6.23M/6.23M [00:59<00:00, 110kB/s]

C:\Users\user_\AppData\Local\Programs\Python\Python310\lib\site-packages\ultralytics\nn\tasks.py:303: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
  ckpt = torch.load(attempt_download(w), map_location='cpu')  # load
Fusing layers... 
YOLOv8n summary: 168 layers, 3151904 parameters, 0 gradients, 8.7 GFLOPs
image 1/1 C:\Users\user_\Desktop\S9\MLOPS\projet\image.jpg: 448x640 8 persons, 1 car, 1 bus, 2 traffic lights, 3 backpacks, 1 handbag, 123.5ms
Speed: 2.9ms pre-process, 123.5ms inference, 15.4ms postprocess per image at shape (1, 3, 640, 640)
Results saved to runs\detect\predict