import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xml.etree.cElementTree as ET
import glob
import os
import json
import random
import shutil
from PIL import Image, ImageOps
from ultralytics import YOLO
import cv2

# Unzip the dataset if needed
import zipfile

dataset_zip_path = 'archive.zip'
dataset_extract_path = 'dataset'

if not os.path.exists(dataset_extract_path):
    with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
        zip_ref.extractall(dataset_extract_path)

# Define paths
input_dir = os.path.join(dataset_extract_path, 'annotations')
output_dir = 'labels'
image_dir = os.path.join(dataset_extract_path, 'images')

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

# Convert XML annotations to YOLO format
def xml_to_yolo_bbox(bbox, w, h):
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]

def yolo_to_xml_bbox(bbox, w, h):
    w_half_len = (bbox[2] * w) / 2
    h_half_len = (bbox[3] * h) / 2
    xmin = int((bbox[0] * w) - w_half_len)
    ymin = int((bbox[1] * h) - h_half_len)
    xmax = int((bbox[0] * w) + w_half_len)
    ymax = int((bbox[1] * h) + h_half_len)
    return [xmin, ymin, xmax, ymax]

classes = []

files = glob.glob(os.path.join(input_dir, '*.xml'))
for fil in files:
    basename = os.path.basename(fil)
    filename = os.path.splitext(basename)[0]
    if not os.path.exists(os.path.join(image_dir, f'{filename}.png')):
        print(f'{filename} image does not exist')
        continue

    result = []

    tree = ET.parse(fil)
    root = tree.getroot()
    width = int(root.find('size').find('width').text)
    height = int(root.find('size').find('height').text)

    for obj in root.findall('object'):
        label = obj.find('name').text

        if label not in classes:
            classes.append(label)

        index = classes.index(label)
        pil_bbox = [int(x.text) for x in obj.find('bndbox')]
        yolo_bbox = xml_to_yolo_bbox(pil_bbox, width, height)

        bbox_string = ' '.join([str(x) for x in yolo_bbox])
        result.append(f'{index} {bbox_string}')

    if result:
        with open(os.path.join(output_dir, f'{filename}.txt'), 'w', encoding='utf-8') as f:
            f.write('\n'.join(result))

with open(f'{output_dir}/classes.txt', 'w', encoding='utf-8') as f:
    f.write(json.dumps(classes))

# Split data into train, test, and validation sets
metarial = [i[:-4] for i in os.listdir(image_dir)]
train_size = int(len(metarial) * 0.7)
test_size = int(len(metarial) * 0.15)
val_size = int(len(metarial) * 0.15)

def preparinbdata(main_txt_file, main_img_file, train_size, test_size, val_size):
    for i in range(0, train_size):
        source_txt = os.path.join(main_txt_file, f"{metarial[i]}.txt")
        source_img = os.path.join(main_img_file, f"{metarial[i]}.png")
        train_destination_txt = os.path.join('data/train/labels', f"{metarial[i]}.txt")
        train_destination_png = os.path.join('data/train/images', f"{metarial[i]}.png")
        shutil.copy(source_txt, train_destination_txt)
        shutil.copy(source_img, train_destination_png)

    for l in range(train_size, train_size + test_size):
        source_txt = os.path.join(main_txt_file, f"{metarial[l]}.txt")
        source_img = os.path.join(main_img_file, f"{metarial[l]}.png")
        test_destination_txt = os.path.join('data/test/labels', f"{metarial[l]}.txt")
        test_destination_png = os.path.join('data/test/images', f"{metarial[l]}.png")
        shutil.copy(source_txt, test_destination_txt)
        shutil.copy(source_img, test_destination_png)

    for n in range(train_size + test_size, train_size + test_size + val_size):
        source_txt = os.path.join(main_txt_file, f"{metarial[n]}.txt")
        source_img = os.path.join(main_img_file, f"{metarial[n]}.png")
        val_destination_txt = os.path.join('data/val/labels', f"{metarial[n]}.txt")
        val_destination_png = os.path.join('data/val/images', f"{metarial[n]}.png")
        shutil.copy(source_txt, val_destination_txt)
        shutil.copy(source_img, val_destination_png)

# Create directories for train, test, and validation data
os.makedirs('data/train/images', exist_ok=True)
os.makedirs('data/train/labels', exist_ok=True)
os.makedirs('data/test/images', exist_ok=True)
os.makedirs('data/test/labels', exist_ok=True)
os.makedirs('data/val/images', exist_ok=True)
os.makedirs('data/val/labels', exist_ok=True)

preparinbdata(main_txt_file=output_dir, main_img_file=image_dir, train_size=train_size, test_size=test_size, val_size=val_size)

# Get absolute paths
import os
train_path = os.path.abspath('data/train/images')
val_path = os.path.abspath('data/val/images')

# Create YAML configuration for YOLO with absolute paths
yaml_text = f"""train: {train_path}
val: {val_path}
nc: 3
names: ["with_mask", "mask_weared_incorrect", "without_mask"]"""

with open("data/data.yaml", 'w') as file:
    file.write(yaml_text)



# Debug directory structure
print("Current working directory:", os.getcwd())
print("Checking if directories exist:")
print("Train images:", os.path.exists("data/train/images"))
print("Train labels:", os.path.exists("data/train/labels"))
print("Val images:", os.path.exists("data/val/images"))
print("Val labels:", os.path.exists("data/val/labels"))
print("data.yaml:", os.path.exists("data/data.yaml"))

# List contents of directories
print("\nContents of train/images:", os.listdir("data/train/images"))
print("Contents of val/images:", os.listdir("data/val/images"))

# Train the YOLO model
model = YOLO('yolo11n.pt')
results = model.train(data="data/data.yaml", epochs=2, imgsz=640, save=True, device='cpu')

# Validate the model
metrics = model.val(split='val')
print(f"Mean Average Precision @.5:.95 : {metrics.box.map}")
print(f"Mean Average Precision @ .50   : {metrics.box.map50}")
print(f"Mean Average Precision @ .70   : {metrics.box.map75}")
print(metrics.box.maps)

# Predict on test images
image_dir = "data/test/images"
all_images = os.listdir(image_dir)
selected_images = all_images[:45]

for img_name in selected_images:
    img_path = os.path.join(image_dir, img_name)
    results = model.predict(img_path)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for result in results:
        plotted_img = result.plot()
        plt.figure(figsize=(8, 6))
        plt.imshow(plotted_img)
        plt.axis('off')
        plt.show()
