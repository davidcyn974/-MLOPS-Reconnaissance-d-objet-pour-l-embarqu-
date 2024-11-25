import os
import shutil
import random
from ultralytics import YOLO
import torch

def initialize_env():
    print("Initializing environment...")
    data_path = os.path.join(os.getcwd(), "glasses.finetuning.yolov8")
    if os.path.exists(data_path):
        shutil.rmtree(data_path)
    os.makedirs(data_path)
    return data_path


def prepare_dataset(data_path):
    print("Preparing dataset...")

    dataset_path = os.path.join(os.getcwd(), "glasses_dataset")

    # Define dataset splits and subfolders
    splits = ["train", "valid", "test"]
    subfolders = ["images", "labels"]

    # Prepare paths for splits
    split_paths = {
        split: {
            subfolder: os.path.join(data_path, split, subfolder)
            for subfolder in subfolders
        }
        for split in splits
    }

    # Create necessary directories
    for split in split_paths.values():
        for path in split.values():
            os.makedirs(path, exist_ok=True)

    # List and shuffle image files
    image_files = os.listdir(os.path.join(dataset_path, "images"))
    random.shuffle(image_files)

    # Split data into train, valid, and test sets
    train_count = int(0.8 * len(image_files))
    valid_count = int(0.1 * len(image_files))

    train_images = image_files[:train_count]
    valid_images = image_files[train_count:train_count + valid_count]
    test_images = image_files[train_count + valid_count:]

    splits_data = {
        "train": train_images,
        "valid": valid_images,
        "test": test_images
    }

    # Helper function to copy files
    def copy_files(files, src_folder, dest_folder, src_extension=None, dest_extension=None):
        for file in files:
            base_name = os.path.splitext(file)[0]  # Extract the base name
            src_path = os.path.join(src_folder, file if src_extension is None else f"{base_name}{src_extension}")
            dest_path = os.path.join(dest_folder, file if dest_extension is None else f"{base_name}{dest_extension}")
            if os.path.exists(src_path):  # Ensure the source file exists before copying
                shutil.copy(src_path, dest_path)
            else:
                print(f"Warning: {src_path} not found. Skipping.")

    # Copy images and labels to their respective directories
    for split, images in splits_data.items():
        copy_files(images, os.path.join(dataset_path, "images"), split_paths[split]["images"])
        copy_files(images, os.path.join(dataset_path, "labels"), split_paths[split]["labels"], src_extension=".txt", dest_extension=".txt")



def write_config(data_path):
    new_object_name = "glasses"

    data_yaml_path = os.path.join(data_path, "data.yaml")

    yaml_content = f"train: ../train/images\nval: ../valid/images\ntest: ../test/images\nnc: 1\nnames: ['{new_object_name}']"

    with open(data_yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"Wrote YAML content to {data_yaml_path}")

if __name__ == "__main__":
    #1: prepare the context
    data_path = initialize_env()
    prepare_dataset(data_path=data_path)
    write_config(data_path=data_path)
    model = YOLO('yolo11n.pt')
    results = model.train(data=f"{data_path}/data.yaml",
                        epochs=100, 
                        imgsz=640, 
                        save=True, 
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        project='yolo_glasses_finetuning',
                        name='glasses_finetuned_yolo11n')
    #2: train the model
    #3: save the results