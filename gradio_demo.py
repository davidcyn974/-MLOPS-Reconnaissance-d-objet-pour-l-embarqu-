import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO('kaggle_finetuned.pt')  # ou le modèle quantifié

def predict(input_image):
    results = model(input_image)
    return results[0].plot()  # Retourne l'image annotée

# Création de l'interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(sources="webcam", streaming=True),
    outputs="image",
    live=True,
    title="YOLOv11n Mask Detection"
)

interface.launch(share=True)  # share=True crée un lien public temporaire