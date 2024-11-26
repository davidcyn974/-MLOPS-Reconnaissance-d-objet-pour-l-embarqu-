from ultralytics import YOLO
import cv2
from typing import Dict, Union, Literal, List
from pathlib import Path
import math

# Démarrage de la webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # largeur
cap.set(4, 480)  # hauteur

ModelNames = Literal["base", "quantized", "mask_finetuned", "glasses_finetuned"]

models: Dict[ModelNames, Union[str, Path]] = {
    'base': 'yolo11n.pt',
    'quantized': 'yolo11n_quant.pt',
    'mask_finetuned': 'yolo_mask_finetuning/mask_finetuned_yolo11n/weights/best.pt',
    'glasses_finetuned': 'yolo_glasses_finetuning/glasses_finetuned_yolo11n/weights/best.pt',
}

class_names: Dict[ModelNames, List[str]] = {
    'base': ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"],
    'quantized': ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"],
    'mask_finetuned': ["with_mask", "mask_weared_incorrect", "without_mask"],
    'glasses_finetuned': ["glasses"],
}
class ModelManager:
    class_names: List[str]
    model: Union[str, Path]
    def __init__(self, model: ModelNames = 'base'):
        self.model = models[model]
        self.class_names = class_names[model]

# Charger le modèle
loaded_model = ModelManager('glasses_finetuned')
model = YOLO(loaded_model.model)
classNames = loaded_model.class_names

while True:
    success, img = cap.read()  # Lire l'image de la webcam
    if not success:
        print("Erreur de lecture de l'image depuis la webcam")
        break

    # Convertir l'image en RGB avant de la passer au modèle
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Passer l'image dans le modèle pour obtenir les résultats
    results = model(img_rgb, stream=True)

    # Parcourir les résultats de détection
    for result in results:
        boxes = result.boxes
        print(f"Nombre de détections : {len(boxes)}")  # Debug info

        for box in boxes:
            # Coordonnées du cadre de détection
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convertir en entiers

            # Obtenir la classe et la confiance
            cls = int(box.cls[0])
            confidence = math.ceil((box.conf[0]*100))/100
            
            if cls < len(classNames):  # Vérifier que l'index est valide
                print(f"Classe détectée : {classNames[cls]} avec confiance : {confidence}")  # Debug info

                # Dessiner le cadre de détection sur l'image
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                
                # Afficher le nom de la classe et la confiance
                cv2.putText(img, f'{classNames[cls]} {confidence}', (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

    # Afficher l'image avec les résultats de détection
    cv2.imshow('Webcam', img)

    # Quitter si la touche 'q' est pressée
    if cv2.waitKey(1) == ord('q'):
        break

# Libérer la caméra et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()
