from ultralytics import YOLO
import cv2
import math

# Démarrage de la webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # largeur
cap.set(4, 480)  # hauteur

# Charger le modèle
model = YOLO("yolo11n.pt")

# Liste des classes d'objets
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

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

        for box in boxes:
            # Coordonnées du cadre de détection
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # Convertir en entiers

            # Dessiner le cadre de détection sur l'image
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

            # Calculer et afficher la confiance de la détection
            confidence = math.ceil((box.conf[0]*100))/100
            print("Confiance --->", confidence)

            # Nom de la classe de l'objet détecté
            cls = int(box.cls[0])
            print("Nom de la classe -->", classNames[cls])

            # Afficher le nom de la classe sur l'image
            org = (x1, y1 - 10)
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2

            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

    # Afficher l'image avec les résultats de détection
    cv2.imshow('Webcam', img)

    # Quitter si la touche 'q' est pressée
    if cv2.waitKey(1) == ord('q'):
        break

# Libérer la caméra et fermer les fenêtres
cap.release()
cv2.destroyAllWindows()
