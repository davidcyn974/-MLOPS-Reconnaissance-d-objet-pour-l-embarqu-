# Launch the Python interpreter
# Check if PyTorch is available
import torch
import cv2

import cv2

cap = cv2.VideoCapture(0)  # 0 pour la première caméra

if not cap.isOpened():
    print("Erreur : Impossible d'accéder à la caméra")
else:
    print("Caméra détectée avec succès")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur : Impossible de lire le flux vidéo")
            break
        cv2.imshow("Webcam", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Appuyez sur 'q' pour quitter
            break

cap.release()
cv2.destroyAllWindows()
