from ultralytics import YOLO
import torch

def quantize_model(model_path, export_path=None):
    # Charger le modèle
    model = YOLO(model_path)
    
    # Configuration de l'export avec quantification
    model.export(
        format='torchscript',
        optimize=True,  # Optimisation TorchScript
        dynamic=True,  # Quantification dynamique
        half=True,     # Conversion en float16
        int8=True,     # Quantification en int8
        simplify=True, # Simplification du modèle
    )
    
    print(f"Modèle quantifié sauvegardé avec succès!")
    print(f"Taille du modèle original: {model_path}")
    print(f"Taille du modèle quantifié: {model_path.replace('.pt', '_quant.pt')}")

if __name__ == "__main__":
    # Chemin vers votre modèle fine-tuné
    model_path = "kaggle_finetuned.pt"
    
    # Quantifier le modèle
    quantize_model(model_path)
