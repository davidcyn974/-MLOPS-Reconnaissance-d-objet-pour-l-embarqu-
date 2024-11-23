# Projet MLOps S9
Heyo!

Voici le projet de MLOps de la S9.
Il s'agit d'un projet de détection d'objets à l'aide de YOLOv8.

Attention : pour ma part la version 3.12 de Python n'est pas encore supportée par YOLOv8, j'ai donc utilisé la version 3.10.
Quelques commandes utiles pour utiliser différentes versions de Python:

```shell
py -3.10 fichier.py
py -list # pour voir les versions de Python installées
py -3.10 -m pip install mon_package
```

Dans requirements.txt, on retrouve les dépendances nécessaires pour le bon fonctionnement du projet.
Ce qui correspond à :

```shell
py -3.10 -m pip freeze
```

sur mon pc.

Pour installer les dépendances, il suffit de taper la commande suivante dans le terminal :

```shell
py -3.10 -m pip install -r requirements.txt
```

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

```