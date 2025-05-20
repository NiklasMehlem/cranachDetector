# =============================================================================
# Dieses Modul verwendet:
# - Dlib CNN (Boost Software License 1.0)
# - MTCNN (MIT License)
# - InsightFace Code (MIT License)
# - RetinaFace (MIT License)
# =============================================================================

### imports
import dlib
import matplotlib.pyplot as plt
import tkinter as tk
from insightface.app import FaceAnalysis
from mtcnn import MTCNN
from pathlib import Path
from tkinter import filedialog


def CranachDetector(images=None):
    root = tk.Tk()
    root.withdraw()
    imageList = formatImages(images)

    # Ordner Auswahl, wenn noch kein Bild oder Bilder mitgegeben wurden
    if imageList == []:
        folder = filedialog.askdirectory(title="Wähle einen Ordner mit Bildern")
        root.destroy()

        if not folder:
            print("Kein Ordner ausgewählt.")
            return []

        imageList = formatImages(folder)

    # print(imageList)


# Überprüft die Art der Bild eingabe und formatiert sie zu einer Liste
def formatImages(images) -> list:
    if images is None:
        return []
    if isinstance(images, list):
        return images

    path = Path(images)
    exts = (".jpg", ".jpeg", ".jfif", ".png")
    if path.is_dir():
        imgs = []
        # for folder in path:
        imgs += path.rglob("*")
        return [p for p in imgs if p.suffix.lower() in exts]
    elif path.is_file():
        return [path] if path.suffix.lower() in exts else []
    else:
        raise TypeError(f"Unbekannter Typ oder Pfad: {images!r}")


# CranachDetector()
