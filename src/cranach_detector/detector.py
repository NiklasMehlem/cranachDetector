# =============================================================================
# Dieses Modul verwendet:
# - Dlib CNN (Boost Software License 1.0)
# - MTCNN (MIT License)
# - InsightFace Code (MIT License)
# - RetinaFace (MIT License)
# =============================================================================

### imports
import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
from insightface.app import FaceAnalysis
from mtcnn import MTCNN
from pathlib import Path
from PIL import Image, ImageTk
from tkinter import filedialog

### variables
cnn_confidence = 0.6
CNN_COLOR = (228, 37, 54) # Rot
mtcnn_confidence = 0.82
MTCNN_COLOR = (248, 156, 32) # Orange
retina_confidence = 0.52
RETINA_COLOR = (87, 144, 252) # Blau
EXTRA_MODEL_COLOR = (150, 74, 139) # Lila 

DETECTION_OFFSET = 10
DETECTION_THICKNESS = 2
DETECTION_FONT_SCALE = 0.6

EXCLUSION_ZONES = []

#_models = ("retinaFace", "mtcnn", "dlib_cnn",)
#_models = ("retinaFace",)
#retinaFace_mode = False
#mtcnn_mode = False
#dlib_cnn_mode = False

BASE_DIR = Path(__file__).resolve().parent
DLIB_CNN_PATH = BASE_DIR / "models" / "dlib_cnn" / "mmod_human_face_detector.dat"
### initierung
cnn_detector = dlib.cnn_face_detection_model_v1(str(DLIB_CNN_PATH))
detector = MTCNN()
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0)

def CranachDetector(images=None, use_retinaFace=True, use_mtcnn=False, use_dlib_cnn=False):
    #root = tk.Tk()
    #root.withdraw()
    #global retinaFace_mode, mtcnn_mode, dlib_cnn_mode
    #retinaFace_mode = use_retinaFace
    #mtcnn_mode = use_mtcnn
    #dlib_cnn_mode = use_dlib_cnn

    imageList = formatImages(images)

    # Ordner Auswahl, wenn noch kein Bild oder Bilder mitgegeben wurden
    while imageList == []:
        folder = filedialog.askdirectory(title="Wähle einen Ordner mit Bildern")
        #root.destroy()

        if not folder:
            print("Kein Ordner ausgewählt.")
            break
        imageList = formatImages(folder)

    #if models is not None:
    start_process(imageList, use_retinaFace, use_mtcnn, use_dlib_cnn)
    
    # Frag nach jedem Bild nicht am Ende des ganzen Ordners!!! TODO
    #if imageList:
        #show_image_gui(imageList[0])
     
    #print(EXCLUSION_ZONES)

def start_process(imageList, use_retinaFace, use_mtcnn, use_dlib_cnn):
    retinaFace_mode = use_retinaFace
    mtcnn_mode = use_mtcnn
    dlib_cnn_mode = use_dlib_cnn
    for image_path in imageList:
        pil_image = Image.open(image_path)
        image = np.asarray(pil_image)
        #overlay = use_models(image_path, retinaFace_mode, mtcnn_mode, dlib_cnn_mode)
        start_gui(image, image_path, retinaFace_mode, mtcnn_mode, dlib_cnn_mode)

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
    
# wendet alle Modelle auf alle Bilder in image_paths an
def use_models(image, image_path, use_retinaFace, use_mtcnn, use_dlib_cnn):
    print("testing modells")
    #pil_bild = Image.open(image_path)
    #bild = np.asarray(pil_bild) 
    #bild = cv2.imread(img_path)
    #bild_hoehe, bild_breite = bild.shape[:2]

    # Modelle ausführen
    if use_dlib_cnn:
        print(f"Teste {image_path.name} mit Dlib CNN")
        faces = cnn_detector(image)
        for face in faces:
            confidence = face.confidence
            if confidence >= cnn_confidence:
                start_x, start_y, width, height = (
                    face.rect.left(),
                    face.rect.top(),
                    face.rect.width(),
                    face.rect.height(),
                )

                EXCLUSION_ZONES.append({
                    "x": start_x,
                    "y": start_y,
                    "w": width,
                    "h": height,
                    "model": "Dlib CNN",
                    "confidence": round(float(confidence), 2),
                    "image_name": image_path.name
                })

    if use_mtcnn:
        print(f"Teste {image_path.name} mit MTCNN")
        faces = detector.detect_faces(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for face in faces:
            confidence = face["confidence"]
            if confidence >= mtcnn_confidence:
                start_x, start_y, width, height = face["box"]

                EXCLUSION_ZONES.append({
                    "x": start_x,
                    "y": start_y,
                    "w": width,
                    "h": height,
                    "model": "MTCNN",
                    "confidence": round(float(confidence), 2),
                    "image_name": image_path.name
                })
    
    if use_retinaFace:
        print(f"Teste {image_path.name} mit RetinaFace")
        faces = app.get(image)
        for face in faces:
            confidence = face.det_score
            if confidence >= retina_confidence:
                start_x, start_y, end_x, end_y = map(int, face.bbox)

                EXCLUSION_ZONES.append({
                    "x": start_x,
                    "y": start_y,
                    "w": end_x - start_x,
                    "h": end_y - start_y,
                    "model": "RetinaFace",
                    "confidence": round(float(confidence), 2),
                    "image_name": image_path.name
                })

    if not any([use_retinaFace, use_mtcnn, use_dlib_cnn]):
        print("Es wurde kein Modell angewendet.")
                
    return mark_faces(image, image_path.name)
    
# markiert alle erkannten Bereiche auf dem Bild
def mark_faces(image, image_name):
    overlay = image.copy()
    for zone in EXCLUSION_ZONES:
        if zone["image_name"] == image_name:
            match zone["model"]:
                case "Dlib CNN":
                    model_color = CNN_COLOR
                case "MTCNN":
                    model_color = MTCNN_COLOR
                case "RetinaFace":
                    model_color = RETINA_COLOR
                case _:
                    model_color = EXTRA_MODEL_COLOR

            cv2.rectangle(
                overlay,
                (zone["x"], zone["y"]),
                (zone["x"] + zone["w"], zone["y"] + zone["h"]),
                model_color,
                DETECTION_THICKNESS,
            )
            cv2.putText(
                overlay,
                zone["model"] + ": " + str(zone["confidence"]),
                (zone["x"], zone["y"] - DETECTION_OFFSET),
                cv2.FONT_HERSHEY_SIMPLEX,
                DETECTION_FONT_SCALE,
                model_color,
                DETECTION_THICKNESS,
            )
    
    return overlay


def start_gui(original_image, image_path, use_retinaFace, use_mtcnn, use_dlib_cnn):
    root = tk.Tk()
    root.title("Cranach Detector GUI")

    # Haupt-Container
    frame = tk.Frame(root)
    frame.pack(padx=10, pady=10)

    print("gomme mode: " + str(use_retinaFace) + " " + str(use_mtcnn) + " " + str(use_dlib_cnn))
    retinaFace_check = tk.BooleanVar(value=use_retinaFace)
    mtcnn_check = tk.BooleanVar(value=use_mtcnn)
    dlib_cnn_check = tk.BooleanVar(value=use_dlib_cnn)

    # Bild laden und proportional skalieren
    #overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    def show_image(image):
        img = Image.fromarray(image)

        max_width = 1080
        max_height = 1080
        original_width, original_height = img.size
        ratio = min(max_width / original_width, max_height / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        img_tk = ImageTk.PhotoImage(img)

        # Bildanzeige
        img_label = tk.Label(frame, image=img_tk)
        img_label.image = img_tk
        img_label.grid(row=0, column=0, columnspan=2)
    
    def start_detection(image):
        overlay = use_models(image, image_path, retinaFace_check.get(), mtcnn_check.get(), dlib_cnn_check.get())
        show_image(overlay)

    # Buttons unter dem Bild
    btn_next = tk.Button(frame, text="Weiter", command=lambda: print("Nächstes Bild"))
    btn_next.grid(row=1, column=1, sticky="e", padx=5, pady=5)

    btn_process = tk.Button(frame, text="Gesichter erkennen", command=lambda: start_detection(original_image))
    btn_process.grid(row=1, column=0, sticky="w", padx=5, pady=5)

    check_retinaFace = tk.Checkbutton(root, text="RetinaFace", variable=retinaFace_check)
    check_retinaFace.pack(padx=20, pady=20)

    check_mtcnn = tk.Checkbutton(root, text="MTCNN", variable=mtcnn_check)
    check_mtcnn.pack(padx=20, pady=20)

    check_dlib_cnn = tk.Checkbutton(root, text="Dlib CNN", variable=dlib_cnn_check)
    check_dlib_cnn.pack(padx=20, pady=20)

    show_image(original_image)

    root.mainloop()



CranachDetector()
