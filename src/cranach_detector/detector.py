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
dlib_cnn_confidence = 0.6
DLIB_CNN_COLOR = (228, 37, 54)  # Rot
DLIB_CNN_COLOR_DARK = (255, 0, 4)
mtcnn_confidence = 0.82
MTCNN_COLOR = (248, 156, 32)  # Orange
MTCNN_COLOR_DARK = (255, 106, 0)
retina_confidence = 0.52
RETINA_COLOR = (87, 144, 252)  # Blau
RETINA_COLOR_DARK = (37, 94, 255)
EXTRA_MODEL_COLOR = (150, 74, 139)  # Lila


DETECTION_OFFSET = 10
DETECTION_THICKNESS = 2
DETECTION_FONT_SCALE = 0.6

EXCLUSION_ZONES = []

### initierung
BASE_DIR = Path(__file__).resolve().parent
DLIB_CNN_PATH = BASE_DIR / "models" / "dlib_cnn" / "mmod_human_face_detector.dat"
cnn_detector = dlib.cnn_face_detection_model_v1(str(DLIB_CNN_PATH))
detector = MTCNN()
app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
app.prepare(ctx_id=0)


def CranachDetector(
    images=None, use_retinaFace=True, use_mtcnn=False, use_dlib_cnn=False
):
    # root = tk.Tk()
    # root.withdraw()
    imageList = formatImages(images)

    while imageList == []:
        folder = filedialog.askdirectory(title="Wähle einen Ordner mit Bildern")
        # root.destroy()

        if not folder:
            print("Kein Ordner ausgewählt.")
            break
        imageList = formatImages(folder)

    start_process(imageList, use_retinaFace, use_mtcnn, use_dlib_cnn)
    return EXCLUSION_ZONES


def start_process(imageList, use_retinaFace, use_mtcnn, use_dlib_cnn):
    retinaFace_mode = use_retinaFace
    mtcnn_mode = use_mtcnn
    dlib_cnn_mode = use_dlib_cnn
    for image_path in imageList:
        pil_image = Image.open(image_path).convert("RGB")
        image = np.asarray(pil_image)
        # overlay = use_models(image_path, retinaFace_mode, mtcnn_mode, dlib_cnn_mode)
        start_gui(image, image_path.name, retinaFace_mode, mtcnn_mode, dlib_cnn_mode)


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
def use_models(
    image,
    image_name,
    use_retinaFace,
    use_mtcnn,
    use_dlib_cnn,
    retina_threshold=retina_confidence,
    mtcnn_threshold=mtcnn_confidence,
    dlib_cnn_threshold=dlib_cnn_confidence,
):
    print("testing modells")
    # pil_bild = Image.open(image_path)
    # bild = np.asarray(pil_bild)
    # bild = cv2.imread(img_path)
    # bild_hoehe, bild_breite = bild.shape[:2]

    # Modelle ausführen
    if use_dlib_cnn:
        print(f"Teste {image_name} mit Dlib CNN")
        faces = cnn_detector(image)
        for face in faces:
            confidence = face.confidence
            if confidence >= dlib_cnn_threshold:
                start_x, start_y, width, height = (
                    face.rect.left(),
                    face.rect.top(),
                    face.rect.width(),
                    face.rect.height(),
                )

                EXCLUSION_ZONES.append(
                    {
                        "x": start_x,
                        "y": start_y,
                        "w": width,
                        "h": height,
                        "model": "Dlib CNN",
                        "confidence": round(float(confidence), 2),
                        "image_name": image_name,
                    }
                )

    if use_mtcnn:
        print(f"Teste {image_name} mit MTCNN")
        faces = detector.detect_faces(image)
        for face in faces:
            confidence = face["confidence"]
            if confidence >= mtcnn_threshold:
                start_x, start_y, width, height = face["box"]

                EXCLUSION_ZONES.append(
                    {
                        "x": start_x,
                        "y": start_y,
                        "w": width,
                        "h": height,
                        "model": "MTCNN",
                        "confidence": round(float(confidence), 2),
                        "image_name": image_name,
                    }
                )

    if use_retinaFace:
        print(f"Teste {image_name} mit RetinaFace")
        faces = app.get(image)
        for face in faces:
            confidence = face.det_score
            if confidence >= retina_threshold:
                start_x, start_y, end_x, end_y = map(int, face.bbox)

                EXCLUSION_ZONES.append(
                    {
                        "x": start_x,
                        "y": start_y,
                        "w": end_x - start_x,
                        "h": end_y - start_y,
                        "model": "RetinaFace",
                        "confidence": round(float(confidence), 2),
                        "image_name": image_name,
                    }
                )

    if not any([use_retinaFace, use_mtcnn, use_dlib_cnn]):
        print("Es wurde kein Modell angewendet.")

    return mark_faces(image, image_name)


# markiert alle erkannten Bereiche auf dem Bild
def mark_faces(image, image_name, use_dark_colors=False):
    overlay = image.copy()
    for zone in EXCLUSION_ZONES:
        if zone["image_name"] == image_name:
            if use_dark_colors:
                match zone["model"]:
                    case "Dlib CNN":
                        model_color = DLIB_CNN_COLOR_DARK
                    case "MTCNN":
                        model_color = MTCNN_COLOR_DARK
                    case "RetinaFace":
                        model_color = RETINA_COLOR_DARK
                    case _:
                        model_color = EXTRA_MODEL_COLOR
            else:
                match zone["model"]:
                    case "Dlib CNN":
                        model_color = DLIB_CNN_COLOR
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


# Entfernt alle makierungen aus EXCLUSION_ZONES von Modellen die nicht auf das Bild angewendet werden
def clean_marks(image_name, use_retinaFace, use_mtcnn, use_dlib_cnn):
    if all([use_retinaFace, use_mtcnn, use_dlib_cnn]):
        return

    EXCLUSION_ZONES[:] = [
        zone
        for zone in EXCLUSION_ZONES
        if zone["image_name"] != image_name
        or (zone["model"] == "RetinaFace" and use_retinaFace)
        or (zone["model"] == "MTCNN" and use_mtcnn)
        or (zone["model"] == "Dlib CNN" and use_dlib_cnn)
    ]


def start_gui(original_image, image_name, use_retinaFace, use_mtcnn, use_dlib_cnn):
    root = tk.Tk()
    root.title("Cranach Detector GUI")

    # Haupt-Container
    frame = tk.Frame(root)
    frame.pack(padx=10, pady=10)

    use_dark_color = False
    retina_used = (False, 0.0)
    mtcnn_used = (False, 0.0)
    dlib_cnn_used = (False, 0.0)
    retinaFace_check = tk.BooleanVar(value=use_retinaFace)
    mtcnn_check = tk.BooleanVar(value=use_mtcnn)
    dlib_cnn_check = tk.BooleanVar(value=use_dlib_cnn)
    retina_threshold = tk.DoubleVar(value=retina_confidence)
    mtcnn_threshold = tk.DoubleVar(value=mtcnn_confidence)
    dlib_cnn_threshold = tk.DoubleVar(value=dlib_cnn_confidence)

    input_frame = tk.Frame(root)
    input_frame.pack(pady=10)

    # Bild laden und proportional skalieren
    def show_image(image):
        img = Image.fromarray(image)
        max_w, max_h = 1080, 1080
        w, h = img.size
        ratio = min(max_w / w, max_h / h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        lbl = tk.Label(frame, image=img_tk)
        lbl.image = img_tk
        lbl.grid(row=0, column=0, columnspan=2)

    def start_detection(image):
        nonlocal use_dark_color
        nonlocal retina_used
        nonlocal mtcnn_used
        nonlocal dlib_cnn_used
        retina_needs_rebuild = True
        mtcnn_needs_rebuild = True
        dlib_cnn_needs_rebuild = True

        if retina_used == (retinaFace_check.get(), retina_threshold.get()):
            retina_needs_rebuild = False
        else:
            retina_used = retinaFace_check.get(), retina_threshold.get()

        if mtcnn_used == (mtcnn_check.get(), mtcnn_threshold.get()):
            mtcnn_needs_rebuild = False
        else:
            mtcnn_used = mtcnn_check.get(), mtcnn_threshold.get()

        if dlib_cnn_used == (dlib_cnn_check.get(), dlib_cnn_threshold.get()):
            dlib_cnn_needs_rebuild = False
        else:
            dlib_cnn_used = dlib_cnn_check.get(), dlib_cnn_threshold.get()

        clean_marks(
            image_name,
            retinaFace_check.get() and not retina_needs_rebuild,
            mtcnn_check.get() and not mtcnn_needs_rebuild,
            dlib_cnn_check.get() and not dlib_cnn_needs_rebuild,
        )
        overlay = use_models(
            image,
            image_name,
            retinaFace_check.get() and retina_needs_rebuild,
            mtcnn_check.get() and mtcnn_needs_rebuild,
            dlib_cnn_check.get() and dlib_cnn_needs_rebuild,
            retina_threshold.get(),
            mtcnn_threshold.get(),
            dlib_cnn_threshold.get(),
        )
        if use_dark_color:
            show_image(mark_faces(original_image, image_name, use_dark_color))
        else:
            show_image(overlay)
        print(EXCLUSION_ZONES)

    def toggle_detection_color():
        nonlocal use_dark_color
        if use_dark_color:
            use_dark_color = False
        else:
            use_dark_color = True
        show_image(mark_faces(original_image, image_name, use_dark_color))

    # Buttons
    # RetinaFace
    tk.Checkbutton(input_frame, text="RetinaFace", variable=retinaFace_check).grid(
        row=0, column=0, sticky="w"
    )
    tk.Entry(input_frame, textvariable=retina_threshold, validate="key", width=5).grid(
        row=0, column=1
    )
    tk.Label(input_frame, text="Confidence Grenzwert").grid(
        row=0, column=2, padx=(0, 20)
    )

    # MTCNN
    tk.Checkbutton(input_frame, text="MTCNN", variable=mtcnn_check).grid(
        row=1, column=0, sticky="w"
    )
    tk.Entry(input_frame, textvariable=mtcnn_threshold, validate="key", width=5).grid(
        row=1, column=1
    )
    tk.Label(input_frame, text="Confidence Grenzwert").grid(
        row=1, column=2, padx=(0, 20)
    )

    # Dlib CNN
    tk.Checkbutton(input_frame, text="Dlib CNN", variable=dlib_cnn_check).grid(
        row=2, column=0, sticky="w"
    )
    tk.Entry(
        input_frame, textvariable=dlib_cnn_threshold, validate="key", width=5
    ).grid(row=2, column=1)
    tk.Label(input_frame, text="Confidence Grenzwert").grid(
        row=2, column=2, padx=(0, 20)
    )

    tk.Button(frame, text="Weiter", command=lambda: print("Nächstes Bild")).grid(
        row=1, column=1, sticky="e", padx=5, pady=5
    )
    tk.Button(
        frame,
        text="Gesichter erkennen",
        command=lambda: start_detection(original_image),
    ).grid(row=1, column=0, sticky="w", padx=5, pady=5)
    tk.Button(frame, text="Dunkele Makierungen", command=toggle_detection_color).grid(
        row=2, column=0, columnspan=2, pady=(0, 5)
    )

    start_detection(original_image)
    root.mainloop()


CranachDetector()
