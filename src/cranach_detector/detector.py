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
MAX_OVERLAP = 0.75

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


# Main Funktion
def CranachDetector(
    images=None, use_retinaFace=True, use_mtcnn=False, use_dlib_cnn=False
):
    """
    Erstellt eine Liste mit allen erkannten Gesichtern aus images.

    Args:
        images (None):
        images (string_path): (x, y, width, height)
        boxB: (x, y, width, height)

    Returns:
        bool: True, wenn sich die Rechtecke überschneiden.
    """
    image_list = format_images(images)

    while image_list == []:
        folder = filedialog.askdirectory(title="Wähle einen Ordner mit Bildern")
        if not folder:
            print("Kein Ordner ausgewählt.")
            break
        image_list = format_images(folder)

    start_gui(image_list, use_retinaFace, use_mtcnn, use_dlib_cnn)
    return EXCLUSION_ZONES


"""
# Läd die Bilder aus imageList und startet die GUI anzeige
def start_process(imageList, use_retinaFace, use_mtcnn, use_dlib_cnn):
    retinaFace_mode = use_retinaFace
    mtcnn_mode = use_mtcnn
    dlib_cnn_mode = use_dlib_cnn
    for image_path in imageList:
        pil_image = Image.open(image_path).convert("RGB")
        image = np.asarray(pil_image)
        start_gui(image, image_path.name, retinaFace_mode, mtcnn_mode, dlib_cnn_mode)
"""


# Überprüft die Art der Bild eingabe und formatiert sie zu einer Liste
def format_images(images) -> list:
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

    remove_intersections(image_name)
    # print(EXCLUSION_ZONES)
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


def remove_intersections(image_name):
    index_to_remove = set()
    for i, boxA in enumerate(EXCLUSION_ZONES):
        if boxA["image_name"] != image_name:
            continue

        box_tupelA = (boxA["x"], boxA["y"], boxA["w"], boxA["h"])
        for j, boxB in enumerate(EXCLUSION_ZONES):
            if (
                boxB["image_name"] != image_name
                or boxB["model"] == boxA["model"]
                or j <= i
            ):
                continue

            box_tupelB = (boxB["x"], boxB["y"], boxB["w"], boxB["h"])
            if not isIntersecting(box_tupelA, box_tupelB):
                continue

            ratio = overlap_ratio(box_tupelA, box_tupelB)
            if ratio >= MAX_OVERLAP:
                areaA = boxA["w"] * boxA["h"]
                areaB = boxB["w"] * boxB["h"]
                if areaA > areaB:
                    index_to_remove.add(i)
                else:
                    index_to_remove.add(j)

    for index in sorted(index_to_remove, reverse=True):
        del EXCLUSION_ZONES[index]


def isIntersecting(
    boxA: tuple[int, int, int, int], boxB: tuple[int, int, int, int]
) -> bool:
    """
    Prüft, ob sich zwei Rechtecke überschneiden.

    Args:
        boxA: (x, y, width, height)
        boxB: (x, y, width, height)

    Returns:
        bool: True, wenn sich die Rechtecke überschneiden.
    """
    a_x_start, a_y_start, a_width, a_height = boxA
    a_x_end = a_x_start + a_width
    a_y_end = a_y_start + a_height
    b_x_start, b_y_start, b_width, b_height = boxB
    b_x_end = b_x_start + b_width
    b_y_end = b_y_start + b_height

    overlap_x = (a_x_start < b_x_end) and (b_x_start < a_x_end)
    overlap_y = (a_y_start < b_y_end) and (b_y_start < a_y_end)
    return overlap_x and overlap_y


def position_isIntersecting(
    position: tuple[
        int,
        int,
    ],
    image_name: str,
    margin: int = 0,
) -> bool:
    """
    Prüft, ob sich die Position in einem markierten Bereich des Bildes befindet.
    Das Bild muss zuvor mit CranachDetector() bearbeitet worden sein.

    Args:
        position (tuple): (x, y)
        image_name (str): Bsp: "img/Bild.jpg"
        margin (int (Optional)): Mindestabstand in px, der zu position eingehalten werden muss.

    Returns:
        bool: True, wenn sich die Position in einem makierten Bereich befindet.
    """
    position_x, position_y = position
    for zone in EXCLUSION_ZONES:
        if zone["image_name"] != image_name:
            continue

        if margin > 0:
            position_area_x = position_x - margin
            position_area_y = position_y - margin
            position_area_length = margin * 2
            position_area = (
                position_area_x,
                position_area_y,
                position_area_length,
                position_area_length,
            )
            zone_tupel = (zone["x"], zone["y"], zone["w"], zone["h"])

            if isIntersecting(position_area, zone_tupel):
                return True

        zone_x_end = zone["x"] + zone["w"]
        zone_y_end = zone["y"] + zone["h"]

        overlap_x = (position_x < zone_x_end) and (zone["x"] < position_x)
        overlap_y = (position_y < zone_y_end) and (zone["y"] < position_y)
        if overlap_x and overlap_y:
            return True

    return False


def area_isIntersecting(
    area: tuple[int, int, int, int], image_name: str, margin: int = 0
) -> bool:
    """
    Prüft, ob sich area in einem markierten Bereich des Bildes befindet.
    Das Bild muss zuvor mit CranachDetector() bearbeitet worden sein.

    Args:
        position (tuple): (x, y, width, height)
        image_name (str): Bsp: "img/Bild.jpg"
        margin (int (Optional)): Mindestabstand in px, der zu position eingehalten werden muss.

    Returns:
        bool: True, wenn sich die Position in einem makierten Bereich befindet.
    """
    (
        area_x_start,
        area_y_start,
        area_width,
        area_height,
    ) = area
    area_x_end = area_x_start + area_width
    area_y_end = area_y_start + area_height
    for zone in EXCLUSION_ZONES:
        if zone["image_name"] != image_name:
            continue

        zone_x_end = zone["x"] + zone["w"]
        zone_y_end = zone["y"] + zone["h"]

        if margin > 0:
            margin_area_x = area_x_start - margin
            margin_area_y = area_y_start - margin
            margin_area_width = area_width + margin * 2
            margin_area_height = area_height + margin * 2
            margin_area = (
                margin_area_x,
                margin_area_y,
                margin_area_width,
                margin_area_height,
            )
            zone_tupel = (zone["x"], zone["y"], zone["w"], zone["h"])

            if isIntersecting(margin_area, zone_tupel):
                return True

        overlap_x = (area_x_start < zone_x_end) and (zone["x"] < area_x_end)
        overlap_y = (area_y_start < zone_y_end) and (zone["y"] < area_y_end)
        if overlap_x and overlap_y:
            return True

    return False


# Berechnet wie groß der Anteil der Fläche ist, mit welcher sich zwei Rechtecke überschneiden
def overlap_ratio(boxA, boxB):
    a_x_start, a_y_start, a_width, a_height = boxA
    b_x_start, b_y_start, b_width, b_height = boxB

    a_x_end = a_x_start + a_width
    a_y_end = a_y_start + a_height
    b_x_end = b_x_start + b_width
    b_y_end = b_y_start + b_height

    inner_x1 = max(a_x_start, b_x_start)
    inner_y1 = max(a_y_start, b_y_start)
    inner_x2 = min(a_x_end, b_x_end)
    inner_y2 = min(a_y_end, b_y_end)

    inner_w = max(0, inner_x2 - inner_x1)
    inner_h = max(0, inner_y2 - inner_y1)
    inner_area = inner_w * inner_h

    areaA = a_width * a_height
    areaB = b_width * b_height
    smaller_area = min(areaA, areaB)

    if smaller_area == 0:
        return 0.0
    return inner_area / smaller_area


# Main Funktion des GUI, inklusiver nötiger Funktionen
def start_gui(image_list, use_retinaFace, use_mtcnn, use_dlib_cnn):
    ### GUI Fenster
    root = tk.Tk()
    root.title("Cranach Detector")
    root.geometry("1280x720")
    root.minsize(854, 480)
    root.rowconfigure(0, weight=1)
    root.columnconfigure(0, weight=1)
    root.columnconfigure(1, weight=0)

    canvas = tk.Canvas(root)
    v_scroll = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
    h_scroll = tk.Scrollbar(root, orient="horizontal", command=canvas.xview)
    canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
    canvas.bind_all(
        "<MouseWheel>",
        lambda e: canvas.yview_scroll(int(-1 * (e.delta / 120)), "units"),
    )
    canvas.bind_all("<Button-4>", lambda e: canvas.yview_scroll(-1, "units"))
    canvas.bind_all("<Button-5>", lambda e: canvas.yview_scroll(1, "units"))

    canvas.grid(row=0, column=0, sticky="nsew")
    v_scroll.grid(row=0, column=0, sticky="nse")
    h_scroll.grid(row=1, column=0, sticky="ew")

    frame = tk.Frame(canvas)
    frame.grid_columnconfigure(0, weight=1)
    canvas.create_window((0, 0), window=frame, anchor="nw")
    frame.bind(
        "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    input_frame = tk.Frame(root)
    input_frame.grid(row=0, column=1, sticky="nw", padx=5, pady=5)
    for col in range(3):
        input_frame.columnconfigure(col, weight=1)

    ### Variablen
    use_dark_color = False
    image_index = 0
    image_name = ""
    original_image = ""
    retinaFace_check = tk.BooleanVar(value=use_retinaFace)
    mtcnn_check = tk.BooleanVar(value=use_mtcnn)
    dlib_cnn_check = tk.BooleanVar(value=use_dlib_cnn)
    retina_threshold = tk.DoubleVar(value=retina_confidence)
    mtcnn_threshold = tk.DoubleVar(value=mtcnn_confidence)
    dlib_cnn_threshold = tk.DoubleVar(value=dlib_cnn_confidence)
    retina_used = (False, 0.0, 0)
    mtcnn_used = (False, 0.0, 0)
    dlib_cnn_used = (False, 0.0, 0)

    ### Fuktionen die für Variablen Initierung gebraucht werden
    def loading_image():
        nonlocal image_index
        nonlocal image_name
        nonlocal original_image
        image_path = image_list[image_index]
        image_name = image_path.name
        pil_image = Image.open(image_path).convert("RGB")
        original_image = np.asarray(pil_image)

    def count_entries(image_name, model):
        return sum(
            1
            for box in EXCLUSION_ZONES
            if box["image_name"] == image_name and box["model"] == model
        )

    loading_image()

    # Bild laden und proportional skalieren
    def show_image(image):
        for w in frame.winfo_children():
            w.destroy()
        img = Image.fromarray(image)
        max_w, max_h = 1080, 1080
        w, h = img.size
        ratio = min(max_w / w, max_h / h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        lbl = tk.Label(frame, image=img_tk)
        lbl.image = img_tk
        lbl.pack(anchor="ne", expand=True)

    def start_detection(image):
        nonlocal use_dark_color
        nonlocal retina_used
        nonlocal mtcnn_used
        nonlocal dlib_cnn_used
        retina_needs_rebuild = True
        mtcnn_needs_rebuild = True
        dlib_cnn_needs_rebuild = True

        # Überprüft ob änderungen Vorgenommen wurden
        if retina_used == (
            retinaFace_check.get(),
            retina_threshold.get(),
            count_entries(image_name, "RetinaFace"),
        ):
            retina_needs_rebuild = False
        else:
            retina_used = (
                retinaFace_check.get(),
                retina_threshold.get(),
                count_entries(image_name, "RetinaFace"),
            )

        if mtcnn_used == (
            mtcnn_check.get(),
            mtcnn_threshold.get(),
            count_entries(image_name, "MTCNN"),
        ):
            mtcnn_needs_rebuild = False
        else:
            mtcnn_used = (
                mtcnn_check.get(),
                mtcnn_threshold.get(),
                count_entries(image_name, "MTCNN"),
            )

        if dlib_cnn_used == (
            dlib_cnn_check.get(),
            dlib_cnn_threshold.get(),
            count_entries(image_name, "Dlib CNN"),
        ):
            dlib_cnn_needs_rebuild = False
        else:
            dlib_cnn_used = (
                dlib_cnn_check.get(),
                dlib_cnn_threshold.get(),
                count_entries(image_name, "Dlib CNN"),
            )

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

    def toggle_detection_color():
        nonlocal use_dark_color
        if use_dark_color:
            use_dark_color = False
        else:
            use_dark_color = True
        show_image(mark_faces(original_image, image_name, use_dark_color))

    def next_image():
        nonlocal image_index
        nonlocal retina_used
        nonlocal mtcnn_used
        nonlocal dlib_cnn_used
        if image_index < len(image_list) - 1:
            image_index += 1
            loading_image()
            retina_used = (False, 0.0, 0)
            mtcnn_used = (False, 0.0, 0)
            dlib_cnn_used = (False, 0.0, 0)
            start_detection(original_image)
        else:
            root.quit()

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

    # Buttons
    tk.Button(input_frame, text="Weiter", command=next_image).grid(
        row=5, column=0, columnspan=3, pady=40, sticky="ew"
    )
    tk.Button(
        input_frame,
        text="Modell/e anwenden",
        command=lambda: start_detection(original_image),
    ).grid(row=4, column=0, columnspan=3, pady=5, sticky="ew")
    tk.Button(
        input_frame, text="Dunkle Makierungen", command=toggle_detection_color
    ).grid(row=3, column=0, columnspan=3, pady=5, sticky="ew")

    start_detection(original_image)
    root.mainloop()


print(CranachDetector())
