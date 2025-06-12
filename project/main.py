### imports
import cv2
import dlib
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from insightface.app import FaceAnalysis
from mtcnn import MTCNN
from pathlib import Path
from cranach_detector import *

### variables
########## Selection Menu ##########
# | haar | caffe | pipe | hog | landmark | cnn | mtcnn | yunet | retina |
TEST_MODELS = ["caffe"]
########## Selection Menu END ##########

FOLDER_PATHS = ["project/img/1Portrait"]
CAFFE_CONFIDENCE = 0.14
CAFFE_COLOR = (255, 50, 50)  # Blau
CNN_CONFIDENCE = 0.6
CNN_COLOR = (0, 255, 0)  # Grün
MTCNN_CONFIDENCE = 0.82
MTCNN_COLOR = (0, 0, 255)  # Rot
YUNET_CONFIDENCE = 125
YUNET_COLOR = (255, 255, 0)  # Türkis
RETINA_CONFIDENCE = 0.52
RETINA_COLOR = (0, 255, 255)  # Gelb

DETECTION_OFFSET = 10
DETECTION_THICKNESS = 2
DETECTION_FONT_SCALE = 0.6

### initierung
for model in TEST_MODELS:
    match model:
        case "haar":
            haar_cascade = cv2.CascadeClassifier(
                "project/models/haar_cascade/haarcascade_frontalface_default.xml"
            )
        case "caffe":
            model = "project/models/caffe/res10_300x300_ssd_iter_140000.caffemodel"
            prototxt = "project/models/caffe/deploy.prototxt"
            netz = cv2.dnn.readNetFromCaffe(prototxt, model)
        case "pipe":
            mp_face_detection = mp.solutions.face_detection
            mp_drawing = mp.solutions.drawing_utils
        case "hog":
            # Initialisiere dlib Gesichtserkenner (HOG-basiert)
            detector = dlib.get_frontal_face_detector()
        case "cnn":
            # dlib CNN-Modell
            cnn_detector = dlib.cnn_face_detection_model_v1(
                "project\models\dlib_cnn\mmod_human_face_detector.dat"
            )
        case "landmark":
            landmark_model = (
                "project/models/dlib_hog_landmark/shape_predictor_68_face_landmarks.dat"
            )
            # Initialisiere dlib-Modelle
            detector = dlib.get_frontal_face_detector()  # HOG-Gesichtserkennung
            predictor = dlib.shape_predictor(landmark_model)  # Landmark-Predictor
        case "mtcnn":
            detector = MTCNN()
        case "yunet":
            detektor = cv2.FaceDetectorYN.create(
                model="project/models/yunet/face_detection_yunet_2023mar.onnx",
                config="",
                input_size=(320, 320),
                score_threshold=0.8,  # 0.9 ist default
            )
        case "retina":
            app = FaceAnalysis(
                name="buffalo_l", providers=["CPUExecutionProvider"]
            )  # oder CUDA für GPU
            app.prepare(ctx_id=0)
        case _:
            print("Initiniere nichts.")


## Erzeugt eine Liste aus allen .jfif Bildern aus folder_paths
def get_images(folder_paths, exts=(".jfif",)):
    print("getting images")
    imgs = []
    for folder in folder_paths:
        imgs += list(Path(folder).rglob("*"))
    return [p for p in imgs if p.suffix.lower() in exts]


## Wendet das Modell auf alle Bilder in image_paths an
def test_models(image_paths, models):
    print("testing modells")
    for img_path in image_paths:
        bild = cv2.imread(img_path)
        overlay = bild.copy()
        bild_hoehe, bild_breite = bild.shape[:2]

        for model in models:
            print(f"Teste {img_path.name} mit {model}")
            match model:
                case "haar":
                    grau = cv2.cvtColor(
                        bild, cv2.COLOR_BGR2GRAY
                    )  # In Graustufen umwandeln (erforderlich für Haar-Cascades)
                    gesichter = haar_cascade.detectMultiScale(
                        grau, scaleFactor=1.1, minNeighbors=3, minSize=(30, 30)
                    )

                    for start_x, start_y, width, height in gesichter:
                        cv2.rectangle(
                            overlay,
                            (start_x, start_y),
                            (start_x + width, start_y + height),
                            (255, 0, 0),
                            DETECTION_THICKNESS,
                        )
                case "caffe":
                    blob = cv2.dnn.blobFromImage(
                        bild, 1.0, (300, 300), (104.0, 177.0, 123.0)
                    )
                    netz.setInput(blob)
                    ergebnisse = netz.forward()

                    for i in range(0, ergebnisse.shape[2]):
                        confidence = ergebnisse[0, 0, i, 2]
                        if confidence >= CAFFE_CONFIDENCE:
                            box = ergebnisse[0, 0, i, 3:7] * [
                                bild_breite,
                                bild_hoehe,
                                bild_breite,
                                bild_hoehe,
                            ]
                            (start_x, start_y, end_x, end_y) = box.astype("int")

                            cv2.rectangle(
                                overlay,
                                (start_x, start_y),
                                (end_x, end_y),
                                CAFFE_COLOR,
                                DETECTION_THICKNESS,
                            )
                            cv2.putText(
                                overlay,
                                f"caffe: {confidence:.2f}",
                                (start_x, start_y - DETECTION_OFFSET),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                DETECTION_FONT_SCALE,
                                CAFFE_COLOR,
                                DETECTION_THICKNESS,
                            )
                case "pipe":
                    ## MediaPipe Face Detection Setup (MediaPipe arbeitet mit RGB)
                    with mp_face_detection.FaceDetection(
                        min_detection_confidence=0.2
                    ) as face_detection:
                        rgb_image = cv2.cvtColor(bild, cv2.COLOR_BGR2RGB)

                        results = face_detection.process(rgb_image)
                        if results.detections:
                            for detection in results.detections:
                                mp_drawing.draw_detection(overlay, detection)
                case "hog":
                    ## Konvertiere Bild in Graustufen
                    gray = cv2.cvtColor(bild, cv2.COLOR_BGR2GRAY)
                    faces, scores, _ = detector.run(
                        gray, 2, -0.5
                    )  ## Zweiter Wert: Confidence-Schwelle (Standard ist 0.0).

                    for face in faces:
                        start_x, start_y, width, height = (
                            face.left(),
                            face.top(),
                            face.width(),
                            face.height(),
                        )
                        cv2.rectangle(
                            overlay,
                            (start_x, start_y),
                            (start_x + width, start_y + height),
                            (0, 255, 0),
                            DETECTION_THICKNESS,
                        )
                case "cnn":
                    faces = cnn_detector(bild)
                    for face in faces:
                        confidence = face.confidence
                        if confidence >= CNN_CONFIDENCE:
                            start_x, start_y, width, height = (
                                face.rect.left(),
                                face.rect.top(),
                                face.rect.width(),
                                face.rect.height(),
                            )
                            cv2.rectangle(
                                overlay,
                                (start_x, start_y),
                                (start_x + width, start_y + height),
                                CNN_COLOR,
                                DETECTION_THICKNESS,
                            )
                            cv2.putText(
                                overlay,
                                f"cnn: {confidence:.2f}",
                                (start_x, start_y - DETECTION_OFFSET),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                DETECTION_FONT_SCALE,
                                CNN_COLOR,
                                DETECTION_THICKNESS,
                            )
                case "landmark":
                    ###### ChatGPT Code Start. (Filter für erkannte Landmark Gesichter für genauere Ergbnisse) ######
                    # Funktionen zur Landmark-Validierung
                    def validate_landmarks(
                        rect,
                        landmarks,
                        sym_thresh=0.2,
                        area_thresh=(0.5, 1.0),
                        ratio_thresh=0.3,
                    ):
                        # Symmetrie-Check
                        left_eye = np.array(
                            [
                                (landmarks.part(i).x, landmarks.part(i).y)
                                for i in range(36, 42)
                            ]
                        )
                        right_eye = np.array(
                            [
                                (landmarks.part(i).x, landmarks.part(i).y)
                                for i in range(42, 48)
                            ]
                        )
                        le_center = left_eye.mean(axis=0)
                        re_center = right_eye.mean(axis=0)
                        eye_dist = np.linalg.norm(le_center - re_center)
                        sym_diff = abs(
                            (le_center[0] + re_center[0]) / 2 - rect.center().x
                        )
                        if sym_diff / eye_dist > sym_thresh:
                            return False

                        # Abstandsverhältnisse
                        nose = np.array([landmarks.part(30).x, landmarks.part(30).y])
                        mouth = np.array([landmarks.part(62).x, landmarks.part(62).y])
                        eye_nose = np.linalg.norm(((le_center + re_center) / 2) - nose)
                        nose_mouth = np.linalg.norm(nose - mouth)
                        if (
                            abs(eye_nose / eye_dist - 0.5) > ratio_thresh
                            or abs(nose_mouth / eye_dist - 0.7) > ratio_thresh
                        ):
                            return False

                        # Konvexhüllen-Check
                        pts = np.array(
                            [
                                [landmarks.part(i).x, landmarks.part(i).y]
                                for i in range(68)
                            ]
                        )
                        hull = cv2.convexHull(pts)
                        hull_area = cv2.contourArea(hull)
                        box_area = rect.width() * rect.height()
                        ratio = hull_area / box_area
                        if not (area_thresh[0] <= ratio <= area_thresh[1]):
                            return False

                        return True

                    def filter_faces_with_landmarks(gray, faces, predictor):
                        valid_faces = []
                        for rect in faces:
                            shape = predictor(gray, rect)
                            if validate_landmarks(rect, shape):
                                valid_faces.append((rect, shape))
                        return valid_faces

                    ###### ChatGPT Code Ende. ######

                    gray = cv2.cvtColor(bild, cv2.COLOR_BGR2GRAY)
                    faces, scores, _ = detector.run(gray, 2, -0.5)  ## HOG-Detektion
                    faces_validated = filter_faces_with_landmarks(
                        gray, faces, predictor
                    )  ## Filtern mit Landmark-Validierung

                    for rect, landmarks in faces_validated:
                        print(f"Bestätigtes Gesicht bei {rect}")

                        for j in range(68):
                            x, y = landmarks.part(j).x, landmarks.part(j).y
                            cv2.circle(
                                overlay, (x, y), DETECTION_THICKNESS, (0, 255, 0), -1
                            )
                case "mtcnn":
                    ## Erkenne Gesichter (MTCNN braucht RGB)
                    faces = detector.detect_faces(cv2.cvtColor(bild, cv2.COLOR_BGR2RGB))
                    for face in faces:
                        confidence = face["confidence"]
                        if confidence >= MTCNN_CONFIDENCE:
                            start_x, start_y, width, height = face["box"]
                            cv2.rectangle(
                                overlay,
                                (start_x, start_y),
                                (start_x + width, start_y + height),
                                MTCNN_COLOR,
                                DETECTION_THICKNESS,
                            )
                            cv2.putText(
                                overlay,
                                f"mtcnn: {confidence:.2f}",
                                (start_x, start_y - DETECTION_OFFSET),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                DETECTION_FONT_SCALE,
                                MTCNN_COLOR,
                                DETECTION_THICKNESS,
                            )
                case "yunet":
                    detektor.setInputSize((bild_breite, bild_hoehe))

                    _, gesichter = detektor.detect(bild)
                    if gesichter is not None and len(gesichter) > 0:
                        for gesicht in gesichter:
                            start_x, start_y, width, height, confidence = gesicht[:5]
                            start_x, start_y, width, height = (
                                int(start_x),
                                int(start_y),
                                int(width),
                                int(height),
                            )
                            if confidence >= YUNET_CONFIDENCE:
                                cv2.rectangle(
                                    overlay,
                                    (start_x, start_y),
                                    (start_x + width, start_y + height),
                                    YUNET_COLOR,
                                    DETECTION_THICKNESS,
                                )
                                cv2.putText(
                                    overlay,
                                    f"yunet: {confidence:.2f}",
                                    (start_x, start_y - DETECTION_OFFSET),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    DETECTION_FONT_SCALE,
                                    YUNET_COLOR,
                                    DETECTION_THICKNESS,
                                )
                    else:
                        print("Kein Gesicht erkannt.")
                case "retina":
                    gesichter = app.get(bild)
                    for gesicht in gesichter:
                        confidence = gesicht.det_score
                        if confidence >= RETINA_CONFIDENCE:
                            start_x, start_y, end_x, end_y = map(int, gesicht.bbox)
                            cv2.rectangle(
                                overlay,
                                (start_x, start_y),
                                (end_x, end_y),
                                RETINA_COLOR,
                                DETECTION_THICKNESS,
                            )
                            cv2.putText(
                                overlay,
                                f"retinaFace: {confidence:.2f}",
                                (start_x, start_y - DETECTION_OFFSET),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                DETECTION_FONT_SCALE,
                                RETINA_COLOR,
                                DETECTION_THICKNESS,
                            )
                case _:
                    print("Ich mach dann mal nichts...")

        if overlay is None:
            print("Bild konnte nicht geladen werden. Überprüfe den Pfad.")
        else:
            bild_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(12, 8))
            plt.imshow(bild_rgb)
            plt.axis("off")
            plt.title(f"{img_path.name} mit {models}")
            plt.show()


### ausführung
#imgs = get_images(FOLDER_PATHS)
#test_models(imgs, TEST_MODELS)
print(cranach_detector("project/img/3Portrait"))
#print(position_isIntersecting((367, 718), "Familie_der_Naturmenschen.jfif", 20))
