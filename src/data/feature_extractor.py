"""Extracción de features tabulares desde videos usando YOLOv8."""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

from src.data.ucf_crime import CATEGORY_TO_LABELS, CATEGORIES

# Features extraídas por segmento de video
EXTRACTED_FEATURE_NAMES = [
    "avg_personas",           # Promedio de personas detectadas por frame
    "max_personas",           # Máximo de personas en un frame
    "avg_confianza_persona",  # Confianza promedio de detección de personas
    "area_persona_max",       # Área máxima de bbox de persona / área del frame
    "intensidad_movimiento",  # Diferencia promedio entre frames consecutivos
    "clases_unicas",          # Número de clases distintas detectadas
    "detecciones_promedio",   # Promedio total de detecciones por frame
    "velocidad_persona",      # Desplazamiento estimado de centroide entre frames
]

_model = None


def _get_model():
    global _model
    if _model is None:
        _model = YOLO("yolov8n.pt")
    return _model


def extract_features_from_video(
    video_path: str,
    frame_interval: int = 30,
    max_frames: int = 60,
) -> np.ndarray | None:
    """Extrae 8 features tabulares de un video usando YOLO.

    Args:
        video_path: Ruta al archivo de video.
        frame_interval: Procesar cada N frames (30 ≈ 1fps para video 30fps).
        max_frames: Máximo de frames a procesar por video.

    Returns:
        Array de 8 features o None si el video no se puede abrir.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    model = _get_model()
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_area = max(frame_w * frame_h, 1)

    person_counts = []
    person_confidences = []
    person_areas = []
    total_detections = []
    unique_classes = set()
    motion_values = []
    person_centroids = []

    prev_gray = None
    frame_idx = 0
    processed = 0

    while processed < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        # Movimiento (diferencia entre frames)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            motion_values.append(float(diff.mean()) / 255.0)
        prev_gray = gray

        # YOLO detección
        results = model(frame, verbose=False)[0]
        boxes = results.boxes

        n_persons = 0
        best_person_area = 0.0
        best_centroid = None

        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            cls_name = model.names[cls_id]
            unique_classes.add(cls_name)

            if cls_name == "person":
                n_persons += 1
                person_confidences.append(conf)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                area = (x2 - x1) * (y2 - y1) / frame_area
                person_areas.append(area)
                if area > best_person_area:
                    best_person_area = area
                    best_centroid = ((x1 + x2) / 2, (y1 + y2) / 2)

        person_counts.append(n_persons)
        total_detections.append(len(boxes))
        person_centroids.append(best_centroid)

        processed += 1
        frame_idx += 1

    cap.release()

    if processed == 0:
        return None

    # Calcular velocidad de persona (desplazamiento de centroide entre frames)
    speeds = []
    for i in range(1, len(person_centroids)):
        if person_centroids[i] is not None and person_centroids[i - 1] is not None:
            dx = person_centroids[i][0] - person_centroids[i - 1][0]
            dy = person_centroids[i][1] - person_centroids[i - 1][1]
            speed = np.sqrt(dx**2 + dy**2) / max(frame_w, 1)
            speeds.append(speed)

    features = np.array([
        np.mean(person_counts) if person_counts else 0,
        np.max(person_counts) if person_counts else 0,
        np.mean(person_confidences) if person_confidences else 0,
        np.max(person_areas) if person_areas else 0,
        np.mean(motion_values) if motion_values else 0,
        len(unique_classes),
        np.mean(total_detections) if total_detections else 0,
        np.mean(speeds) if speeds else 0,
    ], dtype=np.float32)

    return features


def process_dataset(
    video_list: list[tuple[str, str]],
    frame_interval: int = 30,
    max_frames: int = 60,
    progress_callback=None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Procesa una lista de videos y retorna features + labels.

    Args:
        video_list: Lista de (ruta_video, categoría).
        frame_interval: Procesar cada N frames.
        max_frames: Máximo de frames por video.
        progress_callback: Función llamada con (current, total, video_name).

    Returns:
        (X, Y, video_names) — features, labels, nombres de archivos procesados.
    """
    X_list = []
    Y_list = []
    names = []

    for i, (path, category) in enumerate(video_list):
        video_name = Path(path).name
        if progress_callback:
            progress_callback(i + 1, len(video_list), video_name)

        features = extract_features_from_video(path, frame_interval, max_frames)
        if features is None:
            continue

        labels = CATEGORY_TO_LABELS.get(category, [0, 0, 0, 0])
        X_list.append(features)
        Y_list.append(labels)
        names.append(video_name)

    if not X_list:
        return np.array([]), np.array([]), []

    return np.array(X_list), np.array(Y_list), names
