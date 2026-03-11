"""Tabular feature extraction from videos using YOLOv8."""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

from src.data.ucf_crime import CATEGORY_TO_LABELS, CATEGORIES

# Features extracted per video segment
EXTRACTED_FEATURE_NAMES = [
    "avg_personas",           # Average number of persons detected per frame
    "max_personas",           # Maximum number of persons in a single frame
    "avg_confianza_persona",  # Average confidence of person detections
    "area_persona_max",       # Max person bounding-box area / frame area
    "intensidad_movimiento",  # Average difference between consecutive frames
    "clases_unicas",          # Number of distinct classes detected
    "detecciones_promedio",   # Average total detections per frame
    "velocidad_persona",      # Estimated centroid displacement between frames
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
    """Extracts 8 tabular features from a video using YOLO.

    Args:
        video_path: Path to the video file.
        frame_interval: Process every N frames (30 ≈ 1fps for 30fps video).
        max_frames: Maximum number of frames to process per video.

    Returns:
        Array of 8 features, or None if the video cannot be opened.
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

        # Motion (difference between frames)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            motion_values.append(float(diff.mean()) / 255.0)
        prev_gray = gray

        # YOLO detection
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

    # Compute person speed (centroid displacement between frames)
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
    """Processes a list of videos and returns features + labels.

    Args:
        video_list: List of (video_path, category).
        frame_interval: Process every N frames.
        max_frames: Maximum number of frames per video.
        progress_callback: Function called with (current, total, video_name).

    Returns:
        (X, Y, video_names) — features, labels, names of processed files.
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
