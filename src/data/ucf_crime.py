"""Gestión del dataset UCF-Crime y mapeo de categorías a etiquetas multi-label."""

import os
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "ucf-crime"

# Las 13 categorías de anomalía + Normal
CATEGORIES = [
    "Normal",
    "Abuse",
    "Arrest",
    "Arson",
    "Assault",
    "Burglary",
    "Explosion",
    "Fighting",
    "RoadAccidents",
    "Robbery",
    "Shooting",
    "Shoplifting",
    "Stealing",
    "Vandalism",
]

# Mapeo: categoría → [intrusion_probable, requiere_verificacion_visual,
#                       notificar_propietario, despachar_movil]
CATEGORY_TO_LABELS = {
    "Normal":        [0, 0, 0, 0],
    "Abuse":         [1, 1, 1, 1],
    "Arrest":        [0, 1, 1, 0],
    "Arson":         [1, 1, 1, 1],
    "Assault":       [1, 1, 1, 1],
    "Burglary":      [1, 1, 1, 1],
    "Explosion":     [1, 1, 1, 1],
    "Fighting":      [1, 1, 1, 1],
    "RoadAccidents": [0, 1, 1, 1],
    "Robbery":       [1, 1, 1, 1],
    "Shooting":      [1, 1, 1, 1],
    "Shoplifting":   [1, 1, 1, 0],
    "Stealing":      [1, 1, 1, 0],
    "Vandalism":     [1, 1, 1, 0],
}


def get_video_paths() -> list[tuple[str, str]]:
    """Retorna lista de (ruta_video, categoría) para todos los videos encontrados."""
    videos = []
    for category in CATEGORIES:
        cat_dir = DATA_DIR / category
        if not cat_dir.exists():
            continue
        for f in sorted(cat_dir.iterdir()):
            if f.suffix.lower() in (".mp4", ".avi", ".mkv"):
                videos.append((str(f), category))
    return videos


def get_dataset_status() -> dict:
    """Retorna estado actual del dataset: qué categorías tienen videos."""
    status = {}
    total = 0
    for category in CATEGORIES:
        cat_dir = DATA_DIR / category
        if cat_dir.exists():
            count = sum(1 for f in cat_dir.iterdir() if f.suffix.lower() in (".mp4", ".avi", ".mkv"))
        else:
            count = 0
        status[category] = count
        total += count
    return {"categorias": status, "total_videos": total, "path": str(DATA_DIR)}


def setup_directories():
    """Crea la estructura de directorios para el dataset."""
    for category in CATEGORIES:
        (DATA_DIR / category).mkdir(parents=True, exist_ok=True)
    return str(DATA_DIR)
