"""UCF-Crime dataset management and category-to-multi-label mapping."""

import os
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "ucf-crime"

# The 13 anomaly categories + Normal
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

# Mapping: category → [intrusion_probable, requiere_verificacion_visual,
#                        notificar_propietario, despachar_movil]
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
    """Returns a list of (video_path, category) for all videos found."""
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
    """Returns the current dataset status: which categories have videos."""
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
    """Creates the directory structure for the dataset."""
    for category in CATEGORIES:
        (DATA_DIR / category).mkdir(parents=True, exist_ok=True)
    return str(DATA_DIR)
