"""Metrics for multi-label classification: Hamming Loss and per-label F1."""

import numpy as np
from sklearn.metrics import hamming_loss, f1_score, accuracy_score

from src.data.generator import LABEL_NAMES


def compute_all_metrics(Y_true: np.ndarray, Y_pred: np.ndarray) -> dict:
    metrics = {
        "hamming_loss": round(float(hamming_loss(Y_true, Y_pred)), 6),
        "exact_match_ratio": round(float(accuracy_score(Y_true, Y_pred)), 6),
        "f1_micro": round(float(f1_score(Y_true, Y_pred, average="micro", zero_division=0)), 6),
        "f1_macro": round(float(f1_score(Y_true, Y_pred, average="macro", zero_division=0)), 6),
        "f1_per_label": {},
    }

    f1_per_label = f1_score(Y_true, Y_pred, average=None, zero_division=0)
    for i, name in enumerate(LABEL_NAMES):
        metrics["f1_per_label"][name] = round(float(f1_per_label[i]), 6)

    return metrics
