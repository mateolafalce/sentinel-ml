"""MultiOutputClassifier model with scikit-learn."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split

from src.data.generator import LABEL_NAMES
from src.metrics import compute_all_metrics


class SklearnMultiLabel:
    def __init__(self):
        self.model = None
        self.trained = False

    def train(self, X: np.ndarray, Y: np.ndarray) -> dict:
        base = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model = MultiOutputClassifier(base)

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )
        self.model.fit(X_train, Y_train)
        self.trained = True

        Y_pred = self.model.predict(X_test)
        metrics = compute_all_metrics(Y_test, Y_pred)
        metrics["samples_train"] = len(X_train)
        metrics["samples_test"] = len(X_test)
        return metrics

    def predict(self, X: np.ndarray) -> list[dict]:
        preds = self.model.predict(X)
        probas = np.array([est.predict_proba(X)[:, 1] for est in self.model.estimators_]).T

        results = []
        for i in range(len(X)):
            labels = {}
            for j, name in enumerate(LABEL_NAMES):
                labels[name] = {
                    "activo": bool(preds[i][j]),
                    "probabilidad": round(float(probas[i][j]), 4),
                }
            results.append(labels)
        return results
