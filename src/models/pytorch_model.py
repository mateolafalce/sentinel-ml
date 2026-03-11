"""Modelo PyTorch con sigmoid output para clasificación multi-etiqueta."""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from src.data.generator import LABEL_NAMES
from src.metrics import compute_all_metrics


class MultiLabelNet(nn.Module):
    def __init__(self, n_features: int = 8, n_labels: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, n_labels),
        )

    def forward(self, x):
        return self.net(x)


class PyTorchMultiLabel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = MultiLabelNet().to(self.device)
        self.trained = False

    def train(self, X: np.ndarray, Y: np.ndarray, epochs: int = 50, lr: float = 0.001) -> dict:
        n_features = X.shape[1]
        n_labels = Y.shape[1]
        self.model = MultiLabelNet(n_features, n_labels).to(self.device)

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )

        X_train_t = torch.FloatTensor(X_train).to(self.device)
        Y_train_t = torch.FloatTensor(Y_train).to(self.device)
        X_test_t = torch.FloatTensor(X_test).to(self.device)

        dataset = TensorDataset(X_train_t, Y_train_t)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        history = []
        for epoch in range(epochs):
            self.model.train()
            epoch_loss = 0.0
            for X_batch, Y_batch in loader:
                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = criterion(output, Y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            history.append(epoch_loss / len(loader))

        self.model.eval()
        self.trained = True

        with torch.no_grad():
            logits = self.model(X_test_t)
            Y_pred = (torch.sigmoid(logits) > 0.5).cpu().numpy().astype(int)

        metrics = compute_all_metrics(Y_test, Y_pred)
        metrics["samples_train"] = len(X_train)
        metrics["samples_test"] = len(X_test)
        metrics["epochs"] = epochs
        metrics["final_loss"] = round(history[-1], 6)
        return metrics

    def predict(self, X: np.ndarray) -> list[dict]:
        self.model.eval()
        X_t = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            logits = self.model(X_t)
            probas = torch.sigmoid(logits).cpu().numpy()
            preds = (probas > 0.5).astype(int)

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
