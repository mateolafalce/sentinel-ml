"""Servidor Flask para la interfaz gráfica de Sentinel ML."""

import threading
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from src.data.generator import generate_dataset, FEATURE_NAMES, LABEL_NAMES
from src.data.ucf_crime import get_video_paths, get_dataset_status, setup_directories
from src.data.feature_extractor import (
    process_dataset,
    EXTRACTED_FEATURE_NAMES,
)
from src.models.sklearn_model import SklearnMultiLabel
from src.models.pytorch_model import PyTorchMultiLabel
from src.llm.reporter import generate_report

app = Flask(__name__, static_folder="../static")

# Estado global
sklearn_model = SklearnMultiLabel()
pytorch_model = PyTorchMultiLabel()
X_data, Y_data = None, None
data_source = None  # "synthetic" o "ucf-crime"
incidents = []  # historial de incidentes (máx 50)

# Estado de extracción en segundo plano
extraction_status = {
    "running": False,
    "progress": 0,
    "total": 0,
    "current_video": "",
    "done": False,
    "error": None,
}


@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(app.static_folder, path)


@app.route("/api/info", methods=["GET"])
def info():
    feature_names = EXTRACTED_FEATURE_NAMES if data_source == "ucf-crime" else FEATURE_NAMES
    return jsonify({
        "features": feature_names,
        "labels": LABEL_NAMES,
        "sklearn_trained": sklearn_model.trained,
        "pytorch_trained": pytorch_model.trained,
        "data_source": data_source,
        "n_samples": len(X_data) if X_data is not None else 0,
    })


@app.route("/api/generate", methods=["POST"])
def generate():
    global X_data, Y_data, data_source
    body = request.get_json(silent=True) or {}
    n_samples = body.get("n_samples", 2000)
    n_samples = max(100, min(n_samples, 50000))

    X_data, Y_data = generate_dataset(n_samples=n_samples)
    data_source = "synthetic"

    return jsonify({
        "mensaje": f"Dataset sintético generado con {n_samples} muestras",
        "source": "synthetic",
        "n_samples": n_samples,
        "n_features": X_data.shape[1],
        "n_labels": Y_data.shape[1],
        "distribucion_labels": {
            name: int(Y_data[:, i].sum()) for i, name in enumerate(LABEL_NAMES)
        },
    })


# --- UCF-Crime endpoints ---

@app.route("/api/ucf/status", methods=["GET"])
def ucf_status():
    return jsonify(get_dataset_status())


@app.route("/api/ucf/setup", methods=["POST"])
def ucf_setup():
    path = setup_directories()
    return jsonify({
        "mensaje": "Directorios creados. Coloca los videos en las subcarpetas correspondientes.",
        "path": path,
    })


@app.route("/api/ucf/extract", methods=["POST"])
def ucf_extract():
    global extraction_status

    if extraction_status["running"]:
        return jsonify({"error": "Ya hay una extracción en progreso"}), 409

    body = request.get_json(silent=True) or {}
    frame_interval = body.get("frame_interval", 30)
    max_frames = body.get("max_frames", 60)

    videos = get_video_paths()
    if not videos:
        return jsonify({
            "error": "No se encontraron videos. Usa /api/ucf/setup y coloca videos en las carpetas."
        }), 400

    extraction_status = {
        "running": True,
        "progress": 0,
        "total": len(videos),
        "current_video": "",
        "done": False,
        "error": None,
    }

    def run_extraction():
        global X_data, Y_data, data_source, extraction_status
        try:
            def on_progress(current, total, name):
                extraction_status["progress"] = current
                extraction_status["total"] = total
                extraction_status["current_video"] = name

            X, Y, names = process_dataset(
                videos,
                frame_interval=frame_interval,
                max_frames=max_frames,
                progress_callback=on_progress,
            )

            if len(X) == 0:
                extraction_status["error"] = "No se pudieron procesar videos"
            else:
                X_data = X
                Y_data = Y
                data_source = "ucf-crime"

            extraction_status["done"] = True
            extraction_status["running"] = False
        except Exception as e:
            extraction_status["error"] = str(e)
            extraction_status["done"] = True
            extraction_status["running"] = False

    thread = threading.Thread(target=run_extraction, daemon=True)
    thread.start()

    return jsonify({
        "mensaje": f"Extracción iniciada para {len(videos)} videos",
        "total_videos": len(videos),
    })


@app.route("/api/ucf/extract/status", methods=["GET"])
def ucf_extract_status():
    global X_data
    status = dict(extraction_status)
    if status["done"] and X_data is not None and data_source == "ucf-crime":
        status["n_samples"] = len(X_data)
        status["distribucion_labels"] = {
            name: int(Y_data[:, i].sum()) for i, name in enumerate(LABEL_NAMES)
        }
    return jsonify(status)


# --- Train & Predict ---

@app.route("/api/train", methods=["POST"])
def train():
    global X_data, Y_data
    if X_data is None:
        return jsonify({"error": "Primero genera o extrae un dataset"}), 400

    body = request.get_json(silent=True) or {}
    model_type = body.get("model", "sklearn")

    if model_type == "sklearn":
        metrics = sklearn_model.train(X_data, Y_data)
    elif model_type == "pytorch":
        epochs = body.get("epochs", 50)
        metrics = pytorch_model.train(X_data, Y_data, epochs=epochs)
    else:
        return jsonify({"error": f"Modelo no soportado: {model_type}"}), 400

    metrics["data_source"] = data_source
    return jsonify({"model": model_type, "metrics": metrics})


@app.route("/api/predict", methods=["POST"])
def predict():
    body = request.get_json()
    if not body or "signals" not in body:
        return jsonify({"error": "Se requiere 'signals' en el body"}), 400

    model_type = body.get("model", "sklearn")
    signals = body["signals"]

    expected_features = EXTRACTED_FEATURE_NAMES if data_source == "ucf-crime" else FEATURE_NAMES
    if len(signals) != len(expected_features):
        return jsonify({"error": f"Se requieren {len(expected_features)} señales"}), 400

    X = np.array([signals], dtype=float)

    if model_type == "sklearn":
        if not sklearn_model.trained:
            return jsonify({"error": "El modelo sklearn no ha sido entrenado"}), 400
        results = sklearn_model.predict(X)
    elif model_type == "pytorch":
        if not pytorch_model.trained:
            return jsonify({"error": "El modelo pytorch no ha sido entrenado"}), 400
        results = pytorch_model.predict(X)
    else:
        return jsonify({"error": f"Modelo no soportado: {model_type}"}), 400

    return jsonify({"model": model_type, "prediccion": results[0]})


@app.route("/api/report", methods=["POST"])
def report():
    """Ejecuta predicción y genera reporte de incidente con LLM."""
    global incidents
    body = request.get_json()
    if not body or "signals" not in body:
        return jsonify({"error": "Se requiere 'signals' en el body"}), 400

    model_type = body.get("model", "sklearn")
    signals = body["signals"]

    feature_names = EXTRACTED_FEATURE_NAMES if data_source == "ucf-crime" else FEATURE_NAMES
    if len(signals) != len(feature_names):
        return jsonify({"error": f"Se requieren {len(feature_names)} señales"}), 400

    X = np.array([signals], dtype=float)

    if model_type == "sklearn":
        if not sklearn_model.trained:
            return jsonify({"error": "El modelo sklearn no ha sido entrenado"}), 400
        results = sklearn_model.predict(X)
    elif model_type == "pytorch":
        if not pytorch_model.trained:
            return jsonify({"error": "El modelo pytorch no ha sido entrenado"}), 400
        results = pytorch_model.predict(X)
    else:
        return jsonify({"error": f"Modelo no soportado: {model_type}"}), 400

    prediction = results[0]
    report_data = generate_report(prediction, signals, feature_names)

    incident = {
        "id": len(incidents) + 1,
        "model": model_type,
        "prediction": prediction,
        "report": report_data,
        "signals": signals,
    }
    incidents.append(incident)
    if len(incidents) > 50:
        incidents.pop(0)

    return jsonify({"prediction": prediction, "report": report_data, "incident_id": incident["id"]})


@app.route("/api/incidents", methods=["GET"])
def get_incidents():
    return jsonify({"incidents": incidents})


def main():
    app.run(host="0.0.0.0", port=5000, debug=True)


if __name__ == "__main__":
    main()
