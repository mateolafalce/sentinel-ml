# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the web server (http://localhost:5000)
OPENAI_API_KEY="sk-..." uv run python -m src.server

# Quick smoke test (data generation + training + prediction)
uv run python -c "
from src.data.generator import generate_dataset
from src.models.sklearn_model import SklearnMultiLabel
X, Y = generate_dataset(500)
m = SklearnMultiLabel()
print(m.train(X, Y))
"

# Install dependencies
uv sync
```

## Architecture

**Sentinel ML** is a multi-label security anomaly detector + AI incident reporter. Given 8 sensor signals, it predicts 4 action labels and generates a natural-language incident report via OpenAI GPT-4o.

### Data flow

```
Signals (8) → ML Classifier → Prediction (4 labels)
                                      ↓
                              OpenAI GPT-4o
                                      ↓
                         Incident Report (EN) + Log
```

**Inputs (8 features):** `sensor_movimiento`, `camara_activa`, `hora_nocturna`, `zona_riesgo`, `sensor_puerta`, `sensor_ventana`, `nivel_ruido`, `historico_incidentes`

**Outputs (4 labels):** `intrusion_probable`, `requiere_verificacion_visual`, `notificar_propietario`, `despachar_movil`

### Backend (`src/`)

- **`server.py`** — Flask REST API. Globals: `sklearn_model`, `pytorch_model`, `X_data`, `Y_data`, `incidents` (list, max 50). No persistence; state resets on restart.
- **`llm/reporter.py`** — Calls OpenAI GPT-4o with sensor readings + prediction results. Computes severity (NORMAL/LOW/MEDIUM/HIGH/CRITICAL) and returns structured report dict.
- **`data/generator.py`** — Generates synthetic data using deterministic rules with 5% label noise. `FEATURE_NAMES` and `LABEL_NAMES` are the canonical source of truth.
- **`data/feature_extractor.py`** — YOLOv8-based feature extraction from video frames. `EXTRACTED_FEATURE_NAMES` is the canonical source for UCF-Crime features.
- **`data/ucf_crime.py`** — UCF-Crime dataset management; maps 14 anomaly categories to 4 labels.
- **`models/sklearn_model.py`** — `MultiOutputClassifier(RandomForestClassifier)`. Returns per-label probabilities via `estimators_`.
- **`models/pytorch_model.py`** — 3-layer MLP, `BCEWithLogitsLoss`, threshold 0.5 for binarization.
- **`metrics.py`** — Wraps `hamming_loss`, `f1_score`, `accuracy_score` into a single dict.

### Frontend (`static/`)

Single-page SOC (Security Operations Center) dashboard. No framework, no build step.
- **3-column ops grid:** Signals panel | Detection labels | AI Incident Report
- **Setup panel** (collapsible): dataset source tabs + model training
- **Incident log** at the bottom (last 50 incidents, reverse chronological)

### API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET  | `/api/info`               | Features, labels, training status, data source |
| POST | `/api/generate`           | Generate synthetic dataset `{ n_samples }` |
| POST | `/api/train`              | Train model `{ model: "sklearn"\|"pytorch", epochs? }` |
| POST | `/api/predict`            | Raw prediction `{ model, signals: [8 floats] }` |
| POST | `/api/report`             | Predict + LLM report `{ model, signals }` → stores in incidents |
| GET  | `/api/incidents`          | Returns incident log `{ incidents: [...] }` |
| GET  | `/api/ucf/status`         | UCF-Crime dataset status |
| POST | `/api/ucf/setup`          | Create UCF-Crime directory structure |
| POST | `/api/ucf/extract`        | Start async YOLO feature extraction |
| GET  | `/api/ucf/extract/status` | Extraction progress |

### Key design notes

- `/api/report` is the primary endpoint for the dashboard — it runs prediction + LLM in one call and appends to `incidents`.
- `OPENAI_API_KEY` env var must be set; the OpenAI client is lazy-initialized on first call.
- Both data sources (synthetic + UCF-Crime) produce 8-feature vectors compatible with the same models.
- Severity is computed deterministically server-side before calling the LLM, so the report always has a severity even if the LLM call fails.
