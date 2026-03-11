# Sentinel ML — Security Operations Center

Multi-label security anomaly detection system with AI-powered incident report generation (OpenAI GPT-4o). Web dashboard for security operators.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- OpenAI API key

## Setup

```bash
uv sync
export OPENAI_API_KEY="sk-..."
uv run python -m src.server
# → http://localhost:5000
```

## Usage flow

1. **Generate dataset** — synthetic data (or extract features from UCF-Crime videos with YOLO)
2. **Train model** — Random Forest (sklearn) or Neural Network (PyTorch)
3. **Analyze event** — adjust the 8 signal sliders and click "Analyze Event"
4. The system runs the ML prediction and generates an **incident report with GPT-4o**
5. The report is recorded in the **incident history**

## Architecture

```
Signals (8) → ML Classifier → Prediction (4 labels)
                                      ↓
                              OpenAI GPT-4o
                                      ↓
                         Incident Report (EN)
                         severity / analysis / actions
```

### Inputs (8 features)

**Synthetic data:** `sensor_movimiento`, `camara_activa`, `hora_nocturna`, `zona_riesgo`, `sensor_puerta`, `sensor_ventana`, `nivel_ruido`, `historico_incidentes`

**UCF-Crime (YOLO):** `avg_personas`, `max_personas`, `avg_confianza_persona`, `area_persona_max`, `intensidad_movimiento`, `clases_unicas`, `detecciones_promedio`, `velocidad_persona`

### Outputs (4 labels)

`intrusion_probable`, `requiere_verificacion_visual`, `notificar_propietario`, `despachar_movil`

### Computed severity

| Severity | Condition |
|----------|-----------|
| NORMAL   | 0 active labels |
| LOW      | 1 active label (no intrusion) |
| MEDIUM   | intrusion_probable only, or 2+ labels without dispatch |
| HIGH     | intrusion_probable + another label |
| CRITICAL | despachar_movil active |

## Structure

```
src/
├── server.py              # Flask REST API
├── llm/
│   └── reporter.py        # OpenAI GPT-4o integration
├── data/
│   ├── generator.py       # Synthetic dataset
│   ├── feature_extractor.py  # YOLO feature extraction from video
│   └── ucf_crime.py       # UCF-Crime dataset management
├── models/
│   ├── sklearn_model.py   # RandomForest MultiOutputClassifier
│   └── pytorch_model.py   # 3-layer MLP, BCEWithLogitsLoss
└── metrics.py             # hamming_loss, f1, exact_match

static/
├── index.html             # SOC Dashboard (SPA)
├── app.js                 # Client logic
└── style.css              # Dark SOC-style theme
```

## API

| Method | Path | Description |
|--------|------|-------------|
| GET  | `/api/info`               | System status, features, labels |
| POST | `/api/generate`           | Generate synthetic dataset `{ n_samples }` |
| POST | `/api/train`              | Train model `{ model, epochs? }` |
| POST | `/api/predict`            | Raw prediction `{ model, signals }` |
| POST | `/api/report`             | Prediction + LLM report `{ model, signals }` |
| GET  | `/api/incidents`          | Incident history (max 50) |
| GET  | `/api/ucf/status`         | UCF-Crime dataset status |
| POST | `/api/ucf/setup`          | Create folder structure |
| POST | `/api/ucf/extract`        | Extract features with YOLO (async) |
| GET  | `/api/ucf/extract/status` | Extraction progress |
