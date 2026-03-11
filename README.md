# Sentinel ML — Centro de Operaciones de Seguridad

Sistema de detección de anomalías en eventos de seguridad con generación de reportes de incidentes usando IA (OpenAI GPT-4o). Dashboard web para operadores de seguridad.

## Requisitos

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)
- API key de OpenAI

## Setup

```bash
uv sync
export OPENAI_API_KEY="sk-..."
uv run python -m src.server
# → http://localhost:5000
```

## Flujo de uso

1. **Generar dataset** — datos sintéticos (o extraer features de videos UCF-Crime con YOLO)
2. **Entrenar modelo** — Random Forest (sklearn) o Red Neuronal (PyTorch)
3. **Analizar evento** — ajusta los 8 sliders de señales y pulsa "Analizar Evento"
4. El sistema ejecuta la predicción ML y genera un **reporte de incidente con GPT-4o**
5. El reporte queda registrado en el **historial de incidentes**

## Arquitectura

```
Señales (8) → Clasificador ML → Predicción (4 labels)
                                       ↓
                               OpenAI GPT-4o
                                       ↓
                          Reporte de Incidente (ES)
                          severidad / análisis / acciones
```

### Inputs (8 features)

**Datos sintéticos:** `sensor_movimiento`, `camara_activa`, `hora_nocturna`, `zona_riesgo`, `sensor_puerta`, `sensor_ventana`, `nivel_ruido`, `historico_incidentes`

**UCF-Crime (YOLO):** `avg_personas`, `max_personas`, `avg_confianza_persona`, `area_persona_max`, `intensidad_movimiento`, `clases_unicas`, `detecciones_promedio`, `velocidad_persona`

### Outputs (4 labels)

`intrusion_probable`, `requiere_verificacion_visual`, `notificar_propietario`, `despachar_movil`

### Severidad calculada

| Severidad | Condición |
|-----------|-----------|
| NORMAL    | 0 labels activos |
| BAJO      | 1 label activo (no intrusión) |
| MEDIO     | intrusion_probable solo, o 2+ labels sin despacho |
| ALTO      | intrusion_probable + otro label |
| CRÍTICO   | despachar_movil activo |

## Estructura

```
src/
├── server.py              # Flask REST API
├── llm/
│   └── reporter.py        # Integración OpenAI GPT-4o
├── data/
│   ├── generator.py       # Dataset sintético
│   ├── feature_extractor.py  # Extracción YOLO desde video
│   └── ucf_crime.py       # Gestión dataset UCF-Crime
├── models/
│   ├── sklearn_model.py   # RandomForest MultiOutputClassifier
│   └── pytorch_model.py   # MLP 3 capas, BCEWithLogitsLoss
└── metrics.py             # hamming_loss, f1, exact_match

static/
├── index.html             # SOC Dashboard (SPA)
├── app.js                 # Lógica del cliente
└── style.css              # Tema oscuro estilo SOC
```

## API

| Method | Path | Descripción |
|--------|------|-------------|
| GET  | `/api/info`               | Estado del sistema, features, labels |
| POST | `/api/generate`           | Generar dataset sintético `{ n_samples }` |
| POST | `/api/train`              | Entrenar modelo `{ model, epochs? }` |
| POST | `/api/predict`            | Predicción raw `{ model, signals }` |
| POST | `/api/report`             | Predicción + reporte LLM `{ model, signals }` |
| GET  | `/api/incidents`          | Historial de incidentes (máx 50) |
| GET  | `/api/ucf/status`         | Estado del dataset UCF-Crime |
| POST | `/api/ucf/setup`          | Crear estructura de carpetas |
| POST | `/api/ucf/extract`        | Extraer features con YOLO (async) |
| GET  | `/api/ucf/extract/status` | Progreso de extracción |
