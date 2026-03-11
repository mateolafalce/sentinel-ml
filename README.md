# Sentinel ML — Security Operations Center

Multi-label security anomaly detection system with AI-powered incident report generation (OpenAI GPT-4o). Web dashboard for security operators. Runs at http://localhost:5000

## Setup

```bash
uv sync
export OPENAI_API_KEY="sk-..."
uv run python -m src.server
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

