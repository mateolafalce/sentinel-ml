"""Incident report generation with OpenAI GPT-4o."""

import json
from datetime import datetime
from openai import OpenAI

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


_FEATURE_LABELS = {
    "sensor_movimiento": "Motion sensor",
    "camara_activa": "Active camera",
    "hora_nocturna": "Night-time",
    "zona_riesgo": "Risk zone",
    "sensor_puerta": "Door sensor",
    "sensor_ventana": "Window sensor",
    "nivel_ruido": "Noise level",
    "historico_incidentes": "Incident history",
    "avg_personas": "Average persons",
    "max_personas": "Maximum persons",
    "avg_confianza_persona": "Person detection confidence",
    "area_persona_max": "Maximum person area",
    "intensidad_movimiento": "Motion intensity",
    "clases_unicas": "Unique classes detected",
    "detecciones_promedio": "Average detections",
    "velocidad_persona": "Person speed",
}

_LABEL_LABELS = {
    "intrusion_probable": "Probable intrusion",
    "requiere_verificacion_visual": "Requires visual verification",
    "notificar_propietario": "Notify owner",
    "despachar_movil": "Dispatch emergency unit",
}


def compute_severity(prediction: dict) -> str:
    active = [k for k, v in prediction.items() if v["activo"]]
    if "despachar_movil" in active:
        return "CRITICAL"
    if "intrusion_probable" in active and len(active) >= 2:
        return "HIGH"
    if "intrusion_probable" in active or len(active) >= 2:
        return "MEDIUM"
    if len(active) == 1:
        return "LOW"
    return "NORMAL"


def generate_report(prediction: dict, signals: list, feature_names: list) -> dict:
    """Generates an incident report using GPT-4o.

    Args:
        prediction: {label: {activo: bool, probabilidad: float}, ...}
        signals: list of signal values
        feature_names: feature names in the same order as signals
    """
    severity = compute_severity(prediction)
    timestamp = datetime.now().isoformat()

    sensors = "\n".join(
        f"- {_FEATURE_LABELS.get(n, n)}: {v}"
        for n, v in zip(feature_names, signals)
    )
    detections = "\n".join(
        f"- {_LABEL_LABELS.get(k, k)}: {'ACTIVE' if v['activo'] else 'inactive'} ({v['probabilidad'] * 100:.1f}%)"
        for k, v in prediction.items()
    )

    prompt = (
        "You are the incident analysis system of a security operations center.\n\n"
        f"SENSOR READINGS:\n{sensors}\n\n"
        f"ANOMALY DETECTION RESULT:\n{detections}\n\n"
        f"COMPUTED SEVERITY: {severity}\n\n"
        "Generate a concise and professional incident report in English.\n"
        "Respond only with valid JSON with this exact structure:\n"
        '{"titulo": "...", "resumen": "...", "analisis": "...", '
        '"acciones": ["...", "...", "..."], "riesgo": "..."}'
    )

    try:
        response = _get_client().chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=700,
        )
        data = json.loads(response.choices[0].message.content)
        return {
            "severidad": severity,
            "timestamp": timestamp,
            "titulo": data.get("titulo", "Event detected"),
            "resumen": data.get("resumen", ""),
            "analisis": data.get("analisis", ""),
            "acciones": data.get("acciones", []),
            "riesgo": data.get("riesgo", ""),
        }
    except Exception as exc:
        return {
            "severidad": severity,
            "timestamp": timestamp,
            "titulo": "Error generating report",
            "resumen": str(exc),
            "analisis": "",
            "acciones": [],
            "riesgo": "",
            "error": str(exc),
        }
