"""Generación de reportes de incidentes con OpenAI GPT-4o."""

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
    "sensor_movimiento": "Sensor de movimiento",
    "camara_activa": "Cámara activa",
    "hora_nocturna": "Hora nocturna",
    "zona_riesgo": "Zona de riesgo",
    "sensor_puerta": "Sensor de puerta",
    "sensor_ventana": "Sensor de ventana",
    "nivel_ruido": "Nivel de ruido",
    "historico_incidentes": "Histórico de incidentes",
    "avg_personas": "Promedio de personas",
    "max_personas": "Máximo de personas",
    "avg_confianza_persona": "Confianza en detección de personas",
    "area_persona_max": "Área máxima de persona",
    "intensidad_movimiento": "Intensidad de movimiento",
    "clases_unicas": "Clases únicas detectadas",
    "detecciones_promedio": "Detecciones promedio",
    "velocidad_persona": "Velocidad de persona",
}

_LABEL_LABELS = {
    "intrusion_probable": "Intrusión probable",
    "requiere_verificacion_visual": "Requiere verificación visual",
    "notificar_propietario": "Notificar al propietario",
    "despachar_movil": "Despachar móvil de emergencia",
}


def compute_severity(prediction: dict) -> str:
    active = [k for k, v in prediction.items() if v["activo"]]
    if "despachar_movil" in active:
        return "CRÍTICO"
    if "intrusion_probable" in active and len(active) >= 2:
        return "ALTO"
    if "intrusion_probable" in active or len(active) >= 2:
        return "MEDIO"
    if len(active) == 1:
        return "BAJO"
    return "NORMAL"


def generate_report(prediction: dict, signals: list, feature_names: list) -> dict:
    """Genera un reporte de incidente usando GPT-4o.

    Args:
        prediction: {label: {activo: bool, probabilidad: float}, ...}
        signals: lista de valores de señales
        feature_names: nombres de features en el mismo orden que signals
    """
    severity = compute_severity(prediction)
    timestamp = datetime.now().isoformat()

    sensors = "\n".join(
        f"- {_FEATURE_LABELS.get(n, n)}: {v}"
        for n, v in zip(feature_names, signals)
    )
    detections = "\n".join(
        f"- {_LABEL_LABELS.get(k, k)}: {'ACTIVADO' if v['activo'] else 'inactivo'} ({v['probabilidad'] * 100:.1f}%)"
        for k, v in prediction.items()
    )

    prompt = (
        "Eres el sistema de análisis de incidentes de un centro de operaciones de seguridad.\n\n"
        f"LECTURAS DE SENSORES:\n{sensors}\n\n"
        f"RESULTADO DE DETECCIÓN DE ANOMALÍAS:\n{detections}\n\n"
        f"SEVERIDAD CALCULADA: {severity}\n\n"
        "Genera un reporte de incidente conciso y profesional en español.\n"
        "Responde únicamente con JSON válido con esta estructura exacta:\n"
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
            "titulo": data.get("titulo", "Evento detectado"),
            "resumen": data.get("resumen", ""),
            "analisis": data.get("analisis", ""),
            "acciones": data.get("acciones", []),
            "riesgo": data.get("riesgo", ""),
        }
    except Exception as exc:
        return {
            "severidad": severity,
            "timestamp": timestamp,
            "titulo": "Error al generar reporte",
            "resumen": str(exc),
            "analisis": "",
            "acciones": [],
            "riesgo": "",
            "error": str(exc),
        }
