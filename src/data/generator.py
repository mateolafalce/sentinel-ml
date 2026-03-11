"""Generador de datos sintéticos para eventos de seguridad."""

import numpy as np


FEATURE_NAMES = [
    "sensor_movimiento",
    "camara_activa",
    "hora_nocturna",
    "zona_riesgo",
    "sensor_puerta",
    "sensor_ventana",
    "nivel_ruido",
    "historico_incidentes",
]

LABEL_NAMES = [
    "intrusion_probable",
    "requiere_verificacion_visual",
    "notificar_propietario",
    "despachar_movil",
]


def generate_dataset(n_samples: int = 2000, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Genera datos sintéticos con reglas lógicas realistas.

    Features (8):
        0: sensor_movimiento      [0, 1]
        1: camara_activa           [0, 1]
        2: hora_nocturna           [0, 1]
        3: zona_riesgo             [0, 1]
        4: sensor_puerta           [0, 1]
        5: sensor_ventana          [0, 1]
        6: nivel_ruido             [0.0 - 1.0]
        7: historico_incidentes    [0.0 - 1.0]

    Labels (4):
        0: intrusion_probable
        1: requiere_verificacion_visual
        2: notificar_propietario
        3: despachar_movil
    """
    rng = np.random.RandomState(seed)

    # Features binarias
    mov = rng.randint(0, 2, n_samples)
    cam = rng.randint(0, 2, n_samples)
    noche = rng.randint(0, 2, n_samples)
    zona = rng.randint(0, 2, n_samples)
    puerta = rng.randint(0, 2, n_samples)
    ventana = rng.randint(0, 2, n_samples)
    ruido = rng.uniform(0, 1, n_samples)
    historico = rng.uniform(0, 1, n_samples)

    X = np.column_stack([mov, cam, noche, zona, puerta, ventana, ruido, historico])

    # Reglas para generar etiquetas con ruido
    noise = lambda: rng.random(n_samples) < 0.05

    # Intrusión probable: movimiento + (noche o zona_riesgo) + (puerta o ventana o ruido alto)
    intrusion = (
        (mov == 1)
        & ((noche == 1) | (zona == 1))
        & ((puerta == 1) | (ventana == 1) | (ruido > 0.6))
    ).astype(int)
    intrusion = np.abs(intrusion - noise().astype(int))

    # Requiere verificación visual: cámara activa + (movimiento o ruido alto)
    verificacion = ((cam == 1) & ((mov == 1) | (ruido > 0.5))).astype(int)
    verificacion = np.abs(verificacion - noise().astype(int))

    # Notificar propietario: cualquier señal significativa
    notificar = (
        (mov == 1) | ((noche == 1) & (zona == 1)) | (historico > 0.7)
    ).astype(int)
    notificar = np.abs(notificar - noise().astype(int))

    # Despachar móvil: intrusión probable + zona de riesgo + histórico alto
    despachar = (
        (intrusion == 1) & (zona == 1) & (historico > 0.4)
    ).astype(int)
    despachar = np.abs(despachar - noise().astype(int))

    Y = np.column_stack([intrusion, verificacion, notificar, despachar])

    return X, Y
