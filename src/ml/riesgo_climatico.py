"""
riesgo_climatico.py
====================
Motor de ALERTAS DE RIESGO CLIMÁTICO HACIA ADELANTE -- distinto del
"nivel de riesgo" que ya calcula cafe_service.estimar_cosecha() (que mide
una caída de rendimiento que YA ocurrió, según el histórico del EVA).
Este módulo responde a la otra mitad del reto sectorial ("...predecir
rendimientos agrícolas Y RIESGOS CLIMÁTICOS"): usa la observación
climática más reciente de IDEAM para señalar condiciones de riesgo que
podrían afectar el cultivo en los próximos días.

Diseño: motor de reglas agroclimáticas explícitas (mismo criterio que
climate_alert.py del proyecto hermano AgroIA-Nariño / FastAPI, portado y
adaptado aquí) -- NO un modelo de caja negra. Para una recomendación de
alto impacto como "no fertilice esta semana", la transparencia total
importa más que la sofisticación del modelo.

Umbrales de referencia (agroclimatología cafetera, Cenicafé/IDEAM):
  - Helada: temperatura promedio reciente < 12°C Y altitud > 2000 msnm
  - Exceso de lluvia: acumulado de los últimos 7 días > 150 mm
  - Déficit hídrico: acumulado de los últimos 7 días < 15 mm
"""

from dataclasses import dataclass

# Altitud aproximada (msnm) de municipios cafeteros de Nariño -- valores de
# referencia, usados solo cuando no se cuenta con un dato más preciso.
# (mismo criterio de respaldo que MUNICIPIOS_ALTITUD del proyecto hermano)
ALTITUD_MUNICIPIOS_NARINO = {
    "SAMANIEGO": 1770, "SANDONA": 1650, "SANDONÁ": 1650, "CONSACA": 1650, "CONSACÁ": 1650,
    "LA FLORIDA": 1800, "LINARES": 1750, "BUESACO": 1900, "TAMINANGO": 1350,
    "EL TAMBO": 1950, "ARBOLEDA": 1950, "LA UNION": 1700, "LA UNIÓN": 1700,
    "COLON": 2150, "COLÓN": 2150, "CHACHAGUI": 1800, "CHACHAGÜÍ": 1800,
}
ALTITUD_DEFECTO = 1850  # promedio típico de la zona cafetera de Nariño


@dataclass
class ObservacionClimatica:
    temperatura_c: float
    precipitacion_7d_mm: float
    altitud_msnm: float = ALTITUD_DEFECTO
    fuente: str = "ideam_tiempo_real"


def altitud_referencia(municipio: str) -> float:
    return ALTITUD_MUNICIPIOS_NARINO.get((municipio or "").strip().upper(), ALTITUD_DEFECTO)


def evaluar_riesgo_climatico(obs: ObservacionClimatica) -> dict:
    """Devuelve alertas de riesgo climático hacia adelante, con
    recomendaciones concretas -- basado en la observación real más
    reciente de IDEAM, no en un promedio histórico."""
    alertas = []
    recomendaciones = []
    nivel = "Bajo"

    if obs.altitud_msnm > 2000 and obs.temperatura_c < 12:
        alertas.append("Riesgo de helada")
        recomendaciones.append(
            "La temperatura reciente está baja para la altitud de la zona. "
            "Evite fertilización nitrogenada esta semana y considere riego "
            "por aspersión nocturno en los lotes más altos."
        )
        nivel = "Alto"

    if obs.precipitacion_7d_mm > 150:
        alertas.append("Riesgo de exceso de lluvia")
        recomendaciones.append(
            "La precipitación acumulada de la última semana es alta. Revise "
            "los drenajes del lote -- la humedad prolongada favorece la "
            "propagación de roya -- y posponga aplicaciones foliares."
        )
        nivel = "Alto"

    if obs.precipitacion_7d_mm < 15:
        alertas.append("Riesgo de déficit hídrico")
        recomendaciones.append(
            "La precipitación acumulada de la última semana es baja. Existe "
            "riesgo de estrés hídrico y aborto floral; priorice riego si "
            "está disponible, sobre todo en época de floración."
        )
        if nivel == "Bajo":
            nivel = "Medio"

    if not alertas:
        recomendaciones.append("Condiciones climáticas recientes dentro de rangos normales para la zona.")

    return {
        "alertas": alertas or ["Sin alerta"],
        "nivel_riesgo": nivel,
        "recomendaciones": recomendaciones,
        "temperatura_c": obs.temperatura_c,
        "precipitacion_7d_mm": obs.precipitacion_7d_mm,
        "fuente": obs.fuente,
    }
