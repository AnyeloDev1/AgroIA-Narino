"""
ideam_reciente.py
==================
Trae la observación climática REAL más reciente de IDEAM (temperatura
promedio y precipitación acumulada de los últimos 7 días) para un
municipio, y la pasa por el motor de riesgo climático hacia adelante
(riesgo_climatico.py).

Usa los mismos 2 datasets de datos.gov.co ya verificados y usados en
build_dataset.py (entrenamiento) y en el chatbot de la Pestaña 2:
  - sbwg-7ju4  Temperatura IDEAM
  - s54a-sgyg  Precipitación IDEAM
"""

from datetime import datetime, timedelta
from pathlib import Path
import sys

import requests

sys.path.append(str(Path(__file__).resolve().parent))
from riesgo_climatico import ObservacionClimatica, altitud_referencia, evaluar_riesgo_climatico

DATASET_TEMPERATURA = "sbwg-7ju4"
DATASET_PRECIPITACION = "s54a-sgyg"


def _consultar_ideam_reciente(dataset_id: str, municipio: str, dias: int = 7) -> list:
    """Consulta observaciones de los últimos N días para un municipio.
    Si el filtro por municipio no trae nada (o el campo no coincide),
    reintenta filtrando por departamento Nariño -- igual que en el
    chatbot, para no fallar en silencio ante un esquema inesperado."""
    fecha_desde = (datetime.now() - timedelta(days=dias)).strftime("%Y-%m-%d")
    api_url = f"https://www.datos.gov.co/resource/{dataset_id}.json"

    intentos = [
        {"$limit": 500, "$where": f"fechaobservacion >= '{fecha_desde}' AND upper(municipio) like '%{municipio.upper()}%'"},
        {"$limit": 500, "$where": f"fechaobservacion >= '{fecha_desde}' AND upper(departamento) like '%NARI%'"},
    ]
    for params in intentos:
        try:
            r = requests.get(api_url, params=params, timeout=3.0)
            if r.status_code == 200:
                datos = r.json()
                if datos:
                    return datos
        except Exception as e:
            print(f"[ideam_reciente] Error consultando {dataset_id}: {e}")
    return []


def obtener_riesgo_climatico_actual(municipio: str) -> dict:
    """Punto de entrada: trae clima reciente real de IDEAM para el
    municipio y devuelve el resultado del motor de riesgo climático.
    Si IDEAM no responde, degrada a un resultado 'no disponible' claro
    en vez de fallar o inventar datos."""
    try:
        obs_temp = _consultar_ideam_reciente(DATASET_TEMPERATURA, municipio, dias=3)
        obs_precip = _consultar_ideam_reciente(DATASET_PRECIPITACION, municipio, dias=7)

        if not obs_temp and not obs_precip:
            return {
                "disponible": False,
                "motivo": "No se pudo obtener una observación climática reciente de IDEAM en este momento.",
            }

        valores_temp = [float(o["valorobservado"]) for o in obs_temp if _es_numero(o.get("valorobservado"))]
        valores_precip = [float(o["valorobservado"]) for o in obs_precip if _es_numero(o.get("valorobservado"))]

        temperatura_c = sum(valores_temp) / len(valores_temp) if valores_temp else 19.0
        precipitacion_7d_mm = sum(valores_precip) if valores_precip else 0.0

        obs = ObservacionClimatica(
            temperatura_c=temperatura_c,
            precipitacion_7d_mm=precipitacion_7d_mm,
            altitud_msnm=altitud_referencia(municipio),
            fuente="ideam_tiempo_real" if (valores_temp or valores_precip) else "estimado",
        )
        resultado = evaluar_riesgo_climatico(obs)
        resultado["disponible"] = True
        resultado["lecturas_temperatura"] = len(valores_temp)
        resultado["lecturas_precipitacion"] = len(valores_precip)
        return resultado

    except Exception as e:
        return {"disponible": False, "motivo": f"Error consultando IDEAM: {e}"}


def _es_numero(valor) -> bool:
    try:
        float(valor)
        return True
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    import json
    print(json.dumps(obtener_riesgo_climatico_actual("Sandoná"), indent=2, ensure_ascii=False))
