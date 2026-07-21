"""
generar_datos_demo.py
======================
Genera un dataset SINTÉTICO calibrado, con la misma estructura exacta que
produce build_dataset.py al consultar las APIs reales (EVA + IDEAM
precipitación + IDEAM temperatura), para poder desarrollar y probar el
pipeline de features/entrenamiento sin depender de acceso a internet.

IMPORTANTE: este dataset es solo para pruebas de desarrollo. Antes de la
entrega final del proyecto, se debe ejecutar build_dataset.py con conexión
real a internet para reemplazar data/processed/dataset_entrenamiento.csv
por datos 100% reales de datos.gov.co. Este script deja eso documentado
también en el propio CSV que genera (columna 'fuente').

Calibración: rangos de rendimiento (0.5-1.6 ton/ha), temperatura y
precipitación por departamento están calibrados con los órdenes de
magnitud reales observados en consultas previas a estas mismas APIs
(ver build_dataset.py y README), no son arbitrarios.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from build_dataset import DEPARTAMENTOS_CAFETEROS, construir_features  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUT_PATH = BASE_DIR / "data" / "processed" / "dataset_entrenamiento.csv"

np.random.seed(7)

MUNICIPIOS_POR_DEPTO = {
    "NARIÑO": ["SANDONA", "CONSACA", "LA FLORIDA", "LINARES", "BUESACO", "TAMINANGO", "EL TAMBO", "ARBOLEDA"],
    "HUILA": ["PITALITO", "GARZON", "ACEVEDO", "TIMANA", "ISNOS"],
    "CAUCA": ["POPAYAN", "TIMBIO", "PIENDAMO", "MORALES"],
    "TOLIMA": ["IBAGUE", "PLANADAS", "RIOBLANCO", "CHAPARRAL"],
    "ANTIOQUIA": ["ANDES", "JARDIN", "CIUDAD BOLIVAR", "URRAO"],
    "CALDAS": ["MANIZALES", "CHINCHINA", "PALESTINA"],
    "QUINDIO": ["ARMENIA", "CALARCA", "MONTENEGRO"],
    "RISARALDA": ["PEREIRA", "SANTA ROSA DE CABAL"],
    "VALLE DEL CAUCA": ["SEVILLA", "CAICEDONIA", "TRUJILLO"],
    "SANTANDER": ["SAN GIL", "SOCORRO"],
}

# Temperatura/precipitación base aproximadas por departamento (calibradas con
# el orden de magnitud real de la zona cafetera andina, 1200-2000 msnm).
CLIMA_BASE = {
    "NARIÑO": (19.0, 1400), "HUILA": (21.5, 1500), "CAUCA": (20.0, 1700),
    "TOLIMA": (22.0, 1600), "ANTIOQUIA": (20.5, 2200), "CALDAS": (19.5, 2400),
    "QUINDIO": (19.8, 2300), "RISARALDA": (20.2, 2500), "VALLE DEL CAUCA": (21.0, 1900),
    "SANTANDER": (22.5, 1300),
}

filas = []
anios = range(2019, 2025)

for depto in DEPARTAMENTOS_CAFETEROS:
    depto_up = depto.upper()
    municipios = MUNICIPIOS_POR_DEPTO.get(depto_up, [f"{depto_up}-M1", f"{depto_up}-M2"])
    temp_base, precip_base = CLIMA_BASE.get(depto_up, (20.5, 1700))

    for municipio in municipios:
        rendimiento_base = np.random.normal(1.0, 0.15)
        area_base = np.random.uniform(300, 1200)

        for ano in anios:
            temp_c = np.clip(np.random.normal(temp_base, 0.6), 14, 26)
            precip_mm = np.clip(np.random.normal(precip_base, 220), 400, 3200)

            # El clima afecta el rendimiento: exceso/déficit de lluvia y
            # temperaturas atípicas reducen el rendimiento (relación
            # agronómica real, no ruido puro).
            estres_precip = abs(precip_mm - precip_base) / precip_base
            estres_temp = abs(temp_c - temp_base) / temp_base
            factor_clima = 1 - 0.35 * estres_precip - 0.25 * estres_temp

            rendimiento = max(0.15, rendimiento_base * factor_clima * np.random.normal(1, 0.08))
            area_sembrada = area_base * np.random.normal(1, 0.05)
            area_cosechada = area_sembrada * np.random.uniform(0.85, 0.98)
            produccion = rendimiento * area_cosechada

            filas.append({
                "ano": ano, "departamento": depto_up, "municipio": municipio,
                "area_sembrada_ha": round(area_sembrada, 1),
                "area_cosechada_ha": round(area_cosechada, 1),
                "produccion_ton": round(produccion, 1),
                "rendimiento_ton_ha": round(rendimiento, 3),
                "precipitacion_mm": round(precip_mm, 1),
                "temperatura_c": round(temp_c, 2),
            })

df_crudo = pd.DataFrame(filas)

df_eva = df_crudo[["ano", "departamento", "municipio", "area_sembrada_ha",
                    "area_cosechada_ha", "produccion_ton", "rendimiento_ton_ha"]]
df_precip = df_crudo[["ano", "departamento", "precipitacion_mm"]].drop_duplicates(["ano", "departamento"])
df_temp = df_crudo[["ano", "departamento", "temperatura_c"]].drop_duplicates(["ano", "departamento"])

df_final = construir_features(df_eva, df_precip, df_temp)
df_final["fuente"] = "SINTETICO_DEMO_NO_USAR_EN_ENTREGA_FINAL"

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df_final.to_csv(OUT_PATH, index=False)
print(f"Dataset demo generado: {len(df_final)} filas, {df_final.shape[1]} columnas")
print(f"Guardado en: {OUT_PATH}")
print(f"Municipios: {df_final['municipio'].nunique()} · Departamentos: {df_final['departamento'].nunique()}")
