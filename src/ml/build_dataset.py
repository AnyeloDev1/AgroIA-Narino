"""
build_dataset.py
=================
Construye el dataset de entrenamiento REAL para los modelos de ML,
integrando 3 conjuntos de datos abiertos de datos.gov.co:

  1. EVA - Evaluaciones Agropecuarias Municipales (uejq-wxrr)
     -> producción, área sembrada/cosechada y rendimiento de café por
        municipio y año, para TODOS los departamentos cafeteros de
        Colombia (no solo Nariño).
  2. Precipitación IDEAM (s54a-sgyg)
     -> observaciones de precipitación por estación, agregadas a
        promedio anual por departamento.
  3. Temperatura IDEAM (sbwg-7ju4)
     -> observaciones de temperatura por estación, agregadas a
        promedio anual por departamento.

¿Por qué entrenar con TODO el país y no solo Nariño?
Nariño por sí solo tiene ~12 municipios cafeteros x ~6 años de EVA
disponible = un dataset demasiado pequeño (~70 filas) para entrenar
modelos de árboles con validación seria. Se entrena con todos los
departamentos cafeteros de Colombia (mucho más robusto estadísticamente)
y luego se aplica el modelo específicamente a los municipios de Nariño
en la app. Esta es una práctica estándar en ML con datos escasos:
ampliar la población de entrenamiento sin cambiar la población objetivo
de uso.

¿Por qué se integra el clima a nivel de DEPARTAMENTO y no de municipio?
Cruzar ~4.000 estaciones del catálogo IDEAM con cada municipio cafetero
uno por uno requiere resolución de nombres ambigua (varias estaciones
por municipio, nombres no siempre coincidentes) -- un proyecto de
limpieza en sí mismo. Se optó por agregar el clima a nivel departamental
por año, que es integrable de forma confiable y sigue capturando la
variabilidad climática inter-anual (El Niño / La Niña) que afecta el
rendimiento. Se documenta como limitación conocida.

Salida: data/processed/dataset_entrenamiento.csv
"""

import sys
import warnings
from pathlib import Path

import pandas as pd
from sodapy import Socrata

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
OUT_PATH = BASE_DIR / "data" / "processed" / "dataset_entrenamiento.csv"

DOMAIN = "www.datos.gov.co"
DATASET_EVA = "uejq-wxrr"
DATASET_PRECIPITACION = "s54a-sgyg"
DATASET_TEMPERATURA = "sbwg-7ju4"

# Departamentos con producción cafetera relevante y reportada en el EVA
# (fuente: FNC / Comité Nacional de Cafeteros - principales zonas cafeteras).
DEPARTAMENTOS_CAFETEROS = [
    "Nariño", "Huila", "Cauca", "Tolima", "Antioquia", "Caldas", "Quindío",
    "Risaralda", "Valle del Cauca", "Cundinamarca", "Santander",
    "Norte de Santander", "Magdalena", "Boyaca", "Meta",
]


def _cliente():
    return Socrata(DOMAIN, None, timeout=60)


def descargar_eva_cafe() -> pd.DataFrame:
    """Descarga el histórico de café del EVA para los departamentos cafeteros."""
    cliente = _cliente()
    lista_deps = ", ".join(f"'{d.upper()}'" for d in DEPARTAMENTOS_CAFETEROS)
    where_query = f"upper(cultivo) = 'CAFÉ' AND upper(departamento) IN ({lista_deps})"

    resultados = cliente.get(DATASET_EVA, limit=50000, where=where_query)
    df_raw = pd.DataFrame.from_records(resultados)
    if df_raw.empty:
        return pd.DataFrame()

    df = pd.DataFrame()
    df["ano"] = pd.to_numeric(df_raw.get("a_o"), errors="coerce").fillna(0).astype(int)
    df["departamento"] = df_raw.get("departamento", "").astype(str).str.upper().str.strip()
    df["municipio"] = df_raw.get("municipio", "").astype(str).str.upper().str.strip()
    df["area_sembrada_ha"] = pd.to_numeric(df_raw.get("rea_sembrada"), errors="coerce").fillna(0.0)
    df["area_cosechada_ha"] = pd.to_numeric(df_raw.get("rea_cosechada"), errors="coerce").fillna(0.0)
    df["produccion_ton"] = pd.to_numeric(df_raw.get("producci_n"), errors="coerce").fillna(0.0)
    df["rendimiento_ton_ha"] = pd.to_numeric(df_raw.get("rendimiento"), errors="coerce").fillna(0.0)

    df = df[(df["ano"] > 0) & (df["rendimiento_ton_ha"] > 0) & (df["area_cosechada_ha"] > 0)]
    df = df.groupby(["ano", "departamento", "municipio"], as_index=False).agg({
        "area_sembrada_ha": "sum",
        "area_cosechada_ha": "sum",
        "produccion_ton": "sum",
        "rendimiento_ton_ha": "mean",
    })
    return df


def descargar_clima_departamental(dataset_id: str, nombre_columna: str) -> pd.DataFrame:
    """Descarga observaciones de IDEAM y las agrega a promedio anual por
    departamento (climatología departamental)."""
    cliente = _cliente()
    lista_deps = ", ".join(f"'{d.upper()}'" for d in DEPARTAMENTOS_CAFETEROS)
    where_query = f"upper(departamento) IN ({lista_deps})"

    resultados = cliente.get(
        dataset_id,
        select="fechaobservacion, valorobservado, departamento",
        where=where_query,
        limit=200000,
    )
    df_raw = pd.DataFrame.from_records(resultados)
    if df_raw.empty:
        return pd.DataFrame()

    df = pd.DataFrame()
    df["ano"] = pd.to_datetime(df_raw["fechaobservacion"], errors="coerce").dt.year
    df["departamento"] = df_raw["departamento"].astype(str).str.upper().str.strip()
    df["valor"] = pd.to_numeric(df_raw["valorobservado"], errors="coerce")
    df = df.dropna(subset=["ano", "valor"])
    df["ano"] = df["ano"].astype(int)

    agregado = df.groupby(["ano", "departamento"], as_index=False)["valor"].mean()
    agregado = agregado.rename(columns={"valor": nombre_columna})
    return agregado


def construir_features(df_eva: pd.DataFrame, df_precip: pd.DataFrame, df_temp: pd.DataFrame) -> pd.DataFrame:
    """Une las 3 fuentes y construye las variables de entrada (features)
    del modelo, evitando fuga de datos (data leakage): las variables de
    rezago (lag) y tendencia SOLO usan años anteriores al año objetivo."""

    df = df_eva.merge(df_precip, on=["ano", "departamento"], how="left")
    df = df.merge(df_temp, on=["ano", "departamento"], how="left")

    df = df.sort_values(["municipio", "ano"]).reset_index(drop=True)

    # --- Variables de rezago y tendencia por municipio (sin fuga de datos) ---
    grupo = df.groupby("municipio")["rendimiento_ton_ha"]
    df["rendimiento_lag1"] = grupo.shift(1)
    df["rendimiento_lag2"] = grupo.shift(2)
    df["rendimiento_media_movil_3y"] = grupo.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    df["tendencia_rendimiento_3y"] = grupo.transform(
        lambda s: s.shift(1).rolling(3, min_periods=2).apply(
            lambda v: (v.iloc[-1] - v.iloc[0]) / len(v) if len(v) > 1 else 0.0, raw=False
        )
    )

    # --- Anomalías climáticas (proxy de variabilidad El Niño / La Niña) ---
    df["precipitacion_total_mm"] = df["precipitacion_mm"] if "precipitacion_mm" in df.columns else df.get("precipitacion")
    for col_clima, col_anomalia in [("precipitacion_mm", "anomalia_precipitacion"),
                                     ("temperatura_c", "anomalia_temperatura")]:
        if col_clima in df.columns:
            promedio_departamental = df.groupby("departamento")[col_clima].transform("mean")
            df[col_anomalia] = df[col_clima] - promedio_departamental

    # --- Otras variables derivadas ---
    df["ratio_area_cosechada_sembrada"] = (
        df["area_cosechada_ha"] / df["area_sembrada_ha"].replace(0, pd.NA)
    ).fillna(1.0).clip(0, 1.5)

    # Se descartan las primeras observaciones de cada municipio sin historial
    # suficiente (rendimiento_lag1 nulo) -- no se pueden usar para entrenar
    # porque les falta la variable de rezago más importante.
    df_final = df.dropna(subset=["rendimiento_lag1"]).reset_index(drop=True)
    return df_final


def main():
    print("1/4 · Descargando histórico EVA de café (todos los departamentos cafeteros)...")
    df_eva = descargar_eva_cafe()
    print(f"      {len(df_eva)} registros municipio-año descargados.")

    print("2/4 · Descargando y agregando precipitación IDEAM por departamento-año...")
    df_precip = descargar_clima_departamental(DATASET_PRECIPITACION, "precipitacion_mm")
    print(f"      {len(df_precip)} combinaciones departamento-año con dato de precipitación.")

    print("3/4 · Descargando y agregando temperatura IDEAM por departamento-año...")
    df_temp = descargar_clima_departamental(DATASET_TEMPERATURA, "temperatura_c")
    print(f"      {len(df_temp)} combinaciones departamento-año con dato de temperatura.")

    print("4/4 · Integrando fuentes y construyendo variables (features)...")
    df_final = construir_features(df_eva, df_precip, df_temp)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(OUT_PATH, index=False)
    print(f"\n✅ Dataset de entrenamiento guardado en: {OUT_PATH}")
    print(f"   {len(df_final)} filas listas para entrenamiento, {df_final.shape[1]} columnas.")
    print(f"   Municipios distintos: {df_final['municipio'].nunique()}")
    print(f"   Rango de años: {df_final['ano'].min()}-{df_final['ano'].max()}")


if __name__ == "__main__":
    main()
