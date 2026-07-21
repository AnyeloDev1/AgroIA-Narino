"""
predict.py
==========
Capa de inferencia: carga los modelos entrenados (RF, GB, KMeans) y expone
una función simple para predecir el rendimiento cafetero de un municipio,
combinando su historial real (EVA) con climatología departamental de
referencia.

Se separa deliberadamente de train_models.py -- este módulo NO entrena
nada, solo carga modelos ya entrenados y sirve predicciones, para que la
app web pueda importarlo sin pagar el costo de reentrenar en cada consulta.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "data" / "processed"

FEATURES_NUM = [
    "area_sembrada_ha", "area_cosechada_ha", "ratio_area_cosechada_sembrada",
    "rendimiento_lag1", "rendimiento_lag2", "rendimiento_media_movil_3y",
    "tendencia_rendimiento_3y", "precipitacion_mm", "temperatura_c",
    "anomalia_precipitacion", "anomalia_temperatura",
]
FEATURES_CAT = ["departamento"]


def modelos_disponibles() -> bool:
    """True si ya se entrenaron y guardaron los modelos (train_models.py
    se ejecutó al menos una vez con datos reales o demo)."""
    return (MODELS_DIR / "model_rf.pkl").exists() and (MODELS_DIR / "model_gb.pkl").exists()


class ModelosIncompatiblesError(Exception):
    """Se lanza cuando los .pkl existen pero no se pueden cargar -- típicamente
    porque se entrenaron con una versión de scikit-learn distinta a la
    instalada. Los modelos de scikit-learn NO garantizan compatibilidad al
    deserializar (pickle) entre versiones diferentes de la librería; esto es
    una limitación conocida y documentada de scikit-learn, no un bug del
    proyecto. La solución es reentrenar en el mismo entorno donde se usa."""
    pass


def _cargar_modelos():
    try:
        modelo_rf = joblib.load(MODELS_DIR / "model_rf.pkl")
        modelo_gb = joblib.load(MODELS_DIR / "model_gb.pkl")
        return modelo_rf, modelo_gb
    except Exception as e:
        raise ModelosIncompatiblesError(
            "No se pudieron cargar los modelos guardados (model_rf.pkl / model_gb.pkl). "
            "La causa más común es que se entrenaron con una versión de scikit-learn "
            "distinta a la que tienes instalada -- los modelos pickle de scikit-learn "
            "no son compatibles entre versiones diferentes de la librería. "
            "Solución: reentrena los modelos en tu propio entorno ejecutando "
            "'python src/ml/generar_datos_demo.py' (o build_dataset.py con datos reales) "
            f"seguido de 'python src/ml/train_models.py'. Detalle técnico: {e}"
        ) from e


def _climatologia_departamental() -> pd.DataFrame:
    """Promedio histórico de clima por departamento, calculado a partir
    del mismo dataset de entrenamiento -- se usa como valor de referencia
    ('clima esperado normal') cuando el usuario no provee un pronóstico
    específico para el próximo ciclo."""
    df = pd.read_csv(MODELS_DIR / "dataset_entrenamiento.csv")
    return df.groupby("departamento").agg(
        precipitacion_mm=("precipitacion_mm", "mean"),
        temperatura_c=("temperatura_c", "mean"),
    )


def obtener_perfil_municipio(municipio: str) -> dict:
    """Devuelve el perfil/cluster de un municipio (calculado por
    train_models.py -> construir_perfiles_y_clusters)."""
    path_perfiles = MODELS_DIR / "perfiles_municipios.csv"
    if not path_perfiles.exists():
        return None
    df = pd.read_csv(path_perfiles)
    fila = df[df["municipio"].str.upper() == municipio.strip().upper()]
    if fila.empty:
        return None
    fila = fila.iloc[0]
    return {
        "perfil": fila["perfil"],
        "rendimiento_promedio_historico": round(float(fila["rendimiento_promedio"]), 3),
        "variabilidad": round(float(fila["rendimiento_variabilidad"]), 3),
    }


def predecir_rendimiento_ml(departamento: str, municipio: str,
                             area_sembrada_ha: float, area_cosechada_ha: float,
                             historial_rendimiento: list,
                             precipitacion_mm: float = None,
                             temperatura_c: float = None) -> dict:
    """Predice el rendimiento (ton/ha) del próximo ciclo usando el
    ensemble Random Forest + Gradient Boosting.

    historial_rendimiento: lista de rendimientos reales de los últimos
    años de ese municipio, ordenados del más reciente al más antiguo
    (ej. [rendimiento_2024, rendimiento_2023, rendimiento_2022]).
    """
    if not modelos_disponibles():
        return {
            "disponible": False,
            "motivo": (
                "Los modelos de Machine Learning aún no se han entrenado en este "
                "entorno. Ejecuta 'python src/ml/build_dataset.py' (con internet) "
                "o 'python src/ml/generar_datos_demo.py' (modo demo) seguido de "
                "'python src/ml/train_models.py'."
            ),
        }

    try:
        modelo_rf, modelo_gb = _cargar_modelos()
    except ModelosIncompatiblesError as e:
        return {"disponible": False, "motivo": str(e)}

    depto_up = departamento.strip().upper()

    lag1 = historial_rendimiento[0] if len(historial_rendimiento) > 0 else np.nan
    lag2 = historial_rendimiento[1] if len(historial_rendimiento) > 1 else np.nan
    media_movil = float(np.nanmean(historial_rendimiento[:3])) if historial_rendimiento else np.nan
    if len(historial_rendimiento) >= 2:
        ultimos = historial_rendimiento[:3][::-1]  # orden cronológico ascendente
        tendencia = (ultimos[-1] - ultimos[0]) / len(ultimos)
    else:
        tendencia = 0.0

    climatologia = _climatologia_departamental()
    if precipitacion_mm is None or temperatura_c is None:
        if depto_up in climatologia.index:
            precipitacion_mm = precipitacion_mm or float(climatologia.loc[depto_up, "precipitacion_mm"])
            temperatura_c = temperatura_c or float(climatologia.loc[depto_up, "temperatura_c"])
        else:
            precipitacion_mm = precipitacion_mm or 1700.0
            temperatura_c = temperatura_c or 20.5

    anomalia_precip = 0.0  # sin pronóstico específico, se asume clima "normal" (sin anomalía)
    anomalia_temp = 0.0

    fila = pd.DataFrame([{
        "area_sembrada_ha": area_sembrada_ha,
        "area_cosechada_ha": area_cosechada_ha,
        "ratio_area_cosechada_sembrada": (area_cosechada_ha / area_sembrada_ha) if area_sembrada_ha else 1.0,
        "rendimiento_lag1": lag1,
        "rendimiento_lag2": lag2,
        "rendimiento_media_movil_3y": media_movil,
        "tendencia_rendimiento_3y": tendencia,
        "precipitacion_mm": precipitacion_mm,
        "temperatura_c": temperatura_c,
        "anomalia_precipitacion": anomalia_precip,
        "anomalia_temperatura": anomalia_temp,
        "departamento": depto_up,
    }])

    pred_rf = float(modelo_rf.predict(fila)[0])
    pred_gb = float(modelo_gb.predict(fila)[0])
    pred_hibrido = (pred_rf + pred_gb) / 2

    return {
        "disponible": True,
        "rendimiento_rf_ton_ha": round(pred_rf, 3),
        "rendimiento_gb_ton_ha": round(pred_gb, 3),
        "rendimiento_hibrido_ton_ha": round(pred_hibrido, 3),
        "precipitacion_usada_mm": round(precipitacion_mm, 1),
        "temperatura_usada_c": round(temperatura_c, 1),
        "perfil_municipio": obtener_perfil_municipio(municipio),
    }


def referencia_rendimiento_alto() -> float:
    """Rendimiento promedio (ton/ha) del perfil de clustering de mejor
    desempeño, usado como referencia de 'techo alcanzable' para calcular
    impacto económico potencial. Si no hay perfiles calculados, devuelve
    None."""
    path_perfiles = MODELS_DIR / "perfiles_municipios.csv"
    if not path_perfiles.exists():
        return None
    df = pd.read_csv(path_perfiles)
    df_alto = df[df["perfil"].str.contains("alto", case=False, na=False)]
    if df_alto.empty:
        return None
    return float(df_alto["rendimiento_promedio"].mean())


def metricas_modelo() -> dict:
    import json
    path = MODELS_DIR / "metricas_validacion.json"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)
