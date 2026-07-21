"""
train_models.py
================
Entrena los modelos de predicción de rendimiento cafetero sobre el dataset
construido por build_dataset.py / generar_datos_demo.py, y agrupa los
municipios en perfiles de riesgo mediante clustering.

Modelos:
  - RandomForestRegressor
  - GradientBoostingRegressor
  - "Modelo híbrido": promedio simple de las predicciones de ambos
    (ensemble), que suele generalizar mejor que cualquiera de los dos
    modelos por separado.

Validación:
  Se usa una separación TEMPORAL (no aleatoria): se entrena con los años
  más antiguos y se valida con el año más reciente disponible. Esto es
  más honesto que un split aleatorio para datos de series de tiempo --
  un split aleatorio dejaría "ver el futuro" al modelo durante el
  entrenamiento (años posteriores mezclados con años anteriores).

Clustering:
  KMeans sobre el perfil histórico de cada municipio (rendimiento
  promedio, variabilidad, tendencia) para segmentarlos en grupos que
  permiten una recomendación básica según el perfil del municipio.

Salidas (en data/processed/):
  - model_rf.pkl, model_gb.pkl
  - model_kmeans.pkl + perfiles_municipios.csv
  - metricas_validacion.json
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "dataset_entrenamiento.csv"
OUT_DIR = BASE_DIR / "data" / "processed"

FEATURES_NUM = [
    "area_sembrada_ha", "area_cosechada_ha", "ratio_area_cosechada_sembrada",
    "rendimiento_lag1", "rendimiento_lag2", "rendimiento_media_movil_3y",
    "tendencia_rendimiento_3y", "precipitacion_mm", "temperatura_c",
    "anomalia_precipitacion", "anomalia_temperatura",
]
FEATURES_CAT = ["departamento"]
TARGET = "rendimiento_ton_ha"


def _preprocesador():
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), FEATURES_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
    ])


def entrenar_y_validar(df: pd.DataFrame) -> dict:
    anio_validacion = df["ano"].max()
    train = df[df["ano"] < anio_validacion]
    test = df[df["ano"] == anio_validacion]

    print(f"Entrenamiento: {len(train)} filas (años {train['ano'].min()}-{train['ano'].max()})")
    print(f"Validación:    {len(test)} filas (año {anio_validacion}, no visto en entrenamiento)")

    X_train, y_train = train[FEATURES_NUM + FEATURES_CAT], train[TARGET]
    X_test, y_test = test[FEATURES_NUM + FEATURES_CAT], test[TARGET]

    modelo_rf = Pipeline([
        ("prep", _preprocesador()),
        ("rf", RandomForestRegressor(n_estimators=300, max_depth=8, min_samples_leaf=2, random_state=42)),
    ])
    modelo_gb = Pipeline([
        ("prep", _preprocesador()),
        ("gb", GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)),
    ])

    modelo_rf.fit(X_train, y_train)
    modelo_gb.fit(X_train, y_train)

    pred_rf = modelo_rf.predict(X_test)
    pred_gb = modelo_gb.predict(X_test)
    pred_hibrido = (pred_rf + pred_gb) / 2

    metricas = {}
    for nombre, pred in [("random_forest", pred_rf), ("gradient_boosting", pred_gb), ("hibrido", pred_hibrido)]:
        metricas[nombre] = {
            "mae_ton_ha": round(float(mean_absolute_error(y_test, pred)), 4),
            "r2": round(float(r2_score(y_test, pred)), 4) if len(y_test) > 1 else None,
        }
        print(f"  {nombre:20s} MAE={metricas[nombre]['mae_ton_ha']} ton/ha   R2={metricas[nombre]['r2']}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo_rf, OUT_DIR / "model_rf.pkl")
    joblib.dump(modelo_gb, OUT_DIR / "model_gb.pkl")

    metricas["anio_validacion"] = int(anio_validacion)
    metricas["filas_entrenamiento"] = len(train)
    metricas["filas_validacion"] = len(test)
    metricas["variables_usadas"] = FEATURES_NUM + FEATURES_CAT
    with open(OUT_DIR / "metricas_validacion.json", "w", encoding="utf-8") as f:
        json.dump(metricas, f, indent=2, ensure_ascii=False)

    return metricas


def construir_perfiles_y_clusters(df: pd.DataFrame, n_clusters: int = 3) -> pd.DataFrame:
    """Agrupa municipios según su perfil histórico de rendimiento:
    promedio, variabilidad (desviación estándar) y tendencia reciente."""
    perfiles = df.groupby(["departamento", "municipio"]).agg(
        rendimiento_promedio=("rendimiento_ton_ha", "mean"),
        rendimiento_variabilidad=("rendimiento_ton_ha", "std"),
        tendencia_promedio=("tendencia_rendimiento_3y", "mean"),
    ).reset_index()
    perfiles["rendimiento_variabilidad"] = perfiles["rendimiento_variabilidad"].fillna(0.0)
    perfiles["tendencia_promedio"] = perfiles["tendencia_promedio"].fillna(0.0)

    X = perfiles[["rendimiento_promedio", "rendimiento_variabilidad", "tendencia_promedio"]]
    escalador = StandardScaler()
    X_esc = escalador.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    perfiles["cluster"] = kmeans.fit_predict(X_esc)

    # Etiquetar cada cluster según sus características promedio (no por
    # índice arbitrario, para que la etiqueta sea consistente sin importar
    # el orden en que KMeans numeró los grupos).
    resumen_clusters = perfiles.groupby("cluster")[["rendimiento_promedio", "tendencia_promedio"]].mean()
    orden_por_rendimiento = resumen_clusters["rendimiento_promedio"].sort_values().index.tolist()

    etiquetas = {}
    nombres = ["Rendimiento bajo — requiere atención", "Rendimiento medio — monitoreo estándar",
               "Rendimiento alto — buen desempeño"]
    for i, cluster_id in enumerate(orden_por_rendimiento):
        etiquetas[cluster_id] = nombres[min(i, len(nombres) - 1)]
    perfiles["perfil"] = perfiles["cluster"].map(etiquetas)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"kmeans": kmeans, "escalador": escalador, "etiquetas": etiquetas}, OUT_DIR / "model_kmeans.pkl")
    perfiles.to_csv(OUT_DIR / "perfiles_municipios.csv", index=False)

    print(f"\nClustering de municipios ({n_clusters} grupos):")
    print(perfiles["perfil"].value_counts().to_string())

    return perfiles


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset cargado: {len(df)} filas, {df['municipio'].nunique()} municipios\n")

    print("=== Entrenamiento y validación de modelos ===")
    entrenar_y_validar(df)

    print("\n=== Clustering de perfiles de municipios ===")
    construir_perfiles_y_clusters(df)

    print(f"\n✅ Modelos y resultados guardados en: {OUT_DIR}")


if __name__ == "__main__":
    main()
