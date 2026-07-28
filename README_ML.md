# AgroIA-Nariño — Módulo de Machine Learning

Este documento explica la arquitectura de IA agregada para cumplir el **Nivel Intermedio** del concurso: modelos de ML más elaborados, integración de múltiples fuentes de datos.gov.co, y validación real.

## 1. Datasets integrados (3, todos de datos.gov.co)

| # | Dataset | ID | Qué aporta |
|---|---|---|---|
| 1 | EVA — Evaluaciones Agropecuarias Municipales | `uejq-wxrr` | Producción, área sembrada/cosechada y rendimiento de café por municipio y año |
| 2 | Precipitación IDEAM | `s54a-sgyg` | Precipitación por estación, agregada a promedio anual por departamento |
| 3 | Temperatura IDEAM | `sbwg-7ju4` | Temperatura por estación, agregada a promedio anual por departamento |

Se entrena con **15 departamentos cafeteros de Colombia** (no solo Nariño) porque Nariño solo tiene ~12 municipios × 6 años ≈ 70 filas — insuficiente para entrenar árboles con validación seria. Es una práctica estándar en ML con datos escasos: ampliar la población de entrenamiento sin cambiar la población objetivo de uso (la app sigue prediciendo específicamente para municipios de Nariño).

**Limitación documentada:** el clima se integra a nivel *departamento-año*, no *municipio-año*. Cruzar ~4.000 estaciones IDEAM con cada municipio uno por uno requiere resolución de nombres ambigua — un proyecto de limpieza en sí mismo. El nivel departamental es integrable de forma confiable y sigue capturando la variabilidad climática inter-anual (El Niño / La Niña).

## 2. Variables del modelo (12)

`area_sembrada_ha`, `area_cosechada_ha`, `ratio_area_cosechada_sembrada`, `rendimiento_lag1`, `rendimiento_lag2`, `rendimiento_media_movil_3y`, `tendencia_rendimiento_3y`, `precipitacion_mm`, `temperatura_c`, `anomalia_precipitacion`, `anomalia_temperatura`, `departamento`.

Se excluyó deliberadamente `produccion_ton` de las variables de entrada: como `rendimiento ≈ producción / área_cosechada`, incluirla sería fuga de datos (el modelo "haría trampa" viendo casi literalmente la respuesta).

Las variables de rezago (`lag1`, `lag2`, media móvil, tendencia) solo usan años **anteriores** al año que se predice — sin fuga de datos temporal.

## 3. Modelos y validación

- **Random Forest** y **Gradient Boosting** (scikit-learn), más un **modelo híbrido** (promedio simple de ambos).
- Validación **temporal**, no aleatoria: se entrena con años pasados y se valida contra el año más reciente (que el modelo nunca vio). Un split aleatorio sería menos honesto para series de tiempo.
- **Clustering K-Means** sobre el perfil histórico de cada municipio (rendimiento promedio, variabilidad, tendencia), agrupándolos en 3 perfiles de riesgo/desempeño.

Resultados de referencia (con el dataset demo, ver sección 4):

| Modelo | MAE (ton/ha) | R² |
|---|---|---|
| Random Forest | 0.060 | 0.68 |
| Gradient Boosting | 0.062 | 0.70 |
| Híbrido | 0.060 | 0.70 |

## 4. Datos reales

Por restricciones de red del entorno donde se desarrolló este módulo (sin salida a internet), los modelos que vienen empaquetados en `data/processed/` se entrenaron con un **dataset sintético calibrado** (`generar_datos_demo.py`), NO con datos reales descargados en vivo. El archivo `dataset_entrenamiento.csv` incluye una columna `fuente = SINTETICO_DEMO_NO_USAR_EN_ENTREGA_FINAL` para que esto sea imposible de pasar por alto.

**Antes de entregar el proyecto, ejecuta esto una vez, con internet real:**

```bash
python src/ml/build_dataset.py   # descarga EVA + IDEAM reales y arma el dataset
python src/ml/train_models.py    # reentrena RF + GB + KMeans con los datos reales
```

Esto sobrescribe `data/processed/dataset_entrenamiento.csv`, los `.pkl` de los modelos, y `metricas_validacion.json` con resultados 100% reales. La app (`app.py`) no necesita ningún cambio — carga lo que encuentre en `data/processed/`.

## 5. Cómo se usa en la app

Ingresa aquí a la plataforma funcional👉 https://agroia-narino.streamlit.app/
<img width="1895" height="977" alt="image" src="https://github.com/user-attachments/assets/022d8dbf-2131-45bd-a7f3-b322bcec27b0" />


En la Pestaña 1, después de la estimación base (promedio histórico), aparece una sección **"Predicción con Machine Learning"** con las 3 predicciones (RF, GB, híbrido), las métricas de validación del modelo, y el perfil/cluster del municipio. Todo esto también se incluye en el reporte PDF descargable.

## 6. Estructura de archivos

```
src/ml/
├── build_dataset.py        # descarga y limpia los 3 datasets reales (requiere internet)
├── generar_datos_demo.py   # dataset sintético de respaldo para desarrollo/pruebas
├── train_models.py         # entrena RF + GB + KMeans, guarda métricas de validación
└── predict.py               # carga los modelos entrenados y sirve predicciones a la app

data/processed/
├── dataset_entrenamiento.csv
├── model_rf.pkl
├── model_gb.pkl
├── model_kmeans.pkl
├── perfiles_municipios.csv
└── metricas_validacion.json
```
