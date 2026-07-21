# AgroIA-Nariño 
"La inteligencia que impulsa el café de Nariño"

# Población Objetivo: Pequeños caficultores minifundistas del departamento de Nariño.

Descripción de la Solución:
AgroIA-Nariño es una plataforma web analítica orientada a mitigar la vulnerabilidad climática y comercial de los pequeños productores de café en Nariño, mediante el aprovechamiento de datos públicos reales. La solución consulta de forma automática el dataset EVA (Evaluaciones Agropecuarias Municipales, UPRA/MinAgricultura) publicado en datos.gov.co, para obtener el histórico real de área sembrada, área cosechada, producción y rendimiento cafetero por municipio.
Con esa información, la herramienta ofrece al caficultor dos servicios: una estimación de cosecha basada en el rendimiento histórico real de su municipio (en vez de cifras genéricas o inventadas), y un indicador de riesgo que compara el rendimiento del último año reportado contra el promedio histórico del municipio, señalando caídas significativas que pueden ameritar inspección en campo. Complementariamente, incluye un asistente conversacional (modelo de lenguaje Gemini, vía OpenRouter) que responde preguntas del caficultor en tono cercano y campesino, consultando datos agrícolas reales de datos.gov.co como contexto para sus respuestas. Los resultados pueden compartirse como mensaje de texto o como reporte descargable en PDF, listo para adjuntar en WhatsApp.



# Planteamiento del Problema Real y Justificación

El departamento de Nariño sustenta gran parte de su economía en la caficultura, caracterizada estructuralmente por un modelo de minifundios donde más del 90% de las familias productoras operan parcelas menores a las 2 hectáreas. Esta escala productiva expone a los pequeños caficultores a una alta vulnerabilidad socioeconómica debido a dos factores críticos crónicos:
Incertidumbre productiva y fitosanitaria: Los microclimas de la cordillera andina nariñense, alterados por fenómenos de variabilidad climática (El Niño y La Niña), generan condiciones que afectan el rendimiento del cultivo de forma difícil de anticipar para el productor individual. Sin acceso a información histórica sistematizada de su propia zona, el caficultor no tiene una referencia objetiva para saber si el comportamiento de su cosecha está dentro de lo normal o si amerita atención.
Asimetría de información en comercialización: Al desconocer el volumen aproximado de su producción futura, el pequeño caficultor carece de poder de negociación frente a los intermediarios del mercado, vendiendo su café bajo esquemas de precios genéricos e inestables.
La información pública capaz de mitigar estos riesgos existe, pero se encuentra dispersa, en formatos puramente tabulares (hojas de datos crudas del EVA) y desconectada de la realidad del campo. AgroIA-Nariño resuelve este problema unificando esos datos abiertos del Estado y traduciéndolos en una estimación clara y un indicador de riesgo entendible, accesible desde una conversación en lenguaje natural o un reporte descargable.
Asimetría de Información en Comercialización: Al desconocer el volumen exacto de su producción futura, el pequeño caficultor carece de poder de negociación frente a los intermediarios del mercado, vendiendo su café especial bajo esquemas de precios genéricos e inestables.

La información pública capaz de mitigar estos riesgos existe, pero se encuentra dispersa, en formatos puramente tabulares y desconectada de la realidad del campo. AgroIA-Nariño resuelve este problema unificando la analítica de datos abiertos del Estado para transformarla en una herramienta de toma de decisiones preventivas en tiempo real.




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

## 4. ⚠️ Antes de la entrega final: reemplazar el dataset demo por datos 100% reales

Por restricciones de red del entorno donde se desarrolló este módulo (sin salida a internet), los modelos que vienen empaquetados en `data/processed/` se entrenaron con un **dataset sintético calibrado** (`generar_datos_demo.py`), NO con datos reales descargados en vivo. El archivo `dataset_entrenamiento.csv` incluye una columna `fuente = SINTETICO_DEMO_NO_USAR_EN_ENTREGA_FINAL` para que esto sea imposible de pasar por alto.

**Antes de entregar el proyecto, ejecuta esto una vez, con internet real:**

```bash
python src/ml/build_dataset.py   # descarga EVA + IDEAM reales y arma el dataset
python src/ml/train_models.py    # reentrena RF + GB + KMeans con los datos reales
```

Esto sobrescribe `data/processed/dataset_entrenamiento.csv`, los `.pkl` de los modelos, y `metricas_validacion.json` con resultados 100% reales. La app (`app.py`) no necesita ningún cambio — carga lo que encuentre en `data/processed/`.

## 5. Cómo se usa en la app

En la Pestaña 1, después de la estimación base (promedio histórico), aparece una sección **"Predicción con Machine Learning"** con las 3 predicciones (RF, GB, híbrido), las métricas de validación del modelo, y el perfil/cluster del municipio. Todo esto también se incluye en el reporte PDF descargable.

## 6. Estructura de archivos nuevos

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

