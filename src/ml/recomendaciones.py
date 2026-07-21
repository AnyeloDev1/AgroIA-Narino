"""
recomendaciones.py
===================
Sistema de recomendación básico (basado en reglas explícitas, no en un
modelo de caja negra) que convierte las salidas de los otros módulos --
el perfil de clustering, el nivel de riesgo por tendencia de rendimiento,
y las anomalías climáticas -- en ACCIONES CONCRETAS para el caficultor.

Por qué basado en reglas y no otro modelo de ML:
Al igual que con las alertas climáticas del otro proyecto hermano
(AgroIA-Nariño FastAPI), una recomendación agronómica accionable debe ser
100% explicable -- el caficultor y el técnico que lo asesora deben poder
entender EXACTAMENTE por qué se sugiere cada acción. Un sistema de reglas
transparente, construido sobre las salidas ya validadas de los modelos de
ML (RF/GB/KMeans), cumple mejor ese estándar que otro modelo predictivo
sobre las recomendaciones mismas.

Este módulo consume:
  - perfil_municipio: dict de predict.obtener_perfil_municipio()
    (cluster de rendimiento histórico: Bajo/Medio/Alto)
  - nivel_riesgo: str de cafe_service.estimar_cosecha()
    (Bajo/Medio/Alto, por caída de rendimiento reciente vs. histórico)
  - resultado_ml: dict de predict.predecir_rendimiento_ml()
    (para leer anomalías climáticas usadas en la predicción)
"""


def _recomendaciones_por_perfil(perfil: str) -> list:
    perfil = (perfil or "").lower()
    if "bajo" in perfil:
        return [
            {
                "prioridad": "Alta",
                "categoria": "Manejo agronómico",
                "accion": (
                    "Solicitar una visita técnica del Comité de Cafeteros para un "
                    "diagnóstico de suelo y un plan de renovación -- el rendimiento "
                    "histórico de este municipio está entre los más bajos de la "
                    "región cafetera de Colombia."
                ),
            },
            {
                "prioridad": "Media",
                "categoria": "Renovación del cultivo",
                "accion": (
                    "Si los lotes tienen más de 12-15 años, evaluar renovación por "
                    "zoca o siembra nueva con variedades resistentes (ej. Castillo) "
                    "-- cafetales envejecidos suelen explicar buena parte de un "
                    "rendimiento bajo sostenido."
                ),
            },
        ]
    if "medio" in perfil:
        return [
            {
                "prioridad": "Media",
                "categoria": "Manejo agronómico",
                "accion": (
                    "Mantener monitoreo periódico de plagas y enfermedades (roya, "
                    "broca) y actualizar el análisis de suelo cada 2 años para "
                    "ajustar la fertilización."
                ),
            },
        ]
    if "alto" in perfil:
        return [
            {
                "prioridad": "Baja",
                "categoria": "Comercialización",
                "accion": (
                    "Con este desempeño productivo por encima del promedio, evaluar "
                    "la certificación de café especial (orgánico, de origen, taza de "
                    "excelencia) para acceder a mejores precios -- consultar con el "
                    "Comité de Cafeteros los requisitos vigentes."
                ),
            },
            {
                "prioridad": "Baja",
                "categoria": "Manejo agronómico",
                "accion": "Mantener las prácticas actuales de manejo, que ya muestran buenos resultados.",
            },
        ]
    return []


def _recomendaciones_por_riesgo(nivel_riesgo: str) -> list:
    nivel_riesgo = (nivel_riesgo or "").lower()
    if nivel_riesgo == "alto":
        return [
            {
                "prioridad": "Alta",
                "categoria": "Inspección de campo",
                "accion": (
                    "El rendimiento del último año reportado cayó de forma marcada "
                    "frente al histórico de este municipio. Programar una inspección "
                    "de campo en las próximas semanas para descartar roya, broca u "
                    "otro factor, y documentar los hallazgos."
                ),
            },
        ]
    if nivel_riesgo == "medio":
        return [
            {
                "prioridad": "Media",
                "categoria": "Monitoreo preventivo",
                "accion": "Programar un monitoreo preventivo del cultivo en el próximo mes, dada la caída moderada de rendimiento reciente.",
            },
        ]
    return []


def _recomendaciones_por_clima(resultado_ml: dict) -> list:
    if not resultado_ml or not resultado_ml.get("disponible"):
        return []

    recomendaciones = []
    precip = resultado_ml.get("precipitacion_usada_mm")
    temp = resultado_ml.get("temperatura_usada_c")

    # Umbrales de referencia agroclimática (consistentes con el motor de
    # alertas climáticas del proyecto hermano AgroIA-Nariño / FastAPI).
    if precip is not None and precip > 2200:
        recomendaciones.append({
            "prioridad": "Media",
            "categoria": "Clima",
            "accion": (
                "La precipitación de referencia para la zona es alta. Revisar "
                "drenajes del lote y considerar posponer aplicaciones foliares -- "
                "la humedad prolongada favorece la propagación de roya."
            ),
        })
    if temp is not None and temp > 23:
        recomendaciones.append({
            "prioridad": "Baja",
            "categoria": "Clima",
            "accion": (
                "La temperatura de referencia está por encima del rango típico "
                "cafetero. Vigilar el desarrollo de broca, que se favorece con "
                "temperaturas más altas."
            ),
        })
    return recomendaciones


def _recomendaciones_por_aptitud(aptitud_resumen: str) -> list:
    if not aptitud_resumen:
        return []
    texto = aptitud_resumen.lower()
    if "no apt" in texto or "baja" in texto:
        return [{
            "prioridad": "Alta",
            "categoria": "Aptitud de suelo",
            "accion": (
                f"La evaluación de tierras de la UPRA para esta zona indica aptitud '{aptitud_resumen}' "
                "para café. Esto es una limitante estructural, no algo que se resuelva con manejo agronómico "
                "-- vale la pena consultar con la UMATA/Comité de Cafeteros si conviene diversificar cultivos "
                "en los lotes de menor aptitud."
            ),
        }]
    if "media" in texto:
        return [{
            "prioridad": "Media",
            "categoria": "Aptitud de suelo",
            "accion": (
                f"La aptitud de suelo reportada por la UPRA es '{aptitud_resumen}'. Con buen manejo "
                "(fertilización y sombrío adecuados) es posible mantener rendimientos aceptables."
            ),
        }]
    return []


def generar_recomendaciones(perfil_municipio: dict, nivel_riesgo: str, resultado_ml: dict, aptitud_resumen: str = None) -> list:
    """Combina las 4 fuentes de reglas y devuelve una lista de recomendaciones
    ordenadas por prioridad (Alta > Media > Baja), sin duplicados."""
    orden_prioridad = {"Alta": 0, "Media": 1, "Baja": 2}

    perfil_texto = perfil_municipio.get("perfil") if perfil_municipio else None
    recomendaciones = (
        _recomendaciones_por_riesgo(nivel_riesgo)
        + _recomendaciones_por_aptitud(aptitud_resumen)
        + _recomendaciones_por_perfil(perfil_texto)
        + _recomendaciones_por_clima(resultado_ml)
    )

    vistos = set()
    unicas = []
    for r in recomendaciones:
        if r["accion"] not in vistos:
            vistos.add(r["accion"])
            unicas.append(r)

    unicas.sort(key=lambda r: orden_prioridad.get(r["prioridad"], 3))
    return unicas
