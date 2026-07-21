"""
aptitud_suelo.py
=================
Consulta el dataset real de UPRA "Evaluación de Tierras con fines
Agrícolas para el cultivo de Café en Nariño" (ggaa-6f3s, datos.gov.co),
para usarlo como parte del análisis de Tab1 -- no solo como contexto de
un chatbot, sino como una señal que se muestra junto a la predicción y
alimenta las recomendaciones.

No se pudo verificar el esquema exacto de este dataset en vivo (entorno
de desarrollo sin salida a internet), así que -- igual que en el
chatbot -- se piden todas las columnas y se busca de forma genérica
cualquier campo cuyo nombre sugiera que describe la categoría de aptitud
(ej. 'aptitud', 'categoria_aptitud', 'clase_aptitud'), en vez de asumir
un nombre exacto.
"""

import requests

DATASET_APTITUD_CAFE_NARINO = "ggaa-6f3s"


def consultar_aptitud_suelo(municipio: str) -> dict:
    api_url = f"https://www.datos.gov.co/resource/{DATASET_APTITUD_CAFE_NARINO}.json"
    try:
        params_filtrado = {"$limit": 5, "$where": f"upper(municipio) like '%{municipio.upper()}%'"}
        try:
            r = requests.get(api_url, params=params_filtrado, timeout=2.5)
            if r.status_code != 200:
                r = requests.get(api_url, params={"$limit": 5}, timeout=2.5)
        except Exception:
            r = requests.get(api_url, params={"$limit": 5}, timeout=2.5)

        if r.status_code != 200:
            return {"disponible": False, "motivo": f"UPRA respondió {r.status_code}."}

        datos = r.json()
        if not datos:
            return {"disponible": False, "motivo": "Sin registros para este municipio en el dataset de aptitud de suelo (UPRA)."}

        categorias = []
        for fila in datos:
            for k, v in fila.items():
                if "aptitud" in k.lower() and v:
                    categorias.append(str(v))

        resumen = ", ".join(sorted(set(categorias))) if categorias else None
        return {
            "disponible": True,
            "registros": datos,
            "resumen": resumen,
            "n_registros": len(datos),
        }
    except Exception as e:
        return {"disponible": False, "motivo": f"Error consultando UPRA: {e}"}
