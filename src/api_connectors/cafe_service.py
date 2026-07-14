import os
import pandas as pd
from sodapy import Socrata
from dotenv import load_dotenv

load_dotenv()


class CafeDataService:
    """Servicio para consultar datos REALES de producción cafetera en Nariño,
    desde el dataset oficial EVA (Evaluaciones Agropecuarias Municipales) de
    UPRA/MinAgricultura, publicado en datos.gov.co.

    -----------------------------------------------------------------------
    NOTA DE CORRECCIÓN (importante):
    La versión anterior de este archivo apuntaba al dataset "g373-n3yy", que
    NO es el EVA agrícola -- es un dataset completamente distinto (de otro
    dominio, sin relación con agricultura). Por eso la consulta nunca traía
    resultados y la app terminaba usando la fórmula fija de respaldo sin que
    fuera evidente el error (la excepción se atrapaba en silencio).

    El dataset correcto, verificado contra la API en vivo de datos.gov.co,
    es "uejq-wxrr" (EVA 2019-2024, Base Agrícola). Además, Socrata expone
    los nombres de columna con las vocales acentuadas recortadas por su
    normalizador interno, así que los nombres reales NO son los que se
    usaban antes:

        campo esperado (antes)   ->   campo real en la API
        ------------------------------------------------
        ano                      ->   a_o
        area_cosechada           ->   rea_cosechada
        area_sembrada            ->   rea_sembrada
        produccion                ->   producci_n
        rendimiento               ->   rendimiento   (este sí coincidía)

    También el valor de 'departamento' y 'cultivo' viene en formato Título
    ("Nariño", "Café"), no en mayúsculas ("NARIÑO", "CAFÉ") como asumía el
    filtro anterior -- por eso ahora se filtra con upper() en el WHERE, para
    que sea robusto sin importar el formato exacto.

    Unidad de 'rendimiento': toneladas por hectárea (verificado contra la
    API), NO kilogramos. Por eso las conversiones a kg se hacen explícitas
    donde correspondan, para no subestimar 1000x la cosecha.
    -----------------------------------------------------------------------
    """

    DOMAIN = "www.datos.gov.co"
    DATASET_EVA_AGRICOLA = "uejq-wxrr"  # EVA 2019-2024, Base Agrícola (verificado)

    def __init__(self):
        self.token = os.getenv("SOCRATA_APP_TOKEN", None)
        self.client = Socrata(self.DOMAIN, self.token, timeout=20)
        self.dataset_id = self.DATASET_EVA_AGRICOLA

    def obtener_datos_cafe_narino(self) -> pd.DataFrame:
        """Consulta el histórico real de producción de café en Nariño."""
        try:
            where_query = "upper(departamento) = 'NARIÑO' AND upper(cultivo) = 'CAFÉ'"
            results = self.client.get(
                self.dataset_id,
                limit=5000,
                where=where_query,
            )

            if not results:
                print("⚠️ La consulta a datos.gov.co no devolvió registros para Nariño/Café.")
                return pd.DataFrame()

            df_raw = pd.DataFrame.from_records(results)

            df = pd.DataFrame()
            df["ano"] = pd.to_numeric(df_raw.get("a_o"), errors="coerce").fillna(0).astype(int)
            df["municipio"] = df_raw.get("municipio", "").astype(str).str.upper()
            df["area_sembrada_ha"] = pd.to_numeric(df_raw.get("rea_sembrada"), errors="coerce").fillna(0.0)
            df["area_cosechada_ha"] = pd.to_numeric(df_raw.get("rea_cosechada"), errors="coerce").fillna(0.0)
            df["produccion_ton"] = pd.to_numeric(df_raw.get("producci_n"), errors="coerce").fillna(0.0)
            df["rendimiento_ton_ha"] = pd.to_numeric(df_raw.get("rendimiento"), errors="coerce").fillna(0.0)

            df = df[(df["produccion_ton"] > 0) & (df["ano"] > 0)]
            df = df.groupby(["ano", "municipio"], as_index=False).agg({
                "area_sembrada_ha": "sum",
                "area_cosechada_ha": "sum",
                "produccion_ton": "sum",
                "rendimiento_ton_ha": "mean",
            })
            return df.sort_values(by=["municipio", "ano"])

        except Exception as e:
            print(f"⚠️ Error de comunicación con datos.gov.co (dataset {self.dataset_id}): {e}")
            return pd.DataFrame()

    def municipios_disponibles(self, df: pd.DataFrame = None) -> list:
        """Lista de municipios de Nariño con datos reales de café disponibles."""
        if df is None:
            df = self.obtener_datos_cafe_narino()
        if df.empty:
            return []
        # Se evita Series.unique() a propósito: en esta combinación de versiones
        # de pandas/pyarrow, llamarlo sobre un DataFrame que pasó por
        # st.cache_data (que usa Arrow internamente) puede producir un
        # segmentation fault nativo. set()/sorted() en Python puro es
        # igual de correcto y no toca el motor de cómputo de pyarrow.
        return sorted(set(df["municipio"].astype(str).tolist()))

    def estimar_cosecha(self, municipio: str, hectareas: float, df: pd.DataFrame = None) -> dict:
        """Estima la cosecha esperada (en toneladas y kg) para una finca,
        usando el rendimiento histórico REAL del municipio seleccionado
        (promedio de los últimos años reportados en el EVA), en vez de un
        multiplicador fijo inventado.

        También calcula un indicador de riesgo basado en la tendencia real
        del rendimiento: si el último año reportado cayó de forma marcada
        frente al promedio histórico del municipio, se marca como riesgo
        más alto (puede deberse a plagas, clima adverso u otros factores;
        no distingue la causa, pero es una señal basada en datos reales,
        no en una lista fija de municipios).
        """
        if df is None:
            df = self.obtener_datos_cafe_narino()

        municipio_norm = municipio.strip().upper()
        datos_municipio = df[df["municipio"] == municipio_norm] if not df.empty else df

        if datos_municipio.empty:
            return {
                "fuente": "sin_datos_reales",
                "rendimiento_ton_ha": None,
                "produccion_estimada_ton": None,
                "produccion_estimada_kg": None,
                "nivel_riesgo": "Desconocido",
                "detalle_riesgo": (
                    f"No se encontraron registros reales del EVA para '{municipio}'. "
                    "Verifica el nombre del municipio o la disponibilidad de datos."
                ),
                "anios_con_datos": 0,
            }

        rendimiento_promedio = datos_municipio["rendimiento_ton_ha"].mean()
        fila_reciente = datos_municipio.loc[datos_municipio["ano"].idxmax()]
        rendimiento_reciente = fila_reciente["rendimiento_ton_ha"]
        anio_reciente = int(fila_reciente["ano"])

        produccion_estimada_ton = round(rendimiento_promedio * hectareas, 3)

        caida_pct = 0.0
        if rendimiento_promedio > 0:
            caida_pct = (rendimiento_promedio - rendimiento_reciente) / rendimiento_promedio * 100

        if caida_pct >= 25:
            nivel_riesgo = "Alto"
            detalle = (
                f"El rendimiento de {anio_reciente} ({rendimiento_reciente:.2f} ton/ha) fue "
                f"{caida_pct:.0f}% menor que el promedio histórico del municipio "
                f"({rendimiento_promedio:.2f} ton/ha). Esto puede indicar afectación "
                f"fitosanitaria, climática u otra causa -- se recomienda inspección en campo."
            )
        elif caida_pct >= 10:
            nivel_riesgo = "Medio"
            detalle = (
                f"El rendimiento de {anio_reciente} está {caida_pct:.0f}% por debajo del "
                f"promedio histórico ({rendimiento_promedio:.2f} ton/ha). Se recomienda "
                "monitoreo preventivo."
            )
        else:
            nivel_riesgo = "Bajo"
            detalle = (
                f"El rendimiento de {anio_reciente} está en línea con el promedio histórico "
                f"del municipio ({rendimiento_promedio:.2f} ton/ha)."
            )

        return {
            "fuente": "eva_datos_gov_co",
            "rendimiento_ton_ha": round(rendimiento_promedio, 3),
            "produccion_estimada_ton": produccion_estimada_ton,
            "produccion_estimada_kg": round(produccion_estimada_ton * 1000, 1),
            "nivel_riesgo": nivel_riesgo,
            "detalle_riesgo": detalle,
            "anios_con_datos": len(set(datos_municipio["ano"].astype(int).tolist())),
            "ultimo_anio_reportado": anio_reciente,
        }
