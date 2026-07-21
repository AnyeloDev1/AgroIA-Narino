import sys
import time
import urllib.parse
from pathlib import Path

import requests
import streamlit as st
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent))
from src.api_connectors.cafe_service import CafeDataService
from src.utils.pdf_report import generar_reporte_pdf
from src.ml.predict import predecir_rendimiento_ml, metricas_modelo, referencia_rendimiento_alto
from src.ml.recomendaciones import generar_recomendaciones
from src.ml.aptitud_suelo import consultar_aptitud_suelo
sys.path.append(str(Path(__file__).resolve().parent / "src" / "ml"))
from ideam_reciente import obtener_riesgo_climatico_actual

# ==============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="AgroIA-Nariño",
    page_icon="☕",
    layout="centered"
)

st.markdown("""
<style>
:root {
    --azul-tech: #2F8FE0;
    --azul-tech-claro: #6BB6F5;
    --verde-campo: #4CAF50;
    --verde-campo-claro: #7BD07F;
    --cafe-producto: #8B5A2B;
    --cafe-oscuro: #2B1D12;
}

/* --- Fondo degradado: azul tecnología -> verde campo, oscurecido para contraste --- */
[data-testid="stAppViewContainer"], .stApp {
    background: linear-gradient(160deg, #0E2338 0%, #123A2E 55%, #163A22 100%) !important;
    background-attachment: fixed !important;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #14110D 0%, #1B1712 100%) !important;
    border-right: 1px solid rgba(47,143,224,0.18);
}

.encabezado-agroia {
    background: linear-gradient(135deg, #2B1D12 0%, #4A2E1A 100%);
    border: 1px solid rgba(47,143,224,0.35);
    border-left: 4px solid var(--azul-tech);
    border-radius: 14px;
    padding: 1.4rem 1.7rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 10px 30px -12px rgba(0,0,0,0.6);
}
.encabezado-agroia .eyebrow {
    font-family: monospace;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--azul-tech-claro);
    margin: 0 0 0.4rem;
}
.encabezado-agroia h1 {
    color: #FBF3E7;
    font-size: 2.1rem;
    margin: 0 0 0.3rem;
    font-weight: 800;
}
.encabezado-agroia h1 span { color: var(--verde-campo-claro); }
.encabezado-agroia p {
    color: #D8CDB8;
    font-size: 0.92rem;
    margin: 0;
}

/* --- Badges de riesgo: semántica universal (rojo/ámbar/verde), saturados --- */
.badge-riesgo {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.85rem;
    letter-spacing: 0.02em;
}
.badge-alto   { background: #D6432E; color: #FFF3EE; }
.badge-medio  { background: #E0A020; color: #2B1D12; }
.badge-bajo   { background: var(--verde-campo); color: #10230F; }
.badge-desconocido { background: #6B5B44; color: #F4EFE4; }

/* --- Tarjetas de métricas: café oscuro sobre el degradado, valores en verde campo --- */
div[data-testid="stMetric"] {
    background: #251B12;
    border: 1px solid rgba(76,175,80,0.28);
    border-radius: 12px;
    padding: 0.9rem 1rem 0.6rem;
    box-shadow: 0 6px 18px -8px rgba(0,0,0,0.5);
}
div[data-testid="stMetricValue"] { color: var(--verde-campo-claro); }

/* --- Contenedores/tarjetas generales: café oscuro semitransparente sobre el degradado --- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(37,27,18,0.55);
    border-radius: 12px;
}

/* --- Cajas st.info / st.warning / st.error --- */
div[data-testid="stAlertContainer"] {
    border-radius: 10px;
    border-width: 1px;
    border-style: solid;
}

/* --- Pestañas y elementos interactivos: azul tecnología --- */
button[data-baseweb="tab"] { font-weight: 600; }
button[data-baseweb="tab"][aria-selected="true"] { color: var(--azul-tech-claro) !important; }
div[data-baseweb="tab-highlight"] { background-color: var(--azul-tech) !important; }

/* --- Etiqueta de sección "Machine Learning": chip azul tecnológico --- */
.chip-ml {
    display: inline-block;
    background: rgba(47,143,224,0.15);
    border: 1px solid var(--azul-tech);
    color: var(--azul-tech-claro);
    border-radius: 6px;
    padding: 0.1rem 0.5rem;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-left: 0.4rem;
}
</style>
<div class="encabezado-agroia">
    <p></p>
    <h1>☕ AgroIA<span>-Nariño</span></h1>
    <p>La inteligencia que impulsa el café de Nariño</p>
</div>
""", unsafe_allow_html=True)

#Configuración de la IA de Google
st.sidebar.header("🔑 Configuración de IA")
gemini_api_key = st.sidebar.text_input(
    "Ingresa tu OpenRouter API Key (empieza con 'sk-or-'):", type="password"
)
st.sidebar.caption(
    "Se usa para llamar al modelo Gemini 2.0 Flash *a través de* OpenRouter "
   
)
st.sidebar.markdown("""
*Consigue una clave gratis en [OpenRouter](https://openrouter.ai/keys)*
""")

# CREACIÓN DE PESTAÑAS PARA ORGANIZAR LA APP
tab1, tab2 = st.tabs(["🔮 Predicciones e Informes", "💬 Agrónomo Virtual"])

# ==============================================================================
# PESTAÑA 1: PREDICCIONES CON DATOS REALES (EVA - datos.gov.co) + WHATSAPP
# ================================================================================
with tab1:
    st.header("📋 Datos de la Finca")

   
    @st.cache_data(ttl=3600, show_spinner=False)
    def _cargar_datos_eva():
        servicio = CafeDataService()
        return servicio.obtener_datos_cafe_narino()

    with st.spinner("📡 Consultando datos reales de café en Nariño (EVA - datos.gov.co)..."):
        df_eva = _cargar_datos_eva()

    if not df_eva.empty:
        # Se fuerzan tipos de dato "planos" (numpy) justo después de leer el
        # caché. st.cache_data puede devolver columnas con backend de Arrow,
        # y en esta combinación de versiones de pandas/pyarrow, operaciones
        # como .unique(), .mean() o .idxmax() sobre esas columnas pueden
        # producir un segmentation fault nativo. Forzar dtypes numpy antes
        # de cualquier cálculo evita por completo esa ruta de código.
        df_eva = df_eva.astype({
            "ano": "int64",
            "municipio": "object",
            "area_sembrada_ha": "float64",
            "area_cosechada_ha": "float64",
            "produccion_ton": "float64",
            "rendimiento_ton_ha": "float64",
        })

    if df_eva.empty:
        st.error(
            "⚠️ No se pudo obtener el histórico real desde datos.gov.co en este momento "
            "(puede ser un problema temporal de conexión o de la API). "
            "Verifica tu conexión a internet y vuelve a intentar."
        )
        municipios_reales = ["Samaniego", "Sandoná", "Consacá", "La Unión", "Buesaco"]
        st.caption("Mostrando una lista de municipios de referencia mientras se restablece la conexión.")
    else:
        municipios_reales = CafeDataService().municipios_disponibles(df_eva)
        st.success(f"🟢 Conectado a datos.gov.co — {len(municipios_reales)} municipios de Nariño con histórico real de café.")

    municipio = st.selectbox(
        "Seleccione el Municipio:",
        municipios_reales,
        key="municipio_select"
    )

    hectareas = st.number_input(
        "Hectáreas sembradas de café:",
        min_value=0.1,
        max_value=100.0,
        value=1.5,
        step=0.5,
        key="hectareas_input"
    )

    whatsapp_num = st.text_input(
        "📱 Número de WhatsApp Destino (Ej: 573101234567):",
        value="57",
        key="whatsapp_input"
    )

    precio_carga = st.number_input(
        "💰 Precio de referencia (COP por carga de 125 kg) — opcional",
        min_value=0, value=0, step=10000,
        help=(
            "Si lo ingresas, calculamos el valor estimado de tu cosecha y el "
            "impacto económico de mejorar tu rendimiento. Consulta el precio "
            "vigente con el botón de la FNC más abajo, o ingresa el que te "
            "paga tu comprador."
        ),
        key="precio_carga_input",
    )

    if st.button("🔮 Generar Predicción con Datos Reales", key="btn_prediccion"):

        with st.status("🤖 Consola de IA — iniciando diagnóstico...", expanded=True) as consola:
            st.write(f"🌱 **Datos recibidos:** municipio *{municipio}*, {hectareas} ha")
            time.sleep(0.3)

            consola.update(label="📡 Verificando conexión con datos.gov.co (EVA)...")
            if df_eva.empty:
                st.write("⚠️ Sin conexión a datos.gov.co en este momento — no hay histórico real disponible.")
            else:
                st.write(
                    f"✅ Conectado a **datos.gov.co** · dataset EVA (`uejq-wxrr`) · "
                    f"{len(municipios_reales)} municipios de Nariño con histórico real de café"
                )
            time.sleep(0.3)

            consola.update(label="🧠 Calculando rendimiento histórico real del municipio...")
            resultado = CafeDataService().estimar_cosecha(municipio, hectareas, df=df_eva)

            if resultado["fuente"] == "sin_datos_reales":
                st.write(f"⚠️ {resultado['detalle_riesgo']}")
                consola.update(label="⚠️ Diagnóstico incompleto — sin datos reales para este municipio", state="error")
            else:
                st.write(
                    f"✅ Rendimiento histórico promedio (**{resultado['anios_con_datos']} año(s)** reportados): "
                    f"**{resultado['rendimiento_ton_ha']} ton/ha**"
                )
                time.sleep(0.3)

                consola.update(label="📊 Comparando último año reportado contra el histórico...")
                st.write(
                    f"📈 Último dato disponible: **{resultado['ultimo_anio_reportado']}** — "
                    f"comparado contra el promedio de los años anteriores"
                )
                st.write(f"🔎 Resultado del análisis de tendencia: **Riesgo {resultado['nivel_riesgo']}**")
                st.caption(resultado["detalle_riesgo"])
                time.sleep(0.2)

                consola.update(label="🌲 Ejecutando modelos de Machine Learning (Random Forest + Gradient Boosting)...")
                datos_municipio_ml = df_eva[df_eva["municipio"] == municipio.strip().upper()].sort_values("ano", ascending=False)
                historial_rendimiento = datos_municipio_ml["rendimiento_ton_ha"].tolist()
                ratio_historico = (
                    (datos_municipio_ml["area_cosechada_ha"] / datos_municipio_ml["area_sembrada_ha"]).mean()
                    if not datos_municipio_ml.empty else 0.93
                )
                area_cosechada_estimada = hectareas * (ratio_historico if pd.notna(ratio_historico) else 0.93)

                resultado_ml = predecir_rendimiento_ml(
                    departamento="NARIÑO", municipio=municipio,
                    area_sembrada_ha=hectareas, area_cosechada_ha=area_cosechada_estimada,
                    historial_rendimiento=historial_rendimiento,
                )

                if not resultado_ml.get("disponible"):
                    st.write(f"⚠️ {resultado_ml.get('motivo')}")
                else:
                    st.write(
                        f"✅ Random Forest: **{resultado_ml['rendimiento_rf_ton_ha']} ton/ha** · "
                        f"Gradient Boosting: **{resultado_ml['rendimiento_gb_ton_ha']} ton/ha**"
                    )
                    st.write(f"🔀 Modelo híbrido (promedio RF+GB): **{resultado_ml['rendimiento_hibrido_ton_ha']} ton/ha**")
                    if resultado_ml.get("perfil_municipio"):
                        st.write(f"🗂️ Perfil del municipio (clustering): **{resultado_ml['perfil_municipio']['perfil']}**")
                time.sleep(0.2)

                consola.update(label="🌦️ Evaluando riesgo climático hacia adelante (IDEAM en vivo)...")
                riesgo_clima_futuro = obtener_riesgo_climatico_actual(municipio)
                if riesgo_clima_futuro.get("disponible"):
                    fuente_txt = "🟢 en vivo" if riesgo_clima_futuro["fuente"] == "ideam_tiempo_real" else "⚪ estimado"
                    st.write(
                        f"✅ Clima reciente ({fuente_txt}): {riesgo_clima_futuro['temperatura_c']:.1f}°C · "
                        f"{riesgo_clima_futuro['precipitacion_7d_mm']:.0f} mm en los últimos 7 días"
                    )
                    st.write(f"🔎 Riesgo climático hacia adelante: **{', '.join(riesgo_clima_futuro['alertas'])}**")
                else:
                    st.write(f"⚠️ {riesgo_clima_futuro.get('motivo')}")
                time.sleep(0.2)

                consola.update(label="🌱 Consultando aptitud de suelo para café (UPRA)...")
                aptitud_info = consultar_aptitud_suelo(municipio)
                if aptitud_info.get("disponible") and aptitud_info.get("resumen"):
                    st.write(f"✅ Aptitud de suelo (UPRA): **{aptitud_info['resumen']}**")
                else:
                    st.write(f"⚠️ Aptitud de suelo no disponible: {aptitud_info.get('motivo', 'sin dato')}")
                time.sleep(0.2)

                consola.update(label="✅ Diagnóstico completo generado con datos reales del EVA", state="complete")

        if resultado["fuente"] != "sin_datos_reales":
            st.metric(
                "Cosecha proyectada",
                f"{resultado['produccion_estimada_ton']:.2f} ton",
                help=f"= rendimiento histórico promedio ({resultado['rendimiento_ton_ha']} ton/ha) × {hectareas} ha"
            )
            clase_badge = f"badge-{resultado['nivel_riesgo'].lower()}" if resultado['nivel_riesgo'].lower() in ("alto", "medio", "bajo") else "badge-desconocido"
            st.markdown(
                f"**Nivel de riesgo** (tendencia real de rendimiento) "
                f"<span class='badge-riesgo {clase_badge}'>{resultado['nivel_riesgo'].upper()}</span>",
                unsafe_allow_html=True,
            )
            st.caption(resultado["detalle_riesgo"])
            st.caption(
                f"Fuente: EVA - datos.gov.co · {resultado['anios_con_datos']} año(s) de histórico · "
                f"último dato: {resultado['ultimo_anio_reportado']}"
            )

            if resultado_ml.get("disponible"):
                st.markdown(
                    "##### 🌲 Predicción con Machine Learning (Random Forest + Gradient Boosting) "
                    "<span class='chip-ml'>IA</span>",
                    unsafe_allow_html=True,
                )
                col_rf, col_gb, col_hib = st.columns(3)
                col_rf.metric("Random Forest", f"{resultado_ml['rendimiento_rf_ton_ha']} ton/ha")
                col_gb.metric("Gradient Boosting", f"{resultado_ml['rendimiento_gb_ton_ha']} ton/ha")
                col_hib.metric("Modelo híbrido", f"{resultado_ml['rendimiento_hibrido_ton_ha']} ton/ha")

                metricas = metricas_modelo()
                if metricas:
                    st.caption(
                        f"Validación temporal (entrenado con años anteriores, validado con "
                        f"{metricas.get('anio_validacion')}, no visto en entrenamiento): "
                        f"MAE Random Forest = {metricas['random_forest']['mae_ton_ha']} ton/ha "
                        f"(R²={metricas['random_forest']['r2']}) · "
                        f"MAE Gradient Boosting = {metricas['gradient_boosting']['mae_ton_ha']} ton/ha "
                        f"(R²={metricas['gradient_boosting']['r2']}) · "
                        f"{metricas.get('filas_entrenamiento')} filas de entrenamiento, "
                        f"{len(metricas.get('variables_usadas', []))} variables."
                    )
                if resultado_ml.get("perfil_municipio"):
                    perfil = resultado_ml["perfil_municipio"]
                    st.info(
                        f"🗂️ **Perfil del municipio** (agrupado por clustering K-Means junto a otros "
                        f"municipios cafeteros de Colombia): *{perfil['perfil']}* — rendimiento histórico "
                        f"promedio {perfil['rendimiento_promedio_historico']} ton/ha, "
                        f"variabilidad {perfil['variabilidad']} ton/ha."
                    )

                if precio_carga and precio_carga > 0:
                    st.markdown("##### 💰 Impacto económico estimado")
                    kg_por_carga = 125
                    valor_cosecha_actual = (resultado["produccion_estimada_kg"] / kg_por_carga) * precio_carga

                    col_val1, col_val2 = st.columns(2)
                    col_val1.metric(
                        "Valor estimado de tu cosecha",
                        f"${valor_cosecha_actual:,.0f} COP",
                        help=f"({resultado['produccion_estimada_kg']:.0f} kg ÷ {kg_por_carga} kg/carga) × precio ingresado",
                    )

                    rendimiento_referencia_alto = referencia_rendimiento_alto()
                    rendimiento_actual_ml = resultado_ml["rendimiento_hibrido_ton_ha"]
                    if rendimiento_referencia_alto and rendimiento_referencia_alto > rendimiento_actual_ml:
                        delta_rendimiento = rendimiento_referencia_alto - rendimiento_actual_ml
                        kg_extra_potencial = delta_rendimiento * hectareas * 1000
                        valor_extra_potencial = (kg_extra_potencial / kg_por_carga) * precio_carga
                        col_val2.metric(
                            "Potencial adicional alcanzable",
                            f"${valor_extra_potencial:,.0f} COP",
                            help=(
                                f"Si tu finca alcanzara el rendimiento promedio del perfil de mejor "
                                f"desempeño ({rendimiento_referencia_alto:.2f} ton/ha), producirías "
                                f"~{kg_extra_potencial:.0f} kg adicionales."
                            ),
                        )
                        st.caption(
                            f"📈 El perfil de mejor desempeño entre los municipios cafeteros analizados "
                            f"rinde en promedio {rendimiento_referencia_alto:.2f} ton/ha. Cerrar esa brecha "
                            f"representaría aproximadamente **${valor_extra_potencial:,.0f} COP adicionales** "
                            f"al precio que ingresaste — este es el techo de las recomendaciones de abajo."
                        )
                    else:
                        col_val2.metric("Potencial adicional alcanzable", "Ya en el rango alto")
                        st.caption("Tu finca ya rinde al nivel del perfil de mejor desempeño analizado.")
            else:
                st.caption(f"ℹ️ Predicción con Machine Learning no disponible: {resultado_ml.get('motivo', '')}")

            st.markdown("##### 🌦️ Riesgo climático hacia adelante <span class='chip-ml'>IDEAM en vivo</span>", unsafe_allow_html=True)
            st.caption(
                "Distinto del 'nivel de riesgo' de arriba (que mide una caída de rendimiento que YA ocurrió, "
                "según el histórico anual del EVA): esto usa el clima real más reciente reportado por IDEAM "
                "para señalar condiciones que podrían afectar el cultivo en los próximos días."
            )
            if riesgo_clima_futuro.get("disponible"):
                clase_badge_clima = f"badge-{riesgo_clima_futuro['nivel_riesgo'].lower()}" if riesgo_clima_futuro['nivel_riesgo'].lower() in ("alto", "medio", "bajo") else "badge-desconocido"
                st.markdown(
                    f"**Alertas:** {', '.join(riesgo_clima_futuro['alertas'])} "
                    f"<span class='badge-riesgo {clase_badge_clima}'>{riesgo_clima_futuro['nivel_riesgo'].upper()}</span>",
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"Temperatura reciente: {riesgo_clima_futuro['temperatura_c']:.1f}°C · "
                    f"Precipitación últimos 7 días: {riesgo_clima_futuro['precipitacion_7d_mm']:.0f} mm "
                    f"({'dato en vivo IDEAM' if riesgo_clima_futuro['fuente'] == 'ideam_tiempo_real' else 'estimado'})"
                )
                for rec in riesgo_clima_futuro["recomendaciones"]:
                    st.markdown(f"🌦️ {rec}")
            else:
                st.caption(f"⚠️ {riesgo_clima_futuro.get('motivo', 'No disponible en este momento.')}")

            st.markdown("##### 🌱 Aptitud de suelo para café <span class='chip-ml'>UPRA</span>", unsafe_allow_html=True)
            if aptitud_info.get("disponible") and aptitud_info.get("resumen"):
                st.markdown(f"**Categoría reportada:** {aptitud_info['resumen']}")
                st.caption(
                    f"Fuente: Evaluación de Tierras con fines Agrícolas para café en Nariño (UPRA, "
                    f"`ggaa-6f3s`) · {aptitud_info['n_registros']} registro(s) encontrado(s)."
                )
            else:
                st.caption(f"⚠️ {aptitud_info.get('motivo', 'No disponible en este momento.')}")

            recomendaciones_accionables = generar_recomendaciones(
                resultado_ml.get("perfil_municipio") if resultado_ml.get("disponible") else None,
                resultado["nivel_riesgo"],
                resultado_ml,
                aptitud_info.get("resumen") if aptitud_info.get("disponible") else None,
            )
            if recomendaciones_accionables:
                st.markdown("##### ✅ Recomendaciones para esta finca")
                iconos_prioridad = {"Alta": "🔴", "Media": "🟡", "Baja": "🟢"}
                for r in recomendaciones_accionables:
                    icono = iconos_prioridad.get(r["prioridad"], "⚪")
                    st.markdown(f"{icono} **[{r['categoria']}]** {r['accion']}")

            linea_ml = ""
            if resultado_ml.get("disponible"):
                linea_ml = f"\n🌲 Predicción ML (RF+GB): {resultado_ml['rendimiento_hibrido_ton_ha']} ton/ha"

            linea_recom = ""
            if recomendaciones_accionables:
                top = recomendaciones_accionables[0]
                linea_recom = f"\n✅ Recomendación principal: {top['accion']}"

            mensaje_chatbot = (
                f"☕ AgroIA - Nariño Informa ☕\n\n"
                f"📢 Reporte generado con datos reales (EVA - datos.gov.co)\n"
                f"📍 Municipio: {municipio}\n"
                f"🔮 Cosecha Proyectada: {resultado['produccion_estimada_ton']:.2f} Toneladas "
                f"({resultado['produccion_estimada_kg']:.0f} kg)"
                f"{linea_ml}\n"
                f"⚠️ Riesgo (tendencia de rendimiento): {resultado['nivel_riesgo']}\n"
                f"💡 {resultado['detalle_riesgo']}"
                f"{linea_recom}"
            )

            st.markdown("##### 📤 Compartir este reporte")
            col_wsp, col_pdf = st.columns(2)

            with col_wsp:
                if not whatsapp_num or whatsapp_num.strip() == "57":
                    st.info("👈 Ingresa un número de WhatsApp válido arriba para poder enviarlo.")
                else:
                    # Enlace directo a WhatsApp

                    texto_codificado = urllib.parse.quote(mensaje_chatbot)
                    link_whatsapp = f"https://wa.me/{whatsapp_num.strip()}?text={texto_codificado}"
                    st.link_button("💬 Enviar mensaje de texto", link_whatsapp, use_container_width=True)

            with col_pdf:
                pdf_bytes = generar_reporte_pdf(
                    municipio, hectareas, resultado, df_historico=df_eva,
                    resultado_ml=resultado_ml, recomendaciones=recomendaciones_accionables,
                )
                nombre_archivo = f"reporte_agroia_{municipio.strip().lower().replace(' ', '_')}.pdf"
                st.download_button(
                    "📄 Descargar reporte en PDF",
                    data=pdf_bytes,
                    file_name=nombre_archivo,
                    mime="application/pdf",
                    use_container_width=True,
                )

            st.caption(
                "💡 WhatsApp no permite adjuntar archivos automáticamente desde un enlace. "
                "Descarga el PDF con el botón de arriba y adjúntalo tú mismo en el chat "
                "(📎 en WhatsApp Web o el celular), igual que adjuntarías cualquier foto o documento."
            )

            st.markdown("---")
            st.markdown("##### 💰 Precio de referencia del café")
            st.caption(
                "Este proyecto no integra un dataset de precios de mercado: no se encontró uno "
                "confiable y específico para café en datos.gov.co (el componente de precios "
                "mayoristas del SIPSA/DANE cubre frutas, verduras y otros perecederos que sí se "
                "transan en plazas de mercado, no café). El precio oficial del café en Colombia lo "
                "publica a diario la Federación Nacional de Cafeteros en su propio sistema."
            )
            st.link_button(
                "📈 Ver precio interno de referencia actual (FNC)",
                "https://federaciondecafeteros.org/wp/publicaciones/",
            )

            # Persistimos lo necesario para el simulador de escenarios de abajo,
            # que debe seguir funcionando en reruns posteriores sin que el
            # usuario tenga que volver a presionar "Generar Predicción"
            # (los botones de Streamlit solo son True en el run inmediato al clic).
            if resultado_ml.get("disponible"):
                st.session_state["sim_inputs"] = {
                    "municipio": municipio,
                    "hectareas": hectareas,
                    "area_cosechada_estimada": area_cosechada_estimada,
                    "historial_rendimiento": historial_rendimiento,
                    "precipitacion_base": resultado_ml["precipitacion_usada_mm"],
                    "temperatura_base": resultado_ml["temperatura_usada_c"],
                    "rendimiento_base": resultado_ml["rendimiento_hibrido_ton_ha"],
                }

    
# =====================================================================================
# PESTAÑA 2: CHATBOT
# ==============================================================================
with tab2:
    st.header("👨‍🌾 Don Agromía - Tu Ingeniero Virtual con API del Estado")
    st.markdown("*Consultando la API de Datos Abiertos de forma optimizada y en tiempo real.*")

    if not gemini_api_key or len(gemini_api_key.strip()) < 10:
        st.info("👈 Por favor, introduce tu OpenRouter Token (sk-or-...) en la barra lateral para activar el Chat.")
    else:
        st.success("🟢 Conexión lista para respuestas rápidas.")
        
        # DEFINICIÓN DEL ROL EXPERTO CON INSTRUCCIÓN DE DATOS EN VIVO
        system_instruction = (
            "Eres 'Don Agromía', un ingeniero agrónomo virtual experto en la caficultura de Nariño, Colombia. "
            "Tu misión es asesorar a pequeños caficultores locales con total precisión técnica. Debes responder "
            "SIEMPRE en un lenguaje campesino amable, cercano y respetuoso (usa palabras como 'pariente', 'mi amigo', 'con juicio'). "
            "IMPORTANTE: Se te proporcionará un bloque de 'DATOS OFICIALES EN TIEMPO REAL' extraído de la API de datos.gov.co. "
            "Si contiene información, úsala brevemente para enriquecer tu consejo. Si dice que no hay datos o hubo error, "
            "no te preocupes y da tu mejor consejo basado en tu conocimiento general de agronomía. "
            "Asegúrate de cerrar bien tus ideas y terminar por completo la última frase."
        )

        # Inicializar el historial del chat en la sesión si no existe
        if "gemini_messages" not in st.session_state:
            st.session_state.gemini_messages = [
                {"role": "assistant", "content": "¡Hola, pariente! Ya estoy aquí con el internet bien conectado. Cuénteme qué le preocupa hoy de su cafetal, de la broca o el abono, y miramos qué nos dicen las bases de datos."}
            ]

        # Mostrar los mensajes anteriores del chat
        for message in st.session_state.gemini_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # Entrada de texto del usuario
        if user_query := st.chat_input("Pregúntele a Don Agromía aquí..."):
            
            st.session_state.gemini_messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.write(user_query)

            # --- CONSULTA EN TIEMPO REAL OPTIMIZADA (LÍMITE ESTRICTO DE TIEMPO Y TAMAÑO) ---
            datos_contexto_api = "No se encontraron registros específicos en esta consulta rápida."

            palabras = user_query.lower()

            # 
            MUNICIPIOS_NARINO_CONOCIDOS = [
                "samaniego", "sandona", "sandoná", "consaca", "consacá", "la florida",
                "linares", "buesaco", "taminango", "el tambo", "arboleda", "pasto",
                "ipiales", "tumaco", "la union", "la unión", "colon", "colón", "chachagui", "chachagüí",
            ]
            PALABRAS_CLIMA = [
                "clima", "tiempo", "lluvia", "lluvioso", "temperatura", "humedad",
                "pronostico", "pronóstico", "precipitacion", "precipitación",
                "seco", "sequia", "sequía", "frio", "frío", "calor", "helada",
            ]
            municipio_detectado = next((m for m in MUNICIPIOS_NARINO_CONOCIDOS if m in palabras), None)
            es_pregunta_clima = any(p in palabras for p in PALABRAS_CLIMA)

            try:
                import requests
                import json

                if es_pregunta_clima:
                    # --- CLIMA: consulta en vivo a IDEAM (temperatura + precipitación) ---
                    # Datasets verificados contra la API real de datos.gov.co (mismos
                    # que usa build_dataset.py para entrenar los modelos de ML):
                    #   sbwg-7ju4 = temperatura por estación   ·   s54a-sgyg = precipitación por estación
                    # Ambos tienen campos 'municipio', 'departamento', 'fechaobservacion',
                    # 'valorobservado' -- se pide la lectura más reciente disponible.
                    filtro_geo = (
                        f"upper(municipio) like '%{municipio_detectado.upper()}%'"
                        if municipio_detectado else "upper(departamento) like '%NARI%'"
                    )
                    partes_clima = []
                    for nombre_var, dataset_id in [("Temperatura (°C)", "sbwg-7ju4"), ("Precipitación (mm)", "s54a-sgyg")]:
                        api_url = f"https://www.datos.gov.co/resource/{dataset_id}.json"
                        params = {"$limit": 1, "$order": "fechaobservacion DESC", "$where": filtro_geo}
                        try:
                            r = requests.get(api_url, params=params, timeout=1.5)
                            if r.status_code != 200:
                                # Reintento sin filtro geográfico si el campo/valor no coincide
                                r = requests.get(
                                    api_url, params={"$limit": 1, "$order": "fechaobservacion DESC"}, timeout=1.5
                                )
                            if r.status_code == 200 and r.json():
                                partes_clima.append(f"{nombre_var}: {json.dumps(r.json()[0], ensure_ascii=False)}")
                            else:
                                print(f"[chatbot] IDEAM {dataset_id} respondió {r.status_code}: {r.text[:200]}")
                        except Exception as e:
                            print(f"[chatbot] Error consultando IDEAM {dataset_id}: {e}")

                    if partes_clima:
                        zona = municipio_detectado.title() if municipio_detectado else "Nariño (departamento)"
                        datos_contexto_api = (
                            f"DATOS CLIMÁTICOS EN VIVO DESDE DATOS.GOV.CO (IDEAM) para {zona}:\n"
                            + "\n".join(partes_clima)
                            + "\n\nNOTA: son lecturas de estaciones IDEAM, no un pronóstico -- "
                              "son el dato medido más reciente disponible en la API."
                        )
                    else:
                        datos_contexto_api = (
                            "No se pudo obtener un dato climático en vivo de IDEAM en este momento "
                            "(la API puede estar ocupada o sin datos recientes para esta zona)."
                        )
                else:
                    # --- APTITUD DE SUELO: dataset UPRA para café en Nariño ---
                    api_url = "https://www.datos.gov.co/resource/ggaa-6f3s.json"


                    params = {"$limit": 3}
                    if municipio_detectado or "nariño" in palabras or "narino" in palabras:
                        campo_filtro = "municipio" if municipio_detectado else "departamento"
                        valor_filtro = municipio_detectado.upper() if municipio_detectado else "NARI"
                        params_con_filtro = {**params, "$where": f"upper({campo_filtro}) like '%{valor_filtro}%'"}
                        response_api = requests.get(api_url, params=params_con_filtro, timeout=1.5)
                        if response_api.status_code != 200:
                            response_api = requests.get(api_url, params=params, timeout=1.5)
                    else:
                        response_api = requests.get(api_url, params=params, timeout=1.5)

                    if response_api.status_code == 200:
                        datos_json = response_api.json()
                        if datos_json and len(datos_json) > 0:
                            datos_contexto_api = (
                                "DATOS EN VIVO DESDE DATOS.GOV.CO "
                                "(Evaluación de Tierras con fines Agrícolas para café en Nariño, UPRA):\n"
                                f"{json.dumps(datos_json, indent=2, ensure_ascii=False)}"
                            )
                    else:
                        print(f"[chatbot] ggaa-6f3s respondió {response_api.status_code}: {response_api.text[:300]}")
            except Exception as e:
                # Si expira el tiempo o falla, el contexto queda limpio para que Gemini responda usando su propio cerebro
                datos_contexto_api = "Servidor de datos abiertos ocupado. Responde con tu conocimiento experto general."
                print(f"[chatbot] Error consultando datos abiertos: {e}")

            # --- LLAMADA A GEMINI 2.0 A TRAVÉS DE OPENROUTER ---
            with st.chat_message("assistant"):
                with st.spinner("Don Agromía está redactando su consejo..."):
                    try:
                        token_clean = gemini_api_key.strip()
                        url_or = "https://openrouter.ai/api/v1/chat/completions"
                        
                        headers = {
                            "Authorization": f"Bearer {token_clean}",
                            "Content-Type": "application/json"
                        }
                        
                        messages_payload = [
                            {"role": "system", "content": system_instruction},
                            {"role": "system", "content": datos_contexto_api}
                        ]
                        
                        for msg in st.session_state.gemini_messages:
                            messages_payload.append({"role": msg["role"], "content": msg["content"]})
                        
                        payload = {
                            "model": "google/gemini-2.0-flash-exp:free",
                            "messages": messages_payload,
                            "temperature": 0.5,
                            "max_tokens": 1200
                        }
                        
                        # Petición a la velocidad del rayo
                        response = requests.post(url_or, headers=headers, json=payload, timeout=8.0)
                        
                        if response.status_code == 400 or response.status_code == 404:
                            payload["model"] = "google/gemini-2.5-flash"
                            response = requests.post(url_or, headers=headers, json=payload)
                            
                        response_data = response.json()
                        
                        if response.status_code == 200:
                            respuesta_ia = response_data['choices'][0]['message']['content'].strip()
                            st.write(respuesta_ia)
                            st.session_state.gemini_messages.append({"role": "assistant", "content": respuesta_ia})
                            st.rerun()
                        else:
                            error_msg = response_data.get('error', {}).get('message', 'Error en el servidor de IA')
                            st.error(f"❌ Error de Servidor (Código {response.status_code}): {error_msg}")
                            
                    except Exception as e:
                        st.error(f"❌ Ocurrió un error inesperado: {str(e)}")

        # Sección Exportar a WhatsApp
        if len(st.session_state.gemini_messages) > 1:
            st.markdown("---")
            st.subheader("📲 Guardar y Compartir Asesoría")
            
            import urllib.parse
            texto_whatsapp = "🌱 *REPORTE DE ASESORÍA - AGROIA-NARIÑO* ☕\n\n"
            for msg in st.session_state.gemini_messages[1:]:
                if msg["role"] == "user":
                    texto_whatsapp += f"👨‍🌾 *Caficultor:* {msg['content']}\n\n"
                else:
                    texto_whatsapp += f"👨‍⚕️ *Don Agromía:* {msg['content']}\n\n"
            
            texto_whatsapp += "✨ *AgroIA-Nariño*"
            texto_codificado = urllib.parse.quote(texto_whatsapp)
            
            col_tel, col_btn = st.columns([2, 1])
            with col_tel:
                numero_celular = st.text_input("📱 Número de WhatsApp (ej: 573123456789):", placeholder="Código de país + número")
            with col_btn:
                st.write(""); st.write("")
                url_wa = f"https://wa.me/{numero_celular.strip()}?text={texto_codificado}" if numero_celular else f"https://wa.me/?text={texto_codificado}"
                st.link_button("📤 Enviar por WhatsApp", url_wa, use_container_width=True)