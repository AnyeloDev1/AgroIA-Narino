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

# ==============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="AgroIA-Nariño", 
    page_icon="☕", 
    layout="centered"
)

st.title("☕ AgroIA - Nariño")
st.markdown("---")

# BARRA LATERAL: Configuración de la IA de Google
st.sidebar.header("🔑 Configuración de IA")
gemini_api_key = st.sidebar.text_input("Ingresa tu Gemini API Key:", type="password")
st.sidebar.markdown("""
*Consigue una clave gratis en [Google AI Studio](https://aistudio.google.com/)*
""")

# CREACIÓN DE PESTAÑAS PARA ORGANIZAR LA APP
tab1, tab2 = st.tabs(["🔮 Predicciones e Informes", "💬 Agrónomo Virtual"])

# ==============================================================================
# PESTAÑA 1: PREDICCIONES CON DATOS REALES (EVA - datos.gov.co) + WHATSAPP
# ==============================================================================
with tab1:
    st.header("📋 Datos de la Finca")

    # Cacheamos la consulta a datos.gov.co por 1 hora para no golpear la API
    # en cada interacción del usuario con el formulario.
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

                consola.update(label="✅ Diagnóstico completo generado con datos reales del EVA", state="complete")

        if resultado["fuente"] != "sin_datos_reales":
            st.metric(
                "Cosecha proyectada",
                f"{resultado['produccion_estimada_ton']:.2f} ton",
                help=f"= rendimiento histórico promedio ({resultado['rendimiento_ton_ha']} ton/ha) × {hectareas} ha"
            )
            st.write(f"**Nivel de riesgo (basado en tendencia real de rendimiento):** {resultado['nivel_riesgo']}")
            st.caption(resultado["detalle_riesgo"])
            st.caption(
                f"Fuente: EVA - datos.gov.co · {resultado['anios_con_datos']} año(s) de histórico · "
                f"último dato: {resultado['ultimo_anio_reportado']}"
            )

            mensaje_chatbot = (
                f"☕ AgroIA - Nariño Informa ☕\n\n"
                f"📢 Reporte generado con datos reales (EVA - datos.gov.co)\n"
                f"📍 Municipio: {municipio}\n"
                f"🔮 Cosecha Proyectada: {resultado['produccion_estimada_ton']:.2f} Toneladas "
                f"({resultado['produccion_estimada_kg']:.0f} kg)\n"
                f"⚠️ Riesgo (tendencia de rendimiento): {resultado['nivel_riesgo']}\n"
                f"💡 {resultado['detalle_riesgo']}"
            )

            st.markdown("##### 📤 Compartir este reporte")
            col_wsp, col_pdf = st.columns(2)

            with col_wsp:
                if not whatsapp_num or whatsapp_num.strip() == "57":
                    st.info("👈 Ingresa un número de WhatsApp válido arriba para poder enviarlo.")
                else:
                    # Enlace directo a WhatsApp -- el usuario confirma el envío con un clic.
                    # (Antes esto se intentaba automatizar con pyautogui simulando un
                    # Enter tras 8 segundos, lo cual solo funciona en un computador local
                    # con pantalla activa y se rompe en cualquier servidor o si el usuario
                    # cambia de ventana durante la espera. Un botón de enlace es 100%
                    # confiable en cualquier entorno.)
                    texto_codificado = urllib.parse.quote(mensaje_chatbot)
                    link_whatsapp = f"https://wa.me/{whatsapp_num.strip()}?text={texto_codificado}"
                    st.link_button("💬 Enviar mensaje de texto", link_whatsapp, use_container_width=True)

            with col_pdf:
                pdf_bytes = generar_reporte_pdf(municipio, hectareas, resultado, df_historico=df_eva)
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

# ==============================================================================
# PESTAÑA 2: CHATBOT OPTIMIZADO - RESPUESTA INMEDIATA Y RAG ULTRA-LIGERO
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
            
            # Realizamos la consulta en segundo plano sin bloquear la experiencia visual
            try:
                import requests
                
                # URL base limpia
                api_url = "https://www.datos.gov.co/resource/ggaa-6f3s.json" 
                
                # 🛠️ OPTIMIZACIÓN MAESTRA: Pedimos solo 3 columnas clave y máximo 3 filas para que cargue instantáneamente
                params = {
                    "$select": "municipio,departamento,cultivo,rendimiento,produccion" if "ggaa-6f3s" in api_url else "*",
                    "$limit": 3
                }
                
                # Filtrado ultra-rápido por texto si el usuario menciona Nariño o Samaniego
                palabras = user_query.lower()
                if "samaniego" in palabras:
                    params["$where"] = "upper(municipio) like '%SAMANIEGO%'"
                elif "nariño" in palabras or "narino" in palabras:
                    params["$where"] = "upper(departamento) like '%NARIÑO%'"
                
                # 🛠️ CONTROL DE TIEMPO: Si el servidor de datos.gov.co no responde en 1.5 segundos, se corta
                response_api = requests.get(api_url, params=params, timeout=1.5)
                
                if response_api.status_code == 200:
                    datos_json = response_api.json()
                    if datos_json and len(datos_json) > 0:
                        import json
                        datos_contexto_api = f"DATOS EN VIVO DESDE DATOS.GOV.CO:\n{json.dumps(datos_json, indent=2, ensure_ascii=False)}"
            except Exception:
                # Si expira el tiempo o falla, el contexto queda limpio para que Gemini responda usando su propio cerebro
                datos_contexto_api = "Servidor de datos abiertos ocupado. Responde con tu conocimiento experto general."

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
            
            texto_whatsapp += "✨ *AgroIA-Nariño* - Datos en Tiempo Real e Inteligencia Artificial."
            texto_codificado = urllib.parse.quote(texto_whatsapp)
            
            col_tel, col_btn = st.columns([2, 1])
            with col_tel:
                numero_celular = st.text_input("📱 Número de WhatsApp (ej: 573123456789):", placeholder="Código de país + número")
            with col_btn:
                st.write(""); st.write("")
                url_wa = f"https://wa.me/{numero_celular.strip()}?text={texto_codificado}" if numero_celular else f"https://wa.me/?text={texto_codificado}"
                st.link_button("📤 Enviar por WhatsApp", url_wa, use_container_width=True)