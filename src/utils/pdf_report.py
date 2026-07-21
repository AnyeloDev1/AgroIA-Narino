"""
pdf_report.py
=============
Genera un reporte PDF de diagnóstico de finca cafetera, listo para
descargar y adjuntar manualmente en WhatsApp (WhatsApp no permite adjuntar
archivos automáticamente desde un enlace wa.me -- solo texto -- así que
este PDF se descarga primero y el usuario lo adjunta él mismo, igual que
adjuntaría cualquier foto o documento en un chat).

Uso:
    from src.utils.pdf_report import generar_reporte_pdf
    pdf_bytes = generar_reporte_pdf(municipio, hectareas, resultado, df_historico)
"""

from datetime import datetime
from io import BytesIO

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# Paleta de color coherente con la identidad del proyecto (cereza / hoja / tierra)
COLOR_CEREZA = colors.HexColor("#B23A2E")
COLOR_HOJA = colors.HexColor("#3F6B4B")
COLOR_ORO = colors.HexColor("#B8860B")
COLOR_TEXTO = colors.HexColor("#231912")
COLOR_SUAVE = colors.HexColor("#6B5B44")
COLOR_FONDO_TABLA = colors.HexColor("#F7F2E9")

COLOR_RIESGO = {
    "Alto": COLOR_CEREZA,
    "Medio": COLOR_ORO,
    "Bajo": COLOR_HOJA,
    "Desconocido": COLOR_SUAVE,
}


def _construir_estilos():
    base = getSampleStyleSheet()
    estilos = {
        "titulo": ParagraphStyle(
            "TituloReporte", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=20, textColor=COLOR_CEREZA, spaceAfter=2,
        ),
        "subtitulo": ParagraphStyle(
            "Subtitulo", parent=base["Normal"], fontName="Helvetica",
            fontSize=10, textColor=COLOR_SUAVE, spaceAfter=14,
        ),
        "seccion": ParagraphStyle(
            "Seccion", parent=base["Heading2"], fontName="Helvetica-Bold",
            fontSize=13, textColor=COLOR_TEXTO, spaceBefore=16, spaceAfter=6,
        ),
        "cuerpo": ParagraphStyle(
            "Cuerpo", parent=base["Normal"], fontName="Helvetica",
            fontSize=10.5, textColor=COLOR_TEXTO, leading=15,
        ),
        "nota": ParagraphStyle(
            "Nota", parent=base["Normal"], fontName="Helvetica-Oblique",
            fontSize=8.5, textColor=COLOR_SUAVE, leading=12,
        ),
        "metrica_valor": ParagraphStyle(
            "MetricaValor", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=22, textColor=COLOR_TEXTO,
        ),
        "metrica_label": ParagraphStyle(
            "MetricaLabel", parent=base["Normal"], fontName="Helvetica",
            fontSize=9, textColor=COLOR_SUAVE,
        ),
    }
    return estilos


def generar_reporte_pdf(municipio: str, hectareas: float, resultado: dict,
                         df_historico: pd.DataFrame = None, resultado_ml: dict = None,
                         recomendaciones: list = None) -> bytes:
    """Construye el PDF en memoria y devuelve los bytes listos para
    st.download_button (no escribe nada en disco)."""

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=letter,
        topMargin=2.2 * cm, bottomMargin=2 * cm,
        leftMargin=2 * cm, rightMargin=2 * cm,
    )
    estilos = _construir_estilos()
    story = []

    # ---------- Encabezado ----------
    story.append(Paragraph("☕ AgroIA-Nariño", estilos["titulo"]))
    story.append(Paragraph(
        f"Reporte de diagnóstico de finca cafetera · generado el "
        f"{datetime.now().strftime('%d/%m/%Y a las %H:%M')}",
        estilos["subtitulo"],
    ))
    story.append(HRFlowable(width="100%", thickness=1.2, color=COLOR_CEREZA, spaceAfter=14))

    # ---------- Datos de la finca ----------
    story.append(Paragraph("Datos de la finca", estilos["seccion"]))
    tabla_finca = Table(
        [
            ["Municipio", municipio.title()],
            ["Área sembrada", f"{hectareas} ha"],
            ["Fuente de los datos", "EVA · Evaluaciones Agropecuarias Municipales (datos.gov.co)"],
            ["Años de histórico disponibles", str(resultado.get("anios_con_datos", "N/D"))],
            ["Último año reportado", str(resultado.get("ultimo_anio_reportado", "N/D"))],
        ],
        colWidths=[6 * cm, 9.5 * cm],
    )
    tabla_finca.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("TEXTCOLOR", (0, 0), (-1, -1), COLOR_TEXTO),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, -2), 0.5, colors.HexColor("#DDD1BC")),
        ("BACKGROUND", (0, 0), (-1, -1), COLOR_FONDO_TABLA),
    ]))
    story.append(tabla_finca)
    story.append(Spacer(1, 16))

    if resultado.get("fuente") == "sin_datos_reales":
        story.append(Paragraph("Resultado", estilos["seccion"]))
        story.append(Paragraph(
            f"⚠️ {resultado.get('detalle_riesgo', 'No hay datos reales disponibles para este municipio.')}",
            estilos["cuerpo"],
        ))
    else:
        # ---------- Métricas principales ----------
        story.append(Paragraph("Predicción de cosecha", estilos["seccion"]))
        tabla_metricas = Table(
            [[
                Paragraph(f"{resultado['produccion_estimada_ton']:.2f} ton", estilos["metrica_valor"]),
                Paragraph(f"{resultado['produccion_estimada_kg']:.0f} kg", estilos["metrica_valor"]),
                Paragraph(f"{resultado['rendimiento_ton_ha']:.2f} ton/ha", estilos["metrica_valor"]),
            ], [
                Paragraph("Cosecha estimada (total)", estilos["metrica_label"]),
                Paragraph("Equivalente en kilogramos", estilos["metrica_label"]),
                Paragraph("Rendimiento histórico promedio", estilos["metrica_label"]),
            ]],
            colWidths=[5.17 * cm, 5.17 * cm, 5.16 * cm],
        )
        tabla_metricas.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), COLOR_FONDO_TABLA),
            ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#DDD1BC")),
            ("INNERGRID", (0, 0), (-1, -1), 0.75, colors.HexColor("#DDD1BC")),
            ("TOPPADDING", (0, 0), (-1, 0), 12),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 2),
            ("BOTTOMPADDING", (0, 1), (-1, 1), 10),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ]))
        story.append(tabla_metricas)
        story.append(Spacer(1, 16))

        # ---------- Predicción con Machine Learning ----------
        if resultado_ml and resultado_ml.get("disponible"):
            story.append(Paragraph("Predicción con Machine Learning (Random Forest + Gradient Boosting)", estilos["seccion"]))
            tabla_ml = Table(
                [
                    ["Random Forest", f"{resultado_ml['rendimiento_rf_ton_ha']} ton/ha"],
                    ["Gradient Boosting", f"{resultado_ml['rendimiento_gb_ton_ha']} ton/ha"],
                    ["Modelo híbrido (RF + GB)", f"{resultado_ml['rendimiento_hibrido_ton_ha']} ton/ha"],
                ],
                colWidths=[8 * cm, 7.5 * cm],
            )
            tabla_ml.setStyle(TableStyle([
                ("FONTNAME", (0, 0), (0, -1), "Helvetica"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9.5),
                ("BACKGROUND", (0, 0), (-1, -1), COLOR_FONDO_TABLA),
                ("LINEBELOW", (0, 0), (-1, -2), 0.5, colors.HexColor("#DDD1BC")),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]))
            story.append(tabla_ml)

            if resultado_ml.get("perfil_municipio"):
                perfil = resultado_ml["perfil_municipio"]
                story.append(Spacer(1, 6))
                story.append(Paragraph(
                    f"<b>Perfil del municipio (clustering K-Means):</b> {perfil['perfil']} — "
                    f"rendimiento histórico promedio {perfil['rendimiento_promedio_historico']} ton/ha, "
                    f"variabilidad {perfil['variabilidad']} ton/ha.",
                    estilos["cuerpo"],
                ))
            story.append(Spacer(1, 16))

        # ---------- Riesgo ----------
        color_riesgo = COLOR_RIESGO.get(resultado["nivel_riesgo"], COLOR_SUAVE)
        story.append(Paragraph("Nivel de riesgo (tendencia de rendimiento)", estilos["seccion"]))
        estilo_badge = ParagraphStyle(
            "Badge", parent=estilos["cuerpo"], fontName="Helvetica-Bold",
            fontSize=12, textColor=colors.white, backColor=color_riesgo,
            borderPadding=(4, 10, 4, 10), alignment=0,
        )
        story.append(Paragraph(f"&nbsp;{resultado['nivel_riesgo'].upper()}&nbsp;", estilo_badge))
        story.append(Spacer(1, 8))
        story.append(Paragraph(resultado["detalle_riesgo"], estilos["cuerpo"]))
        story.append(Spacer(1, 16))

        # ---------- Histórico ----------
        if df_historico is not None and not df_historico.empty:
            datos_municipio = df_historico[df_historico["municipio"] == municipio.strip().upper()]
            if not datos_municipio.empty:
                story.append(Paragraph("Histórico real reportado (EVA)", estilos["seccion"]))
                filas = [["Año", "Área sembrada (ha)", "Área cosechada (ha)", "Producción (ton)", "Rendimiento (ton/ha)"]]
                for _, fila in datos_municipio.sort_values("ano").iterrows():
                    filas.append([
                        str(int(fila["ano"])),
                        f"{fila['area_sembrada_ha']:.1f}",
                        f"{fila['area_cosechada_ha']:.1f}",
                        f"{fila['produccion_ton']:.1f}",
                        f"{fila['rendimiento_ton_ha']:.2f}",
                    ])
                tabla_hist = Table(filas, colWidths=[2.3 * cm, 3.3 * cm, 3.3 * cm, 3.1 * cm, 3.5 * cm])
                tabla_hist.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), COLOR_HOJA),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, COLOR_FONDO_TABLA]),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#DDD1BC")),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]))
                story.append(tabla_hist)
                story.append(Spacer(1, 16))

    # ---------- Recomendaciones accionables ----------
    if recomendaciones:
        story.append(Paragraph("Recomendaciones para esta finca", estilos["seccion"]))
        color_prioridad = {"Alta": COLOR_CEREZA, "Media": COLOR_ORO, "Baja": COLOR_HOJA}
        for r in recomendaciones:
            color = color_prioridad.get(r["prioridad"], COLOR_SUAVE)
            estilo_item = ParagraphStyle(
                "ItemRecom", parent=estilos["cuerpo"], leftIndent=10,
                spaceAfter=6, borderColor=color, borderWidth=0, bulletIndent=0,
            )
            story.append(Paragraph(
                f"<font color='{color.hexval()}'><b>[{r['prioridad']} · {r['categoria']}]</b></font> {r['accion']}",
                estilo_item,
            ))
        story.append(Spacer(1, 10))

    # ---------- Pie de página / disclaimer ----------
    story.append(HRFlowable(width="100%", thickness=0.75, color=colors.HexColor("#DDD1BC"), spaceAfter=8))
    story.append(Paragraph(
        "Este reporte se generó automáticamente a partir de datos históricos abiertos del EVA "
        "(Evaluaciones Agropecuarias Municipales, UPRA/MinAgricultura, publicados en datos.gov.co). "
        "Es una estimación basada en tendencias históricas reales del municipio, no una medición de campo. "
        "AgroIA-Nariño · Concurso Datos al Ecosistema 2026 · IA para Colombia.",
        estilos["nota"],
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()
