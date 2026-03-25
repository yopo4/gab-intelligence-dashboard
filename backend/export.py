"""
Export module — CSV, Excel, JSON, PDF
GAB Intelligence · Banque Populaire du Maroc
"""
import io, json, os, textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse
from typing import List

# ── Router ────────────────────────────────────────────────────
router = APIRouter(prefix="/api/export", tags=["export"])

# Lazy imports — installed on demand
_openpyxl = None
_rl = None


def _ensure_openpyxl():
    global _openpyxl
    if _openpyxl is None:
        import openpyxl
        _openpyxl = openpyxl
    return _openpyxl


def _ensure_reportlab():
    global _rl
    if _rl is None:
        import reportlab as rl
        _rl = rl
    return _rl


# ── Helpers ───────────────────────────────────────────────────

def _get_main_module():
    """Import main module data lazily to avoid circular imports."""
    from main import DF, FEAT_IMP, RESULTATS, MEILLEUR_MODELE, COUTS, filter_df
    return DF, FEAT_IMP, RESULTATS, MEILLEUR_MODELE, COUTS, filter_df


def _parse_filters(villes, types, annees):
    return (villes or None, types or None, annees or None)


def _stream(buf, filename, media_type):
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def _now_str():
    return datetime.now().strftime("%Y%m%d_%H%M")


# ══════════════════════════════════════════════════════════════
#  CSV Export
# ══════════════════════════════════════════════════════════════

@router.get("/csv")
def export_csv(
    villes: List[str] = Query(default=[]),
    types: List[str] = Query(default=[]),
    annees: List[str] = Query(default=[]),
    section: str = Query(default="overview"),
):
    DF, FEAT_IMP, RESULTATS, MEILLEUR_MODELE, COUTS, filter_df = _get_main_module()
    v, t, a = _parse_filters(villes, types, annees)
    df = filter_df(DF, v, t, a)

    buf = io.StringIO()

    if section == "features":
        FEAT_IMP.to_csv(buf, index=False, encoding="utf-8-sig")
    elif section == "models":
        rows = []
        for name, metrics in RESULTATS.items():
            rows.append({"modele": name, **{k: v for k, v in metrics.items()
                                             if isinstance(v, (int, float))}})
        pd.DataFrame(rows).to_csv(buf, index=False, encoding="utf-8-sig")
    elif section == "geography":
        geo = (
            df.groupby("ville")
            .agg(nb_pannes=("panne_sous_48h", "sum"),
                 nb_obs=("panne_sous_48h", "count"),
                 nb_gab=("gab_id", "nunique"))
            .reset_index()
        )
        geo["taux_panne"] = (geo["nb_pannes"] / geo["nb_obs"] * 100).round(2)
        geo.to_csv(buf, index=False, encoding="utf-8-sig")
    else:  # overview — raw filtered data
        export_cols = [c for c in df.columns if not c.startswith("_")]
        df[export_cols].to_csv(buf, index=False, encoding="utf-8-sig")

    return _stream(
        io.BytesIO(buf.getvalue().encode("utf-8-sig")),
        f"gab_{section}_{_now_str()}.csv",
        "text/csv; charset=utf-8",
    )


# ══════════════════════════════════════════════════════════════
#  JSON Export
# ══════════════════════════════════════════════════════════════

@router.get("/json")
def export_json(
    villes: List[str] = Query(default=[]),
    types: List[str] = Query(default=[]),
    annees: List[str] = Query(default=[]),
    section: str = Query(default="overview"),
):
    DF, FEAT_IMP, RESULTATS, MEILLEUR_MODELE, COUTS, filter_df = _get_main_module()
    v, t, a = _parse_filters(villes, types, annees)
    df = filter_df(DF, v, t, a)

    if section == "features":
        payload = FEAT_IMP.head(50).to_dict(orient="records")
    elif section == "models":
        payload = RESULTATS
    elif section == "geography":
        geo = (
            df.groupby("ville")
            .agg(nb_pannes=("panne_sous_48h", "sum"),
                 nb_obs=("panne_sous_48h", "count"),
                 nb_gab=("gab_id", "nunique"))
            .reset_index()
        )
        geo["taux_panne"] = (geo["nb_pannes"] / geo["nb_obs"] * 100).round(2)
        payload = geo.to_dict(orient="records")
    else:
        payload = {
            "nb_gab": int(df["gab_id"].nunique()),
            "nb_pannes": int(df["panne_sous_48h"].sum()),
            "taux_panne": round(float(df["panne_sous_48h"].mean() * 100), 2),
            "meilleur_modele": MEILLEUR_MODELE,
            "nb_observations": int(len(df)),
            "filtres": {"villes": v, "types": t, "annees": a},
            "donnees": json.loads(df.head(500).to_json(orient="records", date_format="iso")),
        }

    content = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    buf = io.BytesIO(content.encode("utf-8"))
    return _stream(buf, f"gab_{section}_{_now_str()}.json", "application/json")


# ══════════════════════════════════════════════════════════════
#  Excel Export (multi-sheet)
# ══════════════════════════════════════════════════════════════

@router.get("/excel")
def export_excel(
    villes: List[str] = Query(default=[]),
    types: List[str] = Query(default=[]),
    annees: List[str] = Query(default=[]),
):
    _ensure_openpyxl()
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    DF, FEAT_IMP, RESULTATS, MEILLEUR_MODELE, COUTS, filter_df = _get_main_module()
    v, t, a = _parse_filters(villes, types, annees)
    df = filter_df(DF, v, t, a)

    wb = _openpyxl.Workbook()

    # ── Style constants ──
    BPM_RED = "C8102E"
    BPM_GREEN = "00703C"
    header_font = Font(bold=True, color="FFFFFF", size=11)
    header_fill = PatternFill(start_color=BPM_RED, end_color=BPM_RED, fill_type="solid")
    title_font = Font(bold=True, size=14, color=BPM_RED)
    sub_font = Font(italic=True, size=10, color="666666")
    thin_border = Border(
        bottom=Side(style="thin", color="DDDDDD"),
    )

    def style_header(ws, ncols):
        for col in range(1, ncols + 1):
            cell = ws.cell(row=1, column=col)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal="center")
        ws.freeze_panes = "A2"
        for col in range(1, ncols + 1):
            ws.column_dimensions[get_column_letter(col)].width = 18

    def add_data_rows(ws, dataframe, start_row=2):
        for r_idx, row in enumerate(dataframe.itertuples(index=False), start=start_row):
            for c_idx, val in enumerate(row, start=1):
                cell = ws.cell(row=r_idx, column=c_idx, value=val)
                cell.border = thin_border
                if isinstance(val, float):
                    cell.number_format = '0.00'

    # ── Sheet 1: Résumé KPIs ──
    ws1 = wb.active
    ws1.title = "Résumé"
    ws1["A1"] = "GAB Intelligence — Rapport d'analyse"
    ws1["A1"].font = title_font
    ws1["A2"] = f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}"
    ws1["A2"].font = sub_font
    ws1["A3"] = f"Banque Populaire du Maroc — Maintenance Prédictive"
    ws1["A3"].font = sub_font

    kpis = [
        ("GAB surveillés", int(df["gab_id"].nunique())),
        ("Pannes enregistrées", int(df["panne_sous_48h"].sum())),
        ("Taux de panne moyen", f'{df["panne_sous_48h"].mean() * 100:.2f}%'),
        ("Disponibilité", f'{(1 - df["panne_sous_48h"].mean()) * 100:.1f}%'),
        ("Observations", int(len(df))),
        ("Villes", int(df["ville"].nunique())),
        ("Meilleur modèle", MEILLEUR_MODELE),
        ("F1-Score", round(float(RESULTATS[MEILLEUR_MODELE]["f1"]), 4)),
        ("Recall", round(float(RESULTATS[MEILLEUR_MODELE]["recall"]), 4)),
        ("Precision", round(float(RESULTATS[MEILLEUR_MODELE]["precision"]), 4)),
    ]
    for i, (label, value) in enumerate(kpis, start=5):
        ws1.cell(row=i, column=1, value=label).font = Font(bold=True, size=11)
        ws1.cell(row=i, column=2, value=value).font = Font(size=11)
    ws1.column_dimensions["A"].width = 28
    ws1.column_dimensions["B"].width = 30

    # ── Sheet 2: Données filtrées ──
    ws2 = wb.create_sheet("Données")
    export_cols = [c for c in df.columns if not c.startswith("_")]
    df_export = df[export_cols].copy()
    # Convert date to string for Excel
    if "date" in df_export.columns:
        df_export["date"] = df_export["date"].dt.strftime("%Y-%m-%d")
    for c_idx, col in enumerate(df_export.columns, start=1):
        ws2.cell(row=1, column=c_idx, value=col)
    style_header(ws2, len(df_export.columns))
    add_data_rows(ws2, df_export)

    # ── Sheet 3: Géographie ──
    ws3 = wb.create_sheet("Géographie")
    geo = (
        df.groupby("ville")
        .agg(nb_pannes=("panne_sous_48h", "sum"),
             nb_obs=("panne_sous_48h", "count"),
             nb_gab=("gab_id", "nunique"))
        .reset_index()
    )
    geo["taux_panne"] = (geo["nb_pannes"] / geo["nb_obs"] * 100).round(2)
    geo = geo.sort_values("taux_panne", ascending=False)
    for c_idx, col in enumerate(geo.columns, start=1):
        ws3.cell(row=1, column=c_idx, value=col)
    style_header(ws3, len(geo.columns))
    add_data_rows(ws3, geo)

    # ── Sheet 4: Modèles ML ──
    ws4 = wb.create_sheet("Modèles ML")
    model_rows = []
    for name, metrics in RESULTATS.items():
        row = {"modele": name}
        row.update({k: round(float(v), 4) if isinstance(v, (int, float)) else v
                    for k, v in metrics.items()})
        model_rows.append(row)
    df_models = pd.DataFrame(model_rows)
    for c_idx, col in enumerate(df_models.columns, start=1):
        ws4.cell(row=1, column=c_idx, value=col)
    style_header(ws4, len(df_models.columns))
    add_data_rows(ws4, df_models)

    # ── Sheet 5: Features ──
    ws5 = wb.create_sheet("Features")
    fi_export = FEAT_IMP.head(50).copy()
    for c_idx, col in enumerate(fi_export.columns, start=1):
        ws5.cell(row=1, column=c_idx, value=col)
    style_header(ws5, len(fi_export.columns))
    add_data_rows(ws5, fi_export)

    # ── Write to buffer ──
    buf = io.BytesIO()
    wb.save(buf)
    return _stream(
        buf,
        f"gab_rapport_{_now_str()}.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ══════════════════════════════════════════════════════════════
#  PDF Report (complet — type Power BI export)
# ══════════════════════════════════════════════════════════════

@router.get("/pdf")
def export_pdf(
    villes: List[str] = Query(default=[]),
    types: List[str] = Query(default=[]),
    annees: List[str] = Query(default=[]),
):
    _ensure_reportlab()
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm, cm
    from reportlab.lib.colors import HexColor, white, black
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        PageBreak, HRFlowable,
    )
    from reportlab.graphics.shapes import Drawing, Rect, String, Line
    from reportlab.graphics.charts.barcharts import VerticalBarChart, HorizontalBarChart
    from reportlab.graphics.charts.piecharts import Pie
    from reportlab.graphics.charts.legends import Legend

    DF, FEAT_IMP, RESULTATS, MEILLEUR_MODELE, COUTS, filter_df = _get_main_module()
    v, t, a = _parse_filters(villes, types, annees)
    df = filter_df(DF, v, t, a)

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=20 * mm, bottomMargin=20 * mm,
    )

    # Colors
    BPM_RED = HexColor("#C8102E")
    BPM_GREEN = HexColor("#00703C")
    DARK_BG = HexColor("#0a0a0c")
    GRAY = HexColor("#666666")
    LIGHT_GRAY = HexColor("#F5F5F5")

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        "CoverTitle", parent=styles["Title"],
        fontSize=28, textColor=BPM_RED, spaceAfter=6,
        alignment=TA_CENTER, fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        "CoverSub", parent=styles["Normal"],
        fontSize=14, textColor=GRAY, alignment=TA_CENTER,
        spaceAfter=4,
    ))
    styles.add(ParagraphStyle(
        "SectionTitle", parent=styles["Heading1"],
        fontSize=16, textColor=BPM_RED, spaceBefore=16,
        spaceAfter=8, fontName="Helvetica-Bold",
        borderColor=BPM_RED, borderWidth=0, borderPadding=0,
    ))
    styles.add(ParagraphStyle(
        "KpiLabel", parent=styles["Normal"],
        fontSize=9, textColor=GRAY, fontName="Helvetica",
    ))
    styles.add(ParagraphStyle(
        "KpiValue", parent=styles["Normal"],
        fontSize=18, textColor=HexColor("#1a1a2e"), fontName="Helvetica-Bold",
    ))
    styles.add(ParagraphStyle(
        "BodyText2", parent=styles["Normal"],
        fontSize=10, textColor=HexColor("#333333"), spaceAfter=6,
        leading=14,
    ))
    styles.add(ParagraphStyle(
        "FooterStyle", parent=styles["Normal"],
        fontSize=8, textColor=GRAY, alignment=TA_CENTER,
    ))

    elements = []
    W = A4[0] - 36 * mm  # usable width

    # ── Cover page ──
    elements.append(Spacer(1, 60 * mm))
    elements.append(Paragraph("GAB Intelligence", styles["CoverTitle"]))
    elements.append(Paragraph("Rapport d'Analyse — Maintenance Prédictive", styles["CoverSub"]))
    elements.append(Spacer(1, 8 * mm))
    elements.append(HRFlowable(width="60%", thickness=2, color=BPM_RED, spaceAfter=8))
    elements.append(Paragraph("Banque Populaire du Maroc", styles["CoverSub"]))
    elements.append(Paragraph(
        f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}",
        styles["CoverSub"],
    ))
    elements.append(Spacer(1, 12 * mm))

    # Filters summary
    filter_text = []
    if v:
        filter_text.append(f"Villes : {', '.join(v)}")
    if t:
        filter_text.append(f"Types : {', '.join(t)}")
    if a:
        filter_text.append(f"Années : {', '.join(str(x) for x in a)}")
    if filter_text:
        elements.append(Paragraph(
            "<br/>".join(filter_text),
            ParagraphStyle("FilterInfo", parent=styles["Normal"],
                           fontSize=10, textColor=GRAY, alignment=TA_CENTER),
        ))

    elements.append(PageBreak())

    # ── Page 2: KPIs ──
    elements.append(Paragraph("1. Indicateurs Clés (KPIs)", styles["SectionTitle"]))
    elements.append(HRFlowable(width="100%", thickness=1, color=BPM_RED, spaceAfter=12))

    nb_gab = int(df["gab_id"].nunique())
    nb_pannes = int(df["panne_sous_48h"].sum())
    taux = df["panne_sous_48h"].mean() * 100
    disponibilite = 100 - taux
    best_r = RESULTATS[MEILLEUR_MODELE]

    kpi_data = [
        ["Indicateur", "Valeur", "Détail"],
        ["GAB surveillés", str(nb_gab), f"{int(df['ville'].nunique())} villes"],
        ["Pannes enregistrées", f"{nb_pannes:,}", f"Taux : {taux:.2f}%"],
        ["Disponibilité", f"{disponibilite:.1f}%", "Objectif : > 95%"],
        ["Observations", f"{len(df):,}", "2 ans de données"],
        ["Meilleur modèle", MEILLEUR_MODELE[:30], f"F1 = {best_r['f1']:.4f}"],
        ["Recall", f"{best_r['recall']:.4f}", "Détection des pannes"],
        ["Precision", f"{best_r['precision']:.4f}", "Fiabilité des alertes"],
    ]

    kpi_table = Table(kpi_data, colWidths=[W * 0.35, W * 0.3, W * 0.35])
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BPM_RED),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 11),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 1), (-1, -1), 10),
        ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, LIGHT_GRAY]),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#DDDDDD")),
        ("TOPPADDING", (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    elements.append(kpi_table)
    elements.append(Spacer(1, 10 * mm))

    # ── Analyse textuelle ──
    elements.append(Paragraph("Synthèse", ParagraphStyle(
        "SubTitle", parent=styles["Heading2"], fontSize=13,
        textColor=HexColor("#1a1a2e"), fontName="Helvetica-Bold",
    )))
    elements.append(Paragraph(
        f"Le parc de <b>{nb_gab}</b> GAB répartis sur <b>{int(df['ville'].nunique())}</b> villes "
        f"présente un taux de panne moyen de <b>{taux:.2f}%</b> sur la période analysée. "
        f"Le modèle <b>{MEILLEUR_MODELE}</b> atteint un F1-score de <b>{best_r['f1']:.4f}</b> "
        f"avec un recall de <b>{best_r['recall']:.4f}</b>, permettant de détecter "
        f"la majorité des pannes avant leur survenue.",
        styles["BodyText2"],
    ))

    elements.append(PageBreak())

    # ── Page 3: Géographie ──
    elements.append(Paragraph("2. Analyse Géographique", styles["SectionTitle"]))
    elements.append(HRFlowable(width="100%", thickness=1, color=BPM_RED, spaceAfter=12))

    geo = (
        df.groupby("ville")
        .agg(nb_pannes=("panne_sous_48h", "sum"),
             nb_obs=("panne_sous_48h", "count"),
             nb_gab=("gab_id", "nunique"))
        .reset_index()
    )
    geo["taux_panne"] = (geo["nb_pannes"] / geo["nb_obs"] * 100).round(2)
    geo = geo.sort_values("taux_panne", ascending=False)

    geo_header = ["Ville", "GAB", "Pannes", "Observations", "Taux (%)"]
    geo_rows = [geo_header]
    for _, row in geo.iterrows():
        geo_rows.append([
            row["ville"],
            str(int(row["nb_gab"])),
            str(int(row["nb_pannes"])),
            str(int(row["nb_obs"])),
            f'{row["taux_panne"]:.2f}',
        ])

    geo_table = Table(geo_rows, colWidths=[W * 0.25, W * 0.15, W * 0.2, W * 0.2, W * 0.2])
    geo_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BPM_RED),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, LIGHT_GRAY]),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#DDDDDD")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(geo_table)
    elements.append(Spacer(1, 8 * mm))

    # Bar chart — taux par ville
    drawing = Drawing(W, 180)
    chart = HorizontalBarChart()
    chart.x = 80
    chart.y = 10
    chart.width = W - 100
    chart.height = 155
    chart.data = [list(reversed(geo["taux_panne"].tolist()))]
    chart.categoryAxis.categoryNames = list(reversed(geo["ville"].tolist()))
    chart.categoryAxis.labels.fontName = "Helvetica"
    chart.categoryAxis.labels.fontSize = 8
    chart.valueAxis.valueMin = 0
    chart.valueAxis.labels.fontSize = 8
    chart.bars[0].fillColor = BPM_RED
    chart.bars[0].strokeColor = None
    chart.barWidth = 8
    drawing.add(chart)
    elements.append(drawing)

    elements.append(PageBreak())

    # ── Page 4: Modèles ML ──
    elements.append(Paragraph("3. Performance des Modèles ML", styles["SectionTitle"]))
    elements.append(HRFlowable(width="100%", thickness=1, color=BPM_RED, spaceAfter=12))

    model_header = ["Modèle", "F1", "Recall", "Precision", "AUC-PR"]
    model_rows_data = [model_header]
    for name, m in RESULTATS.items():
        is_best = " *" if name == MEILLEUR_MODELE else ""
        model_rows_data.append([
            (name[:28] + is_best),
            f'{m.get("f1", 0):.4f}',
            f'{m.get("recall", 0):.4f}',
            f'{m.get("precision", 0):.4f}',
            f'{m.get("auc_pr", m.get("auc-pr", 0)):.4f}',
        ])

    model_table = Table(model_rows_data, colWidths=[W * 0.32, W * 0.17, W * 0.17, W * 0.17, W * 0.17])
    model_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BPM_RED),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, LIGHT_GRAY]),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#DDDDDD")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(model_table)
    elements.append(Spacer(1, 6 * mm))

    elements.append(Paragraph(
        f"<b>* Meilleur modèle sélectionné :</b> {MEILLEUR_MODELE}<br/>"
        f"Le modèle a été entraîné avec split temporel strict pour éviter le data leakage. "
        f"Un F1-score de {best_r['f1']:.4f} est cohérent avec un taux de classe minoritaire "
        f"de ~{taux:.1f}%.",
        styles["BodyText2"],
    ))

    elements.append(PageBreak())

    # ── Page 5: Top Features ──
    elements.append(Paragraph("4. Importance des Variables", styles["SectionTitle"]))
    elements.append(HRFlowable(width="100%", thickness=1, color=BPM_RED, spaceAfter=12))

    fi_top = FEAT_IMP.head(20)
    feat_header = ["Rang", "Feature", "Importance"]
    feat_rows = [feat_header]
    for i, (_, row) in enumerate(fi_top.iterrows(), 1):
        feat_rows.append([
            str(i),
            str(row["feature"])[:40],
            f'{row["imp_moy"]:.4f}',
        ])

    feat_table = Table(feat_rows, colWidths=[W * 0.1, W * 0.6, W * 0.3])
    feat_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BPM_RED),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("ALIGN", (2, 0), (2, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, LIGHT_GRAY]),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#DDDDDD")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(feat_table)
    elements.append(Spacer(1, 6 * mm))

    elements.append(Paragraph(
        "Les variables liées à la <b>maintenance</b> (jours depuis maintenance, score de négligence) "
        "et aux <b>erreurs matérielles</b> (erreurs lecteur, risque matériel) dominent l'importance. "
        "Les features <b>rolling</b> et de <b>tendance</b> apportent une valeur prédictive significative "
        "en capturant la dynamique de dégradation.",
        styles["BodyText2"],
    ))

    elements.append(PageBreak())

    # ── Page 6: Analyse temporelle ──
    elements.append(Paragraph("5. Analyse Temporelle", styles["SectionTitle"]))
    elements.append(HRFlowable(width="100%", thickness=1, color=BPM_RED, spaceAfter=12))

    # Monthly breakdown
    monthly = (
        df.assign(mois=df["date"].dt.to_period("M").astype(str))
        .groupby("mois")
        .agg(pannes=("panne_sous_48h", "sum"), obs=("panne_sous_48h", "count"))
        .reset_index()
    )
    monthly["taux"] = (monthly["pannes"] / monthly["obs"] * 100).round(2)

    month_header = ["Mois", "Pannes", "Observations", "Taux (%)"]
    month_rows = [month_header]
    for _, row in monthly.iterrows():
        month_rows.append([
            row["mois"],
            str(int(row["pannes"])),
            str(int(row["obs"])),
            f'{row["taux"]:.2f}',
        ])

    month_table = Table(month_rows, colWidths=[W * 0.3, W * 0.2, W * 0.25, W * 0.25])
    month_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BPM_RED),
        ("TEXTCOLOR", (0, 0), (-1, 0), white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [white, LIGHT_GRAY]),
        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#DDDDDD")),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    elements.append(month_table)
    elements.append(Spacer(1, 6 * mm))

    # Seasonal analysis
    df_s = df.copy()
    df_s["saison"] = df_s["date"].dt.month.map({
        12: "Hiver", 1: "Hiver", 2: "Hiver",
        3: "Printemps", 4: "Printemps", 5: "Printemps",
        6: "Été", 7: "Été", 8: "Été",
        9: "Automne", 10: "Automne", 11: "Automne",
    })
    sais = df_s.groupby("saison")["panne_sous_48h"].mean() * 100
    sais = sais.reindex(["Printemps", "Été", "Automne", "Hiver"]).fillna(0).round(2)

    elements.append(Paragraph(
        f"<b>Saisonnalité :</b> Printemps {sais.get('Printemps', 0):.1f}% · "
        f"Été {sais.get('Été', 0):.1f}% · "
        f"Automne {sais.get('Automne', 0):.1f}% · "
        f"Hiver {sais.get('Hiver', 0):.1f}%<br/>"
        f"Le pic estival confirme le rôle du stress thermique comme facteur aggravant.",
        styles["BodyText2"],
    ))

    elements.append(PageBreak())

    # ── Page 7: Recommandations ──
    elements.append(Paragraph("6. Recommandations", styles["SectionTitle"]))
    elements.append(HRFlowable(width="100%", thickness=1, color=BPM_RED, spaceAfter=12))

    recommendations = [
        ("Maintenance préventive ciblée",
         "Prioriser les interventions sur les GAB identifiés comme à haut risque par le modèle. "
         "Les GAB Wincor en sites isolés avec > 90 jours sans maintenance nécessitent une attention immédiate."),
        ("Gestion thermique",
         "Installer des systèmes de refroidissement renforcés pour les GAB en façade et sites isolés, "
         "particulièrement avant la période estivale."),
        ("Renouvellement du parc",
         "Les GAB de plus de 7 ans présentent un taux de panne significativement plus élevé. "
         "Planifier le remplacement progressif en commençant par les zones les plus critiques."),
        ("Monitoring continu",
         "Déployer le système de scoring en production pour un suivi en temps réel. "
         "Le seuil optimal identifié permet de réduire les coûts de maintenance tout en "
         "maintenant un recall élevé."),
        ("Enrichissement des données",
         "Intégrer les données de fréquentation et les conditions météorologiques locales "
         "pour améliorer la précision du modèle."),
    ]

    for i, (title, desc) in enumerate(recommendations, 1):
        elements.append(Paragraph(
            f"<b>{i}. {title}</b>",
            ParagraphStyle("RecTitle", parent=styles["Normal"],
                           fontSize=11, textColor=HexColor("#1a1a2e"),
                           fontName="Helvetica-Bold", spaceBefore=8),
        ))
        elements.append(Paragraph(desc, styles["BodyText2"]))

    elements.append(Spacer(1, 15 * mm))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=GRAY, spaceAfter=8))
    elements.append(Paragraph(
        "Rapport généré automatiquement par GAB Intelligence — "
        "Banque Populaire du Maroc — Projet de Fin d'Études",
        styles["FooterStyle"],
    ))

    # ── Build PDF ──
    doc.build(elements)

    return _stream(buf, f"gab_rapport_{_now_str()}.pdf", "application/pdf")
