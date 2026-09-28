"""
Generates reports/final/MAIN_REPORT.pdf from the research monograph.

Uses reportlab to generate a publication-grade PDF report with:
- Formal cover page and Zetheta Algorithms corporate headers
- Page count target >= 40 pages (comprehensive monograph)
- Zetheta CIN footer on every page (CIN: U72900MH2021PTC367891)
- Styled tables, mathematical equations, and diagnostic figures
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak,
    KeepTogether,
)
from reportlab.lib import colors
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#7f8c8d"))

        # Header (Pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 750, "ZETHETA ALGORITHMS | Bayesian Regime Detection Engine")
            self.drawRightString(612 - 54, 750, "CONFIDENTIAL & PROPRIETARY")
            self.setStrokeColor(colors.HexColor("#bdc3c7"))
            self.setLineWidth(0.5)
            self.line(54, 742, 612 - 54, 742)

        # Footer (All pages)
        self.setStrokeColor(colors.HexColor("#bdc3c7"))
        self.setLineWidth(0.5)
        self.line(54, 50, 612 - 54, 50)
        self.drawString(54, 38, "Zetheta Algorithms Private Limited | CIN: U72900MH2021PTC367891")
        self.drawRightString(612 - 54, 38, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()


def build_pdf() -> None:
    out_dir = Path("reports/final")
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / "MAIN_REPORT.pdf"

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=72,
        bottomMargin=72,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "CoverTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=24,
        leading=30,
        textColor=colors.HexColor("#1a252f"),
        alignment=1,
    )
    subtitle_style = ParagraphStyle(
        "CoverSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=13,
        leading=18,
        textColor=colors.HexColor("#34495e"),
        alignment=1,
    )
    h1_style = ParagraphStyle(
        "Heading1_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=20,
        textColor=colors.HexColor("#2c3e50"),
        spaceBefore=14,
        spaceAfter=8,
    )
    h2_style = ParagraphStyle(
        "Heading2_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#2980b9"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body_Custom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#2c3e50"),
        spaceAfter=8,
    )
    callout_style = ParagraphStyle(
        "Callout_Custom",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#16a085"),
        spaceBefore=6,
        spaceAfter=8,
    )

    story = []

    # Cover Page
    story.append(Spacer(1, 100))
    story.append(Paragraph("ZETHETA ALGORITHMS PRIVATE LIMITED", ParagraphStyle("Corp", fontName="Helvetica-Bold", fontSize=12, leading=16, textColor=colors.HexColor("#7f8c8d"), alignment=1)))
    story.append(Spacer(1, 20))
    story.append(Paragraph("Bayesian Regime Detection Engine for Equity Direction Forecasting", title_style))
    story.append(Spacer(1, 12))
    story.append(Paragraph("A Unified Quantitative Platform for Macro Regime Modeling, Adaptive Conformal Calibration, and Tactical Asset Allocation", subtitle_style))
    story.append(Spacer(1, 40))

    meta_text = """
    <b>Author:</b> Quantitative Architecture & Advanced Research Group<br/>
    <b>Date:</b> September 2026<br/>
    <b>Classification:</b> Strictly Confidential — Institutional Research Monograph<br/>
    <b>Target Environment:</b> Python 3.10 (Linux / macOS ARM64)<br/>
    <b>CIN:</b> U72900MH2021PTC367891<br/>
    <b>Cryptographic Snapshot:</b> Market Data SHA-256 <code>c6c46ed7...</code> | Features SHA-256 <code>61969b7b...</code><br/>
    <b>Release Tag:</b> v1.0.0-institutional
    """
    story.append(Paragraph(meta_text, ParagraphStyle("Meta", fontName="Helvetica", fontSize=9, leading=14, textColor=colors.HexColor("#34495e"), alignment=1)))
    story.append(PageBreak())

    # Read and parse MAIN_REPORT.md sections into reportlab elements
    md_path = Path("reports/final/MAIN_REPORT.md")
    if not md_path.exists():
        raise FileNotFoundError("MAIN_REPORT.md not found.")

    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()

    lines = md_text.splitlines()
    in_table = False
    table_lines = []

    for line in lines:
        line_s = line.strip()

        # Handle table
        if line_s.startswith("|") and line_s.endswith("|"):
            in_table = True
            table_lines.append(line_s)
            continue
        else:
            if in_table:
                # Render table
                in_table = False
                if len(table_lines) >= 3:
                    table_data = []
                    for t_row in table_lines:
                        if "---" in t_row:
                            continue
                        cells = [c.strip() for c in t_row.strip("|").split("|")]
                        cell_paras = [Paragraph(f"<b>{c}</b>" if t_row == table_lines[0] else c, body_style) for c in cells]
                        table_data.append(cell_paras)

                    t = Table(table_data, colWidths=[120] + [(384 // (len(table_data[0]) - 1))] * (len(table_data[0]) - 1))
                    t.setStyle(TableStyle([
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#ecf0f1")),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
                    ]))
                    story.append(t)
                    story.append(Spacer(1, 8))
                table_lines = []

        if not line_s:
            story.append(Spacer(1, 4))
            continue

        if line_s.startswith("# "):
            story.append(Paragraph(line_s[2:], h1_style))
        elif line_s.startswith("## "):
            story.append(Spacer(1, 6))
            story.append(Paragraph(line_s[3:], h1_style))
        elif line_s.startswith("### "):
            story.append(Spacer(1, 4))
            story.append(Paragraph(line_s[4:], h2_style))
        elif line_s.startswith("> "):
            story.append(Paragraph(line_s[2:], callout_style))
        elif line_s.startswith("---"):
            story.append(Spacer(1, 6))
        else:
            story.append(Paragraph(line_s, body_style))

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF monograph: {pdf_path}")


if __name__ == "__main__":
    build_pdf()
