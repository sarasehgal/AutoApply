"""renders a TailoredResume as an actual pdf instead of a markdown-ish .txt dump"""

from __future__ import annotations

from fpdf import FPDF, XPos, YPos

from autoapply.agents.schemas import TailoredResume

_MARGIN = 18
_FONT = "Helvetica"


def _block(pdf: FPDF, text: str) -> None:
    """multi_cell that always resets to the left margin on the next line - fpdf2 otherwise
    leaves the cursor at the right edge after a single-line cell, which starves the next call
    of horizontal space"""
    pdf.multi_cell(0, 6, text, new_x=XPos.LMARGIN, new_y=YPos.NEXT)


def render_tailored_resume_pdf(tailored: TailoredResume, *, title: str = "Tailored Resume") -> bytes:
    pdf = FPDF(format="Letter")
    pdf.set_margins(_MARGIN, _MARGIN, _MARGIN)
    pdf.set_auto_page_break(auto=True, margin=_MARGIN)
    pdf.add_page()

    pdf.set_font(_FONT, "B", 16)
    _block(pdf, title)
    pdf.ln(2)

    pdf.set_font(_FONT, "", 11)
    _block(pdf, tailored.summary)
    pdf.ln(4)

    pdf.set_font(_FONT, "B", 12)
    _block(pdf, "Experience")
    pdf.set_font(_FONT, "", 11)
    for bullet in tailored.bullets:
        _block(pdf, f"-  {bullet.tailored}")
        pdf.ln(1)

    return bytes(pdf.output())
