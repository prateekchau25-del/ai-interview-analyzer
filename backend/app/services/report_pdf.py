"""HR-style PDF report (ReportLab + matplotlib charts) built from interview_service.report_data()."""
import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

LEVELS = ["Basic", "Beginner", "Medium", "Intermediate", "Advanced", "Professional"]
BLUE, INK, MUTED, GRID = "#2a78d6", "#0b0b0b", "#898781", "#e1e0d9"
GOOD, SERIOUS, CRITICAL = "#0ca30c", "#ec835a", "#d03b3b"

styles = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=styles["Title"], fontSize=20, spaceAfter=4, textColor=colors.HexColor(INK))
H2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13, spaceBefore=10, spaceAfter=4,
                    textColor=colors.HexColor(BLUE))
BODY = ParagraphStyle("b", parent=styles["BodyText"], fontSize=9.5, leading=13)
SMALL = ParagraphStyle("s", parent=BODY, fontSize=8, leading=10, textColor=colors.HexColor("#52514e"))
CELL = ParagraphStyle("c", parent=BODY, fontSize=8.5, leading=10.5)
CENTER = ParagraphStyle("ce", parent=BODY, alignment=TA_CENTER)


def _esc(text) -> str:
    return (str(text or "")).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _result(a: dict) -> str:
    if a["skipped"]:
        return "Skipped"
    if a["kind"] == "soft":
        return (a.get("verdict") or "-").title()
    s = a["score"] or 0
    return "Correct" if s >= 85 else "Partly correct" if s >= 50 else "Not correct"


def _png(fig) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160, bbox_inches="tight")
    buf.seek(0)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return buf


def _axes(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors="#52514e", labelsize=8)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _subject_chart(subject_scores: dict) -> io.BytesIO:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    items = sorted(subject_scores.items(), key=lambda kv: kv[1]["avg"])
    fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(items) + 0.8))
    ax.barh([k[:40] for k, _ in items], [v["avg"] for _, v in items], color=BLUE, height=0.55)
    ax.axvline(85, color=MUTED, linewidth=1)
    ax.text(85, len(items) - 0.4, " 85% = correct", color=MUTED, fontsize=7, va="bottom")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Average % correct", fontsize=8, color="#52514e")
    _axes(ax)
    return _png(fig)


def _level_chart(trail: list[int]) -> io.BytesIO:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 2.2))
    x = list(range(1, len(trail) + 1))
    ax.step(x, trail, where="post", color=BLUE, linewidth=2)
    ax.plot(x, trail, "o", color=BLUE, markersize=5, markeredgecolor="white")
    ax.set_yticks(range(1, 7), LEVELS)
    ax.set_ylim(0.5, 6.5)
    ax.set_xticks(x)
    ax.set_xlabel("Technical question #", fontsize=8, color="#52514e")
    _axes(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    return _png(fig)


def _kv_table(rows: list[tuple[str, str]], widths=(60 * mm, 110 * mm)) -> Table:
    t = Table([[Paragraph(f"<b>{_esc(k)}</b>", CELL), Paragraph(_esc(v), CELL)] for k, v in rows],
              colWidths=widths)
    t.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor(GRID)),
                           ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 3),
                           ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]))
    return t


def _fmt(v, suffix="%") -> str:
    return "n/a" if v is None else f"{v:.0f}{suffix}"


def build_pdf(rep: dict) -> bytes:
    c, s = rep["candidate"], rep.get("summary") or {}
    dl = s.get("delivery") or {}
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=16 * mm, title=f"Interview report - {c['name']}",
                            author="AI Interview Analyzer")
    el = []

    # ---- header
    el.append(Paragraph("AI Interview Analyzer - Interview Report", H1))
    who = (f"{_esc(c['name'])} (Faculty) · {_esc(c['college'])} · ID {_esc(c['roll_no'])}"
           if c.get("role") == "teacher" else
           f"{_esc(c['name'])} · {_esc(c['college'])} · Roll {_esc(c['roll_no'])} · "
           f"{_esc(c['course'])} · {_esc(c['year'])}")
    el.append(Paragraph(who, BODY))
    if c.get("teaching"):
        el.append(Paragraph("Subjects taught: " + _esc("; ".join(
            f"{t['course']} Y{t['year']} - {t['subject']}" for t in c["teaching"])), SMALL))
    el.append(Paragraph(f"Generated {datetime.now():%d %b %Y, %H:%M} · Interview #{rep['interview_id']}",
                        SMALL))
    el.append(Spacer(1, 6))

    # ---- headline scores
    overall = rep.get("overall_score") or 0
    head = Table([[Paragraph(f"<font size=26><b>{overall:.0f}%</b></font>", CENTER),
                   Paragraph(f"<font size=13><b>{_esc(rep.get('verdict'))}</b></font><br/>"
                             f"<font size=8 color='{MUTED}'>Overall score (technical 65%, soft skills 20%, "
                             "communication 15%, body language 10% when the camera was on)</font>", BODY)]],
                 colWidths=[40 * mm, 130 * mm])
    head.setStyle(TableStyle([("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor(GRID)),
                              ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("TOPPADDING", (0, 0), (-1, -1), 8),
                              ("BOTTOMPADDING", (0, 0), (-1, -1), 8)]))
    el += [head, Spacer(1, 6)]
    ats = s.get("ats_score")
    el.append(_kv_table([
        ("Technical knowledge (level-weighted)", _fmt(rep.get("technical_score"))),
        ("Soft skills & resume answers", _fmt(rep.get("soft_score"))),
        ("Communication", _fmt(s.get("communication"))),
        ("Body language (webcam)", _fmt(dl.get("body_language"))),
        ("Resume ATS score", "no resume" if ats is None
         else f"{ats:.0f}% - {'PASS' if s.get('ats_pass') else 'below 85%'}"),
        ("Correct answers (>= 85%)", f"{s.get('correct_count', 0)} of {s.get('total_answered', 0)}"),
        ("Highest level reached", s.get("peak_level", "-")),
        ("Counter questions asked", str(s.get("followup_count", 0))),
    ]))

    # ---- per-question table
    el.append(Paragraph("Score for every question", H2))
    data = [[Paragraph(f"<b>{h}</b>", CELL) for h in ("#", "Question", "Type / level", "Score", "Result")]]
    style = [("LINEBELOW", (0, 0), (-1, -1), 0.4, colors.HexColor(GRID)), ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0efec"))]
    for i, a in enumerate(rep["answers"], 1):
        res = _result(a)
        color = GOOD if res == "Correct" else CRITICAL if res in ("Not correct", "Skipped") else SERIOUS
        data.append([Paragraph(str(i), CELL), Paragraph(_esc(a["question"]), CELL),
                     Paragraph(_esc(a["kind"] + (f" · {a['level_name']}" if a.get("level_name") else "")),
                               CELL),
                     Paragraph(f"{a['score'] or 0:.0f}%", CELL),
                     Paragraph(f"<font color='{color}'><b>{res}</b></font>", CELL)])
    t = Table(data, colWidths=[8 * mm, 92 * mm, 32 * mm, 14 * mm, 26 * mm], repeatRows=1)
    t.setStyle(TableStyle(style))
    el.append(t)

    # ---- charts
    if s.get("subject_scores"):
        el.append(KeepTogether([Paragraph("Knowledge by subject", H2),
                                Image(_subject_chart(s["subject_scores"]), width=165 * mm,
                                      height=(0.42 * len(s["subject_scores"]) + 0.8) * 25 * mm)]))
    if s.get("level_trail"):
        el.append(KeepTogether([Paragraph("How the difficulty changed", H2),
                                Image(_level_chart(s["level_trail"]), width=165 * mm, height=55 * mm),
                                Paragraph("An answer at least 85% correct moved the next question one level "
                                          "up; two misses in a row moved it one level down.", SMALL)]))

    # ---- delivery
    if dl:
        el.append(Paragraph("How you spoke and presented yourself", H2))
        rows = [("Fluency", _fmt(dl.get("fluency"), "/100")),
                ("Voice confidence", _fmt(dl.get("voice_confidence"), "/100")),
                ("Confident language", _fmt(dl.get("confident_language"), "/100")),
                ("Average pace", _fmt(dl.get("avg_wpm"), " words/min")),
                ("Filler words / hedging phrases",
                 f"{dl.get('total_fillers', 0)} / {dl.get('total_hedges', 0)}")]
        if dl.get("body_language") is not None:
            rows += [("Eye contact", _fmt(dl.get("eye_contact"))),
                     ("Face in frame", _fmt(dl.get("face_present"))),
                     ("Smiling", _fmt(dl.get("smile")))]
        el.append(_kv_table(rows))
        tips = (dl.get("frequent_voice_notes") or []) + (dl.get("frequent_body_notes") or [])
        if tips:
            el.append(Spacer(1, 4))
            el.append(Paragraph("<b>Top tips</b>", BODY))
            el += [Paragraph(f"• {_esc(t)}", BODY) for t in tips[:6]]

    # ---- resume
    resume = rep.get("resume")
    if resume:
        el.append(Paragraph("Resume (ATS check)", H2))
        el.append(Paragraph(f"ATS score <b>{resume['ats_score']:.0f}%</b> - "
                            f"{'passes' if resume['ats_pass'] else 'does not yet pass'} the 85% mark.", BODY))
        el += [Paragraph(f"• {_esc(f)}", BODY) for f in resume.get("improvements", [])[:8]]

    # ---- missed questions with answers
    missed = [a for a in rep["answers"] if a["kind"] != "soft" and (a["skipped"] or (a["score"] or 0) < 85)]
    if missed:
        el += [PageBreak(), Paragraph("Questions to revise", H2)]
        for a in missed:
            block = [Paragraph(f"<b>{_esc(a['question'])}</b>", BODY),
                     Paragraph(f"Your answer ({a['score'] or 0:.0f}%): "
                               f"{_esc(a['transcript']) or '(skipped)'}", SMALL)]
            if a.get("explanation") or a.get("model_answer"):
                answer = a.get("explanation") or a["model_answer"]
                block.append(Paragraph(f"Correct answer: {_esc(answer)}", BODY))
            elif a.get("feedback"):
                block.append(Paragraph(f"Feedback: {_esc(a['feedback'])}", BODY))
            block.append(Spacer(1, 6))
            el.append(KeepTogether(block))

    # ---- personality indicator
    b5 = s.get("big_five")
    if b5:
        el.append(Paragraph("Communication style indicator (experimental)", H2))
        el.append(Paragraph(_esc(b5["disclaimer"]), SMALL))
        el += [Paragraph(f"• {t.title()}: leans {_esc(v['leaning'])}", BODY) for t, v in b5["traits"].items()]

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(18 * mm, 10 * mm, "AI Interview Analyzer · practice report, not a hiring decision")
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"Page {doc_.page}")
        canvas.restoreState()

    doc.build(el, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()
