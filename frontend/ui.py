"""Shared UI helpers: interviewer voice, styling, score tiles and charts."""
import html
import json

import api
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

LEVELS = ["Basic", "Beginner", "Medium", "Intermediate", "Advanced", "Professional"]
PASS = 85

# Reference palette (dataviz skill): one series -> slot 1 blue; status colors only with icon + label.
SERIES = "#2a78d6"
GRID = "#e1e0d9"
MUTED = "#898781"
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

CSS = """
<style>
.block-container {max-width: 1100px; padding-top: 2rem;}
.q-card {border: 1px solid rgba(128,128,128,.25); border-radius: 14px; padding: 1.4rem 1.6rem;
         margin: .4rem 0 1rem 0;}
.q-meta {font-size: .85rem; opacity: .75; margin-bottom: .4rem;}
.q-text {font-size: 1.45rem; line-height: 1.45; font-weight: 600;}
.pill {display:inline-block; padding:.15rem .6rem; border-radius:999px; font-size:.8rem;
       border:1px solid rgba(128,128,128,.35); margin-right:.35rem;}
.verdict {font-size: 1.15rem; font-weight: 600;}
</style>
"""


def setup_page(title: str) -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    st.title(title)


def speak(text: str, key: str) -> None:
    """Play the interviewer's voice once per `key` (edge-tts MP3, or the browser's voice as fallback)."""
    played = st.session_state.setdefault("_spoken", set())
    autoplay = key not in played
    played.add(key)
    audio = None
    try:
        audio = api.tts(text)
    except api.ApiError:
        pass
    if audio:
        st.audio(audio, format="audio/mp3", autoplay=autoplay)
    elif autoplay:
        components.html(
            "<script>const u=new SpeechSynthesisUtterance(" + json.dumps(text) + ");"
            "u.lang='en-IN';u.rate=0.95;window.speechSynthesis.cancel();window.speechSynthesis.speak(u);</script>",
            height=0)


def status_of(score: float | None) -> tuple[str, str]:
    """(icon+label, status color) - color never carries meaning alone."""
    if score is None:
        return "-", MUTED
    if score >= PASS:
        return "✅ Correct", STATUS["good"]
    if score >= 50:
        return "🟠 Partly correct", STATUS["serious"]
    return "❌ Not correct", STATUS["critical"]


def question_card(q: dict) -> None:
    meta = []
    if q.get("phase") == "followup":
        meta.append("🔁 Follow-up")
    elif q.get("kind") == "technical":
        meta.append(f"Technical {q['technical_done'] + 1}/{q['technical_total']}")
    elif q.get("kind") == "resume":
        meta.append("📄 Resume")
    else:
        meta.append("🤝 Soft skills")
    if q.get("level_name"):
        meta.append(f"Level: {q['level_name']}")
    if q.get("subject"):
        meta.append(html.escape(q["subject"]))
    pills = "".join(f"<span class='pill'>{m}</span>" for m in meta)
    st.markdown(f"<div class='q-card'><div class='q-meta'>{pills}</div>"
                f"<div class='q-text'>{html.escape(q['text'])}</div></div>", unsafe_allow_html=True)


def _layout(fig: go.Figure, height: int, **kw) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=20, t=30, b=40), font=dict(family=FONT, size=13),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
                      hoverlabel=dict(font_family=FONT), **kw)
    fig.update_xaxes(gridcolor=GRID, gridwidth=1, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, gridwidth=1, zeroline=False, linecolor=GRID)
    return fig


def question_scores_chart(answers: list[dict]) -> go.Figure:
    """Horizontal bars: % correct per question, with the 85% pass line."""
    rows = [a for a in answers if a.get("score") is not None]
    labels = [f"Q{i + 1} · {(a.get('level_name') or a['kind']).title()}" for i, a in enumerate(rows)]
    fig = go.Figure(go.Bar(
        x=[a["score"] for a in rows], y=labels, orientation="h", marker=dict(color=SERIES,
        cornerradius=4), width=0.6,
        customdata=[[a["question"][:90], status_of(a["score"])[0]] for a in rows],
        hovertemplate="<b>%{y}</b> - %{x:.0f}%<br>%{customdata[1]}<br>%{customdata[0]}<extra></extra>",
    ))
    fig.add_vline(x=PASS, line_width=1.5, line_color=MUTED,
                  annotation_text="85% = correct", annotation_position="top", annotation_font_color=MUTED)
    _layout(fig, height=max(260, 26 * len(rows) + 80))
    fig.update_xaxes(range=[0, 100], ticksuffix="%")
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def level_trail_chart(trail: list[int]) -> go.Figure:
    """How the difficulty moved across the technical questions."""
    x = list(range(1, len(trail) + 1))
    fig = go.Figure(go.Scatter(
        x=x, y=trail, mode="lines+markers", line=dict(color=SERIES, width=2, shape="hv"),
        marker=dict(size=9, color=SERIES, line=dict(color="white", width=2)),
        hovertemplate="Question %{x}<br>Level: %{text}<extra></extra>", text=[LEVELS[v - 1] for v in trail],
    ))
    _layout(fig, height=280)
    fig.update_yaxes(range=[0.5, 6.5], tickvals=list(range(1, 7)), ticktext=LEVELS)
    fig.update_xaxes(title="Technical question #", dtick=1, showgrid=False)
    return fig


def require_candidate() -> dict | None:
    cand = st.session_state.get("candidate")
    if not cand:
        st.info("Please register first.")
        st.page_link("pages/1_register.py", label="Go to registration", icon="📝")
        return None
    return cand
