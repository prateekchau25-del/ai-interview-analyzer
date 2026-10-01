import api
import pandas as pd
import streamlit as st
import ui

ui.setup_page("📊 Interview Report")
ss = st.session_state

# A report can be opened from the interview (session) or from the HR dashboard (?interview=ID).
iid = (st.query_params.get("interview") or (ss.get("report") or {}).get("interview_id")
       or ss.get("interview_id"))
if not iid:
    st.info("Finish an interview to see your report, or open one from the HR dashboard.")
    st.stop()
try:
    rep = api.report(int(iid))
except api.ApiError as e:
    st.error(str(e))
    st.stop()
if rep["status"] != "completed":
    st.warning("This interview is not finished yet.")
    st.stop()

c = rep["candidate"]
s = rep["summary"] or {}
if c.get("role") == "teacher":
    taught = "; ".join(f"{t['course']} Y{t['year']} - {t['subject']}" for t in c.get("teaching") or [])
    st.markdown(f"**{c['name']}** (Faculty) · {c['college']} · ID {c['roll_no']}")
    st.caption(f"Subjects taught: {taught}")
else:
    st.markdown(f"**{c['name']}** · {c['college']} · Roll {c['roll_no']} · {c['course']} · {c['year']}")

# ---------------------------------------------------------------- headline
overall = rep["overall_score"] or 0
label, _ = ui.status_of(overall)
h1, h2 = st.columns([1, 2])
h1.metric("Overall score", f"{overall:.0f}%")
h2.markdown(f"<p class='verdict' style='margin-top:1.2rem'>{rep['verdict']}</p>", unsafe_allow_html=True)

t1, t2, t3, t4 = st.columns(4)
t1.metric("Technical (level-weighted)", f"{rep['technical_score'] or 0:.0f}%")
t2.metric("Soft skills & resume answers", f"{rep['soft_score'] or 0:.0f}%")
comm = s.get("communication")
t3.metric("Communication (fluency)", f"{comm:.0f}%" if comm is not None else "n/a")
ats = s.get("ats_score")
t4.metric("Resume ATS", f"{ats:.0f}%" if ats is not None else "no resume",
          delta=("Pass" if s.get("ats_pass") else "Below 85%") if ats is not None else None,
          delta_color="normal" if s.get("ats_pass") else "inverse")

k1, k2, k3, k4 = st.columns(4)
k1.metric("Correct answers (≥85%)", f"{s.get('correct_count', 0)} / {s.get('total_answered', 0)}")
k2.metric("Skipped", s.get("skipped_count", 0))
k3.metric("Highest level reached", s.get("peak_level", "-"))
k4.metric("Follow-up questions", s.get("followup_count", 0))

# ---------------------------------------------------------------- per-question scores
answers = rep["answers"]
st.subheader("Score for every question")
tab_chart, tab_table = st.tabs(["Chart", "Table"])
with tab_chart:
    st.plotly_chart(ui.question_scores_chart(answers))
with tab_table:
    df = pd.DataFrame([{
        "#": i + 1, "Type": a["kind"], "Level": a.get("level_name") or "-",
        "Question": a["question"], "Score %": round(a["score"] or 0),
        "Result": "⏭️ Skipped" if a["skipped"] else (
            ui.status_of(a["score"])[0] if a["kind"] != "soft" else f"🤝 {a['verdict']}"),
    } for i, a in enumerate(answers)])
    st.dataframe(df, hide_index=True)

if s.get("subject_scores"):
    st.subheader("Knowledge by subject")
    st.plotly_chart(ui.subject_scores_chart(s["subject_scores"]))
    st.caption("Average % across the main and counter questions asked from each subject.")

if s.get("level_trail"):
    st.subheader("How the difficulty changed")
    st.plotly_chart(ui.level_trail_chart(s["level_trail"]))
    st.caption("≥85% correct → next question one level harder. Two misses in a row → one level easier.")

# ---------------------------------------------------------------- delivery (how you spoke)
dl = s.get("delivery") or {}
if any(dl.get(k) is not None for k in ("fluency", "voice_confidence", "confident_language")):
    st.subheader("How you spoke")
    d1, d2, d3, d4 = st.columns(4)
    for col, key, label in ((d1, "fluency", "Fluency"), (d2, "voice_confidence", "Voice confidence"),
                            (d3, "confident_language", "Confident language")):
        col.metric(label, f"{dl[key]:.0f}/100" if dl.get(key) is not None else "n/a")
    d4.metric("Average pace", f"{dl['avg_wpm']:.0f} wpm" if dl.get("avg_wpm") else "n/a")
    extra = [f"filler words: {dl.get('total_fillers', 0)}", f"hedging phrases: {dl.get('total_hedges', 0)}"]
    if dl.get("emotion_counts"):
        extra.append("tone across answers: " + ", ".join(
            f"{k} ×{v}" for k, v in sorted(dl["emotion_counts"].items(), key=lambda kv: -kv[1])))
    st.caption(" · ".join(extra))
    for note in dl.get("frequent_voice_notes", []):
        st.markdown(f"- 💡 {note}")
    st.caption("Communication score = average of fluency, voice confidence and confident language. "
               "Tone is estimated by a speech-emotion model (≈63% accurate on unseen speakers), so treat "
               "it as a hint.")

if dl.get("body_language") is not None:
    st.subheader("Body language (webcam)")
    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Body language", f"{dl['body_language']:.0f}/100")
    b2.metric("Eye contact", f"{dl['eye_contact']:.0f}%")
    b3.metric("Face in frame", f"{dl['face_present']:.0f}%")
    b4.metric("Smiling", f"{dl['smile']:.0f}%")
    if dl.get("face_emotion_counts"):
        st.caption("Facial expression across answers: " + ", ".join(
            f"{k} ×{v}" for k, v in sorted(dl["face_emotion_counts"].items(), key=lambda kv: -kv[1])))
    for note in dl.get("frequent_body_notes", []):
        st.markdown(f"- 💡 {note}")
    st.caption(f"Measured on {dl['answers_with_video']} answer(s) with the camera on; counts for 10% of the "
               "overall score.")

b5 = s.get("big_five")
if b5:
    with st.expander("🧭 Communication style indicator (Big Five) - experimental"):
        st.caption(b5["disclaimer"] + f" Model accuracy ≈ {b5['model_accuracy'] * 100:.0f}% per trait; "
                   f"based on {b5['words_analyzed']} words you spoke.")
        for trait, v in b5["traits"].items():
            st.markdown(f"**{trait.title()}**: leans *{v['leaning']}*")
            st.progress(v["score"] / 100)

# ---------------------------------------------------------------- answers in detail
st.subheader("Your answers")
for i, a in enumerate(answers, 1):
    icon = "⏭️" if a["skipped"] else ("🤝" if a["kind"] == "soft" else ui.status_of(a["score"])[0].split()[0])
    with st.expander(f"{icon} Q{i}. {a['question'][:100]} · {a['score'] or 0:.0f}%"):
        st.markdown(f"**Your answer:** {a['transcript'] or '_(skipped)_'}")
        if a.get("feedback"):
            st.markdown(f"**Feedback:** {a['feedback']}")
        if a.get("explanation"):
            st.markdown(f"**Correct answer explained:** {a['explanation']}")

# ---------------------------------------------------------------- resume
resume = rep.get("resume")
if resume:
    st.subheader("Resume")
    status = "✅ ATS pass" if resume["ats_pass"] else "❌ Not ATS-ready"
    st.markdown(f"{status} · {resume['ats_score']:.0f}%")
    if resume["improvements"]:
        st.markdown("**Top fixes:**\n" + "\n".join(f"- {x}" for x in resume["improvements"][:6]))
