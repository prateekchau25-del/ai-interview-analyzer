import api
import streamlit as st
import ui

ui.setup_page("🎙️ Interview")
cand = ui.require_candidate()
if not cand:
    st.stop()

ss = st.session_state


def submit(fn, *args):
    try:
        with st.spinner("Listening to your answer and checking it..."):
            result = fn(ss["interview_id"], *args)
    except api.ApiError as e:
        st.error(str(e))
        return
    ss["last_result"] = result
    ss["question"] = result["next"]
    st.rerun()


# ---------------------------------------------------------------- start
if "interview_id" not in ss:
    resume_part = "questions on your resume, " if cand.get("has_resume") else ""
    if cand.get("role") == "teacher":
        subjects = ", ".join(sorted({t["subject"] for t in cand.get("teaching") or []}))
        st.markdown(f"Welcome **{cand['name']}**! This faculty interview has an introduction, "
                    f"**10 technical questions** from the subjects you teach ({subjects}) at Intermediate "
                    "level and above, **1-2 counter questions on every answer**, " + resume_part
                    + "and questions on your teaching practice.")
    else:
        st.markdown(f"Hi **{cand['name']}**! This interview has an introduction, **15 technical questions** "
                    f"from the {cand['course_name']} syllabus (up to {cand['year_name']}), follow-up "
                    "questions on your answers, " + resume_part + "and soft-skill questions.")
    st.markdown("- Allow microphone access when your browser asks.\n"
                "- Listen to the question, click the mic, answer, then **Submit**.\n"
                "- Don't know an answer? Press **I don't know**: we will explain it at the end.")
    if st.button("Start interview", type="primary"):
        try:
            started = api.start_interview(cand["id"])
        except api.ApiError as e:
            st.error(str(e))
            st.stop()
        ss["interview_id"], ss["question"] = started["interview_id"], started["question"]
        ss.pop("last_result", None)
        ss.pop("review", None)
        st.rerun()
    st.stop()

iid, q = ss["interview_id"], ss["question"]

# ---------------------------------------------------------------- feedback on the previous answer
last = ss.get("last_result")
if last:
    with st.container(border=True):
        label, _ = ui.status_of(None if last["skipped"] else last["score"])
        if last["skipped"]:
            st.markdown(f"<p class='verdict'>⏭️ {last['message']}</p>", unsafe_allow_html=True)
        else:
            st.markdown(f"<p class='verdict'>{label.split()[0]} {last['message']} "
                        f"· {last['score']:.0f}%</p>", unsafe_allow_html=True)
            if last.get("feedback"):
                st.write(last["feedback"])
        if last.get("level_change") == "up":
            st.caption(f"⬆️ Level up: next questions are **{last['level_name']}**")
        elif last.get("level_change") == "down":
            st.caption(f"⬇️ Level down: next questions are **{last['level_name']}**")
        if last.get("speech"):
            sp = last["speech"]
            st.caption(f"Pace {sp['words_per_minute']:.0f} words/min ({sp['pace_label']}) · "
                       f"filler words {sp['filler_count']} · long pauses {sp['long_pause_count']}")
        if last.get("transcript"):
            with st.expander("What we heard"):
                st.write(last["transcript"])

# ---------------------------------------------------------------- question
if q.get("phase") == "error":
    st.error(q["message"])
    st.stop()

if q.get("phase") not in ("review", "finished"):
    done = q["number"] - 1
    st.progress(min(1.0, done / q["planned_total"]),
                text=f"Question {q['number']} of about {q['planned_total']}")
    ui.question_card(q)
    ui.speak(q["text"], key=f"q{iid}-{q['question_id']}")
    st.caption(f"Suggested time: about {q['time_limit_sec'] // 60} min {q['time_limit_sec'] % 60:02d} s")

    rec = st.audio_input("Record your answer", key=f"rec-{q['question_id']}")
    c1, c2 = st.columns([1, 1])
    if c1.button("Submit answer", type="primary", disabled=rec is None):
        submit(api.answer_audio, "answer.wav", rec.getvalue())
    if c2.button("I don't know / Skip"):
        submit(api.skip)
    with st.expander("Can't use the microphone? Type your answer"):
        typed = st.text_area("Your answer", key=f"typed-{q['question_id']}")
        if st.button("Submit typed answer", disabled=not typed.strip()):
            submit(api.answer_text, typed)
    st.stop()

# ---------------------------------------------------------------- end-of-interview review
st.subheader("Interview complete 🎉")
if "review" not in ss:
    ss["review"] = api.review(iid)
rv = ss["review"]

if rv["count"] and "review_decision" not in ss:
    st.markdown(f"You could not answer **{rv['count']} question(s)** correctly.")
    st.markdown("<p class='verdict'>Do you want me to explain the answers?</p>", unsafe_allow_html=True)
    ui.speak("You could not answer some questions correctly. Do you want me to explain the answers?",
             key=f"review-{iid}")
    c1, c2 = st.columns(2)
    if c1.button("✅ Yes, explain", type="primary"):
        with st.spinner("Preparing explanations..."):
            ss["review_decision"] = api.review_decision(iid, True)
        st.rerun()
    if c2.button("❌ No, thanks"):
        ss["review_decision"] = api.review_decision(iid, False)
        st.rerun()
    st.stop()

decision = ss.get("review_decision")
if decision:
    st.info(decision["message"])
    for n, item in enumerate(decision["items"], 1):
        with st.container(border=True):
            st.markdown(f"**Q{n}. {item['question']}**")
            st.caption(f"Your answer ({item['score']:.0f}%): " + (item["your_answer"] or "(skipped)"))
            st.write(item["explanation"])
            if st.button("🔊 Listen", key=f"listen-{item['question_id']}"):
                ui.speak(item["explanation"], key=f"exp-{iid}-{item['question_id']}")
    if not decision["items"]:
        ui.speak(decision["message"], key=f"bye-{iid}")
elif not rv["count"]:
    st.success("Excellent! You answered every question correctly.")

if st.button("See my report", type="primary"):
    with st.spinner("Calculating your scores..."):
        ss["report"] = api.finish(iid)
    st.switch_page("pages/3_report.py")
