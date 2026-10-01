import api
import pandas as pd
import streamlit as st
import ui

ui.setup_page("📝 Register & Resume Check")

COURSES = {"BSC_IT": "B.Sc IT", "BSC_DS": "B.Sc DS", "BSC_AI": "B.Sc AI", "BCA": "BCA"}
YEARS = {1: "FY (1st year)", 2: "SY (2nd year)", 3: "TY (3rd year)"}

with st.form("register"):
    c1, c2 = st.columns(2)
    name = c1.text_input("Full name *")
    roll_no = c2.text_input("Roll number *")
    college = st.text_input("College name *", value="K.E.S. Shroff College")
    c3, c4 = st.columns(2)
    course = c3.selectbox("Course *", list(COURSES), format_func=COURSES.get)
    year = c4.radio("Year *", list(YEARS), format_func=YEARS.get, horizontal=True)
    resume = st.file_uploader("Resume (PDF or DOCX, max 5 MB)", type=["pdf", "docx"])
    submitted = st.form_submit_button("Save & analyse resume", type="primary")

if submitted:
    if len(name.strip()) < 2 or not roll_no.strip() or len(college.strip()) < 2:
        st.error("Please fill name, roll number and college.")
        st.stop()
    try:
        with st.spinner("Saving your details..."):
            cand = api.register(name, college, roll_no, course, year)
        st.session_state["candidate"] = cand
        st.session_state.pop("interview_id", None)
        if resume is not None:
            with st.spinner("Reading your resume and running the ATS check..."):
                st.session_state["ats"] = api.upload_resume(cand["id"], resume.name, resume.getvalue())
        else:
            st.session_state.pop("ats", None)
    except api.ApiError as e:
        st.error(str(e))
        st.stop()

cand = st.session_state.get("candidate")
if not cand:
    st.stop()

st.success(f"Registered: **{cand['name']}** · {cand['college_name']} · Roll {cand['roll_no']} · "
           f"{cand['course_name']} · {cand['year_name']}")

ats = st.session_state.get("ats")
if ats:
    st.subheader("Resume ATS report")
    score = ats["ats_score"]
    c1, c2 = st.columns([1, 2])
    c1.metric("ATS score", f"{score:.0f}%")
    if ats["ats_pass"]:
        c2.markdown("<p class='verdict'>✅ ATS pass (≥ 85%)</p>Your resume is likely to get through "
                    "applicant tracking systems.", unsafe_allow_html=True)
    else:
        c2.markdown(f"<p class='verdict'>❌ Not ATS-ready yet ({85 - score:.0f} points short of 85%)</p>"
                    "Fix the points below and upload again.", unsafe_allow_html=True)

    table = pd.DataFrame([{"Check": c["name"], "Points": f"{c['points']:.0f} / {c['max_points']:.0f}",
                           "Status": "✅" if c["points"] >= c["max_points"] else "⚠️"}
                          for c in ats["checks"]])
    st.dataframe(table, hide_index=True)

    if ats["issues"]:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### ❌ What is wrong")
            for i in ats["issues"]:
                st.markdown(f"- {i}")
        with c2:
            st.markdown("##### ✅ How to fix it")
            for f in ats["improvements"]:
                st.markdown(f"- {f}")

    review = ats.get("llm_review")
    if review:
        st.markdown("##### 🤖 AI recruiter review")
        st.write(review.get("summary", ""))
        for m in review.get("mistakes", []):
            st.markdown(f"- ❌ {m}")
        for imp in review.get("improvements", []):
            st.markdown(f"- 💡 {imp}")
        for rb in review.get("rewritten_bullets", []):
            st.markdown(f"> **Before:** {rb.get('before', '')}\n>\n> **After:** {rb.get('after', '')}")
    d = ats.get("detected", {})
    if d.get("skills_found"):
        st.caption("Skills found: " + ", ".join(d["skills_found"]))
elif not cand.get("has_resume"):
    st.info("No resume uploaded. The interview will skip resume-based questions.")

st.divider()
st.page_link("pages/2_interview.py", label="Start the interview", icon="🎙️")
