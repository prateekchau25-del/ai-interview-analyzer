import api
import pandas as pd
import streamlit as st
import ui

ui.setup_page("🗂️ HR / Faculty Dashboard")

COURSES = {"": "All courses", "BSC_IT": "B.Sc IT", "BSC_DS": "B.Sc DS", "BSC_AI": "B.Sc AI", "BCA": "BCA"}
ROLES = {"": "Students & teachers", "student": "Students", "teacher": "Teachers"}
f1, f2 = st.columns(2)
role = f1.selectbox("Who", list(ROLES), format_func=ROLES.get)
course = f2.selectbox("Course", list(COURSES), format_func=COURSES.get, disabled=role == "teacher")
try:
    rows = api.candidates(None if role == "teacher" else (course or None), role or None)
except api.ApiError as e:
    st.error(str(e))
    st.stop()
if not rows:
    st.info("No candidates yet.")
    st.stop()

df = pd.DataFrame([{
    "Name": r["name"], "Role": r["role"].title(), "Roll / emp. no": r["roll_no"],
    "College": r["college_name"], "Course": r["course_name"] or "Faculty",
    "Year": r["year_name"] or ("; ".join(sorted({t["subject"] for t in r["teaching"] or []}))[:60] or "-"),
    "Interviews": r["interviews"],
    "Last score %": round(r["last_score"]) if r["last_score"] is not None else None,
    "Verdict": r["last_verdict"] or "-",
    "ATS %": round(r["ats_score"]) if r["ats_score"] is not None else None,
    "ATS": ("✅ Pass" if r["ats_score"] >= 85 else "❌ Below 85") if r["ats_score"] is not None else "-",
    "_iid": r["last_interview_id"],
} for r in rows])

scored = df["Last score %"].dropna()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Candidates", len(df))
c2.metric("Interviews done", int(df["Interviews"].sum()))
c3.metric("Average score", f"{scored.mean():.0f}%" if len(scored) else "-")
ats = df["ATS %"].dropna()
c4.metric("ATS pass rate", f"{(ats >= 85).mean() * 100:.0f}%" if len(ats) else "-")

st.dataframe(df.drop(columns="_iid"), hide_index=True)
st.download_button("Download CSV", df.drop(columns="_iid").to_csv(index=False).encode(),
                   "candidates.csv", "text/csv")

st.subheader("Open a report")
options = {f"{r['Name']} ({r['Roll / emp. no']})": r["_iid"] for _, r in df.iterrows() if pd.notna(r["_iid"])}
if options:
    pick = st.selectbox("Candidate", list(options))
    if st.button("Open report"):
        st.query_params["interview"] = str(int(options[pick]))
        st.switch_page("pages/3_report.py", query_params={"interview": str(int(options[pick]))})
