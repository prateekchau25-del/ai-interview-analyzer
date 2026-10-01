import api
import streamlit as st
import ui

ui.setup_page("🎙️ AI Interview Analyzer")
st.markdown(
    "Practice a real technical interview for **B.Sc IT, B.Sc Data Science, B.Sc AI and BCA**. "
    "The AI interviewer **speaks each question**, listens to your answer, checks it against the correct "
    "answer and adapts the difficulty as you go."
)

c1, c2, c3 = st.columns(3)
c1.markdown("#### 1. Register\nName, college, roll no, course, year and your resume. "
            "You get an **ATS score** with exact mistakes to fix.")
c2.markdown("#### 2. Interview\n**15+ technical** questions from your syllabus, "
            "**follow-up** questions on what you say, plus resume and soft-skill questions.")
c3.markdown("#### 3. Report\n**% score for every answer**, your level graph, "
            "explanations for questions you missed and an overall verdict.")

st.divider()
st.markdown("##### How the difficulty adapts")
st.markdown(" → ".join(f"**{lv}**" for lv in ui.LEVELS))
st.caption("An answer that is at least 85% correct moves you one level up. "
           "Two answers in a row that are not correct move you one level down.")

try:
    h = api.health()
    st.success(f"Backend connected · LLM: {'on (' + h['llm_provider'] + ')' if h['llm_enabled'] else 'off'} "
               f"· speech model: {h['whisper_model']}")
except api.ApiError as e:
    st.error(str(e))

st.page_link("pages/1_register.py", label="Start: register now", icon="📝")
