"""Streamlit entry point.  Run:  streamlit run frontend/app.py"""
import streamlit as st

st.set_page_config(page_title="AI Interview Analyzer", page_icon="🎙️", layout="wide")

pages = [
    st.Page("pages/0_home.py", title="Home", icon="🏠", default=True),
    st.Page("pages/1_register.py", title="Register & Resume", icon="📝"),
    st.Page("pages/2_interview.py", title="Interview", icon="🎙️"),
    st.Page("pages/3_report.py", title="Report", icon="📊"),
    st.Page("pages/4_dashboard.py", title="HR Dashboard", icon="🗂️"),
]
st.navigation(pages).run()
