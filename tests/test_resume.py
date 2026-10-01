from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate

from backend.app.analyzers.resume import (
    ATS_PASS,
    detect_contact,
    detect_skills,
    extract_text,
    normalize_text,
    resume_questions,
    score_resume,
)

STRONG = """Priya Nair
priya.nair@gmail.com | +91 98765 43210 | linkedin.com/in/priyanair | github.com/priyanair
Summary
Data Science undergraduate skilled in Python, SQL and machine learning with an analytics internship.
Education
B.Sc Data Science, K.E.S. Shroff College, Mumbai - CGPA 8.9
Technical Skills
Python, SQL, Pandas, NumPy, Scikit-learn, Matplotlib, Power BI, Tableau, Excel, Statistics,
Machine Learning, Deep Learning, NLP, Git, Data Cleaning, Data Visualization, Spark
Experience
Data Analyst Intern - FinEdge Analytics
• Built 6 Power BI dashboards tracking sales for 40+ stores, used weekly by the regional team.
• Automated a daily Excel report with Python and Pandas, saving 5 hours per week.
• Cleaned and merged 2 lakh transaction records, reducing missing values by 90%.
Projects
Customer Churn Prediction | Python, Scikit-learn, Pandas
• Developed a random forest model with 91% accuracy on 7,000 customer records.
• Engineered 12 features and explained results with SHAP plots for the business team.
Movie Review Sentiment Analyzer | Python, NLP, Flask
• Trained a TF-IDF and logistic regression classifier reaching 88% F1 score on reviews.
• Deployed a Flask web app used by 150+ students in a college demo day.
COVID Data Dashboard | SQL, Tableau
• Designed SQL queries and a Tableau dashboard visualizing 3 years of state-wise data.
Certifications
Google Data Analytics Certificate; NPTEL Python for Data Science - Elite
Achievements
• Won 2nd place among 60 teams at the college Datathon 2025.
• Led a 4-member team to publish a mini research paper on student performance analytics.
"""

WEAK = """Rahul Sharma
rahul.sharma@gmail.com
Objective
I am a hardworking and team player student. I want a job in a good company where I can grow.
Education
B.Sc Data Science, 2024-2027
Skills
MS Office, Python, Excel
Project
Student Management
• Responsible for making the project
• Worked on the frontend part
"""


def test_strong_resume_passes_and_weak_fails():
    strong = score_resume(STRONG, pages=1, course="BSC_DS")
    weak = score_resume(WEAK, pages=1, course="BSC_DS")
    assert strong.ats_score >= ATS_PASS and strong.ats_pass
    assert weak.ats_score < 50 and not weak.ats_pass
    assert sum(c.max_points for c in strong.checks) == 100
    # every lost point comes with an issue and a fix
    for c in weak.checks:
        if c.points < c.max_points:
            assert c.issues and c.fixes


def test_weak_resume_issues_are_specific():
    issues = " ".join(score_resume(WEAK, pages=1, course="BSC_DS").issues).lower()
    for expected in ("phone", "github", "quantified", "responsible for", "fewer than 2 projects"):
        assert expected in issues


def test_contact_and_skills_detection():
    c = detect_contact(STRONG)
    assert all(c.values())
    found, missing = detect_skills("Worked with C++ and Java, not c#", "BCA")
    assert "c++" in found and "java" in found and "c" not in found


def test_bullet_glyphs_are_normalised():
    text = normalize_text("Projects\n\x7f Built an app\n Deployed it\n● Tested it")
    assert text.count("•") == 3


def test_resume_questions_use_project_titles():
    qs = resume_questions(STRONG, "BSC_DS", n=3)
    assert len(qs) == 3
    assert "Customer Churn Prediction" in qs[0]["text"]
    assert all(q["ideal_answer"] and q["keywords"] for q in qs)


def test_pdf_extraction_roundtrip(tmp_path):
    pdf = tmp_path / "r.pdf"
    styles = getSampleStyleSheet()
    SimpleDocTemplate(str(pdf), pagesize=A4).build(
        [Paragraph(line or "&nbsp;", styles["Normal"]) for line in STRONG.splitlines()])
    text, pages = extract_text(pdf)
    assert pages == 1
    assert score_resume(text, pages, "BSC_DS").ats_pass
