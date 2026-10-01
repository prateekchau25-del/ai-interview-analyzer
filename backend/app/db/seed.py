"""Default data, inserted idempotently on every setup/start.

- data/soft_skills.json        -> soft-skill / HR questions (kind="soft")
- data/question_bank/*.json    -> syllabus technical questions with model answers (kind="technical")
- ROLES (below)                -> legacy job-role demo questions
"""
import json
import logging
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.db.models import JobRole, Question

log = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SOFT_SKILLS_FILE = DATA_DIR / "soft_skills.json"
QUESTION_BANK_DIR = DATA_DIR / "question_bank"

ROLES = [
    {
        "title": "Software Developer",
        "description": "Build and maintain backend services and web applications using Python/Java, "
        "REST APIs, databases, Git and cloud deployment.",
        "required_skills": ["python", "java", "sql", "git", "rest api", "data structures",
                            "algorithms", "docker", "oop", "testing"],
        "questions": [
            {
                "text": "Explain the difference between a process and a thread.",
                "ideal_answer": (
                    "A process is an independent program in execution with its own memory space, "
                    "while a thread is a lightweight unit of execution inside a process that shares "
                    "the process memory. Threads are cheaper to create and switch, and communicate "
                    "through shared memory, but need synchronization like locks to avoid race "
                    "conditions. Processes are isolated, so a crash in one does not affect others, "
                    "and they communicate through inter-process communication."
                ),
                "keywords": ["memory", "shared", "independent", "lightweight", "context switch",
                             "synchronization", "race condition", "isolation"],
            },
            {
                "text": "What is a REST API and what makes an API RESTful?",
                "ideal_answer": (
                    "REST is an architectural style where resources are identified by URLs and "
                    "manipulated with standard HTTP methods like GET, POST, PUT and DELETE. A RESTful "
                    "API is stateless, uses a client-server model, returns representations such as "
                    "JSON, uses proper status codes, and supports caching and a uniform interface."
                ),
                "keywords": ["resource", "http", "get", "post", "stateless", "json",
                             "status code", "client server"],
            },
            {
                "text": "How would you find and fix a slow database query in production?",
                "ideal_answer": (
                    "First measure using slow query logs or APM to find the query, then run EXPLAIN "
                    "ANALYZE to inspect the execution plan. Look for full table scans and add suitable "
                    "indexes, avoid SELECT *, fix N+1 queries, paginate results and consider caching. "
                    "Test the change on staging, deploy, and monitor the latency afterwards."
                ),
                "keywords": ["explain", "index", "execution plan", "n+1", "cache", "monitor",
                             "full table scan", "logs"],
            },
        ],
    },
    {
        "title": "Data Analyst",
        "description": "Clean, analyse and visualise business data using SQL, Excel, Python "
        "(pandas) and BI tools like Power BI or Tableau; communicate insights to stakeholders.",
        "required_skills": ["sql", "excel", "python", "pandas", "power bi", "tableau",
                            "statistics", "data visualization", "data cleaning"],
        "questions": [
            {
                "text": "How do you handle missing values in a dataset?",
                "ideal_answer": (
                    "First understand why data is missing and how much is missing per column. Then "
                    "choose a strategy: drop rows or columns when the missing share is small or the "
                    "column is not useful, impute with mean, median or mode, use forward fill for time "
                    "series, or model-based imputation. Optionally add an indicator flag and always "
                    "validate the impact on the analysis."
                ),
                "keywords": ["drop", "impute", "mean", "median", "mode", "forward fill",
                             "percentage", "bias"],
            },
            {
                "text": "Explain the difference between INNER JOIN, LEFT JOIN and FULL OUTER JOIN.",
                "ideal_answer": (
                    "An INNER JOIN returns only rows with matching keys in both tables. A LEFT JOIN "
                    "returns all rows from the left table and matching rows from the right table, with "
                    "NULLs where there is no match. A FULL OUTER JOIN returns all rows from both "
                    "tables, filling NULLs on whichever side has no match."
                ),
                "keywords": ["matching", "both tables", "left table", "null", "all rows", "key"],
            },
            {
                "text": "Tell me how you would present a data insight to a non-technical manager.",
                "ideal_answer": (
                    "Start with the business question and the key takeaway, use one simple, clean "
                    "visual, avoid jargon, quantify the impact in business terms like revenue or cost, "
                    "give a clear recommendation and next steps, and be ready with supporting detail "
                    "if asked."
                ),
                "keywords": ["business", "visual", "simple", "impact", "recommendation",
                             "story", "jargon"],
            },
        ],
    },
    {
        "title": "IT Executive",
        "description": "Manage IT infrastructure, networks, user support, security and vendor "
        "coordination for the organisation.",
        "required_skills": ["networking", "windows server", "active directory", "troubleshooting",
                            "security", "backup", "linux", "itil", "firewall"],
        "questions": [
            {
                "text": (
                    "Explain how you would handle a sudden network outage affecting multiple departments."
                ),
                "ideal_answer": (
                    "Identify the scope and root cause by checking monitoring, switches, routers, "
                    "firewall and server logs. Communicate the outage and expected timelines to the "
                    "affected departments, apply a workaround or failover to restore service quickly, "
                    "fix the root cause, then document the incident and preventive measures in a "
                    "post-incident review."
                ),
                "keywords": ["root cause", "logs", "switch", "communicate", "failover",
                             "document", "monitoring", "escalate"],
            },
            {
                "text": "How do you ensure company data is backed up and recoverable?",
                "ideal_answer": (
                    "Follow the 3-2-1 rule: three copies of data, on two different media, one "
                    "offsite or in the cloud. Automate scheduled full and incremental backups, encrypt "
                    "them, monitor backup jobs, and regularly test restores against defined RPO and RTO "
                    "targets."
                ),
                "keywords": ["3-2-1", "offsite", "cloud", "incremental", "encrypt",
                             "test restore", "rpo", "rto", "schedule"],
            },
        ],
    },
    {
        "title": "Machine Learning Engineer",
        "description": "Design, train, evaluate and deploy machine learning models; build data "
        "pipelines and MLOps workflows.",
        "required_skills": ["python", "machine learning", "deep learning", "pytorch", "tensorflow",
                            "scikit-learn", "sql", "mlops", "docker", "statistics", "nlp"],
        "questions": [
            {
                "text": "What is overfitting and how do you prevent it?",
                "ideal_answer": (
                    "Overfitting is when a model learns noise in the training data and performs well "
                    "on training but poorly on unseen data. Detect it with a validation set or "
                    "cross-validation. Prevent it with more data, data augmentation, regularization "
                    "such as L1, L2 or dropout, simpler models, early stopping and feature selection."
                ),
                "keywords": ["training", "validation", "generalize", "cross validation",
                             "regularization", "dropout", "early stopping", "more data"],
            },
            {
                "text": "How would you deploy and monitor a machine learning model in production?",
                "ideal_answer": (
                    "Package the model with its preprocessing into a versioned artifact, serve it "
                    "behind an API such as FastAPI in a Docker container, and deploy with CI/CD. "
                    "Monitor latency, errors, input data drift and prediction drift, log predictions, "
                    "and retrain or roll back based on alerts."
                ),
                "keywords": ["api", "docker", "version", "ci/cd", "monitor", "drift",
                             "latency", "retrain", "rollback"],
            },
        ],
    },
]


def load_soft_skills() -> list[dict]:
    return json.loads(SOFT_SKILLS_FILE.read_text(encoding="utf-8"))["questions"]


def load_question_bank() -> list[dict]:
    """Flatten data/question_bank/*.json into question dicts (one row per course/year using the subject).

    File format: {"subject", "applies_to": [{"course", "year", "semester"}, ...], "questions": [
        {"level": 1-6, "topic", "text", "ideal_answer", "keywords": [...]}, ...]}
    """
    rows = []
    for f in sorted(QUESTION_BANK_DIR.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        targets = {(a["course"], a["year"]): a.get("semester") for a in data["applies_to"]}
        for (course, year), semester in sorted(targets.items()):
            for q in data["questions"]:
                rows.append({**q, "course": course, "year": year, "semester": semester,
                             "subject": data["subject"]})
    return rows


def _seed_roles(db: Session) -> None:
    if db.scalar(select(JobRole.id).limit(1)) is not None:
        return
    for role_data in ROLES:
        role = JobRole(title=role_data["title"], description=role_data["description"],
                       required_skills=role_data["required_skills"])
        for i, q in enumerate(role_data["questions"]):
            role.questions.append(Question(order_index=i, text=q["text"], ideal_answer=q["ideal_answer"],
                                           keywords=q["keywords"], kind="technical", source="seed"))
        db.add(role)


def _insert_missing(db: Session, kind: str, rows: list[dict]) -> int:
    """Insert questions not already present (matched on kind + course + text)."""
    existing = set(db.execute(select(Question.course, Question.text).where(Question.kind == kind)).all())
    added = 0
    for r in rows:
        if (r.get("course"), r["text"]) in existing:
            continue
        db.add(Question(
            kind=kind, text=r["text"], ideal_answer=r["ideal_answer"], keywords=r.get("keywords", []),
            category="technical" if kind == "technical" else "soft",
            course=r.get("course"), year=r.get("year"), semester=r.get("semester"),
            subject=r.get("subject"), topic=r.get("topic"), level=r.get("level", 0),
            source="bank", time_limit_sec=r.get("time_limit_sec", 120 if kind == "technical" else 90),
        ))
        existing.add((r.get("course"), r["text"]))
        added += 1
    return added


def seed_defaults(db: Session) -> None:
    _seed_roles(db)
    soft = _insert_missing(db, "soft", load_soft_skills())
    tech = _insert_missing(db, "technical", load_question_bank())
    db.commit()
    if soft or tech:
        log.info("Seeded %d soft-skill and %d technical questions", soft, tech)
