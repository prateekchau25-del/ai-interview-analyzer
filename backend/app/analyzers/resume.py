"""Resume parsing, transparent ATS scoring, LLM review and resume-based interview questions.

ATS score (0-100) is a sum of explainable checks, so every lost point maps to a concrete issue and fix.
Pass mark: 85. The optional LLM review adds qualitative feedback but never changes the score.
"""
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from backend.app.analyzers import llm

log = logging.getLogger(__name__)

ATS_PASS = 85.0

# Skills an ATS would look for, per course (derived from the course syllabus + common job postings).
COURSE_SKILLS: dict[str, list[str]] = {
    "BSC_IT": ["python", "java", "sql", "html", "css", "javascript", "react", "node", "git", "linux",
               "data structures", "oop", "dbms", "networking", "cloud", "rest api", "android", "testing"],
    "BSC_DS": ["python", "sql", "pandas", "numpy", "statistics", "machine learning", "data visualization",
               "power bi", "tableau", "excel", "scikit-learn", "matplotlib", "deep learning", "nlp",
               "big data", "spark", "git", "data cleaning"],
    "BSC_AI": ["python", "machine learning", "deep learning", "tensorflow", "pytorch", "nlp",
               "computer vision", "opencv", "scikit-learn", "sql", "numpy", "pandas", "prompt engineering",
               "llm", "git", "cloud", "statistics", "neural network"],
    "BCA": ["c", "c++", "java", "python", "sql", "html", "css", "javascript", "php", "dbms", "oop",
            "data structures", "operating system", "networking", "git", "web development", "react", "cloud"],
}

SECTION_PATTERNS = {
    "summary": r"\b(summary|objective|profile|about me)\b",
    "education": r"\b(education|academic|qualification)s?\b",
    "skills": r"\b(skills|technical skills|technologies|tech stack)\b",
    "projects": r"\bprojects?\b",
    "experience": r"\b(experience|internships?|work history|employment)\b",
    "certifications": r"\b(certifications?|certificates?|courses|achievements|awards)\b",
}
SECTION_POINTS = {"summary": 3, "education": 4, "skills": 4, "projects": 4, "experience": 3,
                  "certifications": 2}

ACTION_VERBS = {
    "built", "developed", "designed", "implemented", "created", "led", "managed", "improved", "optimized",
    "automated", "analyzed", "analysed", "deployed", "integrated", "reduced", "increased", "achieved",
    "trained", "engineered", "launched", "collaborated", "organized", "organised", "won", "published",
    "migrated", "tested", "debugged", "researched", "visualized", "presented", "coordinated",
}
WEAK_PHRASES = ["responsible for", "worked on", "helped with", "hardworking", "team player",
                "go-getter", "detail oriented", "references available"]


@dataclass
class Check:
    name: str
    points: float
    max_points: float
    issues: list[str] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)


@dataclass
class ResumeReport:
    ats_score: float
    ats_pass: bool
    checks: list[Check]
    detected: dict
    issues: list[str]
    improvements: list[str]
    llm_review: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- text extraction
# Bullet glyphs as they come out of PDF/DOCX text extraction (Word "Symbol"/"Wingdings" private-use chars,
# DEL from some PDF fonts, arrows, check marks...). All are normalised to "•".
BULLET_GLYPHS = (
    "\x7f\u2022\u25cf\u25aa\u25e6\u2023\u2043\u27a2\u2713\u2714\u25ba\u25b8"
    "\uf0b7\uf0a7\uf076\uf0d8\uf0fc\u00b7"
)


def normalize_text(text: str) -> str:
    text = text.translate({ord(ch): "\u2022" for ch in BULLET_GLYPHS})
    text = text.replace("\u00a0", " ").replace("\r", "")
    return "\n".join(ln.rstrip() for ln in text.splitlines())


def extract_text(path: str | Path) -> tuple[str, int]:
    text, pages = _extract_raw(path)
    return normalize_text(text), pages


def _extract_raw(path: str | Path) -> tuple[str, int]:
    """Return (text, page_count). Supports PDF and DOCX."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join((p.extract_text() or "") for p in reader.pages), len(reader.pages)
    if suffix == ".docx":
        import docx

        d = docx.Document(str(path))
        parts = [p.text for p in d.paragraphs]
        for table in d.tables:
            parts += [cell.text for row in table.rows for cell in row.cells]
        text = "\n".join(parts)
        return text, max(1, round(len(text.split()) / 450))
    if suffix == ".txt":
        text = path.read_text(encoding="utf-8", errors="ignore")
        return text, 1
    raise ValueError(f"Unsupported resume format: {suffix} (use PDF or DOCX)")


# ---------------------------------------------------------------- detection helpers
def _has(pattern: str, text: str) -> bool:
    return re.search(pattern, text, re.I | re.M) is not None


def detect_contact(text: str) -> dict:
    return {
        "email": bool(re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)),
        "phone": bool(re.search(r"(\+91[\s-]?)?\b[6-9]\d{4}[\s-]?\d{5}\b", text)),
        "linkedin": _has(r"linkedin\.com/|linkedin", text),
        "github": _has(r"github\.com/|github", text),
    }


def detect_sections(text: str) -> dict[str, bool]:
    # Headings are usually on their own short line; fall back to anywhere in the text.
    lines = [ln.strip() for ln in text.splitlines() if 0 < len(ln.strip()) <= 40]
    head = "\n".join(lines)
    return {name: _has(rf"^\W*{pat}\W*$", head) or _has(pat, text) for name, pat in SECTION_PATTERNS.items()}


def detect_skills(text: str, course: str) -> tuple[list[str], list[str]]:
    low = text.lower()
    found, missing = [], []
    for skill in COURSE_SKILLS.get(course, COURSE_SKILLS["BSC_IT"]):
        pattern = rf"(?<![\w+#]){re.escape(skill)}(?![\w+#])"
        (found if re.search(pattern, low) else missing).append(skill)
    return found, missing


def _first_word(line: str) -> str:
    """First real word of a line, skipping bullet symbols and numbering."""
    for tok in line.split():
        word = re.sub(r"[^a-z]", "", tok.lower())
        if word:
            return word
    return ""


def bullet_lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if re.match(r"^\s*([•\-–*]|\d+\.)\s*\S", ln)]


# ---------------------------------------------------------------- scoring
def score_resume(text: str, pages: int, course: str) -> ResumeReport:
    words = re.findall(r"[A-Za-z][A-Za-z+#.]*", text)
    n_words = len(words)
    low = text.lower()
    checks: list[Check] = []

    # 1. Parseability (10)
    c = Check("Readable by ATS", 0, 10)
    if n_words >= 150:
        c.points = 10
    elif n_words >= 40:
        c.points = 5
        c.issues.append("Very little text could be read from the file.")
        c.fixes.append("Use a text-based PDF (export from Word/Google Docs), not a scanned image or photo.")
    else:
        c.issues.append("The resume text could not be read; ATS systems will see an almost empty resume.")
        c.fixes.append("Do not upload a scanned/image resume. Export a text PDF from Word or Google Docs.")
    checks.append(c)

    # 2. Contact (10)
    contact = detect_contact(text)
    c = Check("Contact details", 0, 10)
    for key, pts, fix in (("email", 3, "Add a professional email address."),
                          ("phone", 3, "Add a 10-digit mobile number."),
                          ("linkedin", 2, "Add your LinkedIn profile URL."),
                          ("github", 2, "Add your GitHub profile URL with your project code.")):
        if contact[key]:
            c.points += pts
        else:
            c.issues.append(f"Missing {key}.")
            c.fixes.append(fix)
    checks.append(c)

    # 3. Sections (20)
    sections = detect_sections(text)
    c = Check("Standard sections", 0, 20)
    for name, ok in sections.items():
        if ok:
            c.points += SECTION_POINTS[name]
        else:
            c.issues.append(f"No '{name.title()}' section found.")
            c.fixes.append(f"Add a clearly titled '{name.title()}' heading; "
                           "ATS parsers look for standard names.")
    checks.append(c)

    # 4. Skills match (25)
    found, missing = detect_skills(text, course)
    total = len(COURSE_SKILLS.get(course, COURSE_SKILLS["BSC_IT"]))
    # 60% of the list is a strong match for a student resume; scale to full marks there.
    c = Check("Skills / keyword match", round(min(1.0, len(found) / (0.6 * total)) * 25, 1), 25)
    if c.points < 25:
        c.issues.append(f"Only {len(found)} of {total} key skills for your course are mentioned.")
        c.fixes.append("Add skills you genuinely know from: " + ", ".join(missing[:8]) + ".")
    checks.append(c)

    # 5. Impact (15): action verbs + quantified results
    bullets = bullet_lines(text)
    lines_for_impact = bullets or [ln for ln in text.splitlines() if len(ln.split()) >= 5]
    verb_lines = sum(1 for ln in lines_for_impact if _first_word(ln) in ACTION_VERBS)
    numbers = len(re.findall(r"\b\d+(\.\d+)?\s*(%|\+|x\b|users|students|records|ms|sec|hours|days|"
                             r"lakh|k\b|percent)", low))
    weak = [p for p in WEAK_PHRASES if p in low]
    c = Check("Impact (action verbs & numbers)", 0, 15)
    c.points += min(8.0, verb_lines * 2.0)
    c.points += min(7.0, numbers * 2.5)
    if verb_lines < 4:
        c.issues.append("Few bullet points start with strong action verbs.")
        c.fixes.append("Start bullets with verbs like Built, Developed, Automated, Improved, Led.")
    if numbers < 3:
        c.issues.append("Achievements are not quantified.")
        c.fixes.append("Add numbers: e.g. 'Reduced query time by 40%', 'Used by 200+ students', "
                       "'95% accuracy'.")
    if weak:
        c.points = max(0.0, c.points - 2)
        c.issues.append("Weak/generic phrases used: " + ", ".join(weak) + ".")
        c.fixes.append("Replace generic phrases with what you actually did and the result.")
    checks.append(c)

    # 6. Projects (10)
    proj_section = re.search(r"projects?\s*\n(.*?)(\n\s*(experience|internship|education|skills|"
                             r"certification|achievement|award)s?\b|\Z)", text, re.I | re.S)
    proj_text = proj_section.group(1) if proj_section else ""
    proj_count = len([ln for ln in proj_text.splitlines()
                      if ln.strip() and not re.match(r"^\s*[•\-–*]", ln) and len(ln.split()) <= 12])
    tech_in_projects = len(detect_skills(proj_text, course)[0]) if proj_text else 0
    c = Check("Projects", 0, 10)
    c.points += min(6.0, proj_count * 2.0)
    c.points += min(4.0, tech_in_projects * 1.0)
    if proj_count < 2:
        c.issues.append("Fewer than 2 projects listed.")
        c.fixes.append("List 2-3 projects with a title, the tech stack and 2 bullets on what you "
                       "built/achieved.")
    if tech_in_projects < 3 and proj_count:
        c.issues.append("Projects do not mention the technologies used.")
        c.fixes.append("Write the tech stack for each project, e.g. 'Python, Flask, MySQL'.")
    checks.append(c)

    # 7. Format & length (10)
    c = Check("Format & length", 10, 10)
    if pages > 2:
        c.points -= 4
        c.issues.append(f"Resume is {pages} pages long.")
        c.fixes.append("Keep a student resume to 1 page (max 2).")
    if n_words < 250:
        c.points -= 3
        c.issues.append("Resume content is too short.")
        c.fixes.append("Add project details, skills and achievements (aim for 350-700 words).")
    elif n_words > 1000:
        c.points -= 3
        c.issues.append("Resume is too wordy.")
        c.fixes.append("Cut long paragraphs; use short bullet points.")
    if len(bullets) < 5:
        c.points -= 2
        c.issues.append("Few bullet points; long paragraphs are hard for ATS and recruiters to scan.")
        c.fixes.append("Use bullet points under projects and experience.")
    pronouns = len(re.findall(r"\b(i|my|me)\b", low))
    if pronouns > 6:
        c.points -= 1
        c.issues.append("Too many first-person words (I, my, me).")
        c.fixes.append("Write bullets without 'I': 'Developed a web app...' instead of 'I developed...'.")
    c.points = max(0.0, c.points)
    checks.append(c)

    score = round(sum(ch.points for ch in checks), 1)
    return ResumeReport(
        ats_score=score,
        ats_pass=score >= ATS_PASS,
        checks=checks,
        detected={"contact": contact, "sections": sections, "skills_found": found,
                  "skills_missing": missing, "word_count": n_words, "pages": pages,
                  "bullets": len(bullets), "quantified_results": numbers, "project_count": proj_count},
        issues=[i for ch in checks for i in ch.issues],
        improvements=[f for ch in checks for f in ch.fixes],
    )


# ---------------------------------------------------------------- LLM review + questions
REVIEW_SYSTEM = """You are an expert technical recruiter reviewing a student's resume (Indian college,
course: {course}). Be specific and practical; refer to actual lines from the resume.
Return JSON: {{"summary": "2-3 sentence overall assessment",
"mistakes": ["specific mistake (grammar, formatting, vague claim, missing info)", ...],
"improvements": ["specific rewrite or addition", ...],
"rewritten_bullets": [{{"before": "original line", "after": "improved line"}}, ...]}}
Max 6 items per list, max 3 rewritten bullets."""

QUESTIONS_SYSTEM = """You are a technical interviewer. Using ONLY the candidate's resume, write {n} interview
questions that test whether they really did what they claim (projects, internships, skills). Each question
must name the specific project/skill from the resume. For each, give the key points a strong answer covers.
Return JSON: {{"questions": [{{"text": "...", "ideal_answer": "what a strong answer explains",
"keywords": ["3-6 key terms"], "topic": "project/skill name"}}]}}"""


def llm_review(resume_text: str, course: str) -> dict | None:
    if not llm.is_enabled():
        return None
    try:
        return llm.chat_json(REVIEW_SYSTEM.format(course=course), resume_text[:12000])
    except llm.LLMUnavailable as e:
        log.warning("Resume LLM review failed: %s", e)
        return None


def _project_titles(text: str) -> list[str]:
    m = re.search(r"projects?\s*\n(.*?)(\n\s*(experience|internship|education|skills|certification|"
                  r"achievement|award)s?\b|\Z)", text, re.I | re.S)
    if not m:
        return []
    titles = []
    for ln in m.group(1).splitlines():
        ln = ln.strip()
        if ln and not re.match(r"^[•\-–*]", ln) and 1 < len(ln.split()) <= 12:
            titles.append(re.split(r"\s[|–-]\s|\(", ln)[0].strip())
    return titles[:4]


def resume_questions(resume_text: str, course: str, n: int = 3) -> list[dict]:
    """Questions about the candidate's own projects/skills, each with expected answer points."""
    if llm.is_enabled():
        try:
            data = llm.chat_json(QUESTIONS_SYSTEM.format(n=n), resume_text[:12000], temperature=0.5)
            qs = [q for q in data.get("questions", []) if q.get("text") and q.get("ideal_answer")]
            if qs:
                return qs[:n]
        except llm.LLMUnavailable as e:
            log.warning("Resume question generation failed: %s", e)

    # Fallback: template questions from detected projects and skills.
    out = []
    for title in _project_titles(resume_text):
        out.append({
            "text": f"In your resume you mention the project '{title}'. Explain what it does, your role, "
                    "the technologies you used and one challenge you solved.",
            "ideal_answer": f"Clear explanation of the purpose of {title}, the candidate's own contribution, "
                            "the tech stack and why it was chosen, how a key challenge was solved, and the "
                            "result or what was learned.",
            "keywords": ["purpose", "role", "technology", "challenge", "result"],
            "topic": title,
        })
    found, _ = detect_skills(resume_text, course)
    for skill in found:
        if len(out) >= n:
            break
        out.append({
            "text": f"You have listed {skill} as a skill. Where have you used {skill} and what did you "
                    "build with it?",
            "ideal_answer": f"A concrete example of using {skill} in a project or internship: the problem, "
                            f"how {skill} was applied, specific features or concepts used, and the outcome.",
            "keywords": [skill, "project", "used", "example", "result"],
            "topic": skill,
        })
    return out[:n]


def analyze_resume(path: str | Path, course: str) -> tuple[ResumeReport, str]:
    text, pages = extract_text(path)
    report = score_resume(text, pages, course)
    report.llm_review = llm_review(text, course)
    return report, text
