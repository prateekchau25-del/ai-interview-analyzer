"""Read-only access to data/syllabus.json (KES Shroff IT department, FY-TY)."""
import json
from functools import lru_cache
from pathlib import Path

SYLLABUS_FILE = Path(__file__).resolve().parents[1] / "data" / "syllabus.json"


def clean_subject(name: str) -> str:
    """Subject names as used in the question bank (electives lose the ' (elective)' suffix)."""
    return name.replace(" (elective)", "").strip()


@lru_cache
def load() -> dict:
    return json.loads(SYLLABUS_FILE.read_text(encoding="utf-8"))


def catalog() -> dict[str, dict[int, list[dict]]]:
    """{course: {year: [{"subject", "semester", "topics"}]}} with cleaned, de-duplicated subject names."""
    out: dict[str, dict[int, list[dict]]] = {}
    for course, cd in load()["courses"].items():
        for year, yd in cd["years"].items():
            seen, subjects = set(), []
            for sem, subs in sorted(yd["semesters"].items()):
                for s in subs:
                    name = clean_subject(s["subject"])
                    if name not in seen:
                        seen.add(name)
                        subjects.append({"subject": name, "semester": int(sem), "topics": s["topics"]})
            out.setdefault(course, {})[int(year)] = subjects
    return out


def subjects_for(course: str, year: int) -> list[str]:
    return [s["subject"] for s in catalog().get(course, {}).get(year, [])]


def validate_teaching(items: list[dict]) -> list[dict]:
    """Normalise and check [{"course", "year", "subject"}] against the syllabus. Raises ValueError."""
    if not items:
        raise ValueError("Select at least one subject you have taught.")
    clean, seen = [], set()
    for it in items:
        course, year, subject = it.get("course"), int(it.get("year", 0)), clean_subject(it.get("subject", ""))
        if subject not in subjects_for(course, year):
            raise ValueError(f"'{subject}' is not in the {course} year {year} syllabus.")
        key = (course, year, subject)
        if key not in seen:
            seen.add(key)
            clean.append({"course": course, "year": year, "subject": subject})
    return clean


def topics_for(course: str, year: int, subject: str) -> list[tuple[int, str]]:
    """[(semester, topic), ...] for one subject."""
    for s in catalog().get(course, {}).get(year, []):
        if s["subject"] == subject:
            return [(s["semester"], t) for t in s["topics"]]
    return []
