import random

from backend.app.services.interview_engine import (
    MAX_LEVEL,
    MIN_LEVEL,
    RESUME_COUNT,
    SOFT_COUNT,
    START_LEVEL,
    TECHNICAL_COUNT,
    InterviewState,
    QuestionRef,
    next_step,
    record_answer,
    summary,
)


class FakePool:
    """6 levels x 3 years x 4 subjects x 3 questions of technical + soft + resume questions."""

    def __init__(self):
        self.qs: list[QuestionRef] = []
        i = 0
        for year in (1, 2, 3):
            for subj in ("DBMS", "Python", "OS", "Networks"):
                for level in range(1, 7):
                    for _ in range(3):
                        i += 1
                        self.qs.append(QuestionRef(i, "technical", level, year, f"{subj}-Y{year}"))
        self.intro_ids = set()
        for kind in ("intro", "soft", "resume"):
            for _ in range(10):
                i += 1
                if kind == "intro":
                    self.intro_ids.add(i)
                self.qs.append(QuestionRef(i, "soft" if kind == "intro" else kind, 0))

    def candidates(self, kind, course, max_year, level, exclude_ids):
        want = "soft" if kind == "intro" else kind
        return [q for q in self.qs if q.kind == want and q.id not in exclude_ids
                and (q.year is None or q.year <= max_year) and (level is None or q.level == level)
                and (kind != "intro" or q.id in self.intro_ids)
                and (kind != "soft" or q.id not in self.intro_ids)]


def run_interview(answer_fn, year=3, has_resume=True, seed=0):
    pool, rng = FakePool(), random.Random(seed)
    state = InterviewState(course="BSC_IT", year=year, has_resume=has_resume)
    phases, asked = [], []
    while True:
        step = next_step(state, pool, rng)
        if step["phase"] in ("review", "finished"):
            return state, phases, asked, step["phase"]
        q = step["question"]
        phases.append(step["phase"])
        asked.append(q)
        score = answer_fn(q, state)
        record_answer(state, q, score, skipped=score == 0)


def test_flow_order_and_counts():
    state, phases, asked, end = run_interview(lambda q, s: 90)
    assert phases[0] == "intro"
    assert asked[0].id in FakePool().intro_ids  # opens with a "tell me about yourself" question
    assert all(q.id not in FakePool().intro_ids for q in asked[1:])
    assert phases.count("technical") == TECHNICAL_COUNT
    assert phases.count("resume") == RESUME_COUNT
    assert phases.count("intro") + phases.count("soft") == SOFT_COUNT
    assert phases == sorted(phases, key=["intro", "technical", "resume", "soft"].index)
    assert end == "finished"  # all correct -> nothing to review
    assert len({q.id for q in asked}) == len(asked)  # never repeats a question


def test_no_resume_skips_resume_phase():
    _, phases, _, _ = run_interview(lambda q, s: 90, has_resume=False)
    assert "resume" not in phases


def test_level_goes_up_on_correct_and_caps():
    state, _, asked, _ = run_interview(lambda q, s: 95)
    tech = [q for q in asked if q.kind == "technical"]
    assert tech[0].level == START_LEVEL
    assert tech[1].level == START_LEVEL + 1
    assert state.level == MAX_LEVEL


def test_level_goes_down_after_two_wrong_and_floors():
    state, _, asked, end = run_interview(lambda q, s: 40)
    tech_levels = [q.level for q in asked if q.kind == "technical"]
    assert tech_levels[:4] == [START_LEVEL, START_LEVEL, START_LEVEL - 1, START_LEVEL - 1]
    assert state.level == MIN_LEVEL
    assert end == "review"
    assert len(state.parked) == len(asked)  # every wrong answer is parked for the end


def test_single_wrong_does_not_drop_level():
    answers = iter([90, 90, 40, 90] + [90] * 30)  # intro, t1 ok, t2 wrong, t3 ok
    state, _, asked, _ = run_interview(lambda q, s: next(answers))
    tech_levels = [q.level for q in asked if q.kind == "technical"]
    assert tech_levels[:3] == [START_LEVEL, START_LEVEL + 1, START_LEVEL + 1]


def test_first_year_only_gets_first_year_questions():
    _, _, asked, _ = run_interview(lambda q, s: 90, year=1)
    assert all(q.year in (None, 1) for q in asked)


def test_different_candidates_get_different_sets():
    sets = [frozenset(q.id for q in run_interview(lambda q, s: 90, seed=seed)[2]) for seed in range(5)]
    assert len(set(sets)) == 5
    # some overlap is fine, but no two candidates share most of their questions
    for a in sets:
        for b in sets:
            if a is not b:
                assert len(a & b) / len(a) < 0.5


def test_state_roundtrip_and_summary():
    state, *_ = run_interview(lambda q, s: 90 if q.level < 4 else 50)
    restored = InterviewState.from_dict(state.to_dict())
    assert restored == state
    s = summary(state)
    assert s["total_answered"] == TECHNICAL_COUNT + RESUME_COUNT + SOFT_COUNT
    assert s["peak_level"] in ("Intermediate", "Advanced")
