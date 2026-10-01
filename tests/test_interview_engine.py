import random

from backend.app.services.interview_engine import (
    MAX_FOLLOWUPS,
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


def run_interview(answer_fn, year=3, has_resume=True, seed=0, mode="student"):
    pool, rng = FakePool(), random.Random(seed)
    state = InterviewState(course="BSC_IT", year=year, has_resume=has_resume, mode=mode)
    phases, asked = [], []
    while True:
        step = next_step(state, pool, rng)
        if step["phase"] in ("review", "finished"):
            return state, phases, asked, step["phase"]
        q = step["question"]
        if step["phase"] == "followup":
            parent = next(a for a in asked if a.id == step["parent_id"])
            q = QuestionRef(100000 + len(asked), "followup", parent.level, parent.year, parent.subject)
        phases.append(step["phase"])
        asked.append(q)
        score = answer_fn(q, state)
        record_answer(state, q, score, skipped=score == 0, rng=rng)


def test_flow_order_and_counts():
    state, phases, asked, end = run_interview(lambda q, s: 90)
    assert phases[0] == "intro"
    assert asked[0].id in FakePool().intro_ids  # opens with a "tell me about yourself" question
    assert all(q.id not in FakePool().intro_ids for q in asked[1:])
    assert phases.count("technical") == TECHNICAL_COUNT
    assert phases.count("resume") == RESUME_COUNT
    assert phases.count("intro") + phases.count("soft") == SOFT_COUNT
    main = [ph for ph in phases if ph != "followup"]
    assert main == sorted(main, key=["intro", "technical", "resume", "soft"].index)
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
    # every wrong non-soft answer is parked for the end; soft-skill answers only get feedback
    assert len(state.parked) == sum(1 for q in asked if q.kind != "soft")


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
    assert s["total_answered"] == TECHNICAL_COUNT + RESUME_COUNT + SOFT_COUNT + s["followup_count"]
    assert s["peak_level"] in ("Intermediate", "Advanced")


def test_partial_answer_triggers_followup_without_changing_level():
    # intro ok, then every technical answer is partial (70), follow-ups answered correctly (90)
    state, phases, asked, _ = run_interview(lambda q, s: 90 if q.kind in ("soft", "followup") else 70)
    assert phases[1:3] == ["technical", "followup"]  # counter question right after the partial answer
    assert phases.count("followup") == MAX_FOLLOWUPS  # capped
    assert phases.count("technical") == TECHNICAL_COUNT  # follow-ups are extra, not part of the 15
    for i, q in enumerate(asked):
        if q.kind == "followup":
            assert asked[i - 1].kind in ("technical", "resume")


def test_no_followup_after_wrong_or_skipped():
    _, phases, _, _ = run_interview(lambda q, s: 90 if q.kind == "soft" else 20)
    assert "followup" not in phases
    _, phases, _, _ = run_interview(lambda q, s: 90 if q.kind == "soft" else 0)
    assert "followup" not in phases


def test_followup_answers_do_not_move_level():
    state = InterviewState(course="BSC_IT", year=3)
    record_answer(state, QuestionRef(1, "technical", START_LEVEL, 3, "OS"), 70)
    assert state.followup_for == 1 and state.level == START_LEVEL
    record_answer(state, QuestionRef(2, "followup", START_LEVEL, 3, "OS"), 95)  # correct follow-up
    assert state.level == START_LEVEL and state.followup_for is None
    record_answer(state, QuestionRef(3, "followup", START_LEVEL, 3, "OS"), 10)  # wrong follow-up
    assert state.level == START_LEVEL and 3 in state.parked



# ------------------------------------------------------------------ teacher mode
from backend.app.services.interview_engine import PROFILES  # noqa: E402

TEACHER = PROFILES["teacher"]


def test_teacher_every_answer_gets_one_or_two_counter_questions():
    state, phases, asked, _ = run_interview(lambda q, s: 90, mode="teacher", seed=3)
    assert phases.count("technical") == TEACHER.technical
    # walk the sequence: every technical/resume answer is followed by 1 or 2 follow-ups
    i = 0
    while i < len(asked):
        if asked[i].kind in ("technical", "resume"):
            n = 0
            while i + 1 + n < len(asked) and asked[i + 1 + n].kind == "followup":
                n += 1
            assert 1 <= n <= 2, f"question {i} got {n} follow-ups"
            i += n
        i += 1


def test_teacher_gets_second_followup_when_first_is_weak():
    state = InterviewState(course="TEACHER", year=3, mode="teacher")
    record_answer(state, QuestionRef(1, "technical", 4, 1, "DBMS"), 95)
    assert state.followup_for == 1 and state.followup_depth == 1
    record_answer(state, QuestionRef(2, "followup", 4, 1, "DBMS"), 60)  # weak -> must probe again
    assert state.followup_for == 2 and state.followup_depth == 2
    record_answer(state, QuestionRef(3, "followup", 4, 1, "DBMS"), 40)  # chain capped at 2
    assert state.followup_for is None and state.followup_depth == 0


def test_teacher_levels_never_easy():
    state, _, asked, _ = run_interview(lambda q, s: 10 if q.kind == "technical" else 90, mode="teacher")
    tech = [q.level for q in asked if q.kind == "technical"]
    assert tech[0] == TEACHER.start_level == 4  # Intermediate
    assert min(tech) == TEACHER.min_level == 3  # floors at Medium, never Basic/Beginner


def test_student_followups_unchanged_and_no_chains():
    _, phases, asked, _ = run_interview(lambda q, s: 70 if q.kind == "technical" else 90)
    for i, q in enumerate(asked):
        if q.kind == "followup":
            assert asked[i - 1].kind in ("technical", "resume")  # never a follow-up of a follow-up


def test_subject_scores_in_summary():
    state, *_ = run_interview(lambda q, s: 90, mode="teacher")
    subj = summary(state)["subject_scores"]
    assert subj and all(v["questions"] >= 1 and 0 <= v["avg"] <= 100 for v in subj.values())
