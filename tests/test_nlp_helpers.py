from backend.app.analyzers.nlp import is_dont_know, keyword_coverage


def test_keyword_coverage_matches_word_roots():
    cov, missing = keyword_coverage("Threads share the same memory and processes are isolated.",
                                    ["shared memory", "isolation", "race condition"])
    assert missing == ["race condition"]
    assert round(cov, 2) == 0.67


def test_dont_know_detection():
    assert is_dont_know("Sorry, I don't know this one.")
    assert is_dont_know("pata nahi sir")
    assert is_dont_know("um")
    assert not is_dont_know("A primary key uniquely identifies each row and cannot be null in a table.")
    # long answers that mention uncertainty are still graded
    assert not is_dont_know("I am not sure about the exact name but a thread is a lightweight unit that "
                            "shares memory with other threads of the same process.")


def test_soft_answers_get_quality_feedback_not_right_wrong():
    from backend.app.analyzers.nlp import grade_answer

    guidance = "Introduction with course, skills, a project and a career goal."
    kw = ["course", "skills", "project", "goal"]
    strong = ("I am Asha, a second year B.Sc IT student. My key skills are Python and SQL. "
              "In my last semester "
              "I built a library management project used by 200 students, where I designed the database. "
              "My goal is to become a backend developer and keep learning cloud technologies.")
    weak = "I am Asha. I like computers."
    g1 = grade_answer("Tell me about yourself.", guidance, kw, strong, kind="soft")
    g2 = grade_answer("Tell me about yourself.", guidance, kw, weak, kind="soft")
    assert g1.score > 70 > g2.score
    assert g1.verdict == "good" and g2.verdict == "weak"
    assert "example" in g2.feedback.lower() or "fuller" in g2.feedback.lower()
