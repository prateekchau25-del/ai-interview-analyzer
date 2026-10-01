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
