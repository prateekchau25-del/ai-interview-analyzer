from backend.app.analyzers.speech import Word, analyze_words, count_fillers, fluency_score


def words_from(text: str, start: float = 0.0, step: float = 0.4, gaps: dict[int, float] | None = None):
    """Build fake Whisper words; `gaps` inserts silence (seconds) before word index i."""
    out, t = [], start
    for i, tok in enumerate(text.split()):
        t += (gaps or {}).get(i, 0.0)
        out.append(Word(text=" " + tok, start=t, end=t + step * 0.8, prob=0.9))
        t += step
    return out


def test_context_fillers_only_counted_at_clause_start():
    ws = words_from("Umm, so a thread shares memory so you need locks. Like, basically that is it.")
    counts = count_fillers(ws)
    assert counts["umm"] == 1
    assert counts["so"] == 1  # "so you need" is not a filler
    assert counts["like"] == 1
    assert counts["basically"] == 1


def test_filler_phrases():
    ws = words_from("You know I mean it is kind of simple")
    counts = count_fillers(ws)
    assert counts["you know"] == 1 and counts["i mean"] == 1 and counts["kind of"] == 1


def test_long_pauses_detected():
    ws = words_from("one two three four five", gaps={2: 2.0, 4: 3.1})
    r = analyze_words(ws, duration_sec=10)
    assert r.long_pause_count == 2
    assert r.longest_pause_sec >= 3.0


def test_empty_speech():
    r = analyze_words([], duration_sec=5)
    assert r.word_count == 0 and r.fluency_score == 0 and r.pace_label == "no speech"


def test_fluency_score_bounds():
    assert fluency_score(140, 0, 0, 1) == 100
    assert fluency_score(60, 20, 10, 1) == 0
    assert 0 < fluency_score(175, 5, 1, 1) < 100
