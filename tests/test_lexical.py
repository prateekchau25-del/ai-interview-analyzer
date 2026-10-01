from backend.app.analyzers.lexical import analyze


def test_confident_vs_hesitant():
    confident = analyze("I designed and implemented the database myself and we successfully delivered it "
                        "two weeks early. It clearly improved the query speed for the whole team.")
    hesitant = analyze("Umm I think maybe it is kind of a database, I guess, I am not sure, probably "
                       "something like storing data I believe.")
    assert confident.confidence_language_score > 80 > 50 > hesitant.confidence_language_score
    # i think, maybe, kind of, i guess, i am not sure, probably, i believe
    assert hesitant.hedge_count == 7 and confident.confident_count >= 4


def test_pronouns_sentiment_and_keywords():
    r = analyze("We built a great project and our team enjoyed it. I learned a lot about normalization.",
                keywords=["normalization", "primary key"])
    assert r.we_ratio > r.i_ratio and r.sentiment > 0
    assert r.keyword_density > 0


def test_empty():
    r = analyze("")
    assert r.word_count == 0 and r.confidence_language_score == 0
