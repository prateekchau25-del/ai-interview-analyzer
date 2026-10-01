"""Body-language summary logic (fast, synthetic frames) + one real-photo check with the actual models."""
import numpy as np
import pytest

from backend.app.analyzers.vision import FrameResult, summarize


def frames(n, **kw):
    base = dict(face=True, faces=1, eye_contact=True, smile=0.0, blink=False, nose=(0.5, 0.5), face_width=0.3)
    return [FrameResult(**{**base, **kw}) for _ in range(n)]


def ts(n, step=0.2):
    return [i * step for i in range(n)]


def test_good_body_language_scores_high():
    rs = frames(20, smile=0.6)
    s = summarize(rs, ts(20))
    assert s["face_present_pct"] == 100 and s["eye_contact_pct"] == 100 and s["smile_pct"] == 100
    assert s["body_language_score"] == 100 and not s["notes"]


def test_looking_away_and_out_of_frame_are_penalised():
    rs = frames(10, eye_contact=False) + [FrameResult() for _ in range(10)]
    s = summarize(rs, ts(20))
    assert s["face_present_pct"] == 50 and s["eye_contact_pct"] == 0
    assert s["body_language_score"] < 60
    assert any("out of frame" in n for n in s["notes"]) and any("Look at the camera" in n for n in s["notes"])


def test_fidgety_hands_and_nervous_face():
    rs = frames(20)
    for i, r in enumerate(rs):
        r.hands = [(0.2 + 0.3 * (i % 2), 0.8)]  # wrist jumps back and forth every frame
        r.emotion = np.array([0, 0, 0, 0.7, 0, 0.3, 0, 0])  # mostly fear
    s = summarize(rs, ts(20))
    assert s["hand_movement"] == "fidgety" and s["dominant_emotion"] == "nervous"
    assert any("hands" in n for n in s["notes"]) and any("tense or anxious" in n for n in s["notes"])


def test_blinks_counted_on_transitions_only():
    rs = frames(30)
    for i in (5, 6, 7, 15, 25):  # 3 blink onsets (5-7 is one blink)
        rs[i].blink = True
    assert summarize(rs, ts(30, step=1.0))["blink_per_min"] == pytest.approx(3 / 29 * 60, abs=0.2)


def test_too_few_frames_returns_none():
    assert summarize(frames(2), ts(2)) is None


def test_real_photo_with_models():
    cv2 = pytest.importorskip("cv2")
    from backend.app.analyzers import vision

    try:
        vision.ensure_models()
    except OSError:
        pytest.skip("vision models not downloadable (offline)")
    rgb = cv2.cvtColor(cv2.imread("tests/data/portrait.jpg"), cv2.COLOR_BGR2RGB)
    r = vision.analyze_frame(rgb)
    assert r.face and r.eye_contact and r.smile > 0.5
    assert vision.FACE_EMOTIONS[int(np.argmax(r.emotion))] == "happiness"
