"""Live webcam body-language tracking for the interview page.

Frames are analysed inside the Streamlit server process (MediaPipe + HSEmotion); no video is stored
or uploaded.
Only the per-answer summary (eye contact %, smile %, ...) is sent to the API with the answer.
"""
import json
import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # so the frontend can use backend.app.analyzers.vision
    sys.path.insert(0, str(ROOT))

RTC = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}


def _tracker():
    from backend.app.analyzers.vision import VisionTracker

    if "vision_tracker" not in st.session_state:
        # ~5 analysed frames per second at 30 fps keeps CPU use low.
        st.session_state["vision_tracker"] = VisionTracker(sample_every=6, emotion_every=3)
    return st.session_state["vision_tracker"]


def render(question_key: str) -> None:
    """Show the webcam widget; restart the per-answer tracker whenever the question changes."""
    from streamlit_webrtc import WebRtcMode, webrtc_streamer

    tracker = _tracker()
    if st.session_state.get("vision_question") != question_key:
        tracker.reset()
        st.session_state["vision_question"] = question_key

    def on_frame(frame):  # runs on the WebRTC worker thread
        tracker.add_frame(frame.to_ndarray(format="rgb24"), frame.time or time.time())
        return frame

    try:
        ctx = webrtc_streamer(
            key="interview-cam", mode=WebRtcMode.SENDRECV, rtc_configuration=RTC,
            media_stream_constraints={"video": True, "audio": False},
            video_frame_callback=on_frame, async_processing=True,
        )
    except Exception as e:  # the webcam is optional - never block the interview
        st.session_state["webcam_on"] = False
        st.info(f"Webcam analysis is not available here ({type(e).__name__}). "
                "You can continue the interview without it.")
        return
    st.session_state["webcam_on"] = bool(ctx.state.playing)
    if ctx.state.playing:
        live = tracker.summary()
        if live:
            st.caption(f"👀 eye contact {live['eye_contact_pct']:.0f}% · 🙂 smile {live['smile_pct']:.0f}% · "
                       f"face in frame {live['face_present_pct']:.0f}%")
    else:
        st.caption("Click **START** to turn on the camera for body-language feedback (optional).")


def answer_summary() -> str | None:
    """JSON summary for the current answer, or None if the webcam was off / too few frames."""
    if not st.session_state.get("webcam_on"):
        return None
    summary = _tracker().summary()
    return json.dumps(summary) if summary else None
