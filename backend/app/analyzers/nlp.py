"""Answer grading and explanations.

Primary: LLM examiner that compares the candidate's answer with the stored model answer.
Fallback (no LLM): semantic similarity (sentence-transformers) + keyword coverage.
"""
import logging
import re
import threading
from dataclasses import asdict, dataclass, field

from backend.app.analyzers import llm
from backend.app.core.config import get_settings

log = logging.getLogger(__name__)

PASS_THRESHOLD = 85.0

DONT_KNOW_PATTERNS = [
    r"\bi (really )?(don'?t|do not) know\b", r"\bno idea\b", r"\bnot sure\b", r"\bi forgot\b",
    r"\bpata nahi?\b", r"\bnahi pata\b", r"\bmaloom nahi\b", r"\bskip\b", r"\bpass\b",
    r"\bcan'?t (answer|remember)\b", r"\byaad nahi\b",
]

_embedder = None
_embedder_lock = threading.Lock()


@dataclass
class Grade:
    score: float  # 0-100: how correct/complete the answer is
    is_correct: bool  # score >= PASS_THRESHOLD
    skipped: bool
    verdict: str  # correct | partially_correct | incorrect | skipped
    feedback: str
    missing_points: list[str] = field(default_factory=list)
    method: str = "llm"  # llm | embedding

    def to_dict(self) -> dict:
        return asdict(self)


def get_embedder():
    """(tokenizer, onnx session) for MiniLM sentence embeddings.

    ONNX Runtime + `tokenizers` is 2-3x faster on CPU than PyTorch and avoids the heavy
    transformers/scikit-learn import chain. Mean pooling matches sentence-transformers exactly.
    """
    global _embedder
    with _embedder_lock:
        if _embedder is None:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download
            from tokenizers import Tokenizer

            repo = get_settings().embedding_model
            tok = Tokenizer.from_file(hf_hub_download(repo, "tokenizer.json"))
            tok.enable_truncation(max_length=256)
            tok.enable_padding()
            sess = ort.InferenceSession(hf_hub_download(repo, "onnx/model.onnx"),
                                        providers=["CPUExecutionProvider"])
            _embedder = (tok, sess)
        return _embedder


def embed(texts: list[str]):
    """L2-normalised sentence embeddings, shape (len(texts), dim)."""
    import numpy as np

    tok, sess = get_embedder()
    enc = tok.encode_batch(texts)
    ids = np.array([e.ids for e in enc], dtype=np.int64)
    mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
    feeds = {"input_ids": ids, "attention_mask": mask}
    if "token_type_ids" in {i.name for i in sess.get_inputs()}:
        feeds["token_type_ids"] = np.zeros_like(ids)
    hidden = sess.run(None, feeds)[0]
    m = mask[..., None].astype(np.float32)
    pooled = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
    return pooled / np.linalg.norm(pooled, axis=1, keepdims=True)


def similarity(a: str, b: str) -> float:
    e = embed([a, b])
    return float(e[0] @ e[1])


def is_dont_know(transcript: str) -> bool:
    text = transcript.lower().strip()
    if len(re.findall(r"\w+", text)) < 3:
        return True
    # Only treat as "don't know" when the answer is short; long answers may mention "not sure" in passing.
    return len(text.split()) < 15 and any(re.search(p, text) for p in DONT_KNOW_PATTERNS)


def _stem(word: str) -> str:
    # Crude prefix stem: "shared"/"share"/"sharing" -> "share", "isolation"/"isolated" -> "isola".
    return word if len(word) <= 4 else word[:5]


def keyword_coverage(transcript: str, keywords: list[str]) -> tuple[float, list[str]]:
    """Fraction of keywords present. A keyword matches when every word in it appears in the answer
    with the same root, so "shared memory" matches "threads share the same memory"."""
    if not keywords:
        return 1.0, []
    stems = {_stem(t) for t in re.findall(r"[a-z0-9+#/]+", transcript.lower())}
    missing = [
        k for k in keywords
        if not all(_stem(t) in stems for t in re.findall(r"[a-z0-9+#/]+", k.lower()))
    ]
    return 1 - len(missing) / len(keywords), missing


def _verdict(score: float) -> str:
    return "correct" if score >= PASS_THRESHOLD else "partially_correct" if score >= 50 else "incorrect"


GRADER_SYSTEM = """You are a strict but fair technical interviewer grading a spoken answer from an Indian
undergraduate (B.Sc IT / Data Science / AI / BCA). The answer was transcribed from speech, so ignore
grammar, filler words, transcription typos and Hinglish; judge only technical correctness and completeness
against the model answer. Different wording or examples that are correct must get full credit.
Wrong statements reduce the score even if keywords are present.

Scoring (0-100): 90-100 fully correct and complete; 85-89 correct with a minor omission;
60-84 partly correct, key points missing; 30-59 mostly incomplete or some errors; 0-29 wrong/irrelevant.

Return JSON: {"score": int, "feedback": "2 short sentences addressed to the candidate",
"missing_points": ["key point the candidate missed", ...]}"""


def _grade_llm(question: str, model_answer: str, transcript: str) -> Grade:
    data = llm.chat_json(
        GRADER_SYSTEM,
        f"Question: {question}\n\nModel answer: {model_answer}\n\nCandidate's answer: {transcript}",
    )
    score = float(max(0, min(100, data.get("score", 0))))
    return Grade(score=score, is_correct=score >= PASS_THRESHOLD, skipped=False, verdict=_verdict(score),
                 feedback=str(data.get("feedback", "")).strip(),
                 missing_points=[str(p) for p in data.get("missing_points", [])][:5], method="llm")


def _grade_embedding(model_answer: str, keywords: list[str], transcript: str) -> Grade:
    sim = similarity(model_answer, transcript)
    # Measured on MiniLM: off-topic ~0.0, wrong-but-on-topic 0.55-0.76, good 0.83-0.95.
    # Similarity cannot tell "wrong but on-topic" from "right", so it acts mainly as an off-topic gate
    # and keyword coverage carries more weight. Conservative on purpose: the LLM grader is primary.
    sim_score = max(0.0, min(1.0, (sim - 0.45) / (0.92 - 0.45)))
    cov, missing = keyword_coverage(transcript, keywords)
    score = round(100 * (0.4 * sim_score + 0.6 * cov), 1)
    if sim < 0.3:  # off-topic answers get no keyword credit
        score = min(score, 15.0)
    if score >= PASS_THRESHOLD:
        feedback = "Your answer covers the key concepts correctly."
    elif missing:
        feedback = "Your answer is not complete. You missed: " + ", ".join(missing[:4]) + "."
    else:
        feedback = "Your answer does not match the expected explanation closely enough."
    return Grade(score=score, is_correct=score >= PASS_THRESHOLD, skipped=False, verdict=_verdict(score),
                 feedback=feedback, missing_points=missing[:5], method="embedding")


SOFT_SYSTEM = """You are an experienced HR interviewer evaluating a spoken answer from an Indian college
student to a soft-skill/HR question. There is no single correct answer: judge structure (e.g. STAR),
relevance, a concrete personal example, clarity and professionalism, using the guidance on what a strong
answer contains. Ignore grammar, filler words and Hinglish.
Scoring (0-100): 85+ strong, 70-84 good, 50-69 average, below 50 weak.
Return JSON: {"score": int, "feedback": "2 short encouraging sentences with one specific tip",
"missing_points": ["what would make it stronger", ...]}"""


def _grade_soft(question: str, guidance: str, keywords: list[str], transcript: str) -> Grade:
    if llm.is_enabled():
        try:
            data = llm.chat_json(SOFT_SYSTEM, f"Question: {question}\nA strong answer contains: {guidance}\n"
                                              f"Candidate's answer: {transcript}")
            score = float(max(0, min(100, data.get("score", 0))))
            return Grade(score=score, is_correct=score >= PASS_THRESHOLD, skipped=False,
                         verdict="good" if score >= 70 else "average" if score >= 50 else "weak",
                         feedback=str(data.get("feedback", "")).strip(),
                         missing_points=[str(p) for p in data.get("missing_points", [])][:4], method="llm")
        except llm.LLMUnavailable as e:
            log.warning("LLM soft grading failed, using heuristic: %s", e)
    # Heuristic: coverage of expected elements, enough substance, and a concrete example (numbers/"when I").
    cov, missing = keyword_coverage(transcript, keywords)
    words = len(transcript.split())
    substance = min(1.0, words / 70)
    example = 1.0 if re.search(r"\b(when i|for example|in my|i (built|led|worked|did|created)|\d+)\b",
                               transcript.lower()) else 0.0
    score = round(100 * (0.45 * cov + 0.35 * substance + 0.20 * example), 1)
    tips = []
    if words < 40:
        tips.append("Give a fuller answer (about 45-90 seconds).")
    if not example:
        tips.append("Add a concrete example from your own experience.")
    if missing:
        tips.append("You could also mention: " + ", ".join(missing[:3]) + ".")
    return Grade(score=score, is_correct=score >= PASS_THRESHOLD, skipped=False,
                 verdict="good" if score >= 70 else "average" if score >= 50 else "weak",
                 feedback=" ".join(tips) or "Good, well-structured answer.", missing_points=missing[:4],
                 method="heuristic")


def grade_answer(question: str, model_answer: str, keywords: list[str], transcript: str,
                 kind: str = "technical") -> Grade:
    """kind='soft' uses an HR rubric (no right/wrong); every other kind is graded for correctness."""
    if is_dont_know(transcript):
        return Grade(score=0, is_correct=False, skipped=True, verdict="skipped",
                     feedback="No problem, we will come back to this question at the end.", method="rule")
    if kind == "soft":
        return _grade_soft(question, model_answer, keywords, transcript)
    if llm.is_enabled():
        try:
            return _grade_llm(question, model_answer, transcript)
        except llm.LLMUnavailable as e:
            log.warning("LLM grading failed, using embedding fallback: %s", e)
    return _grade_embedding(model_answer, keywords, transcript)


FOLLOWUP_SYSTEM = """You are a sharp but friendly technical interviewer. Ask ONE short counter/follow-up
question that builds directly on what the candidate just said (quote or refer to their own words).
- If their answer was incomplete, probe the most important missing point.
- If it was correct, go one level deeper: ask "why", "what happens if", an edge case or a real-world use.
Keep it answerable verbally in under a minute. Also give the model answer and 3-6 keywords.
Return JSON: {"text": "...", "ideal_answer": "...", "keywords": ["..."]}"""


def generate_followup(question: str, model_answer: str, transcript: str, grade: Grade) -> dict:
    """Counter question based on the candidate's own answer, with its model answer for grading."""
    if llm.is_enabled():
        try:
            data = llm.chat_json(
                FOLLOWUP_SYSTEM,
                f"Original question: {question}\nModel answer: {model_answer}\n"
                f"Candidate said: {transcript}\nScore: {grade.score:.0f}/100\n"
                f"Missing points: {', '.join(grade.missing_points) or 'none'}",
                temperature=0.5,
            )
            if data.get("text") and data.get("ideal_answer"):
                return {"text": data["text"], "ideal_answer": data["ideal_answer"],
                        "keywords": [str(k) for k in data.get("keywords", [])][:6], "method": "llm"}
        except llm.LLMUnavailable as e:
            log.warning("Follow-up generation failed, using template: %s", e)

    if grade.missing_points:
        point = grade.missing_points[0]
        text = (f"You did not mention '{point}'. Can you explain what {point} means here "
                f"and how it relates to your answer?")
        keywords = [point]
    else:
        text = "Good. Can you give a real-world example of this, and explain why it works that way?"
        keywords = []
    return {"text": text, "ideal_answer": model_answer, "keywords": keywords, "method": "template"}


def explain_answer(question: str, model_answer: str, candidate_answer: str | None = None) -> str:
    """Teacher-style explanation for the end-of-interview 'show me the answer' step."""
    if llm.is_enabled():
        try:
            return llm.chat(
                [
                    {"role": "system", "content": (
                        "You are a friendly teacher explaining an interview question to an Indian college "
                        "student. Explain in simple English in 5-8 short sentences: the core concept, a "
                        "real-life example or analogy, and one line on how to say it in an interview. "
                        "If the student's attempt is given, first gently point out what was wrong or "
                        "missing. Plain text, no markdown headings.")},
                    {"role": "user", "content": (
                        f"Question: {question}\nCorrect answer: {model_answer}\n"
                        f"Student's attempt: {candidate_answer or '(did not answer)'}")},
                ],
                temperature=0.4, max_tokens=500,
            ).strip()
        except llm.LLMUnavailable as e:
            log.warning("LLM explanation failed: %s", e)
    return model_answer
