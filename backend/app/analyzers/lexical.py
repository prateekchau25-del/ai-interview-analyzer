"""Lexical (word-choice) features of a spoken answer - no model needed.

Inspired by interview-performance research (e.g. Naim et al., MIT, 2015): good interviewees use fewer
hedges/fillers, more confident and outcome words, richer vocabulary and a positive tone.
"""
import re
from dataclasses import asdict, dataclass

HEDGES = {"maybe", "probably", "perhaps", "possibly", "somewhat", "guess", "think", "suppose", "unsure",
          "kinda", "sorta", "might", "hopefully", "shayad", "lagta"}
HEDGE_PHRASES = ["i think", "i guess", "i'm not sure", "i am not sure", "not sure", "kind of", "sort of",
                 "i believe", "more or less", "i don't know"]
CONFIDENT = {"definitely", "certainly", "clearly", "exactly", "always", "successfully", "achieved",
             "built", "designed", "implemented", "developed", "led", "solved", "delivered", "improved",
             "optimized", "created", "ensure", "ensured", "confident", "specifically", "proven"}
POSITIVE = {"good", "great", "excellent", "happy", "enjoy", "enjoyed", "love", "passionate", "interested",
            "excited", "learn", "learned", "success", "successful", "improve", "improved", "best", "strong",
            "efficient", "effective", "benefit", "opportunity", "achieve", "achieved", "proud", "glad"}
NEGATIVE = {"bad", "poor", "hate", "difficult", "problem", "fail", "failed", "boring", "worst", "weak",
            "hard", "confused", "worried", "nervous", "stress", "stressed", "never", "cannot", "can't",
            "unfortunately", "struggle", "struggled"}
FIRST_SINGULAR = {"i", "me", "my", "mine", "myself"}
FIRST_PLURAL = {"we", "us", "our", "ours", "ourselves"}
STOPWORDS = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be", "been", "to", "of", "in",
             "on", "for", "with", "as", "by", "at", "it", "this", "that", "these", "those", "so", "then",
             "there", "they", "them", "he", "she", "you", "your", "from", "into", "can", "will", "would",
             "should", "do", "does", "did", "has", "have", "had", "which", "what", "when", "where", "how",
             "also", "very", "just", "about", "if", "not", "no", "yes", "i", "we", "my", "our", "um", "uh"}


@dataclass
class LexicalResult:
    word_count: int
    unique_ratio: float  # type-token ratio on content words (vocabulary richness)
    avg_word_length: float
    hedge_count: int
    confident_count: int
    sentiment: float  # -1 (negative) .. +1 (positive)
    i_ratio: float  # share of first-person singular among first-person pronouns
    we_ratio: float
    keyword_density: float  # share of answer words that are expected technical keywords
    confidence_language_score: float  # 0-100

    def to_dict(self) -> dict:
        return asdict(self)


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z][a-z'+#]*", text.lower())


def analyze(text: str, keywords: list[str] | None = None) -> LexicalResult:
    toks = tokens(text)
    n = len(toks)
    if n == 0:
        return LexicalResult(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    low = " ".join(toks)
    content = [t for t in toks if t not in STOPWORDS]
    unique_ratio = len(set(content)) / len(content) if content else 0.0

    # Count multi-word hedges first (longest first) and remove them so "i think" is not counted twice.
    rest, phrase_hits = f" {low} ", 0
    for p in sorted(HEDGE_PHRASES, key=len, reverse=True):
        phrase_hits += rest.count(f" {p} ")
        rest = rest.replace(f" {p} ", " | ")
    hedges = phrase_hits + sum(1 for t in rest.split() if t in HEDGES)
    confident = sum(1 for t in toks if t in CONFIDENT)
    pos, neg = sum(1 for t in toks if t in POSITIVE), sum(1 for t in toks if t in NEGATIVE)
    sentiment = (pos - neg) / (pos + neg) if pos + neg else 0.0
    i_n, we_n = sum(t in FIRST_SINGULAR for t in toks), sum(t in FIRST_PLURAL for t in toks)
    pron = i_n + we_n

    kw_roots = {w[:5] for k in (keywords or []) for w in tokens(k) if w not in STOPWORDS}
    density = sum(1 for t in content if t[:5] in kw_roots) / n if kw_roots else 0.0

    # Confidence language: penalise hedges per 100 words, reward confident wording and a non-negative tone.
    hedge_rate = hedges / n * 100
    confident_rate = confident / n * 100
    score = 70 - min(40, hedge_rate * 6) + min(20, confident_rate * 5) + 10 * sentiment
    if n < 15:
        score -= 15  # very short answers rarely sound confident
    return LexicalResult(
        word_count=n, unique_ratio=round(unique_ratio, 3),
        avg_word_length=round(sum(len(t) for t in toks) / n, 2), hedge_count=hedges,
        confident_count=confident, sentiment=round(sentiment, 2),
        i_ratio=round(i_n / pron, 2) if pron else 0.0, we_ratio=round(we_n / pron, 2) if pron else 0.0,
        keyword_density=round(density, 3), confidence_language_score=round(max(0.0, min(100.0, score)), 1),
    )
