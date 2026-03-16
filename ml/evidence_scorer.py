"""Feature extraction and inference for evidence quality scoring.

Given a piece of cited evidence and the source visit notes, extract
features that capture how well the evidence is grounded in the notes,
then score it with a trained classifier.
"""

import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alphanumeric characters."""
    return re.findall(r"[a-z0-9]+", text.lower())


def jaccard_similarity(a: list[str], b: list[str]) -> float:
    """Jaccard index between two token lists."""
    set_a, set_b = set(a), set(b)
    if not set_a and not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def word_overlap_ratio(evidence_tokens: list[str], notes_tokens: list[str]) -> float:
    """Fraction of evidence tokens that appear in the notes."""
    if not evidence_tokens:
        return 0.0
    notes_set = set(notes_tokens)
    found = sum(1 for t in evidence_tokens if t in notes_set)
    return found / len(evidence_tokens)


def weighted_word_overlap(evidence_tokens: list[str], notes_tokens: list[str]) -> float:
    """Word overlap weighted by inverse frequency in the notes.

    Rare words matching matters more than common words like 'the' or 'patient'.
    Tokens not found in notes contribute 0 to the score.
    """
    if not evidence_tokens or not notes_tokens:
        return 0.0
    notes_freq = Counter(notes_tokens)
    total = sum(notes_freq.values())

    score = 0.0
    max_possible = 0.0
    for token in evidence_tokens:
        # Weight for this token if it were found
        weight = np.log(total / (notes_freq.get(token, 0) + 1))
        max_possible += weight
        if token in notes_freq:
            score += weight

    return score / max_possible if max_possible > 0 else 0.0


def longest_common_substring_ratio(evidence: str, notes: str) -> float:
    """Length of longest common substring / length of evidence."""
    if not evidence:
        return 0.0
    ev = evidence.lower()
    nt = notes.lower()
    m, n = len(ev), len(nt)
    # Optimize: only track current and previous row
    prev = [0] * (n + 1)
    best = 0
    for i in range(1, m + 1):
        curr = [0] * (n + 1)
        for j in range(1, n + 1):
            if ev[i - 1] == nt[j - 1]:
                curr[j] = prev[j - 1] + 1
                best = max(best, curr[j])
        prev = curr
    return best / m


def tfidf_cosine_similarity(evidence: str, notes: str) -> float:
    """TF-IDF cosine similarity between evidence and notes."""
    if not evidence.strip() or not notes.strip():
        return 0.0
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    try:
        tfidf = vectorizer.fit_transform([evidence.lower(), notes.lower()])
        return float(cosine_similarity(tfidf[0:1], tfidf[1:2])[0, 0])
    except ValueError:
        return 0.0


def extract_features(evidence: str, visit_notes: list[str]) -> np.ndarray:
    """Extract feature vector for an evidence-notes pair.

    Features:
        0: tfidf_cosine     - TF-IDF cosine similarity
        1: jaccard           - Jaccard token similarity
        2: word_overlap      - Fraction of evidence tokens in notes
        3: weighted_overlap  - IDF-weighted word overlap
        4: lcs_ratio         - Longest common substring / evidence length
        5: evidence_len      - Log of evidence token count
        6: len_ratio         - Evidence length / notes length (tokens)
    """
    all_notes = " ".join(visit_notes)
    ev_tokens = tokenize(evidence)
    notes_tokens = tokenize(all_notes)

    features = [
        tfidf_cosine_similarity(evidence, all_notes),
        jaccard_similarity(ev_tokens, notes_tokens),
        word_overlap_ratio(ev_tokens, notes_tokens),
        weighted_word_overlap(ev_tokens, notes_tokens),
        longest_common_substring_ratio(evidence, all_notes),
        np.log1p(len(ev_tokens)),
        len(ev_tokens) / max(len(notes_tokens), 1),
    ]
    return np.array(features)


FEATURE_NAMES = [
    "tfidf_cosine",
    "jaccard",
    "word_overlap",
    "weighted_overlap",
    "lcs_ratio",
    "evidence_len_log",
    "len_ratio",
]
