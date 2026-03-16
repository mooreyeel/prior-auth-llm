"""Tests for the evidence quality scorer ML pipeline."""

import numpy as np
import pytest

from ml.evidence_scorer import (
    extract_features,
    jaccard_similarity,
    longest_common_substring_ratio,
    tokenize,
    tfidf_cosine_similarity,
    weighted_word_overlap,
    word_overlap_ratio,
)


class TestTokenize:
    def test_lowercase_and_split(self):
        assert tokenize("BMI is 32.4 kg/m2") == ["bmi", "is", "32", "4", "kg", "m2"]

    def test_empty_string(self):
        assert tokenize("") == []


class TestJaccardSimilarity:
    def test_identical(self):
        tokens = ["patient", "has", "diabetes"]
        assert jaccard_similarity(tokens, tokens) == 1.0

    def test_no_overlap(self):
        assert jaccard_similarity(["a", "b"], ["c", "d"]) == 0.0

    def test_partial_overlap(self):
        a = ["patient", "has", "diabetes"]
        b = ["patient", "has", "hypertension"]
        assert jaccard_similarity(a, b) == pytest.approx(0.5)

    def test_both_empty(self):
        assert jaccard_similarity([], []) == 0.0


class TestWordOverlapRatio:
    def test_full_overlap(self):
        ev = ["bmi", "is", "32"]
        notes = ["patient", "bmi", "is", "32", "kg"]
        assert word_overlap_ratio(ev, notes) == 1.0

    def test_no_overlap(self):
        assert word_overlap_ratio(["foo", "bar"], ["baz", "qux"]) == 0.0

    def test_empty_evidence(self):
        assert word_overlap_ratio([], ["some", "notes"]) == 0.0


class TestWeightedWordOverlap:
    def test_non_matching_tokens_reduce_score(self):
        # When evidence tokens don't appear in notes, score should drop
        notes = ["patient", "has", "diabetes", "and", "hypertension"]
        ev_full_match = ["diabetes", "hypertension"]
        ev_partial = ["diabetes", "cancer"]
        score_full = weighted_word_overlap(ev_full_match, notes)
        score_partial = weighted_word_overlap(ev_partial, notes)
        assert score_full > score_partial

    def test_empty_inputs(self):
        assert weighted_word_overlap([], ["a"]) == 0.0
        assert weighted_word_overlap(["a"], []) == 0.0


class TestLongestCommonSubstringRatio:
    def test_exact_match(self):
        assert longest_common_substring_ratio("BMI is 32", "BMI is 32") == 1.0

    def test_substring_present(self):
        ratio = longest_common_substring_ratio("BMI is 32", "Patient BMI is 32.4 kg/m2")
        assert ratio > 0.8

    def test_no_match(self):
        ratio = longest_common_substring_ratio("qqq zzz xxx", "patient has diabetes")
        assert ratio < 0.2

    def test_empty_evidence(self):
        assert longest_common_substring_ratio("", "some notes") == 0.0


class TestTfidfCosineSimilarity:
    def test_identical_text(self):
        text = "patient has documented history of hypertension"
        assert tfidf_cosine_similarity(text, text) == pytest.approx(1.0, abs=0.01)

    def test_unrelated_text(self):
        score = tfidf_cosine_similarity(
            "quantum physics experiment results",
            "patient BMI is 32 with history of diabetes"
        )
        assert score < 0.3

    def test_empty_strings(self):
        assert tfidf_cosine_similarity("", "some text") == 0.0


class TestExtractFeatures:
    def test_output_shape(self):
        features = extract_features("BMI is 32", ["Patient BMI is 32.4 kg/m2"])
        assert features.shape == (7,)

    def test_real_evidence_scores_high(self):
        notes = ["Patient presents with BMI of 32.4, history of type 2 diabetes and hypertension."]
        real_evidence = "BMI of 32.4, history of type 2 diabetes"
        fake_evidence = "MRI shows herniation at L4-L5 with nerve compression"

        real_features = extract_features(real_evidence, notes)
        fake_features = extract_features(fake_evidence, notes)

        # Real evidence should have higher TF-IDF cosine (feature 0) and word overlap (feature 2)
        assert real_features[0] > fake_features[0]  # tfidf_cosine
        assert real_features[2] > fake_features[2]  # word_overlap

    def test_no_nan_or_inf(self):
        features = extract_features("test evidence", ["some visit notes here"])
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))
