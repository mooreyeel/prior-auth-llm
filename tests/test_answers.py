import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models import AnswerInput, Patient, Question, QuestionSet
from app.services.answer_generator import (
    _coerce_boolean,
    _verify_evidence,
    _parse_visible_if,
)

client = TestClient(app)


# =============================================================================
# Unit Tests - Pure functions, no API calls, run in milliseconds
# =============================================================================


class TestCoerceBoolean:
    """Test boolean coercion for LLM output normalization."""

    def test_coerce_yes_to_true(self):
        assert _coerce_boolean("yes") is True
        assert _coerce_boolean("Yes") is True
        assert _coerce_boolean("YES") is True

    def test_coerce_no_to_false(self):
        assert _coerce_boolean("no") is False
        assert _coerce_boolean("No") is False
        assert _coerce_boolean("NO") is False

    def test_passthrough_actual_booleans(self):
        assert _coerce_boolean(True) is True
        assert _coerce_boolean(False) is False

    def test_coerce_string_true_false(self):
        assert _coerce_boolean("true") is True
        assert _coerce_boolean("false") is False


class TestVerifyEvidence:
    """Test evidence verification against visit notes."""

    def test_evidence_found_exact(self):
        notes = ["Patient BMI is 32.4 kg/m2"]
        assert _verify_evidence("BMI is 32.4", notes) is True

    def test_evidence_not_found(self):
        notes = ["Patient is healthy"]
        assert _verify_evidence("BMI is 32.4", notes) is False

    def test_empty_evidence_passes(self):
        notes = ["Patient is healthy"]
        assert _verify_evidence(None, notes) is True
        assert _verify_evidence("", notes) is True

    def test_evidence_across_multiple_notes(self):
        notes = ["Visit 1: Weight recorded", "Visit 2: BMI is 32.4"]
        assert _verify_evidence("BMI is 32.4", notes) is True

    def test_long_evidence_partial_match(self):
        notes = ["Patient has documented history of hypertension and obesity"]
        # 70% of words should match for longer evidence
        evidence = "documented history of hypertension and diabetes"
        # Most words match except "diabetes" vs "obesity"
        assert _verify_evidence(evidence, notes) is True


class TestParseVisibleIf:
    """Test visible_if condition parsing."""

    def test_simple_condition(self):
        result = _parse_visible_if("{continuation} = true")
        assert result == [("continuation", "=", "true")]

    def test_compound_condition(self):
        result = _parse_visible_if("{continuation} = true and {cont_less_6m} = false")
        assert result == [
            ("continuation", "=", "true"),
            ("cont_less_6m", "=", "false"),
        ]

    def test_empty_condition(self):
        result = _parse_visible_if("")
        assert result == []


# Create a fixture for loading test data from sample data directory
@pytest.fixture
def test_data():
    # Load patient data
    with open("sample_data/patient_data.json") as f:
        patients_data = json.load(f)

    # Load question set
    with open("sample_data/zepbound_question_set.json") as f:
        questions_data = json.load(f)

    # Use the first patient from the sample data and create Patient model
    patient = Patient(**patients_data[0])

    # Create Question models from the loaded data
    questions = [Question(**q) for q in questions_data]

    # Create QuestionSet model
    question_set = QuestionSet(name="Zepbound Prior Authorization", questions=questions)

    # Create AnswerInput model
    answer_input = AnswerInput(patient=patient, question_set=question_set)

    return answer_input


def test_get_answers(test_data):
    response = client.post("/answers", json=test_data.model_dump())
    assert response.status_code == 200
    assert "answers" in response.json()
    assert len(response.json()["answers"]) > 0


def test_health_check():
    """Test the health endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_response_has_completeness_score(test_data):
    """Test that response includes completeness_score and summary."""
    response = client.post("/answers", json=test_data.model_dump())
    data = response.json()

    assert "completeness_score" in data
    assert 0.0 <= data["completeness_score"] <= 1.0
    assert "summary" in data
    assert "total_questions" in data["summary"]
    assert "high_confidence" in data["summary"]
    assert "needs_review" in data["summary"]


def test_empty_questions_returns_error():
    """Test that empty questions list is rejected with ValueError."""
    payload = {
        "patient": {
            "first_name": "Test", "last_name": "User",
            "date_of_birth": "1990-01-01", "gender": "Male",
            "prescription": {"medication": "Test", "dosage": "1mg", "frequency": "daily", "duration": "1 week"},
            "visit_notes": ["Patient is healthy."]
        },
        "question_set": {"name": "Empty", "questions": []}
    }
    with pytest.raises(ValueError, match="No questions provided"):
        client.post("/answers", json=payload)
