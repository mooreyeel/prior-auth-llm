from typing import Literal

from pydantic import BaseModel


class Question(BaseModel):
    type: Literal["text", "boolean"]
    key: str
    content: str
    visible_if: str | None = None


class QuestionSet(BaseModel):
    name: str
    questions: list[Question]


class Answer(BaseModel):
    question: Question
    value: str | bool
    confidence: float | None = None  # 0.0 to 1.0, how certain LLM is
    evidence: str | None = (
        None  # Quote from visit notes supporting the answer. If it does not match any quote, needs review = 1
    )
    needs_review: bool = False  # True if confidence < threshold or no evidence found


class Prescription(BaseModel):
    medication: str
    dosage: str
    frequency: str
    duration: str


class Patient(BaseModel):
    first_name: str
    last_name: str
    date_of_birth: str
    gender: str
    prescription: Prescription
    visit_notes: list[str]


class AnswerInput(BaseModel):
    patient: Patient
    question_set: QuestionSet


class Summary(BaseModel):
    """New, Summary statistics for the answer set."""

    total_questions: int
    hidden_by_conditions: int  # Questions removed by visible_if conditions
    high_confidence: int  # Answers with confidence >= 0.7
    needs_review: int  # Answers flagged for human review
    needs_review_keys: list[str]  # Question keys that need human review


class AnswerOutput(BaseModel):
    answers: list[Answer]
    completeness_score: float
    summary: Summary
