import json
import logging
import re
from pathlib import Path

from openai import AsyncOpenAI

from app.config import (
    CONFIDENCE_THRESHOLD,
    FALSY_VALUES,
    LLM_MODEL,
    NOT_FOUND_VALUE,
    OPENAI_API_KEY,
    TRUTHY_VALUES,
)
from app.models import Answer, AnswerInput, AnswerOutput, Question, Summary

logger = logging.getLogger(__name__)

# Load the system prompt once
PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "prior_auth_answer.txt"
SYSTEM_PROMPT = PROMPT_PATH.read_text()

# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------


def _format_patient_context(data: AnswerInput) -> str:
    """Format patient data into a readable context string for the LLM."""
    patient = data.patient
    prescription = patient.prescription

    context = f"""## Patient Information
- Name: {patient.first_name} {patient.last_name}
- Date of Birth: {patient.date_of_birth}
- Gender: {patient.gender}

## Current Prescription
- Medication: {prescription.medication}
- Dosage: {prescription.dosage}
- Frequency: {prescription.frequency}
- Duration: {prescription.duration}

## Visit Notes"""
    for i, note in enumerate(patient.visit_notes, 1):
        context += f"\n### Visit {i}\n{note}\n"

    return context


def _format_questions(questions: list[Question]) -> str:
    """Format questions into a clear list for the LLM."""
    lines = ["## Questions to Answer\n"]
    for q in questions:
        q_type = "boolean (true/false)" if q.type == "boolean" else "text"
        lines.append(f"- key: {q.key}")
        lines.append(f"  type: {q_type}")
        lines.append(f"  question: {q.content}")
        lines.append("")
    return "\n".join(lines)


def _build_question_lookup(questions: list[Question]) -> dict[str, Question]:
    """Create a lookup dict from question key to Question object."""
    return {q.key: q for q in questions}


# --------------------------------------------------------------------------
# Validation of inputs and outputs
# --------------------------------------------------------------------------


def _coerce_boolean(value: str | bool) -> bool:
    """
    Coerce various boolean-like values to actual booleans.

    LLMs sometimes return "yes"/"no" instead of true/false.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() in TRUTHY_VALUES:
            return True
        if value.lower() in FALSY_VALUES:
            return False
    # If we can't coerce, return as-is and let validation catch it
    return value


def _verify_evidence(evidence: str | None, visit_notes: list[str]) -> bool:
    """
    Check if the evidence actually exists in the visit notes.

    Returns True if evidence is found (or evidence is None/empty).
    Returns False if evidence is provided but not found in notes.
    """
    if not evidence:
        return True  # No evidence to verify

    # Combine all visit notes into one searchable string
    all_notes = " ".join(visit_notes).lower()
    evidence_lower = evidence.lower()

    # Check if evidence appears in notes (allowing for minor differences)
    # We check if most words from evidence appear in notes from a past visit
    # Issue is visits aren't dated, which definitely could complicate things here, but I won't fix that
    evidence_words = set(evidence_lower.split())
    if len(evidence_words) < 3:
        # Short evidence - check exact match
        return evidence_lower in all_notes

    # For longer evidence, check if 70% of words appear
    words_found = sum(1 for word in evidence_words if word in all_notes)
    return words_found / len(evidence_words) >= 0.7


def _validate_input(data: AnswerInput) -> None:
    """
    Validate input data before processing.

    Raises ValueError if validation fails.
    """
    if not data.question_set.questions:
        raise ValueError("No questions provided")

    if not data.patient.visit_notes:
        raise ValueError("No visit notes provided")

    # Could add more validation here, but pydantic does most for us already:
    # - DOB format validation
    # - Required fields check
    # - etc.


# --------------------------------------------------------------------------
# Process Conditional Questions
# 1) Do you take any meds -> 2) How long have you taken those meds
# If 1 is false, we don't look at 2
# --------------------------------------------------------------------------


def _parse_visible_if(condition: str) -> list[tuple[str, str, str]]:
    """
    Parse a visible_if condition string into a list of (key, operator, value) tuples.
    This is so after the LLM API call, we can parse out what to skip or not

    Examples:
        "{continuation} = true" -> [("continuation", "=", "true")]
        "{continuation} = true and {cont_less_6m} = false" -> [("continuation", "=", "true"), ("cont_less_6m", "=", "false")]
    """

    # Split by "and" to handle compound conditions
    parts = condition.split(" and ")
    conditions = []

    for part in parts:
        # Match pattern: {key} = value
        match = re.match(r"\{(\w+)\}\s*=\s*(\w+)", part.strip())
        if match:
            key = match.group(1)
            value = match.group(2).lower()
            conditions.append((key, "=", value))

    return conditions


def _check_visible_if(condition: str, answer_lookup: dict[str, Answer]) -> bool:
    """
    Check if a visible_if condition is satisfied based on existing answers.

    Returns True if the question should be visible (condition met).
    Returns False if the question should be hidden (condition not met).
    """
    if not condition:
        return True  # No condition means always visible

    parsed = _parse_visible_if(condition)
    if not parsed:
        logger.warning(f"Could not parse visible_if condition: {condition}")
        return True  # If we can't parse, show the question

    for key, operator, expected_value in parsed:
        if key not in answer_lookup:
            # Referenced answer doesn't exist yet, hide the question
            return False

        answer = answer_lookup[key]
        actual_value = str(answer.value).lower()

        if operator == "=":
            if actual_value != expected_value:
                return False

    return True  # All conditions passed


def _filter_visible_answers(answers: list[Answer]) -> list[Answer]:
    """
    Filter answers based on visible_if conditions.

    Questions with visible_if conditions are only included if
    the referenced answers satisfy the condition.
    """
    # Build lookup of answers by question key
    answer_lookup = {a.question.key: a for a in answers}

    visible_answers = []
    for answer in answers:
        condition = answer.question.visible_if
        if _check_visible_if(condition, answer_lookup):
            visible_answers.append(answer)
        else:
            logger.debug(
                f"Hiding answer for {answer.question.key} due to visible_if: {condition}"
            )

    return visible_answers


# --------------------------------------------------------------------------
# LLM Summary
# --------------------------------------------------------------------------


def _calculate_completeness(
    answers: list[Answer], hidden_by_conditions: int = 0
) -> tuple[float, Summary]:
    """
    Calculate completeness score and summary statistics.

    Returns:
        Tuple of (completeness_score, summary)
    """
    if not answers:
        return 0.0, Summary(
            total_questions=0,
            hidden_by_conditions=0,
            high_confidence=0,
            needs_review=0,
            needs_review_keys=[],
        )

    high_confidence = sum(
        1 for a in answers if a.confidence and a.confidence >= CONFIDENCE_THRESHOLD
    )
    needs_review_keys = [a.question.key for a in answers if a.needs_review]

    completeness = round(high_confidence / len(answers), 2)

    summary = Summary(
        total_questions=len(answers),
        hidden_by_conditions=hidden_by_conditions,
        high_confidence=high_confidence,
        needs_review=len(needs_review_keys),
        needs_review_keys=needs_review_keys,
    )

    return completeness, summary


# --------------------------------------------------------------------------
# Main LLM Processing Function
# --------------------------------------------------------------------------


def _parse_llm_response(
    response_text: str,
    question_lookup: dict[str, Question],
    visit_notes: list[str],
) -> list[Answer]:
    """
    Parse the LLM's JSON response into Answer objects.

    Handles validation and applies the needs_review flag based on:
    - Confidence below threshold
    - Missing evidence
    - Evidence not found in visit notes
    - Value is "Not found in records"
    """
    # Clean up response - sometimes LLMs wrap JSON in markdown code blocks
    text = response_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        raw_answers = json.loads(text)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        logger.error(f"Response was: {text[:500]}")
        raise ValueError(f"LLM returned invalid JSON: {e}") from e

    answers = []
    for raw in raw_answers:
        key = raw.get("key")
        if key not in question_lookup:
            logger.warning(f"LLM returned unknown question key: {key}")
            continue

        question = question_lookup[key]
        confidence = raw.get("confidence", 0.0)
        evidence = raw.get("evidence")
        value = raw.get("value")

        # Type coercion for boolean questions
        if question.type == "boolean":
            value = _coerce_boolean(value)

        # If value is "Not found in records", confidence is meaningless - set to None
        if value == NOT_FOUND_VALUE:
            confidence = None

        # Verify evidence exists in visit notes
        evidence_verified = _verify_evidence(evidence, visit_notes)
        if not evidence_verified:
            logger.warning(f"Evidence not found in visit notes for key: {key}")

        # Determine if this answer needs human review
        needs_review = (
            confidence is None  # Not found in records
            or confidence < CONFIDENCE_THRESHOLD
            or evidence is None
            or value == NOT_FOUND_VALUE
            or not evidence_verified  # Flag if evidence doesn't match
        )

        answer = Answer(
            question=question,
            value=value,
            confidence=confidence,
            evidence=evidence,
            needs_review=needs_review,
        )
        answers.append(answer)

    return answers


async def generate_answers(data: AnswerInput) -> AnswerOutput:
    """
    Generate answers to prior authorization questions using OpenAI.

    Args:
        data: The input containing patient info and questions

    Returns:
        AnswerOutput with answers, completeness score, and summary

    Raises:
        ValueError: If API key is missing, input invalid, or LLM returns invalid response
    """
    # AUDIT: Log request (in production, would log to database with timestamp, user ID)
    logger.info(
        f"AUDIT: Request for patient {data.patient.first_name} {data.patient.last_name}"
    )
    # Input validation
    _validate_input(data)

    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    # Use async client so the event loop isn't blocked while waiting for the API response
    client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    # Build the prompt
    patient_context = _format_patient_context(data)
    questions_text = _format_questions(data.question_set.questions)

    user_prompt = f"""{patient_context}

{questions_text}
"""

    # Log what we're sending to the LLM
    logger.info(f"Calling OpenAI with {len(data.question_set.questions)} questions")
    logger.debug(f"=== SYSTEM PROMPT ===\n{SYSTEM_PROMPT}")
    logger.debug(f"=== USER PROMPT ===\n{user_prompt}")

    try:
        response = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # Low temperature for consistent, factual answers
        )
        response_text = response.choices[0].message.content
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        raise ValueError(f"Failed to generate answers: {e}") from e

    # Log what we got back
    logger.debug(f"=== LLM RESPONSE ===\n{response_text}")

    # Parse the LLM response
    question_lookup = _build_question_lookup(data.question_set.questions)
    answers = _parse_llm_response(
        response_text,
        question_lookup,
        data.patient.visit_notes,  # Pass visit notes for evidence verification
    )

    # Filter based on visible_if conditions
    visible_answers = _filter_visible_answers(answers)
    hidden_by_conditions = len(answers) - len(visible_answers)
    logger.info(
        f"Filtered {len(answers)} answers to {len(visible_answers)} visible answers"
    )

    # Calculate completeness score and summary
    completeness, summary = _calculate_completeness(visible_answers, hidden_by_conditions)

    # AUDIT: Log response summary (in production, would include full request/response)
    logger.info(
        f"AUDIT: Generated {len(visible_answers)} answers, completeness: {completeness:.2f}"
    )

    return AnswerOutput(
        answers=visible_answers,
        completeness_score=completeness,
        summary=summary,
    )
