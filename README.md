# Prior Authorization LLM Pipeline

An async FastAPI backend that uses LLMs to automatically complete prior authorization forms by extracting clinical information from patient visit notes.

## What it does

Healthcare prior authorization requires medical staff to manually fill out lengthy insurance forms by cross-referencing patient records. This system automates that process:

1. Accepts patient demographics, visit notes, and a prior auth question set via POST request
2. Uses an LLM (GPT-4o by default, swappable) to extract answers from unstructured clinical notes
3. Validates every answer with evidence verification — cited text must actually appear in the visit notes (70% word match threshold)
4. Applies confidence scoring (0.0-1.0) and flags low-confidence answers for human review
5. Evaluates conditional question logic (`visible_if` conditions) to skip irrelevant questions
6. Returns structured answers with confidence scores, evidence citations, and completeness metrics

## Architecture

```
POST /answers
  -> Input validation (Pydantic v2)
  -> Format patient context + questions
  -> LLM call (async, low temperature for consistency)
  -> Parse JSON response
  -> Boolean coercion (yes/no -> true/false)
  -> Evidence verification against visit notes
  -> Conditional question filtering (visible_if)
  -> Completeness scoring + summary stats
  -> Return structured AnswerOutput
```

Key design decisions:
- **Evidence grounding**: Every answer must cite text from the visit notes. If the citation doesn't match the source (70% word overlap), the answer is flagged for review. This prevents hallucinated answers from reaching clinical staff.
- **Confidence-based triage**: Answers below 0.7 confidence are automatically flagged for human review, reducing the review burden to only uncertain cases.
- **Conditional logic**: Questions can depend on previous answers (e.g., "how long on medication?" only shown if "continuing treatment?" is true), reducing form noise.

## Tech stack

- **Python 3.10+**, **FastAPI**, **Pydantic v2**
- **OpenAI** async client (supports GPT-4o, configurable to other providers)
- **Docker** for containerization
- **pytest** + **httpx** for testing
- **ruff** for linting/formatting
- **Pydantic Logfire** for observability (optional)

## Setup

```bash
# Install dependencies
uv sync

# Copy and fill in your API key
cp .env.example .env

# Run the server
uv run uvicorn app.main:app --reload

# Run tests
uv run pytest

# Generate synthetic patient data
uv run python scripts/generate_patient_data.py -n 10
```

## Usage

```bash
curl -X POST "http://localhost:8000/answers" \
  -H "Content-Type: application/json" \
  -d @sample_data/demo_request.json
```

API docs available at `http://localhost:8000/docs` when running.

## Sample data

All patient data in `sample_data/` is synthetic — generated using Faker + AI. No real patient information.

- `demo_request.json` — single complete request example
- `example_request.json` — minimal test case
- `patient_data.json` — 100+ synthetic patients with visit notes
- `zepbound_question_set.json` — 166 prior auth questions with conditional logic

## Tests

```bash
uv run pytest -v
```

Unit tests cover boolean coercion, evidence verification, and conditional parsing. Integration tests hit the full endpoint with sample data.
