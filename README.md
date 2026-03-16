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
- **Evidence grounding**: Every answer must cite text from the visit notes. A trained ML classifier (logistic regression on TF-IDF cosine, Jaccard overlap, weighted word overlap, longest common substring ratio) scores whether cited evidence is actually grounded in the notes — replacing the original hardcoded 70% word-overlap heuristic. Falls back to the heuristic if no checkpoint exists.
- **Confidence-based triage**: Answers below 0.7 confidence are automatically flagged for human review, reducing the review burden to only uncertain cases.
- **Conditional logic**: Questions can depend on previous answers (e.g., "how long on medication?" only shown if "continuing treatment?" is true), reducing form noise.

## ML: Evidence Quality Scorer

The `ml/` directory contains a trained classifier that replaces the naive word-overlap heuristic for evidence verification. See `ml/train_evidence_scorer.py` for the full pipeline.

```bash
# Train the evidence scorer (no API key needed — uses local sample data)
python -m ml.train_evidence_scorer --C 0.001 0.01 0.1 1.0 10.0 100.0 --folds 5 --seed 42
```

- **Task**: Binary classification — is this cited evidence actually grounded in the visit notes?
- **Features**: TF-IDF cosine similarity, Jaccard overlap, IDF-weighted word overlap, longest common substring ratio, evidence length ratio
- **Model**: LogisticRegression (chosen over Random Forest for calibrated probabilities and interpretable coefficients; over SVM for same reasons plus faster inference)
- **Training data**: Generated locally from synthetic patient visit notes — positive pairs from real sentences, negatives from wrong-patient sentences, shuffled tokens, and fabricated medical text
- **Evaluation**: 5-fold stratified CV, optimized on AUROC
- **Results**: CV AUROC 0.999, F1 0.975, log loss 0.123

## Tech stack

- **Python 3.10+**, **FastAPI**, **Pydantic v2**
- **scikit-learn** for evidence quality classifier
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
