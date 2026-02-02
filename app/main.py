import logging
import os

from fastapi import FastAPI

from app.env import setup_env
from app.models import AnswerInput, AnswerOutput
from app.services.answer_generator import generate_answers

setup_env()

# Configure logging - set to DEBUG to see LLM input/output
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

# Initialize the FastAPI application
app = FastAPI(
    title="Prior Authorization API",
    description="API for generating answers to prior authorization questions using patient data",
    version="1.0.0",
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "Prior Authorization API is running",
        "status": "healthy",
    }


@app.post("/answers")
async def get_answers(data: AnswerInput) -> AnswerOutput:
    """
    Generate answers to prior authorization questions based on patient data.

    This endpoint accepts patient information and a list of questions,
    then uses LLM to generate appropriate answers based on the patient's
    medical history, current medications, and other relevant data.

    Returns answers with confidence scores and evidence citations.
    Answers with low confidence are flagged for human review.
    """

    return await generate_answers(data)
