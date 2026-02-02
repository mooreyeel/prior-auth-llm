import os

from dotenv import load_dotenv

# load keys
load_dotenv()

# LLM Settings
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o")
CONFIDENCE_THRESHOLD = 0.7  # Below this, flag for human review

# API Keys (loaded from environment)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# Magic Strings
NOT_FOUND_VALUE = "Not found in records"
TRUTHY_VALUES = {"true", "yes", "1", "y"}
FALSY_VALUES = {"false", "no", "0", "n"}
