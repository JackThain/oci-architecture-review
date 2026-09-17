"""Composition root: the only place that decides which adapters are used."""
import os
from pathlib import Path

from dotenv import load_dotenv

from adapters import FakeLLMClient, LocalStandards, OCIGenAIClient
from domain import LLMClient
from reviewer import ArchitectureReviewer

PROJECT_DIR = Path(__file__).parent
STANDARDS_DIR = PROJECT_DIR / "standards"

# Load settings from a local .env file if one exists. The file is git-ignored.
# Variables already set in the terminal take priority over the file.
load_dotenv(PROJECT_DIR / ".env")


def build_llm() -> LLMClient:
    if os.getenv("USE_FAKE_LLM") == "1":
        return FakeLLMClient()
    return OCIGenAIClient(
        compartment_id=os.environ["COMPARTMENT_ID"],
        model_id=os.environ["MODEL_ID"],
        region=os.getenv("GENAI_REGION") or None,  # optional override
    )


def build_reviewer(llm: LLMClient | None = None) -> ArchitectureReviewer:
    return ArchitectureReviewer(llm or build_llm(), LocalStandards(STANDARDS_DIR))
