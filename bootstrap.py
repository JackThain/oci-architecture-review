"""Composition root: the only place that decides which adapters are used."""
import os
from pathlib import Path

from dotenv import load_dotenv

from adapters import BedrockClient, FakeLLMClient, LocalStandards, OCIGenAIClient
from domain import CLOUDS, DEFAULT_CLOUD, LLMClient
from reviewer import ArchitectureReviewer

PROJECT_DIR = Path(__file__).parent
STANDARDS_DIR = PROJECT_DIR / "standards"

# Load settings from a local .env file if one exists. The file is git-ignored.
# Variables already set in the terminal take priority over the file.
load_dotenv(PROJECT_DIR / ".env")


def selected_cloud() -> str:
    """Which cloud this process targets. Set CLOUD=aws or CLOUD=oci."""
    cloud = os.getenv("CLOUD", DEFAULT_CLOUD).strip().lower()
    if cloud not in CLOUDS:
        raise ValueError(
            f"CLOUD={cloud!r} is not supported. Use one of: {', '.join(sorted(CLOUDS))}."
        )
    return cloud


def _model_id(cloud: str) -> str:
    """A per-cloud model ID if set, otherwise the shared MODEL_ID.

    This lets you flip between clouds by changing CLOUD alone."""
    specific = os.getenv(f"{cloud.upper()}_MODEL_ID")
    if specific:
        return specific
    try:
        return os.environ["MODEL_ID"]
    except KeyError:
        raise KeyError(
            f"Set {cloud.upper()}_MODEL_ID or MODEL_ID to the model you want to use."
        ) from None


def build_llm(cloud: str | None = None) -> LLMClient:
    cloud = cloud or selected_cloud()
    if os.getenv("USE_FAKE_LLM") == "1":
        return FakeLLMClient(cloud)
    if cloud == "aws":
        return BedrockClient(
            model_id=_model_id("aws"),
            region=os.getenv("AWS_REGION") or None,  # else ~/.aws/config
        )
    return OCIGenAIClient(
        compartment_id=os.environ["COMPARTMENT_ID"],
        model_id=_model_id("oci"),
        region=os.getenv("GENAI_REGION") or None,  # optional override
    )


def build_reviewer(llm: LLMClient | None = None) -> ArchitectureReviewer:
    cloud = selected_cloud()
    return ArchitectureReviewer(llm or build_llm(cloud), LocalStandards(STANDARDS_DIR), cloud)
