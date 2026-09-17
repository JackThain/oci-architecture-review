"""Domain model and ports. No cloud SDK or web framework imports in this file."""
from typing import Literal, Protocol

from pydantic import BaseModel, Field, field_validator

# The clouds this reviewer knows how to talk about. The key is what goes in the
# CLOUD environment variable; the value is how the model is told to describe itself.
CLOUDS = {
    "oci": "Oracle Cloud Infrastructure (OCI)",
    "aws": "Amazon Web Services (AWS)",
}
DEFAULT_CLOUD = "oci"


class Standard(BaseModel):
    id: str  # e.g. "SEC-02", taken from the file name
    text: str


class Finding(BaseModel):
    pillar: Literal["security", "reliability", "performance", "cost", "operations"]
    severity: Literal["high", "medium", "low"]
    issue: str
    recommendation: str
    standard_id: str | None = None  # which organisation standard this relates to

    @field_validator("pillar", "severity", mode="before")
    @classmethod
    def _lowercase(cls, value):
        # Models sometimes answer "High" or "Security"; normalise before validating.
        return value.strip().lower() if isinstance(value, str) else value


class Review(BaseModel):
    summary: str
    cloud_services: list[str]
    findings: list[Finding]
    mermaid: str
    standards_considered: list[str] = Field(default_factory=list)  # set by our code
    cloud: str = DEFAULT_CLOUD  # set by our code: which cloud this review targets


class LLMClient(Protocol):
    """Port: anything that can turn a prompt into text."""

    model_name: str

    def complete(self, system: str, user: str) -> str: ...


class StandardsRepository(Protocol):
    """Port: anything that can find the standards relevant to a design."""

    def relevant(self, description: str, k: int = 3) -> list[Standard]: ...
