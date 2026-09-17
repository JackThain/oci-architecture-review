import json

import pytest

from adapters import FakeLLMClient, LocalStandards
from bootstrap import STANDARDS_DIR
from reviewer import ArchitectureReviewer, GuardrailError

RISKY_DESIGN = (
    "Tier-1 client reporting app on a single VM in London. "
    "PostgreSQL database with a public IP so analysts can connect directly. "
    "Backups kept in the same region."
)


def make_reviewer(llm=None) -> ArchitectureReviewer:
    return ArchitectureReviewer(llm or FakeLLMClient(), LocalStandards(STANDARDS_DIR))


def test_retrieves_relevant_standards():
    ids = [s.id for s in LocalStandards(STANDARDS_DIR).relevant(RISKY_DESIGN)]
    assert "SEC-02" in ids
    assert "REL-01" in ids


def test_findings_cite_standards():
    review = make_reviewer().review(RISKY_DESIGN)
    assert {"SEC-02", "REL-01"} <= {f.standard_id for f in review.findings}


def test_invented_citations_are_removed():
    class InventsCitations(FakeLLMClient):
        def complete(self, system, user):
            data = json.loads(super().complete(system, user))
            data["findings"][0]["standard_id"] = "MADE-UP-99"
            return json.dumps(data)

    review = make_reviewer(InventsCitations()).review(RISKY_DESIGN)
    assert review.findings[0].standard_id is None


def test_guardrail_blocks_client_data():
    with pytest.raises(GuardrailError):
        make_reviewer().review(RISKY_DESIGN + " Contact jane.doe@example.com for access.")


def test_bad_model_output_is_rejected():
    class ReturnsProse(FakeLLMClient):
        def complete(self, system, user):
            return "Sorry, I can't help with that."

    with pytest.raises(ValueError):
        make_reviewer(ReturnsProse()).review(RISKY_DESIGN)
