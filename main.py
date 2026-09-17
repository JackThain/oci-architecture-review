"""Web layer: HTTP concerns and audit logging only."""
import json
import logging
import time
import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from bootstrap import build_reviewer
from domain import Review
from reviewer import GuardrailError

logging.basicConfig(level=logging.INFO, format="%(message)s")
audit_log = logging.getLogger("audit")
error_log = logging.getLogger("app")

app = FastAPI(title="Cloud Architecture Review Assistant")
reviewer = build_reviewer()


class ReviewRequest(BaseModel):
    description: str


def audit(request_id: str, outcome: str, started: float, **details) -> None:
    """One structured log line per request. Metadata only, never the design text."""
    audit_log.info(json.dumps({
        "event": "architecture_review",
        "request_id": request_id,
        "outcome": outcome,
        "cloud": reviewer.cloud,
        "model": reviewer.llm.model_name,
        "latency_ms": round((time.perf_counter() - started) * 1000),
        **details,
    }))


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "cloud": reviewer.cloud, "model": reviewer.llm.model_name}


@app.post("/review", response_model=Review)
def review(req: ReviewRequest) -> Review:
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    try:
        result = reviewer.review(req.description)
    except GuardrailError as err:
        audit(request_id, "blocked_by_guardrail", started)
        raise HTTPException(status_code=400, detail=str(err))
    except ValueError as err:  # includes Pydantic validation errors
        audit(request_id, "invalid_model_output", started)
        raise HTTPException(status_code=502, detail=f"Model gave an unusable answer: {err}")
    except Exception:  # e.g. throttling or auth errors from either cloud
        error_log.exception("Model call failed")
        audit(request_id, "model_call_failed", started)
        raise HTTPException(status_code=503, detail="Model service unavailable. Try again shortly.")

    audit(
        request_id, "ok", started,
        input_chars=len(req.description),
        standards_considered=result.standards_considered,
        findings=len(result.findings),
        high_severity=sum(f.severity == "high" for f in result.findings),
    )
    return result
