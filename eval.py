"""Evaluation harness: does the reviewer catch issues we already know about?

Run with:  python eval.py   (uses the same environment variables as the app)
"""
import json
import time
from pathlib import Path

from bootstrap import build_reviewer

# Pause between cases so a batch run does not trip on-demand rate limits.
# A failure usually means we are already being throttled, so back off harder
# there than after a call that succeeded.
PAUSE_AFTER_OK = 3
PAUSE_AFTER_ERROR = 30

cases = json.loads((Path(__file__).parent / "golden_set.json").read_text(encoding="utf-8"))
reviewer = build_reviewer()

expected_total = caught_total = 0
for index, case in enumerate(cases):
    expected = set(case["expected_standards"])
    expected_total += len(expected)
    try:
        review = reviewer.review(case["description"])
    except Exception as err:
        print(f"{case['name']:<26} ERROR: {err}")
        pause = PAUSE_AFTER_ERROR
    else:
        cited = {f.standard_id for f in review.findings if f.standard_id}
        caught = expected & cited
        caught_total += len(caught)
        missed = ", ".join(sorted(expected - cited)) or "none"
        print(f"{case['name']:<26} caught {len(caught)}/{len(expected)}   missed: {missed}")
        pause = PAUSE_AFTER_OK

    if index < len(cases) - 1:  # nothing left to be gentle towards
        time.sleep(pause)

print(f"\nCloud: {reviewer.cloud}")
print(f"Model: {reviewer.llm.model_name}")
print(f"Recall on known issues: {caught_total / expected_total:.0%}")
