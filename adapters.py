"""Adapters: concrete implementations of the ports in domain.py."""
import json
import re
from pathlib import Path

import oci
from oci.generative_ai_inference import GenerativeAiInferenceClient
from oci.generative_ai_inference.models import (
    ChatDetails,
    GenericChatRequest,
    OnDemandServingMode,
    SystemMessage,
    TextContent,
    UserMessage,
)

from domain import Standard


# ---------- LLM adapters ----------

class OCIGenAIClient:
    """Calls an on-demand chat model in OCI Generative AI."""

    def __init__(self, compartment_id: str, model_id: str, region: str | None = None):
        config = oci.config.from_file()  # reads ~/.oci/config
        region = region or config["region"]
        endpoint = f"https://inference.generativeai.{region}.oci.oraclecloud.com"
        self.client = GenerativeAiInferenceClient(
            config,
            service_endpoint=endpoint,
            retry_strategy=oci.retry.DEFAULT_RETRY_STRATEGY,  # backs off on 429 throttling
        )
        self.compartment_id = compartment_id
        self.model_name = model_id

    def complete(self, system: str, user: str) -> str:
        chat_request = GenericChatRequest(
            api_format="GENERIC",
            messages=[
                SystemMessage(content=[TextContent(text=system)]),
                UserMessage(content=[TextContent(text=user)]),
            ],
            max_tokens=2000,  # also caps cost per review
            temperature=0.2,  # low = more consistent, structured answers
        )
        details = ChatDetails(
            compartment_id=self.compartment_id,
            serving_mode=OnDemandServingMode(model_id=self.model_name),
            chat_request=chat_request,
        )
        response = self.client.chat(details)
        return response.data.chat_response.choices[0].message.content[0].text


class FakeLLMClient:
    """For tests and offline demos: no cloud calls, no cost, same answer every time."""

    model_name = "fake"

    def complete(self, system: str, user: str) -> str:
        return json.dumps({
            "summary": "Run the API on Container Instances in private subnets behind a "
                       "Load Balancer, with Autonomous Database and cross-region DR.",
            "oci_services": [
                "Load Balancer - public entry point with TLS",
                "Container Instances - runs the API in a private subnet",
                "Autonomous Database - managed database with a private endpoint",
            ],
            "findings": [
                {
                    "pillar": "security",
                    "severity": "high",
                    "issue": "The database has a public IP address.",
                    "recommendation": "Use a private endpoint and give analysts access via Bastion.",
                    "standard_id": "SEC-02",
                },
                {
                    "pillar": "reliability",
                    "severity": "high",
                    "issue": "Tier-1 system runs on a single VM in one region.",
                    "recommendation": "Add a standby in a second region using Autonomous Data Guard.",
                    "standard_id": "REL-01",
                },
            ],
            "mermaid": "flowchart LR\n  User --> LB[Load Balancer] --> API[Container Instances]"
                       " --> DB[(Autonomous DB)]\n  DB -.-> DR[(Standby DB, second region)]",
        })


# ---------- Standards adapter ----------

STOP_WORDS = {"must", "with", "from", "that", "this", "they", "their", "only", "have",
              "into", "each", "other", "client", "data", "system", "team"}


def _keywords(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9-]{4,}", text.lower())
    return {word.removesuffix("s") for word in words} - STOP_WORDS


class LocalStandards:
    """MVP: markdown files ranked by keyword overlap.
    Swap for vector search (e.g. Autonomous Database) without changing the reviewer."""

    def __init__(self, folder: str | Path):
        self.standards = [
            Standard(id=path.stem, text=path.read_text(encoding="utf-8"))
            for path in sorted(Path(folder).glob("*.md"))
        ]
        if not self.standards:
            raise FileNotFoundError(f"No standards (*.md files) found in {folder}")

    def relevant(self, description: str, k: int = 3) -> list[Standard]:
        wanted = _keywords(description)
        scored = [(len(wanted & _keywords(s.text)), s) for s in self.standards]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [standard for score, standard in scored[:k] if score > 0]
