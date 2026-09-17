"""Adapters: concrete implementations of the ports in domain.py."""
import json
import re
from pathlib import Path

from domain import Standard

# Both cloud SDKs are optional. Importing them here (rather than inside the
# methods) keeps the request path clean, while still letting someone install
# only the SDK for the cloud they actually use.
try:
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
except ImportError:  # pragma: no cover - exercised only on an AWS-only install
    oci = None

try:
    import boto3
    from botocore.config import Config as BotoConfig
except ImportError:  # pragma: no cover - exercised only on an OCI-only install
    boto3 = None


# ---------- LLM adapters ----------

class OCIGenAIClient:
    """Calls an on-demand chat model in OCI Generative AI."""

    def __init__(self, compartment_id: str, model_id: str, region: str | None = None):
        if oci is None:
            raise RuntimeError("The OCI SDK is not installed. Run: pip install oci")
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


class BedrockClient:
    """Calls a chat model in Amazon Bedrock through the Converse API.

    Converse gives every Bedrock model the same request shape, so this adapter
    works for Anthropic, Meta, Amazon and Mistral models without branching."""

    def __init__(self, model_id: str, region: str | None = None):
        if boto3 is None:
            raise RuntimeError("The AWS SDK is not installed. Run: pip install boto3")
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=region,  # None = fall back to ~/.aws/config or AWS_REGION
            # "adaptive" adds client-side rate limiting on top of retries, which
            # behaves better than pure backoff when the account is being throttled.
            config=BotoConfig(retries={"max_attempts": 8, "mode": "adaptive"}),
        )
        self.model_name = model_id

    def complete(self, system: str, user: str) -> str:
        response = self.client.converse(
            modelId=self.model_name,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"maxTokens": 2000, "temperature": 0.2},
        )
        return response["output"]["message"]["content"][0]["text"]


# Canned answers for the fake client, one per cloud, so an offline demo still
# shows plausible service names for whichever cloud is selected.
_FAKE_ANSWERS = {
    "oci": {
        "summary": "Run the API on Container Instances in private subnets behind a "
                   "Load Balancer, with Autonomous Database and cross-region DR.",
        "cloud_services": [
            "Load Balancer - public entry point with TLS",
            "Container Instances - runs the API in a private subnet",
            "Autonomous Database - managed database with a private endpoint",
        ],
        "mermaid": "flowchart LR\n  User --> LB[Load Balancer] --> API[Container Instances]"
                   " --> DB[(Autonomous DB)]\n  DB -.-> DR[(Standby DB, second region)]",
    },
    "aws": {
        "summary": "Run the API on ECS Fargate in private subnets behind an Application "
                   "Load Balancer, with Aurora PostgreSQL and a cross-region replica.",
        "cloud_services": [
            "Application Load Balancer - public entry point with TLS",
            "ECS Fargate - runs the API in a private subnet",
            "Aurora PostgreSQL - managed database reachable only inside the VPC",
        ],
        "mermaid": "flowchart LR\n  User --> ALB[Application Load Balancer] --> API[ECS Fargate]"
                   " --> DB[(Aurora PostgreSQL)]\n  DB -.-> DR[(Global Database, second region)]",
    },
}

_FAKE_FINDINGS = [
    {
        "pillar": "security",
        "severity": "high",
        "issue": "The database has a public IP address.",
        "recommendation": "Use a private endpoint and give analysts access via a bastion.",
        "standard_id": "SEC-02",
    },
    {
        "pillar": "reliability",
        "severity": "high",
        "issue": "Tier-1 system runs on a single VM in one region.",
        "recommendation": "Add a standby in a second region with managed replication.",
        "standard_id": "REL-01",
    },
]


class FakeLLMClient:
    """For tests and offline demos: no cloud calls, no cost, same answer every time."""

    def __init__(self, cloud: str = "oci"):
        self.cloud = cloud if cloud in _FAKE_ANSWERS else "oci"
        self.model_name = f"fake-{self.cloud}"

    def complete(self, system: str, user: str) -> str:
        return json.dumps({**_FAKE_ANSWERS[self.cloud], "findings": _FAKE_FINDINGS})


# ---------- Standards adapter ----------

STOP_WORDS = {"must", "with", "from", "that", "this", "they", "their", "only", "have",
              "into", "each", "other", "client", "data", "system", "team"}


def _keywords(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9-]{4,}", text.lower())
    return {word.removesuffix("s") for word in words} - STOP_WORDS


class LocalStandards:
    """MVP: markdown files ranked by keyword overlap.
    Swap for vector search (e.g. OpenSearch or Autonomous Database) without
    changing the reviewer."""

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
