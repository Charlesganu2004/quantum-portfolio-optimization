"""The LLM review step -- and the wall around it.

The reviewer sees a candidate portfolio plus retrieved evidence, and returns a
structured verdict. What it is *not* allowed to do is the point of this module:

* It never sees or produces order quantities, prices, or account balances.
  Those are computed by deterministic code and are not in its prompt.
* Its output cannot alter a risk limit.
* Every citation it makes is verified against the knowledge base; a fabricated
  `source_id` raises rather than passing through.
* Failure of any kind during a review maps to ABSTAIN, never to APPROVE. A model
  that is broken, unreachable, or confused must not be able to authorize a
  trade. The one exception sits outside the review call: `resolve_reviewer`
  falls back to `OfflineReviewer`, which can approve, when no model is reachable
  at startup. The result is labelled `offline_rules`.

`ReviewerClient` has three implementations. `OllamaReviewer` is the default on
this machine and constrains decoding with a JSON Schema. `LlamaServerReviewer`
talks to a local llama-server with an equivalent GBNF grammar. `OfflineReviewer`
is a deterministic rule-based stand-in so the pipeline and the ablation study run
on a machine with no model downloaded -- it is clearly labelled as such in every
result it produces, and is never presented as an LLM verdict.

Both model backends accept only an http endpoint whose host name is 127.0.0.1,
localhost or ::1, with no embedded credentials. Portfolio state never leaves the
machine and no token is billed.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

import requests

from quantlab.knowledge.formulas import FORMULA_INDEX
from quantlab.knowledge.records import KnowledgeRecord

from .schema import (
    Action,
    Evidence,
    FabricatedCitation,
    ReviewerOutput,
    SchemaViolation,
    VERDICT_GRAMMAR,
    VERDICT_JSON_SCHEMA,
    Verdict,
    parse_reviewer_output,
    require_evidence_for_approval,
    verify_citations,
)

DEFAULT_ENDPOINT = "http://127.0.0.1:8080"
OLLAMA_ENDPOINT = "http://127.0.0.1:11434"

#: Hostnames a reviewer endpoint may use. A prefix test on the URL string is not
#: enough: "http://localhost@attacker.example" and "http://127.0.0.1.nip.io" both
#: start with an allowed prefix but contact another machine.
LOCAL_HOSTNAMES = frozenset({"127.0.0.1", "localhost", "::1"})


def require_local_endpoint(endpoint: str) -> None:
    """Raise ValueError unless `endpoint` is a plain http URL on this machine.

    The URL is parsed rather than prefix-matched: the scheme must be http, the
    URL must carry no user name or password, and the host name must be exactly
    one of `LOCAL_HOSTNAMES`.
    """
    try:
        parts = urlsplit(endpoint)
        hostname = parts.hostname
        parts.port  # noqa: B018 -- raises ValueError on a malformed port
    except ValueError:
        hostname = None
        parts = None
    if (
        parts is None
        or parts.scheme != "http"
        or parts.username is not None
        or parts.password is not None
        or hostname not in LOCAL_HOSTNAMES
    ):
        raise ValueError(
            f"reviewer endpoint must be local, got {endpoint!r}. Sending "
            f"portfolio state off-machine is out of scope for this project."
        )

SYSTEM_PROMPT = """\
You are a risk reviewer for a portfolio optimization system. You are given a
candidate allocation produced by a deterministic optimizer, plus retrieved
reference material.

Your ONLY job is to return a JSON verdict object. You must:
- cite a source_id from the provided evidence for every factual claim you make
- return ABSTAIN when the evidence provided does not support a judgement
- report uncertainties honestly rather than resolving them silently

You must NOT:
- compute or suggest order sizes, share counts, prices, or dollar amounts
- assert that any outcome is guaranteed, certain, or risk-free
- cite a source_id that does not appear in the evidence given to you
- infer a person's reasoning or intent from an observed trade

CITATION FORMAT: copy each source_id EXACTLY as it appears on the "source_id:"
line, including the prefix before the colon, and add nothing around it. Write
"formula:sharpe-ratio" -- not "sharpe-ratio", and not "[formula:sharpe-ratio]".
An id that does not resolve is treated as fabricated.

Abstaining is a correct and expected outcome. It is preferred to guessing.\
"""


@dataclass(frozen=True)
class ReviewRequest:
    """Everything the reviewer is permitted to see."""

    candidate_id: str
    method: str
    symbols: tuple[str, ...]
    weights_pct: tuple[float, ...]
    expected_return: float
    expected_risk: float
    n_positions: int
    evidence: tuple[KnowledgeRecord, ...] = ()
    notes: tuple[str, ...] = ()

    def known_ids(self) -> set[str]:
        return {record.source_id for record in self.evidence}

    def to_prompt(self) -> str:
        holdings = "\n".join(
            f"  {symbol}: {weight:.1%}"
            for symbol, weight in zip(self.symbols, self.weights_pct)
            if abs(weight) > 1e-8
        )
        # Ids are rendered on their own labelled line rather than wrapped in
        # square brackets. Small models copy the delimiters they are shown, and
        # `[formula:sharpe-ratio]` was the single most common malformed citation
        # observed. Removing the brackets removes the thing being copied.
        evidence_block = (
            "\n\n".join(
                f"source_id: {record.source_id}\ntitle: {record.title}\n"
                f"{record.content.strip()}"
                for record in self.evidence
            )
            or "(no evidence retrieved)"
        )
        notes = "\n".join(f"  - {note}" for note in self.notes) or "  (none)"
        return f"""\
CANDIDATE: {self.candidate_id} (produced by method: {self.method})

Proposed allocation ({self.n_positions} positions):
{holdings}

Optimizer estimates (annualized, from deterministic code -- do not recompute):
  expected return: {self.expected_return:.2%}
  expected risk:   {self.expected_risk:.2%}

Notes:
{notes}

EVIDENCE AVAILABLE FOR CITATION:
{evidence_block}

Return only the JSON verdict object."""


@dataclass
class ReviewResult:
    """A verdict plus how it was reached."""

    output: ReviewerOutput
    source: str
    raw_response: str = ""
    error: str | None = None
    #: True when every citation resolved against the knowledge base, False when
    #: the check found one that does not exist, and None when no output was
    #: checked (the model was unreachable or its output failed the schema).
    citations_verified: bool | None = None
    #: Citations that resolved only after a dropped namespace prefix was
    #: repaired. Recorded so the report can state how often it happened rather
    #: than the repair being invisible.
    repaired_citations: tuple[str, ...] = ()
    #: True only when a model backend returned text for this review. False for an
    #: unreachable server, a malformed HTTP response and the offline stand-in. A
    #: schema or citation failure still counts as a response: the model answered,
    #: and its answer was refused.
    model_responded: bool = False

    @property
    def permits_trading(self) -> bool:
        return self.output.permits_trading and self.citations_verified is True


def _abstain(
    candidate_id: str,
    source: str,
    reason: str,
    *,
    citations_verified: bool | None = None,
    model_responded: bool = False,
) -> ReviewResult:
    """Every failure path lands here. Failure must never authorize a trade.

    `citations_verified` is None unless a citation check actually ran, and False
    when it ran and failed. `model_responded` is True only when the backend
    returned text that was then refused.
    """
    return ReviewResult(
        output=ReviewerOutput(
            verdict=Verdict.ABSTAIN,
            candidate_id=candidate_id,
            risk_observations=(f"review unavailable: {reason}",),
            uncertainties=("no reviewer judgement was obtained",),
            recommended_action=Action.NO_TRADE,
        ),
        source=source,
        error=reason,
        citations_verified=citations_verified,
        model_responded=model_responded,
    )


def _validate_response(
    content: str, request: ReviewRequest, source: str
) -> ReviewResult:
    """Parse, schema-check, and verify citations. Shared by every model backend.

    Constrained decoding guarantees *shape*; it cannot know whether a cited
    `source_id` refers to anything real. So every response is re-validated here
    regardless of which backend produced it, and any failure abstains.
    """
    try:
        output = parse_reviewer_output(json.loads(content))
        require_evidence_for_approval(output)
    except (json.JSONDecodeError, SchemaViolation) as exc:
        result = _abstain(
            request.candidate_id,
            source,
            f"schema violation: {exc}",
            model_responded=True,
        )
        result.raw_response = content
        return result

    try:
        repaired = verify_citations(output, request.known_ids() | set(FORMULA_INDEX))
    except FabricatedCitation as exc:
        # Fatal to the verdict rather than a warning: a fabricated citation
        # invalidates the reasoning that produced it.
        result = _abstain(
            request.candidate_id,
            source,
            str(exc),
            citations_verified=False,
            model_responded=True,
        )
        result.raw_response = content
        return result

    return ReviewResult(
        output=output,
        source=source,
        raw_response=content,
        citations_verified=True,
        repaired_citations=repaired,
        model_responded=True,
    )


class ReviewerClient(ABC):
    name: str = "unnamed"

    @abstractmethod
    def review(self, request: ReviewRequest) -> ReviewResult:
        ...


class LlamaServerReviewer(ReviewerClient):
    """Local llama-server over its OpenAI-compatible endpoint.

    Local endpoints only (see `require_local_endpoint`). No API key leaves the
    machine and no token is billed -- the whole point of running the model
    locally.
    """

    name = "llama_server"

    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        model: str = "local",
        temperature: float = 0.0,
        timeout: float = 120.0,
    ) -> None:
        require_local_endpoint(endpoint)
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.timeout = timeout

    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.endpoint}/health", timeout=2.0)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def review(self, request: ReviewRequest) -> ReviewResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.to_prompt()},
            ],
            # Grammar-constrained decoding: malformed JSON is unrepresentable.
            "grammar": VERDICT_GRAMMAR,
        }

        try:
            response = requests.post(
                f"{self.endpoint}/v1/chat/completions",
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        except requests.RequestException as exc:
            return _abstain(request.candidate_id, self.name, f"server unreachable: {exc}")
        except (KeyError, IndexError, ValueError) as exc:
            return _abstain(request.candidate_id, self.name, f"malformed response: {exc}")

        return _validate_response(content, request, self.name)


class OllamaReviewer(ReviewerClient):
    """Local Ollama daemon, structured output constrained by JSON Schema.

    Ollama does not implement llama.cpp's GBNF grammars, so constrained decoding
    goes through its `format` field instead. The guarantee is equivalent -- the
    sampler cannot emit a non-conforming object -- but the mechanism differs, and
    `VERDICT_JSON_SCHEMA` is kept beside `VERDICT_GRAMMAR` so the two definitions
    cannot drift apart.

    Local endpoints only (see `require_local_endpoint`). Nothing leaves the
    machine and no token is billed.
    """

    name = "ollama"

    def __init__(
        self,
        model: str = "llama3.2:3b",
        endpoint: str = OLLAMA_ENDPOINT,
        temperature: float = 0.0,
        timeout: float = 180.0,
    ) -> None:
        require_local_endpoint(endpoint)
        self.model = model
        self.endpoint = endpoint.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

    def is_available(self) -> bool:
        return self.model in self.installed_models()

    def installed_models(self) -> list[str]:
        try:
            response = requests.get(f"{self.endpoint}/api/tags", timeout=3.0)
            response.raise_for_status()
            return [m["name"] for m in response.json().get("models", [])]
        except (requests.RequestException, ValueError, KeyError):
            return []

    def review(self, request: ReviewRequest) -> ReviewResult:
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "format": VERDICT_JSON_SCHEMA,
            "options": {"temperature": self.temperature},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": request.to_prompt()},
            ],
        }

        try:
            response = requests.post(
                f"{self.endpoint}/api/chat", json=payload, timeout=self.timeout
            )
            response.raise_for_status()
            content = response.json()["message"]["content"]
        except requests.RequestException as exc:
            return _abstain(request.candidate_id, self.name, f"ollama unreachable: {exc}")
        except (KeyError, ValueError) as exc:
            return _abstain(request.candidate_id, self.name, f"malformed response: {exc}")

        return _validate_response(content, request, self.name)


class OfflineReviewer(ReviewerClient):
    """Deterministic rule-based stand-in. **Not an LLM.**

    Exists so the pipeline and the with/without-review ablation can run on a
    machine with no model downloaded. Every result it emits is labelled
    `offline_rules` so it can never be mistaken for a model verdict in the report.

    Its rules encode the same conservatism the reviewer prompt asks for:
    concentration and unusual risk trigger abstention rather than approval.
    """

    name = "offline_rules"

    def __init__(
        self, max_single_weight: float = 0.45, max_expected_risk: float = 0.60
    ) -> None:
        self.max_single_weight = max_single_weight
        self.max_expected_risk = max_expected_risk

    def review(self, request: ReviewRequest) -> ReviewResult:
        observations: list[str] = []
        uncertainties = [
            "produced by deterministic rules, not a language model; "
            "carries no qualitative judgement"
        ]

        largest = max((abs(w) for w in request.weights_pct), default=0.0)
        if largest > self.max_single_weight:
            observations.append(
                f"largest position is {largest:.1%}, above the "
                f"{self.max_single_weight:.0%} concentration threshold"
            )
        if request.expected_risk > self.max_expected_risk:
            observations.append(
                f"expected risk {request.expected_risk:.1%} exceeds the "
                f"{self.max_expected_risk:.0%} threshold"
            )
        if request.n_positions <= 1:
            observations.append("allocation is a single position; no diversification")

        blocked = bool(observations)
        output = ReviewerOutput(
            verdict=Verdict.ABSTAIN if blocked else Verdict.APPROVE,
            candidate_id=request.candidate_id,
            risk_observations=tuple(observations),
            evidence=(
                ()
                if blocked
                else (
                    Evidence(
                        source_id="formula:portfolio-variance",
                        claim=(
                            f"Allocation holds {request.n_positions} positions with "
                            f"expected risk {request.expected_risk:.2%}, within "
                            f"configured thresholds."
                        ),
                    ),
                )
            ),
            uncertainties=tuple(uncertainties),
            recommended_action=Action.NO_TRADE if blocked else Action.ENTER,
        )
        return ReviewResult(output=output, source=self.name, citations_verified=True)


def resolve_reviewer(
    prefer_local_model: bool = True, ollama_model: str = "llama3.2:3b"
) -> ReviewerClient:
    """First available local model backend, falling back to the offline rules.

    The fallback is silent: with no reachable model this returns an
    `OfflineReviewer`, whose rules can approve. Check `reviewer.name` before
    treating a run as one that a language model reviewed.

    Ollama is tried first because it is what this machine runs. `llama3.2:3b` is
    the default rather than a larger installed model: the reviewer's task is a
    schema-bound verdict over supplied evidence, which a small instruct-tuned
    model handles well under constrained decoding, and a code-tuned model is a
    poor fit for qualitative judgement.
    """
    if prefer_local_model:
        ollama = OllamaReviewer(model=ollama_model)
        if ollama.is_available():
            return ollama
        llama_server = LlamaServerReviewer()
        if llama_server.is_available():
            return llama_server
    return OfflineReviewer()
