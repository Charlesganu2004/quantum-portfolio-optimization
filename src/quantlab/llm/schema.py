"""Reviewer output schema and the grammar that enforces it at decode time.

Two layers, deliberately redundant:

1. **GBNF grammar** constrains llama.cpp's sampler so malformed JSON is
   *unrepresentable* -- the model cannot emit a token that breaks the schema.
   This is stronger than asking politely in a prompt and retrying.
2. **Post-parse validation** re-checks everything anyway, because the grammar
   constrains shape, not meaning: it cannot know whether a cited `source_id`
   actually exists.

The reviewer's job is narrow by design. It may approve, reject, or abstain, and
it must cite evidence. It may not compute a number, size a position, or place an
order -- those come from deterministic code. An LLM that cannot place an order
cannot hallucinate one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Verdict(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"
    ABSTAIN = "ABSTAIN"


class Action(str, Enum):
    ENTER = "ENTER"
    EXIT = "EXIT"
    HOLD = "HOLD"
    NO_TRADE = "NO_TRADE"


#: GBNF grammar passed to llama-server so the sampler can only produce a
#: conforming object. Enum members are literal alternatives, so an invalid
#: verdict string cannot be generated at all.
VERDICT_GRAMMAR = r"""
root         ::= "{" ws
                 "\"verdict\"" ws ":" ws verdict ws "," ws
                 "\"candidate_id\"" ws ":" ws string ws "," ws
                 "\"risk_observations\"" ws ":" ws stringlist ws "," ws
                 "\"evidence\"" ws ":" ws evidencelist ws "," ws
                 "\"uncertainties\"" ws ":" ws stringlist ws "," ws
                 "\"recommended_action\"" ws ":" ws action ws
                 "}" ws

verdict      ::= "\"APPROVE\"" | "\"REJECT\"" | "\"ABSTAIN\""
action       ::= "\"ENTER\"" | "\"EXIT\"" | "\"HOLD\"" | "\"NO_TRADE\""

evidencelist ::= "[" ws (evidence (ws "," ws evidence)*)? ws "]"
evidence     ::= "{" ws
                 "\"source_id\"" ws ":" ws string ws "," ws
                 "\"claim\"" ws ":" ws string ws
                 "}"

stringlist   ::= "[" ws (string (ws "," ws string)*)? ws "]"
string       ::= "\"" char* "\""
char         ::= [^"\\] | "\\" ["\\/bfnrt]
ws           ::= [ \t\n]*
"""


#: JSON Schema equivalent of `VERDICT_GRAMMAR`, for backends that constrain
#: decoding with a schema rather than a GBNF grammar (Ollama's `format` field).
#: Same guarantee by a different mechanism -- the sampler cannot emit a
#: non-conforming object. Kept adjacent to the grammar so the two cannot drift.
VERDICT_JSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": [v.value for v in Verdict]},
        "candidate_id": {"type": "string"},
        "risk_observations": {"type": "array", "items": {"type": "string"}},
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "claim": {"type": "string"},
                },
                "required": ["source_id", "claim"],
            },
        },
        "uncertainties": {"type": "array", "items": {"type": "string"}},
        "recommended_action": {"type": "string", "enum": [a.value for a in Action]},
    },
    "required": [
        "verdict",
        "candidate_id",
        "risk_observations",
        "evidence",
        "uncertainties",
        "recommended_action",
    ],
}


@dataclass(frozen=True)
class Evidence:
    """One cited claim. `source_id` must resolve in the knowledge base."""

    source_id: str
    claim: str

    def __post_init__(self) -> None:
        if not self.source_id.strip():
            raise ValueError("evidence entry has an empty source_id")
        if not self.claim.strip():
            raise ValueError(f"evidence {self.source_id} has an empty claim")


@dataclass(frozen=True)
class ReviewerOutput:
    """A validated reviewer response."""

    verdict: Verdict
    candidate_id: str
    risk_observations: tuple[str, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    uncertainties: tuple[str, ...] = ()
    recommended_action: Action = Action.NO_TRADE

    @property
    def permits_trading(self) -> bool:
        """Only an explicit APPROVE with an actionable recommendation permits a trade.

        REJECT and ABSTAIN both block. Abstention is a success state, not a
        failure to hide -- withholding under insufficient evidence is the
        behaviour being engineered for.
        """
        return self.verdict is Verdict.APPROVE and self.recommended_action in (
            Action.ENTER,
            Action.EXIT,
        )

    def cited_ids(self) -> set[str]:
        return {e.source_id for e in self.evidence}


class SchemaViolation(ValueError):
    """Raised when a response cannot be parsed into a valid `ReviewerOutput`."""


class FabricatedCitation(ValueError):
    """Raised when the reviewer cites a source_id that does not exist.

    Tracked separately from other schema failures because fabricated-citation
    rate is a headline anti-hallucination metric with a target of exactly zero.
    """


def parse_reviewer_output(payload: dict) -> ReviewerOutput:
    """Validate a decoded JSON object into a `ReviewerOutput`."""
    if not isinstance(payload, dict):
        raise SchemaViolation(f"expected an object, got {type(payload).__name__}")

    missing = {
        "verdict",
        "candidate_id",
        "risk_observations",
        "evidence",
        "uncertainties",
        "recommended_action",
    } - set(payload)
    if missing:
        raise SchemaViolation(f"missing required fields: {sorted(missing)}")

    try:
        verdict = Verdict(payload["verdict"])
    except ValueError:
        raise SchemaViolation(
            f"invalid verdict {payload['verdict']!r}; expected one of "
            f"{[v.value for v in Verdict]}"
        ) from None

    try:
        action = Action(payload["recommended_action"])
    except ValueError:
        raise SchemaViolation(
            f"invalid recommended_action {payload['recommended_action']!r}; "
            f"expected one of {[a.value for a in Action]}"
        ) from None

    candidate_id = payload["candidate_id"]
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise SchemaViolation("candidate_id must be a non-empty string")

    def string_tuple(key: str) -> tuple[str, ...]:
        value = payload[key]
        if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
            raise SchemaViolation(f"{key} must be a list of strings")
        return tuple(value)

    raw_evidence = payload["evidence"]
    if not isinstance(raw_evidence, list):
        raise SchemaViolation("evidence must be a list")
    try:
        evidence = tuple(
            Evidence(source_id=item["source_id"], claim=item["claim"])
            for item in raw_evidence
        )
    except (TypeError, KeyError) as exc:
        raise SchemaViolation(
            f"each evidence entry needs source_id and claim: {exc}"
        ) from None

    return ReviewerOutput(
        verdict=verdict,
        candidate_id=candidate_id,
        risk_observations=string_tuple("risk_observations"),
        evidence=evidence,
        uncertainties=string_tuple("uncertainties"),
        recommended_action=action,
    )


#: Delimiters and punctuation a model wraps around an id rather than making part
#: of it. Observed in practice: `[formula:sharpe-ratio]` (bracket delimiters
#: copied from how the evidence block presents the id) and trailing sentence
#: punctuation. None of these characters occurs inside a real `source_id`, so
#: stripping them from the ends cannot collide with a legitimate id.
_CITATION_DECORATION = " \t\n\r[](){}<>\"'`,;.:"


def normalize_citation(cited: str) -> str:
    """Strip decoration a model wrapped around an id, leaving the id itself.

    Only the ends are touched, and only with characters that cannot appear at
    the boundary of a real id. The internal `:` separating namespace from name
    is untouched because it is never at an end.
    """
    return cited.strip().strip(_CITATION_DECORATION)


def resolve_citation(cited: str, known_ids: set[str]) -> str | None:
    """Resolve a cited id, tolerating decoration and a dropped namespace prefix.

    Small models get the id right and the *formatting* wrong in two recurring
    ways. They cite `[formula:portfolio-variance]`, carrying over the square
    brackets the evidence block used as delimiters; and they cite
    `portfolio-variance`, dropping the namespace prefix. Both are formatting
    slips, not fabrications, and failing them would discard a valid review.

    Repair is deliberately narrow. Decoration is only stripped from the ends,
    and a bare id resolves **only when exactly one** known id ends with
    `:<cited>`. Ambiguous matches and genuinely invented ids both return None,
    so the fabrication check keeps its full strength.
    """
    if cited in known_ids:
        return cited

    bare = normalize_citation(cited)
    if not bare:
        return None
    if bare in known_ids:
        return bare

    suffix_matches = [
        known
        for known in known_ids
        if ":" in known and known.split(":", 1)[1] == bare
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    return None


def verify_citations(
    output: ReviewerOutput, known_ids: set[str]
) -> tuple[str, ...]:
    """Reject any citation that does not resolve.

    Deterministic code, not a model judging a model -- this catches fabricated
    sources at essentially 100%, which is why the target is zero rather than
    "low".

    Returns:
        The citations that needed prefix repair, so the caller can log that it
        happened rather than silently accepting a malformed id.
    """
    fabricated: list[str] = []
    repaired: list[str] = []
    for cited in sorted(output.cited_ids()):
        resolved = resolve_citation(cited, known_ids)
        if resolved is None:
            fabricated.append(cited)
        elif resolved != cited:
            repaired.append(cited)

    if fabricated:
        raise FabricatedCitation(
            f"reviewer cited {len(fabricated)} non-existent source(s): {fabricated}"
        )
    return tuple(repaired)


def require_evidence_for_approval(output: ReviewerOutput) -> None:
    """An APPROVE with no cited evidence is an unsupported assertion."""
    if output.verdict is Verdict.APPROVE and not output.evidence:
        raise SchemaViolation(
            "APPROVE requires at least one cited evidence entry; an approval with "
            "no evidence is an unsupported claim"
        )
