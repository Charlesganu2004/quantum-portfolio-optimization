"""Provenance-carrying knowledge records.

Every fact the LLM reviewer is allowed to cite lives in one of these. The schema
exists so that a citation can be *checked*: `source_id` must resolve to a real
record, and the risk controller verifies that before an answer is accepted.

`stale_after` is load-bearing rather than decorative. Much of the legally
available trader data (13F at 45 days, Form 4 at 2 days, congressional PTRs at up
to 45) is far too old to drive an intraday decision. Recording the staleness
horizon in the record itself means the retrieval layer can refuse to serve it as
a live signal while still allowing it as pattern-analysis context.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum


class SourceType(str, Enum):
    FORMULA = "formula"
    PAPER = "paper"
    MARKET_RULE = "market_rule"
    NEWS = "news"
    FILING = "filing"
    TRADE_RECORD = "trade_record"
    SYSTEM_TRADE = "system_trade"
    RISK_DECISION = "risk_decision"


#: Source types whose publication date lags the event they describe, so they can
#: never drive a live decision.
NEVER_LIVE_SOURCE_TYPES = frozenset({SourceType.FILING, SourceType.TRADE_RECORD})


class Confidence(str, Enum):
    """How much weight a claim from this record may carry.

    `HYPOTHESIS` matters most: a behavioural pattern inferred from someone's
    observed trades is never a fact about their reasoning, and must not be
    presented as one.
    """

    DEFINITIONAL = "definitional"   # a formula; true by definition
    REPORTED = "reported"           # a filed, attributable fact
    DERIVED = "derived"             # computed by this system from data
    HYPOTHESIS = "hypothesis"       # an inferred pattern; unproven


@dataclass(frozen=True)
class KnowledgeRecord:
    """One retrievable, citable unit of knowledge."""

    source_id: str
    source_type: SourceType
    title: str
    content: str
    published_at: datetime | None = None
    retrieved_at: datetime | None = None
    asset_symbols: tuple[str, ...] = ()
    license_or_permission: str = "public"
    confidence: Confidence = Confidence.REPORTED
    stale_after: datetime | None = None
    citation: str = ""
    metadata: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.source_id:
            raise ValueError("source_id is required -- citations must resolve")
        if not self.content.strip():
            raise ValueError(f"{self.source_id}: content cannot be empty")

    @property
    def checksum(self) -> str:
        """SHA-256 of the content, so tampering is detectable."""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    def is_stale(self, now: datetime | None = None) -> bool:
        if self.stale_after is None:
            return False
        return (now or datetime.now(timezone.utc)) > self.stale_after

    def age_days(self, now: datetime | None = None) -> float | None:
        if self.published_at is None:
            return None
        delta = (now or datetime.now(timezone.utc)) - self.published_at
        return delta.total_seconds() / 86_400.0

    def usable_as_live_signal(self, max_age_days: float = 1.0) -> bool:
        """Whether this record is fresh enough to influence a trading decision.

        Deliberately strict. Filing and trade-record sources are never live
        signals, whatever their age: the filing date says nothing about when the
        underlying trade happened (a Form 4 filed an hour ago can describe a
        trade two days old, and a 13F describes holdings up to 45 days old), so
        measuring age from `published_at` would let a just-filed record through.
        Other record types must also be inside `max_age_days` and not stale.
        """
        if self.source_type in NEVER_LIVE_SOURCE_TYPES:
            return False
        if self.is_stale():
            return False
        if self.confidence is Confidence.HYPOTHESIS:
            return False
        age = self.age_days()
        return age is not None and age <= max_age_days

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["source_type"] = self.source_type.value
        payload["confidence"] = self.confidence.value
        for key in ("published_at", "retrieved_at", "stale_after"):
            value = payload.get(key)
            payload[key] = value.isoformat() if isinstance(value, datetime) else None
        payload["checksum"] = self.checksum
        return payload

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True)


def utc(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=timezone.utc)


def staleness(published: datetime, days: float) -> datetime:
    return published + timedelta(days=days)
