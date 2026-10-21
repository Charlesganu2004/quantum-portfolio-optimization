"""Public trader-activity records -- the obtainable version of "top trader tracking".

**There is no public record of individual day traders' trades.** No broker
publishes them, no ranked leaderboard with trade history exists, and one probably
should not. Any system claiming to learn from "the top 30 day traders" is either
using a different data source than it says, or inventing it.

What genuinely exists -- named individuals, real trades, real timestamps, legally
public:

    Form 4       SEC insider transactions        filed within 2 business days
    PTR          Congressional periodic reports  filed within ~30-45 days
    13F          institutional holdings (not trades), >$100M    quarterly, 45-day lag
    13D/G        >5% beneficial ownership stakes                5 business days (13D, passive 13G);
                                                                institutional 13G up to 45 days after quarter end

Three constraints this module enforces rather than documents-and-ignores:

1. **None of it is fast enough to trade on.** The freshest source lags two
   business days; an intraday decision cycle cannot use any of it as a signal.
   Every record carries `stale_after`, and `usable_as_live_signal()` returns
   False for every filing and trade record regardless of its age. They are a
   pattern-analysis corpus, nothing more.
2. **An observed trade is not a stated reason.** Inferring why someone traded is
   a hypothesis. Derived patterns are emitted with `Confidence.HYPOTHESIS`, which
   the reviewer is forbidden from presenting as fact.
3. **Only public filings.** No scraping of private accounts, no broker portals,
   no "leaked" positions.

This module defines the disclosure record and the `FilingSource` interface. The
only implementation is `StaticDisclosureSource`, which replays a fixed list for
tests and offline analysis. No live SEC ingestion is implemented.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone

from .records import Confidence, KnowledgeRecord, SourceType, staleness

#: Filing deadline in calendar days, used to set each record's staleness horizon.
#: "13dg" is 5 business days (about 7 calendar days), the initial Schedule 13D
#: and passive Schedule 13G deadline since the SEC's beneficial-ownership
#: amendments took effect on 5 February 2024 (Release 33-11253). Institutional
#: Schedule 13G filers may file up to 45 days after the quarter end, so this
#: entry is the shortest case, not a bound on every 13G.
FILING_LAG_DAYS = {
    "form4": 2.0,
    "ptr": 45.0,
    "13f": 45.0,
    "13dg": 7.0,
}



class FilingAccessError(RuntimeError):
    """Raised when a filings source cannot be reached or used."""


@dataclass(frozen=True)
class TradeDisclosure:
    """One disclosed transaction by a named filer."""

    filer_name: str
    filer_role: str
    symbol: str
    transaction_date: datetime
    filed_date: datetime
    direction: str            # "buy" | "sell"
    shares: float | None
    value_usd: float | None
    source_type: str          # "form4" | "ptr" | "13f" | "13dg"
    source_url: str
    accession: str

    def disclosure_lag_days(self) -> float:
        return (self.filed_date - self.transaction_date).total_seconds() / 86_400.0

    def to_record(self) -> KnowledgeRecord:
        lag = FILING_LAG_DAYS.get(self.source_type, 45.0)
        size = (
            f"{self.shares:,.0f} shares" if self.shares is not None else "size undisclosed"
        )
        value = f" (~${self.value_usd:,.0f})" if self.value_usd is not None else ""
        return KnowledgeRecord(
            source_id=f"{self.source_type}:{self.accession}",
            source_type=SourceType.TRADE_RECORD,
            title=f"{self.filer_name} {self.direction} {self.symbol}",
            content=(
                f"{self.filer_name} ({self.filer_role}) disclosed a "
                f"{self.direction} of {self.symbol}: {size}{value}. "
                f"Transaction dated {self.transaction_date.date()}, "
                f"filed {self.filed_date.date()} "
                f"({self.disclosure_lag_days():.0f} days later). "
                f"This is a disclosed transaction only. No stated reason accompanies "
                f"it, and none should be inferred as fact."
            ),
            published_at=self.filed_date,
            retrieved_at=datetime.now(timezone.utc),
            asset_symbols=(self.symbol,),
            license_or_permission="public (SEC/congressional disclosure)",
            confidence=Confidence.REPORTED,
            stale_after=staleness(self.filed_date, lag),
            citation=self.source_url,
            metadata={
                "filer": self.filer_name,
                "role": self.filer_role,
                "disclosure_lag_days": f"{self.disclosure_lag_days():.1f}",
            },
        )


class FilingSource(ABC):
    """Ingests public trade disclosures."""

    name: str = "unnamed"

    @abstractmethod
    def fetch(self, symbols: tuple[str, ...], since: datetime) -> list[TradeDisclosure]:
        ...


class StaticDisclosureSource(FilingSource):
    """Replays a fixed disclosure list. For tests and offline pattern analysis."""

    name = "static"

    def __init__(self, disclosures: list[TradeDisclosure]) -> None:
        self.disclosures = disclosures

    def fetch(self, symbols: tuple[str, ...], since: datetime) -> list[TradeDisclosure]:
        wanted = set(symbols)
        return [
            d
            for d in self.disclosures
            if d.symbol in wanted and d.transaction_date >= since
        ]


def summarize_filer_pattern(
    disclosures: list[TradeDisclosure], filer_name: str
) -> KnowledgeRecord:
    """Describe one filer's observed behaviour -- explicitly as a hypothesis.

    Counts and dates are facts. Any characterization of *how someone trades* is
    an inference from a small, lagged, incomplete sample, so the record is marked
    `Confidence.HYPOTHESIS` and the reviewer may not assert it as fact.
    """
    theirs = [d for d in disclosures if d.filer_name == filer_name]
    if not theirs:
        raise ValueError(f"no disclosures found for {filer_name!r}")

    buys = sum(1 for d in theirs if d.direction == "buy")
    sells = len(theirs) - buys
    symbols = sorted({d.symbol for d in theirs})
    mean_lag = sum(d.disclosure_lag_days() for d in theirs) / len(theirs)
    first = min(d.transaction_date for d in theirs)
    last = max(d.transaction_date for d in theirs)

    return KnowledgeRecord(
        source_id=f"pattern:{filer_name.lower().replace(' ', '-')}",
        source_type=SourceType.TRADE_RECORD,
        title=f"Observed disclosure pattern: {filer_name}",
        content=(
            f"Across {len(theirs)} disclosed transactions between {first.date()} "
            f"and {last.date()}, {filer_name} reported {buys} purchases and "
            f"{sells} sales across {len(symbols)} symbols ({', '.join(symbols)}). "
            f"Mean disclosure lag {mean_lag:.1f} days.\n\n"
            f"HYPOTHESIS ONLY. These counts describe what was disclosed, not why. "
            f"Disclosure covers a legally defined subset of activity, arrives days "
            f"to weeks late, and carries no stated rationale. No claim about this "
            f"person's strategy, skill, or intent is supported by this record, and "
            f"the lag makes it unusable as a live trading signal."
        ),
        published_at=max(d.filed_date for d in theirs),
        retrieved_at=datetime.now(timezone.utc),
        asset_symbols=tuple(symbols),
        license_or_permission="public (derived from public disclosures)",
        confidence=Confidence.HYPOTHESIS,
        citation="; ".join(sorted({d.source_url for d in theirs})),
        metadata={
            "transactions": str(len(theirs)),
            "mean_disclosure_lag_days": f"{mean_lag:.1f}",
        },
    )
