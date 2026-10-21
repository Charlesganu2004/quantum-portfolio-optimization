"""Knowledge base: citation integrity, staleness, and hypothesis labelling."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from quantlab.knowledge.filings import (
    FILING_LAG_DAYS,
    StaticDisclosureSource,
    TradeDisclosure,
    summarize_filer_pattern,
)
from quantlab.knowledge.formulas import (
    FORMULA_INDEX,
    FORMULA_RECORDS,
    get_formula,
    search_formulas,
)
from quantlab.knowledge.records import (
    Confidence,
    KnowledgeRecord,
    SourceType,
    staleness,
    utc,
)


def disclosure(
    symbol: str = "AAPL",
    filer: str = "Jane Doe",
    direction: str = "buy",
    transaction_day: int = 1,
    lag_days: int = 2,
) -> TradeDisclosure:
    traded = utc(2026, 6, transaction_day)
    return TradeDisclosure(
        filer_name=filer,
        filer_role="Director",
        symbol=symbol,
        transaction_date=traded,
        filed_date=traded + timedelta(days=lag_days),
        direction=direction,
        shares=1000.0,
        value_usd=250_000.0,
        source_type="form4",
        source_url=f"https://sec.gov/{symbol}/{transaction_day}",
        accession=f"0001-{symbol}-{transaction_day}",
    )


class TestRecords:
    def test_requires_a_resolvable_id(self) -> None:
        with pytest.raises(ValueError, match="source_id"):
            KnowledgeRecord(
                source_id="", source_type=SourceType.FORMULA, title="t", content="c"
            )

    def test_rejects_empty_content(self) -> None:
        with pytest.raises(ValueError, match="content"):
            KnowledgeRecord(
                source_id="x", source_type=SourceType.FORMULA, title="t", content="  "
            )

    def test_checksum_detects_tampering(self) -> None:
        base = dict(source_id="x", source_type=SourceType.FORMULA, title="t")
        assert (
            KnowledgeRecord(**base, content="original").checksum
            != KnowledgeRecord(**base, content="altered").checksum
        )

    def test_staleness_horizon_is_enforced(self) -> None:
        published = utc(2026, 1, 1)
        record = KnowledgeRecord(
            source_id="x",
            source_type=SourceType.FILING,
            title="t",
            content="c",
            published_at=published,
            stale_after=staleness(published, 30),
        )
        assert not record.is_stale(published + timedelta(days=10))
        assert record.is_stale(published + timedelta(days=40))

    def test_serializes_with_checksum(self) -> None:
        record = KnowledgeRecord(
            source_id="x", source_type=SourceType.FORMULA, title="t", content="c"
        )
        payload = record.as_dict()
        assert payload["checksum"] == record.checksum
        assert payload["source_type"] == "formula"


class TestLiveSignalGate:
    """Filing-derived records must never qualify as live signals."""

    def test_hypothesis_records_are_never_live_signals(self) -> None:
        record = KnowledgeRecord(
            source_id="p",
            source_type=SourceType.TRADE_RECORD,
            title="t",
            content="c",
            published_at=datetime.now(timezone.utc),
            confidence=Confidence.HYPOTHESIS,
        )
        assert not record.usable_as_live_signal()

    def test_stale_records_are_rejected(self) -> None:
        published = datetime.now(timezone.utc) - timedelta(days=60)
        record = KnowledgeRecord(
            source_id="f",
            source_type=SourceType.FILING,
            title="t",
            content="c",
            published_at=published,
            stale_after=staleness(published, 45),
        )
        assert not record.usable_as_live_signal()

    def test_a_two_day_old_filing_still_fails_an_intraday_gate(self) -> None:
        """The freshest legal source is still too slow for an intraday cycle."""
        published = datetime.now(timezone.utc) - timedelta(days=2)
        record = KnowledgeRecord(
            source_id="f4",
            source_type=SourceType.TRADE_RECORD,
            title="t",
            content="c",
            published_at=published,
            confidence=Confidence.REPORTED,
        )
        assert not record.usable_as_live_signal(max_age_days=1.0)

    def test_no_filing_lag_is_short_enough_for_intraday(self) -> None:
        assert min(FILING_LAG_DAYS.values()) >= 2.0

    @pytest.mark.parametrize(
        "source_type", [SourceType.FILING, SourceType.TRADE_RECORD]
    )
    def test_a_just_filed_record_is_still_not_a_live_signal(
        self, source_type: SourceType
    ) -> None:
        """The filing date is not the trade date, so a fresh filing is not fresh.

        A Form 4 filed a minute ago can describe a trade two days old. Age must
        not be measured from `published_at` for these source types.
        """
        record = KnowledgeRecord(
            source_id="just-filed",
            source_type=source_type,
            title="t",
            content="c",
            published_at=datetime.now(timezone.utc),
            confidence=Confidence.REPORTED,
        )
        assert record.age_days() == pytest.approx(0.0, abs=0.01)
        assert not record.usable_as_live_signal()
        assert not record.usable_as_live_signal(max_age_days=365.0)

    def test_a_fresh_derived_record_can_still_pass(self) -> None:
        """The gate is specific to filings; it is not a blanket refusal."""
        record = KnowledgeRecord(
            source_id="derived",
            source_type=SourceType.SYSTEM_TRADE,
            title="t",
            content="c",
            published_at=datetime.now(timezone.utc),
            confidence=Confidence.DERIVED,
        )
        assert record.usable_as_live_signal()

    def test_a_disclosure_filed_just_now_converts_to_a_non_live_record(self) -> None:
        now = datetime.now(timezone.utc)
        just_filed = TradeDisclosure(
            filer_name="Jane Doe",
            filer_role="Director",
            symbol="AAPL",
            transaction_date=now - timedelta(days=2),
            filed_date=now,
            direction="buy",
            shares=1000.0,
            value_usd=None,
            source_type="form4",
            source_url="https://example.invalid/form4",
            accession="0001-test-now",
        )
        record = just_filed.to_record()
        assert record.age_days() == pytest.approx(0.0, abs=0.01)
        assert not record.usable_as_live_signal()

    def test_the_13dg_lag_reflects_the_2024_five_business_day_rule(self) -> None:
        assert FILING_LAG_DAYS["13dg"] <= 7.0


class TestFormulas:
    def test_every_record_is_definitional_and_cited(self) -> None:
        for record in FORMULA_RECORDS:
            assert record.confidence is Confidence.DEFINITIONAL
            assert record.citation, f"{record.source_id} has no citation"

    def test_ids_are_unique(self) -> None:
        ids = [r.source_id for r in FORMULA_RECORDS]
        assert len(ids) == len(set(ids))

    def test_covers_the_metrics_the_code_computes(self) -> None:
        for required in (
            "formula:sharpe-ratio",
            "formula:sortino-ratio",
            "formula:max-drawdown",
            "formula:qubo-portfolio",
            "formula:ledoit-wolf",
            "formula:transaction-costs",
        ):
            assert required in FORMULA_INDEX

    def test_unknown_citation_raises_rather_than_returning_nothing(self) -> None:
        """Fabricated ids must fail loudly -- this is the anti-hallucination check."""
        with pytest.raises(KeyError, match="unknown formula id"):
            get_formula("formula:invented-by-the-model")

    def test_known_citation_resolves(self) -> None:
        assert get_formula("formula:sharpe-ratio").title == "Sharpe ratio"

    def test_search_finds_relevant_records(self) -> None:
        assert any(
            r.source_id == "formula:ledoit-wolf" for r in search_formulas("shrinkage")
        )

    def test_empty_search_returns_nothing(self) -> None:
        assert search_formulas("   ") == []


class TestDisclosures:
    def test_records_the_disclosure_lag(self) -> None:
        assert disclosure(lag_days=3).disclosure_lag_days() == pytest.approx(3.0)

    def test_converts_to_a_provenanced_record(self) -> None:
        record = disclosure().to_record()
        assert record.source_type is SourceType.TRADE_RECORD
        assert record.asset_symbols == ("AAPL",)
        assert record.stale_after is not None
        assert "no stated reason" in record.content.lower()

    def test_static_source_filters_by_symbol_and_date(self) -> None:
        source = StaticDisclosureSource(
            [disclosure("AAPL", transaction_day=1), disclosure("MSFT", transaction_day=20)]
        )
        found = source.fetch(("MSFT",), since=utc(2026, 6, 10))
        assert len(found) == 1
        assert found[0].symbol == "MSFT"


class TestPatternSummaries:
    def test_pattern_is_labelled_a_hypothesis(self) -> None:
        records = [
            disclosure("AAPL", transaction_day=1),
            disclosure("MSFT", transaction_day=5, direction="sell"),
        ]
        summary = summarize_filer_pattern(records, "Jane Doe")
        assert summary.confidence is Confidence.HYPOTHESIS
        assert "HYPOTHESIS ONLY" in summary.content

    def test_pattern_never_qualifies_as_a_live_signal(self) -> None:
        summary = summarize_filer_pattern([disclosure()], "Jane Doe")
        assert not summary.usable_as_live_signal()

    def test_pattern_states_the_lag_limitation(self) -> None:
        summary = summarize_filer_pattern([disclosure()], "Jane Doe")
        assert "lag" in summary.content.lower()

    def test_unknown_filer_raises(self) -> None:
        with pytest.raises(ValueError, match="no disclosures"):
            summarize_filer_pattern([disclosure()], "Nobody At All")
