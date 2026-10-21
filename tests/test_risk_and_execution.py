"""Risk gate and execution. This is where the project's 100% target is tested.

Profit cannot be guaranteed. These controls firing every time can be, so every
limit gets an explicit test, and the default-deny behaviour is tested against
each failure mode independently.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone

import pytest
import requests

from quantlab.execution.broker import (
    LIVE_TRADING_DISABLED,
    AlpacaPaperBroker,
    BrokerError,
    LiveDisabledBroker,
    LiveTradingDisabled,
    OrderStatus,
    SimulatedBroker,
    is_paper_endpoint,
    new_client_order_id,
)
from quantlab.config import Settings, load_env_file, normalize_base_url, project_root
from quantlab.data.sources import AlpacaBarSource
from quantlab.knowledge.formulas import FORMULA_INDEX
from quantlab.llm.reviewer import (
    LlamaServerReviewer,
    OfflineReviewer,
    OllamaReviewer,
    ReviewRequest,
)
from quantlab.llm.schema import (
    Action,
    FabricatedCitation,
    SchemaViolation,
    Verdict,
    normalize_citation,
    parse_reviewer_output,
    require_evidence_for_approval,
    resolve_citation,
    verify_citations,
)
from quantlab.risk.gate import (
    AccountState,
    Decision,
    OrderIntent,
    RejectReason,
    RiskGate,
    RiskPolicy,
)

NOW = datetime(2026, 8, 3, 15, 0, tzinfo=timezone.utc)


def account(**overrides) -> AccountState:
    defaults = dict(
        equity_usd=10.0,
        peak_equity_usd=10.0,
        gross_exposure_usd=0.0,
        open_positions=0,
        session_realized_pnl_usd=0.0,
        consecutive_losses=0,
        orders_this_interval=0,
    )
    defaults.update(overrides)
    return AccountState(**defaults)


def intent(**overrides) -> OrderIntent:
    defaults = dict(
        client_order_id="ql-test-1",
        symbol="AAPL",
        side="buy",
        notional_usd=1.50,
        reference_price=200.0,
        stop_loss_price=186.0,
        quote_timestamp=NOW - timedelta(seconds=1),
        spread_percent=0.02,
    )
    defaults.update(overrides)
    return OrderIntent(**defaults)


class TestApproval:
    def test_a_clean_order_is_approved(self) -> None:
        decision = RiskGate().evaluate(intent(), account(), True, now=NOW)
        assert decision.approved
        assert decision.reasons == ()


class TestDefaultDeny:
    """Every failure mode must block independently."""

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({"notional_usd": 5.0}, RejectReason.POSITION_SIZE),
            ({"notional_usd": 0.0}, RejectReason.NON_POSITIVE_SIZE),
            ({"notional_usd": -1.0}, RejectReason.NON_POSITIVE_SIZE),
            ({"stop_loss_price": None}, RejectReason.MISSING_STOP),
            ({"spread_percent": 1.5}, RejectReason.WIDE_SPREAD),
            ({"is_short": True}, RejectReason.SHORTING_DISABLED),
            ({"uses_leverage": True}, RejectReason.LEVERAGE_DISABLED),
            (
                {"quote_timestamp": NOW - timedelta(seconds=30)},
                RejectReason.STALE_DATA,
            ),
        ],
    )
    def test_order_level_violations_block(self, kwargs, expected) -> None:
        decision = RiskGate().evaluate(intent(**kwargs), account(), True, now=NOW)
        assert not decision.approved
        assert expected in decision.reasons

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({"gross_exposure_usd": 9.5}, RejectReason.GROSS_EXPOSURE),
            ({"open_positions": 4}, RejectReason.POSITION_COUNT),
            ({"session_realized_pnl_usd": -0.50}, RejectReason.SESSION_LOSS),
            ({"equity_usd": 9.0, "peak_equity_usd": 10.0}, RejectReason.DRAWDOWN),
            ({"consecutive_losses": 3}, RejectReason.CONSECUTIVE_LOSSES),
            ({"orders_this_interval": 2}, RejectReason.INTERVAL_ORDER_CAP),
        ],
    )
    def test_account_level_violations_block(self, kwargs, expected) -> None:
        decision = RiskGate().evaluate(intent(), account(**kwargs), True, now=NOW)
        assert not decision.approved
        assert expected in decision.reasons

    def test_reviewer_disapproval_blocks(self) -> None:
        decision = RiskGate().evaluate(intent(), account(), False, now=NOW)
        assert not decision.approved
        assert RejectReason.REVIEWER_BLOCKED in decision.reasons

    def test_duplicate_order_id_blocks_within_the_window(self) -> None:
        state = account(recent_client_order_ids={"ql-test-1": NOW - timedelta(seconds=30)})
        decision = RiskGate().evaluate(intent(), state, True, now=NOW)
        assert RejectReason.DUPLICATE_ORDER in decision.reasons

    def test_same_id_is_allowed_once_the_window_expires(self) -> None:
        state = account(recent_client_order_ids={"ql-test-1": NOW - timedelta(seconds=300)})
        assert RiskGate().evaluate(intent(), state, True, now=NOW).approved

    def test_all_failing_reasons_are_reported_not_just_the_first(self) -> None:
        """The activity log needs every reason, not the earliest one."""
        decision = RiskGate().evaluate(
            intent(notional_usd=99.0, spread_percent=5.0, stop_loss_price=None),
            account(open_positions=4),
            False,
            now=NOW,
        )
        assert len(decision.reasons) >= 4


class TestMalformedInput:
    """Bad numbers and unknown sides are rejected, never approved.

    Every comparison with NaN is false, so a limit written as `x > cap` lets a NaN
    through. Each case below was approved before the gate checked for it.
    """

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"notional_usd": math.nan},
            {"notional_usd": math.inf},
            {"reference_price": math.nan},
            {"reference_price": math.inf},
            {"reference_price": 0.0},
            {"reference_price": -5.0},
            {"stop_loss_price": math.nan},
            {"stop_loss_price": math.inf},
            {"spread_percent": math.nan},
            {"reference_price": 100.0, "stop_loss_price": 150.0},
            {"reference_price": 100.0, "stop_loss_price": 100.0},
        ],
    )
    def test_non_finite_or_wrong_side_order_fields_are_rejected(self, kwargs) -> None:
        decision = RiskGate().evaluate(intent(**kwargs), account(), True, now=NOW)
        assert not decision.approved
        assert RejectReason.INVALID_ORDER in decision.reasons

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"stop_loss_price": 0.0},
            {"stop_loss_price": -5.0},
            {"spread_percent": -3.0},
            {"quote_timestamp": NOW + timedelta(hours=1)},
        ],
        ids=["stop_zero", "stop_negative", "spread_negative", "quote_in_the_future"],
    )
    def test_a_stop_spread_or_quote_that_cannot_work_is_rejected(self, kwargs) -> None:
        """A stop at or below zero never triggers, and a future quote reads as fresh."""
        decision = RiskGate().evaluate(intent(**kwargs), account(), True, now=NOW)
        assert not decision.approved
        assert RejectReason.INVALID_ORDER in decision.reasons

    def test_a_stop_just_above_zero_and_a_zero_spread_are_still_approved(self) -> None:
        decision = RiskGate().evaluate(
            intent(stop_loss_price=0.01, spread_percent=0.0), account(), True, now=NOW
        )
        assert decision.approved

    @pytest.mark.parametrize("side", ["BUY", "Buy", "long", "SELL", "", "short"])
    def test_a_side_other_than_buy_or_sell_is_rejected(self, side: str) -> None:
        decision = RiskGate().evaluate(intent(side=side), account(), True, now=NOW)
        assert not decision.approved
        assert RejectReason.INVALID_ORDER in decision.reasons

    def test_an_unrecognised_side_cannot_skip_the_entry_limits(self) -> None:
        """A full book and no stop was approved when the side read "BUY"."""
        full_book = account(gross_exposure_usd=9.9, open_positions=4)
        decision = RiskGate().evaluate(
            intent(side="BUY", stop_loss_price=None), full_book, True, now=NOW
        )
        assert not decision.approved

    @pytest.mark.parametrize(
        "field",
        [
            "equity_usd",
            "peak_equity_usd",
            "gross_exposure_usd",
            "session_realized_pnl_usd",
        ],
    )
    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_a_non_finite_account_halts_trading(self, field: str, bad: float) -> None:
        decision = RiskGate().evaluate(
            intent(), account(**{field: bad}), True, now=NOW
        )
        assert not decision.approved
        assert RejectReason.NON_FINITE_ACCOUNT in decision.reasons

    def test_drawdown_is_unknown_not_zero_when_equity_is_nan(self) -> None:
        """0.0 would read as "no drawdown" and switch the limit off."""
        assert math.isnan(account(equity_usd=math.nan).drawdown_percent)
        assert math.isnan(account(peak_equity_usd=math.inf).drawdown_percent)

    def test_a_sell_exit_needs_no_stop_and_is_still_approved(self) -> None:
        """Tightening the entry checks must not block an exit."""
        decision = RiskGate().evaluate(
            intent(side="sell", stop_loss_price=None), account(), True, now=NOW
        )
        assert decision.approved

    def test_every_malformed_reason_is_reported_with_detail(self) -> None:
        decision = RiskGate().evaluate(
            intent(notional_usd=math.nan, spread_percent=math.nan),
            account(equity_usd=math.nan),
            True,
            now=NOW,
        )
        assert RejectReason.INVALID_ORDER in decision.reasons
        assert RejectReason.NON_FINITE_ACCOUNT in decision.reasons
        assert len(decision.detail) == len(decision.reasons)


class TestKillSwitch:
    def test_blocks_everything_once_engaged(self) -> None:
        gate = RiskGate()
        gate.engage_kill_switch()
        decision = gate.evaluate(intent(), account(), True, now=NOW)
        assert not decision.approved
        assert RejectReason.KILL_SWITCH in decision.reasons

    def test_requires_explicit_reset(self) -> None:
        gate = RiskGate()
        gate.engage_kill_switch()
        assert gate.kill_switch_engaged
        gate.reset_kill_switch()
        assert gate.evaluate(intent(), account(), True, now=NOW).approved


class TestStopsAndFlatten:
    def test_stop_price_caps_loss_at_the_configured_dollar_limit(self) -> None:
        gate = RiskGate(RiskPolicy(max_loss_per_trade_usd=0.10))
        stop = gate.stop_price_for(entry_price=100.0, notional_usd=2.0, is_long=True)
        # A $0.10 loss on $2.00 notional is a 5% adverse move.
        assert stop == pytest.approx(95.0)

    def test_short_stop_is_above_entry(self) -> None:
        gate = RiskGate(RiskPolicy(max_loss_per_trade_usd=0.10))
        assert gate.stop_price_for(100.0, 2.0, is_long=False) == pytest.approx(105.0)

    def test_stop_breach_detected_for_long_and_short(self) -> None:
        gate = RiskGate()
        assert gate.stop_loss_breached(100.0, 94.0, 95.0, is_long=True)
        assert not gate.stop_loss_breached(100.0, 96.0, 95.0, is_long=True)
        assert gate.stop_loss_breached(100.0, 106.0, 105.0, is_long=False)

    def test_forced_flatten_after_the_interval(self) -> None:
        gate = RiskGate(RiskPolicy(interval_minutes=10))
        opened = NOW - timedelta(minutes=11)
        assert gate.must_flatten(opened, now=NOW)
        assert not gate.must_flatten(NOW - timedelta(minutes=3), now=NOW)

    def test_flatten_ignores_whether_the_position_is_winning(self) -> None:
        """No discretion: the timer does not consult P&L."""
        gate = RiskGate(RiskPolicy(interval_minutes=10))
        assert gate.must_flatten(NOW - timedelta(minutes=10), now=NOW)

    def test_stop_price_rejects_bad_inputs(self) -> None:
        gate = RiskGate()
        with pytest.raises(ValueError, match="entry_price"):
            gate.stop_price_for(0.0, 1.0, True)
        with pytest.raises(ValueError, match="notional_usd"):
            gate.stop_price_for(100.0, 0.0, True)


class TestPolicyValidation:
    def test_rejects_position_cap_above_gross_cap(self) -> None:
        with pytest.raises(ValueError, match="exceeds"):
            RiskPolicy(max_gross_usd=10.0, max_single_position_usd=20.0)

    def test_rejects_non_positive_capital(self) -> None:
        with pytest.raises(ValueError, match="starting_capital"):
            RiskPolicy(starting_capital_usd=0.0)


class TestLiveTradingIsOff:
    """'Live is disabled' must be a tested property, not a claim in a document."""

    def test_live_broker_refuses_every_order(self) -> None:
        with pytest.raises(LiveTradingDisabled, match=LIVE_TRADING_DISABLED):
            LiveDisabledBroker().submit("id", "AAPL", "buy", 1.0, 100.0)

    def test_alpaca_adapter_refuses_a_non_paper_endpoint(self) -> None:
        settings = Settings(
            alpaca_key_id="k",
            alpaca_secret_key="s",
            trading_base_url="https://api.alpaca.markets",
        )
        with pytest.raises(LiveTradingDisabled, match="not the paper endpoint"):
            AlpacaPaperBroker(settings=settings)

    def test_no_broker_reports_itself_as_live(self) -> None:
        assert not SimulatedBroker().is_live
        assert not LiveDisabledBroker().is_live


class TestSimulatedBroker:
    def test_buy_reduces_cash_and_opens_a_position(self) -> None:
        broker = SimulatedBroker(starting_cash_usd=10.0)
        fill = broker.submit(new_client_order_id(), "AAPL", "buy", 2.0, 100.0)
        assert fill.succeeded
        assert broker.cash_usd == pytest.approx(8.0)
        assert "AAPL" in broker.positions

    def test_slippage_always_works_against_the_order(self) -> None:
        broker = SimulatedBroker(slippage_bps=10.0)
        buy = broker.submit(new_client_order_id(), "AAPL", "buy", 1.0, 100.0)
        assert buy.filled_price > 100.0
        sell = broker.submit(new_client_order_id(), "AAPL", "sell", 1.0, 100.0)
        assert sell.filled_price < 100.0

    def test_rejects_a_buy_beyond_available_cash(self) -> None:
        broker = SimulatedBroker(starting_cash_usd=1.0)
        fill = broker.submit(new_client_order_id(), "AAPL", "buy", 5.0, 100.0)
        assert fill.status is OrderStatus.REJECTED
        assert "insufficient cash" in fill.note

    def test_rejects_selling_what_is_not_held(self) -> None:
        fill = SimulatedBroker().submit(new_client_order_id(), "AAPL", "sell", 1.0, 100.0)
        assert fill.status is OrderStatus.REJECTED

    def test_profitable_round_trip_increases_equity(self) -> None:
        broker = SimulatedBroker(starting_cash_usd=10.0, slippage_bps=0.0)
        broker.submit(new_client_order_id(), "AAPL", "buy", 2.0, 100.0)
        broker.submit(new_client_order_id(), "AAPL", "sell", 2.0, 110.0)
        assert broker.equity_usd() > 10.0

    def test_rejects_invalid_side_and_price(self) -> None:
        broker = SimulatedBroker()
        with pytest.raises(BrokerError, match="side"):
            broker.submit("i", "AAPL", "hold", 1.0, 100.0)
        with pytest.raises(BrokerError, match="reference_price"):
            broker.submit("i", "AAPL", "buy", 1.0, 0.0)

    def test_client_order_ids_are_unique(self) -> None:
        assert len({new_client_order_id() for _ in range(500)}) == 500


class TestReviewerSchema:
    def _valid(self) -> dict:
        return {
            "verdict": "APPROVE",
            "candidate_id": "qubo-16a",
            "risk_observations": ["concentration is within limits"],
            "evidence": [
                {"source_id": "formula:sharpe-ratio", "claim": "risk-adjusted"}
            ],
            "uncertainties": [],
            "recommended_action": "ENTER",
        }

    def test_parses_a_valid_payload(self) -> None:
        output = parse_reviewer_output(self._valid())
        assert output.verdict is Verdict.APPROVE
        assert output.permits_trading

    def test_rejects_an_unknown_verdict(self) -> None:
        payload = self._valid() | {"verdict": "DEFINITELY_BUY"}
        with pytest.raises(SchemaViolation, match="invalid verdict"):
            parse_reviewer_output(payload)

    def test_rejects_missing_fields(self) -> None:
        payload = self._valid()
        del payload["evidence"]
        with pytest.raises(SchemaViolation, match="missing required fields"):
            parse_reviewer_output(payload)

    def test_fabricated_citation_is_caught(self) -> None:
        output = parse_reviewer_output(
            self._valid()
            | {"evidence": [{"source_id": "formula:invented", "claim": "x"}]}
        )
        with pytest.raises(FabricatedCitation, match="non-existent source"):
            verify_citations(output, {"formula:sharpe-ratio"})

    def test_approval_without_evidence_is_rejected(self) -> None:
        output = parse_reviewer_output(self._valid() | {"evidence": []})
        with pytest.raises(SchemaViolation, match="requires at least one cited"):
            require_evidence_for_approval(output)

    def test_abstain_never_permits_trading(self) -> None:
        output = parse_reviewer_output(
            self._valid() | {"verdict": "ABSTAIN", "recommended_action": "NO_TRADE"}
        )
        assert not output.permits_trading

    def test_approve_with_no_trade_action_does_not_permit_trading(self) -> None:
        output = parse_reviewer_output(self._valid() | {"recommended_action": "HOLD"})
        assert not output.permits_trading


class TestOfflineReviewer:
    def _request(self, **overrides) -> ReviewRequest:
        defaults = dict(
            candidate_id="qubo-16a",
            method="qubo_annealing",
            symbols=("AAPL", "MSFT", "JPM"),
            weights_pct=(0.34, 0.33, 0.33),
            expected_return=0.20,
            expected_risk=0.12,
            n_positions=3,
        )
        defaults.update(overrides)
        return ReviewRequest(**defaults)

    def test_approves_a_reasonable_allocation(self) -> None:
        result = OfflineReviewer().review(self._request())
        assert result.output.verdict is Verdict.APPROVE
        assert result.permits_trading

    def test_abstains_on_excess_concentration(self) -> None:
        result = OfflineReviewer().review(
            self._request(weights_pct=(0.90, 0.05, 0.05))
        )
        assert result.output.verdict is Verdict.ABSTAIN
        assert not result.permits_trading

    def test_abstains_on_excessive_risk(self) -> None:
        result = OfflineReviewer().review(self._request(expected_risk=0.95))
        assert result.output.verdict is Verdict.ABSTAIN

    def test_abstains_on_a_single_position(self) -> None:
        result = OfflineReviewer().review(
            self._request(symbols=("AAPL",), weights_pct=(1.0,), n_positions=1)
        )
        assert result.output.verdict is Verdict.ABSTAIN

    def test_labels_itself_as_not_a_language_model(self) -> None:
        """It must never be mistaken for an LLM verdict in the report."""
        result = OfflineReviewer().review(self._request())
        assert result.source == "offline_rules"
        assert any(
            "not a language model" in u for u in result.output.uncertainties
        )

    def test_only_cites_resolvable_sources(self) -> None:
        result = OfflineReviewer().review(self._request())
        verify_citations(result.output, set(__import__(
            "quantlab.knowledge.formulas", fromlist=["FORMULA_INDEX"]
        ).FORMULA_INDEX))

    def test_prompt_excludes_order_sizing_information(self) -> None:
        """The reviewer must not see dollar amounts it could echo back as an order."""
        prompt = self._request().to_prompt()
        assert "$" not in prompt
        assert "notional" not in prompt.lower()


class _FakeResponse:
    """The slice of `requests.Response` a reviewer reads."""

    def __init__(self, body) -> None:
        self._body = body

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._body


class TestFailedReviewRecords:
    """A failed review says what failed and never claims a check it did not run.

    Before these fields were separated, every failure path reported
    `citations_verified=True`, including the step whose citation was refused as
    fabricated, and nothing recorded whether the model had answered at all.
    """

    def _request(self) -> ReviewRequest:
        return ReviewRequest(
            candidate_id="qubo-16a",
            method="qubo_annealing",
            symbols=("AAPL", "MSFT"),
            weights_pct=(0.5, 0.5),
            expected_return=0.20,
            expected_risk=0.12,
            n_positions=2,
            evidence=(FORMULA_INDEX["formula:sharpe-ratio"],),
        )

    def _verdict(self, source_id: str = "formula:sharpe-ratio") -> str:
        return json.dumps(
            {
                "verdict": "APPROVE",
                "candidate_id": "qubo-16a",
                "risk_observations": ["concentration is within limits"],
                "evidence": [{"source_id": source_id, "claim": "risk-adjusted"}],
                "uncertainties": [],
                "recommended_action": "ENTER",
            }
        )

    def _serve(self, monkeypatch, body) -> None:
        monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(body))

    def _ollama_body(self, content: str) -> dict:
        return {"message": {"content": content}}

    def test_an_answered_and_verified_review_records_both(self, monkeypatch) -> None:
        self._serve(monkeypatch, self._ollama_body(self._verdict()))
        result = OllamaReviewer().review(self._request())
        assert result.error is None
        assert result.citations_verified is True
        assert result.model_responded is True
        assert result.permits_trading

    @pytest.mark.parametrize("backend", [OllamaReviewer, LlamaServerReviewer])
    def test_an_unreachable_model_checked_nothing_and_never_answered(
        self, monkeypatch, backend
    ) -> None:
        def down(*args, **kwargs):
            raise requests.ConnectionError("RemoteDisconnected")

        monkeypatch.setattr(requests, "post", down)
        result = backend().review(self._request())
        assert result.output.verdict is Verdict.ABSTAIN
        assert "RemoteDisconnected" in result.error
        assert result.citations_verified is None
        assert result.model_responded is False
        assert not result.permits_trading

    def test_a_malformed_envelope_checked_nothing_and_never_answered(
        self, monkeypatch
    ) -> None:
        self._serve(monkeypatch, {})
        result = OllamaReviewer().review(self._request())
        assert result.error.startswith("malformed response")
        assert result.citations_verified is None
        assert result.model_responded is False

    def test_a_schema_failure_checked_no_citations_but_the_model_answered(
        self, monkeypatch
    ) -> None:
        self._serve(monkeypatch, self._ollama_body("this is not json"))
        result = OllamaReviewer().review(self._request())
        assert result.error.startswith("schema violation")
        assert result.citations_verified is None
        assert result.model_responded is True
        assert not result.permits_trading

    def test_a_fabricated_citation_is_recorded_as_not_verified(
        self, monkeypatch
    ) -> None:
        """The field must not say True beside a fabricated-citation error."""
        self._serve(
            monkeypatch,
            self._ollama_body(self._verdict("source_id:transaction-costs")),
        )
        result = OllamaReviewer().review(self._request())
        assert "non-existent source" in result.error
        assert result.citations_verified is False
        assert result.model_responded is True
        assert not result.permits_trading

    def test_the_offline_stand_in_is_never_counted_as_a_model(self) -> None:
        result = OfflineReviewer().review(self._request())
        assert result.model_responded is False


class TestLocalOnlyReviewer:
    #: Strings a prefix test accepts but that contact another machine.
    BYPASSES = (
        "http://localhost@attacker.example",
        "http://localhost.attacker.example:11434",
        "http://127.0.0.1.nip.io:11434",
    )

    #: Other forms that must also be refused.
    OTHER_REFUSALS = (
        "https://api.openai.com",
        "https://localhost:11434",
        "http://user:secret@127.0.0.1:11434",
        "http://127.0.0.1@attacker.example",
        "http://127.0.0.1:notaport",
        "ftp://127.0.0.1",
        "127.0.0.1:11434",
        "",
    )

    LOCAL = (
        "http://127.0.0.1:11434",
        "http://127.0.0.1:8080/",
        "http://localhost:11434",
        "http://LOCALHOST:11434",
        "http://[::1]:11434",
    )

    def test_remote_endpoints_are_refused(self) -> None:
        from quantlab.llm.reviewer import LlamaServerReviewer

        with pytest.raises(ValueError, match="must be local"):
            LlamaServerReviewer(endpoint="https://api.openai.com")

    def test_ollama_refuses_a_remote_endpoint(self) -> None:
        from quantlab.llm.reviewer import OllamaReviewer

        with pytest.raises(ValueError, match="must be local"):
            OllamaReviewer(endpoint="https://ollama.example.com")

    @pytest.mark.parametrize("endpoint", BYPASSES)
    def test_both_backends_refuse_the_prefix_bypasses(self, endpoint: str) -> None:
        from quantlab.llm.reviewer import LlamaServerReviewer, OllamaReviewer

        with pytest.raises(ValueError, match="must be local"):
            OllamaReviewer(endpoint=endpoint)
        with pytest.raises(ValueError, match="must be local"):
            LlamaServerReviewer(endpoint=endpoint)

    @pytest.mark.parametrize("endpoint", OTHER_REFUSALS)
    def test_other_non_local_forms_are_refused(self, endpoint: str) -> None:
        from quantlab.llm.reviewer import require_local_endpoint

        with pytest.raises(ValueError, match="must be local"):
            require_local_endpoint(endpoint)

    @pytest.mark.parametrize("endpoint", LOCAL)
    def test_plain_local_endpoints_are_accepted(self, endpoint: str) -> None:
        from quantlab.llm.reviewer import (
            LlamaServerReviewer,
            OllamaReviewer,
            require_local_endpoint,
        )

        require_local_endpoint(endpoint)
        assert OllamaReviewer(endpoint=endpoint).endpoint == endpoint.rstrip("/")
        assert LlamaServerReviewer(endpoint=endpoint).endpoint == endpoint.rstrip("/")

    def test_a_refused_endpoint_is_never_contacted(self, monkeypatch) -> None:
        """Construction fails before any request could be built."""
        import requests

        from quantlab.llm.reviewer import OllamaReviewer

        def boom(*args, **kwargs):
            raise AssertionError("a request was attempted")

        monkeypatch.setattr(requests, "get", boom)
        monkeypatch.setattr(requests, "post", boom)
        with pytest.raises(ValueError):
            OllamaReviewer(endpoint="http://localhost@attacker.example")


class TestCitationRepair:
    """Formatting slips are repaired; fabrication still fails.

    The two failures a small model actually produces are decoration
    (`[formula:sharpe-ratio]`, copied from how evidence used to be delimited)
    and a dropped namespace (`sharpe-ratio`). Both are recoverable. Neither
    repair may weaken the fabrication check, which is the anti-hallucination
    metric with a target of exactly zero.
    """

    KNOWN = {"formula:sharpe-ratio", "formula:portfolio-variance"}

    def test_exact_id_resolves_unchanged(self) -> None:
        assert resolve_citation("formula:sharpe-ratio", self.KNOWN) == (
            "formula:sharpe-ratio"
        )

    @pytest.mark.parametrize(
        "cited",
        [
            "[formula:sharpe-ratio]",
            "(formula:sharpe-ratio)",
            " formula:sharpe-ratio ",
            "formula:sharpe-ratio.",
            "'formula:sharpe-ratio'",
            "[sharpe-ratio]",
        ],
    )
    def test_decoration_is_stripped(self, cited: str) -> None:
        assert resolve_citation(cited, self.KNOWN) == "formula:sharpe-ratio"

    def test_dropped_namespace_prefix_resolves(self) -> None:
        assert resolve_citation("sharpe-ratio", self.KNOWN) == "formula:sharpe-ratio"

    def test_invented_id_is_still_unresolvable(self) -> None:
        assert resolve_citation("[formula:invented]", self.KNOWN) is None
        assert resolve_citation("invented", self.KNOWN) is None

    def test_ambiguous_bare_id_is_refused(self) -> None:
        """Two namespaces, one bare name -- repair must not guess."""
        known = {"formula:beta", "paper:beta"}
        assert resolve_citation("beta", known) is None
        assert resolve_citation("[beta]", known) is None

    def test_empty_after_stripping_is_not_a_citation(self) -> None:
        assert resolve_citation("[]", self.KNOWN) is None
        assert normalize_citation("[ ]") == ""

    def test_internal_colon_survives_normalization(self) -> None:
        assert normalize_citation("[formula:sharpe-ratio]") == "formula:sharpe-ratio"

    def test_repaired_citations_are_reported_not_hidden(self) -> None:
        output = parse_reviewer_output(
            {
                "verdict": "APPROVE",
                "candidate_id": "qubo-16a",
                "risk_observations": [],
                "evidence": [
                    {"source_id": "[formula:sharpe-ratio]", "claim": "risk-adjusted"}
                ],
                "uncertainties": [],
                "recommended_action": "ENTER",
            }
        )
        repaired = verify_citations(output, self.KNOWN)
        assert repaired == ("[formula:sharpe-ratio]",)

    def test_a_clean_citation_is_not_reported_as_repaired(self) -> None:
        output = parse_reviewer_output(
            {
                "verdict": "APPROVE",
                "candidate_id": "qubo-16a",
                "risk_observations": [],
                "evidence": [
                    {"source_id": "formula:sharpe-ratio", "claim": "risk-adjusted"}
                ],
                "uncertainties": [],
                "recommended_action": "ENTER",
            }
        )
        assert verify_citations(output, self.KNOWN) == ()

    def test_decorated_fabrication_still_raises(self) -> None:
        output = parse_reviewer_output(
            {
                "verdict": "APPROVE",
                "candidate_id": "qubo-16a",
                "risk_observations": [],
                "evidence": [{"source_id": "[formula:invented]", "claim": "x"}],
                "uncertainties": [],
                "recommended_action": "ENTER",
            }
        )
        with pytest.raises(FabricatedCitation, match="non-existent source"):
            verify_citations(output, self.KNOWN)


class TestEvidencePresentation:
    """The prompt must not show the model delimiters it will copy into an id."""

    def _request(self) -> ReviewRequest:
        return ReviewRequest(
            candidate_id="qubo-16a",
            method="qubo_annealing",
            symbols=("AAPL", "MSFT"),
            weights_pct=(0.5, 0.5),
            expected_return=0.2,
            expected_risk=0.1,
            n_positions=2,
            evidence=(FORMULA_INDEX["formula:sharpe-ratio"],),
        )

    def test_source_ids_are_not_wrapped_in_brackets(self) -> None:
        prompt = self._request().to_prompt()
        assert "source_id: formula:sharpe-ratio" in prompt
        assert "[formula:sharpe-ratio]" not in prompt

    def test_prompt_still_excludes_order_sizing(self) -> None:
        prompt = self._request().to_prompt()
        assert "$" not in prompt
        assert "notional" not in prompt.lower()


class TestBaseUrlNormalization:
    """Alpaca documents the endpoint with `/v2`; the code appends its own.

    Pasting the documented URL into `.env` is the expected user action, and left
    unnormalized it produces `/v2/v2/orders` -- a 404 that reads like a
    credential failure and wastes an afternoon.
    """

    @pytest.mark.parametrize(
        "given",
        [
            "https://paper-api.alpaca.markets/v2",
            "https://paper-api.alpaca.markets/v2/",
            "https://paper-api.alpaca.markets/",
            "  https://paper-api.alpaca.markets/v2  ",
        ],
    )
    def test_version_suffix_and_slashes_are_removed(self, given: str) -> None:
        assert normalize_base_url(given) == "https://paper-api.alpaca.markets"

    def test_settings_normalize_on_construction(self) -> None:
        settings = Settings(
            alpaca_key_id="k",
            alpaca_secret_key="s",
            trading_base_url="https://paper-api.alpaca.markets/v2",
            data_base_url="https://data.alpaca.markets/v2/",
        )
        assert settings.trading_base_url == "https://paper-api.alpaca.markets"
        assert settings.data_base_url == "https://data.alpaca.markets"

    def test_host_is_not_rewritten(self) -> None:
        """Normalization trims a path; it must never change which host is called."""
        assert normalize_base_url("https://api.alpaca.markets/v2") == (
            "https://api.alpaca.markets"
        )

    def test_documented_paper_url_still_constructs_a_paper_broker(self) -> None:
        broker = AlpacaPaperBroker(
            settings=Settings(
                alpaca_key_id="k",
                alpaca_secret_key="s",
                trading_base_url="https://paper-api.alpaca.markets/v2",
            )
        )
        assert broker.settings.trading_base_url == "https://paper-api.alpaca.markets"
        assert broker.is_live is False

    @pytest.mark.parametrize(
        "url",
        [
            "https://paper-api@api.alpaca.markets",
            "https://paper-api.alpaca.markets@api.alpaca.markets",
            "https://paper-api.alpaca.markets.attacker.example",
            "https://api.alpaca.markets/paper-api",
            "https://api.alpaca.markets?host=paper-api.alpaca.markets",
            "http://paper-api.alpaca.markets",
            "https://user:pw@paper-api.alpaca.markets",
            "paper-api.alpaca.markets",
            "",
        ],
    )
    def test_urls_that_only_mention_paper_api_are_refused(self, url: str) -> None:
        """A substring test passes the first of these; the real host is live."""
        assert not is_paper_endpoint(url)
        with pytest.raises(LiveTradingDisabled):
            AlpacaPaperBroker(
                settings=Settings(
                    alpaca_key_id="k", alpaca_secret_key="s", trading_base_url=url
                )
            )

    @pytest.mark.parametrize(
        "url",
        [
            "https://paper-api.alpaca.markets",
            "https://paper-api.alpaca.markets/v2",
            "https://PAPER-API.alpaca.markets",
        ],
    )
    def test_the_paper_host_is_accepted(self, url: str) -> None:
        assert is_paper_endpoint(url)

    def test_live_url_is_still_refused_after_normalization(self) -> None:
        with pytest.raises(LiveTradingDisabled):
            AlpacaPaperBroker(
                settings=Settings(
                    alpaca_key_id="k",
                    alpaca_secret_key="s",
                    trading_base_url="https://api.alpaca.markets/v2",
                )
            )


class TestProjectRoot:
    """`.env` and the bar cache are found from the working directory.

    The root used to come from `__file__`, which is right for an editable install
    and wrong for a wheel in a virtual environment, where it points into
    site-packages.
    """

    def test_the_env_file_is_read_from_the_current_directory(
        self, tmp_path, monkeypatch
    ) -> None:
        monkeypatch.delenv("QUANTLAB_HOME", raising=False)
        monkeypatch.setenv("QL_ROOT_PROBE", "unset")
        (tmp_path / ".env").write_text("QL_ROOT_PROBE=from-cwd\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path)
        assert project_root() == tmp_path
        assert load_env_file(override=True) == {"QL_ROOT_PROBE": "from-cwd"}

    def test_quantlab_home_overrides_the_current_directory(
        self, tmp_path, monkeypatch
    ) -> None:
        home = tmp_path / "home"
        elsewhere = tmp_path / "elsewhere"
        home.mkdir()
        elsewhere.mkdir()
        (home / ".env").write_text("QL_ROOT_PROBE=from-home\n", encoding="utf-8")
        (elsewhere / ".env").write_text("QL_ROOT_PROBE=from-cwd\n", encoding="utf-8")
        monkeypatch.setenv("QL_ROOT_PROBE", "unset")
        monkeypatch.setenv("QUANTLAB_HOME", str(home))
        monkeypatch.chdir(elsewhere)
        assert project_root() == home
        assert load_env_file(override=True) == {"QL_ROOT_PROBE": "from-home"}

    def test_the_bar_cache_follows_the_root(self, tmp_path, monkeypatch) -> None:
        monkeypatch.delenv("QUANTLAB_HOME", raising=False)
        monkeypatch.chdir(tmp_path)
        assert AlpacaBarSource(settings=Settings()).cache_dir == tmp_path / "data_cache"
