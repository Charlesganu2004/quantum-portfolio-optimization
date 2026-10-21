"""Decision cycle: accounting integrity and the reviewer's confined role."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from quantlab.data.features import InsufficientHistory
from quantlab.data.sources import BarRequest, SyntheticBarSource
from quantlab.execution.broker import Broker, SimulatedBroker
from quantlab.llm.reviewer import (
    OfflineReviewer,
    ReviewRequest,
    ReviewResult,
    _abstain,
)
from quantlab.llm.schema import Action, ReviewerOutput, Verdict
from quantlab.optimizers import HeuristicOptimizer, QuboOptimizer
from quantlab.pipeline.session import (
    REVIEW_EVIDENCE_IDS,
    SessionConfig,
    SessionRunner,
    compare_review_ablation,
)
from quantlab.risk.gate import AccountState


@pytest.fixture(scope="module")
def prices():
    return SyntheticBarSource(seed=20260803).fetch(
        BarRequest(
            tuple(f"SYN{i:02d}" for i in range(12)), date(2023, 1, 1), date(2026, 8, 1)
        )
    )


def config(**overrides) -> SessionConfig:
    base = dict(max_intervals=6, train_sessions=200, review_enabled=True)
    base.update(overrides)
    return SessionConfig(**base)


class AlwaysBlocks(OfflineReviewer):
    """Reviewer that refuses everything, to prove review can veto."""

    name = "always_blocks"

    def review(self, request: ReviewRequest) -> ReviewResult:
        return ReviewResult(
            output=ReviewerOutput(
                verdict=Verdict.REJECT,
                candidate_id=request.candidate_id,
                recommended_action=Action.NO_TRADE,
            ),
            source=self.name,
            citations_verified=True,
        )


class ModelThatGoesDown(OfflineReviewer):
    """A model backend that answers `answers` reviews and then stops responding.

    The answers are the offline rules' verdicts relabelled as model responses, so
    only the availability of the backend differs between steps.
    """

    name = "flaky_model"

    def __init__(self, answers: int) -> None:
        super().__init__()
        self.answers = answers
        self.calls = 0

    def review(self, request: ReviewRequest) -> ReviewResult:
        self.calls += 1
        if self.calls > self.answers:
            return _abstain(
                request.candidate_id,
                self.name,
                "ollama unreachable: RemoteDisconnected",
            )
        result = super().review(request)
        result.source = self.name
        result.model_responded = True
        return result


def load_run_session_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_session.py"
    spec = importlib.util.spec_from_file_location("run_session_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestAccounting:
    def test_pnl_reconciles_with_the_equity_curve(self, prices) -> None:
        """Interval P&L must sum to the total equity change.

        An earlier version derived P&L from fill notionals and reported 0.0000
        every interval while equity visibly moved. The two must agree.
        """
        report = SessionRunner(prices, config=config()).run()
        pnl_sum = sum(i.realized_pnl_usd for i in report.intervals)
        equity_change = report.ending_equity_usd - report.starting_equity_usd
        assert pnl_sum == pytest.approx(equity_change, abs=1e-9)

    def test_equity_curve_matches_interval_records(self, prices) -> None:
        report = SessionRunner(prices, config=config()).run()
        curve = report.equity_curve()
        assert len(curve) == len(report.intervals) + 1
        assert curve[0] == pytest.approx(report.starting_equity_usd)
        assert curve[-1] == pytest.approx(report.ending_equity_usd)

    def test_a_nonzero_pnl_is_actually_produced(self, prices) -> None:
        """Guards the specific regression: all-zero P&L looks plausible."""
        report = SessionRunner(prices, config=config()).run()
        traded = [i for i in report.intervals if i.traded()]
        assert traded, "no interval traded; the test proves nothing"
        assert any(abs(i.realized_pnl_usd) > 1e-9 for i in traded)

    def test_every_interval_is_recorded(self, prices) -> None:
        report = SessionRunner(prices, config=config(max_intervals=5)).run()
        assert len(report.intervals) == 5
        assert [i.index for i in report.intervals] == [0, 1, 2, 3, 4]


class TestReviewerRole:
    def test_a_blocking_reviewer_prevents_all_trading(self, prices) -> None:
        report = SessionRunner(
            prices, config=config(), reviewer=AlwaysBlocks()
        ).run()
        assert report.intervals_traded == 0
        assert all(not i.reviewer_permits for i in report.intervals)

    def test_blocked_intervals_record_the_reason(self, prices) -> None:
        report = SessionRunner(prices, config=config(), reviewer=AlwaysBlocks()).run()
        assert all(
            "reviewer" in i.no_trade_reason for i in report.intervals
        )

    def test_a_blocking_reviewer_leaves_equity_untouched(self, prices) -> None:
        report = SessionRunner(prices, config=config(), reviewer=AlwaysBlocks()).run()
        assert report.ending_equity_usd == pytest.approx(report.starting_equity_usd)

    def test_disabled_review_is_recorded_as_bypassed_not_approved(
        self, prices
    ) -> None:
        """The log must never imply a verdict that was never sought."""
        report = SessionRunner(prices, config=config(review_enabled=False)).run()
        assert all(i.reviewer_verdict == "BYPASSED" for i in report.intervals)
        assert all(i.reviewer_source == "none" for i in report.intervals)

    def test_evidence_offered_is_resolvable(self, prices) -> None:
        from quantlab.knowledge.formulas import FORMULA_INDEX

        report = SessionRunner(prices, config=config()).run()
        for interval in report.intervals:
            assert interval.evidence_ids
            for source_id in interval.evidence_ids:
                assert source_id in FORMULA_INDEX

    def test_evidence_set_is_fixed_across_intervals(self, prices) -> None:
        """Holds retrieval constant so the ablation measures the reviewer."""
        report = SessionRunner(prices, config=config()).run()
        assert {i.evidence_ids for i in report.intervals} == {REVIEW_EVIDENCE_IDS}


class TestModelFailureMidRun:
    """A model that passes the startup check can still fail during the run.

    A failed review abstains exactly like a judgement abstention, so the log has
    to record which steps the model never answered. Otherwise a run where every
    review failed looks like one where the model declined every trade.
    """

    def test_steps_after_the_failure_are_recorded_as_unanswered(self, prices) -> None:
        model = ModelThatGoesDown(answers=2)
        report = SessionRunner(prices, config=config(), reviewer=model).run()
        answered = [i.model_responded for i in report.intervals]
        assert answered == [True, True, False, False, False, False]
        for interval in report.intervals[2:]:
            assert interval.reviewer_verdict == "ABSTAIN"
            assert interval.reviewer_permits is False
            assert interval.citations_verified is None
            assert "RemoteDisconnected" in interval.reviewer_error

    def test_a_run_with_a_failed_step_is_not_reviewed_by_the_model(self, prices) -> None:
        report = SessionRunner(
            prices, config=config(), reviewer=ModelThatGoesDown(answers=2)
        ).run()
        assert not report.reviewed_by_model_every_step

    def test_a_run_where_every_step_was_answered_is_reviewed_by_the_model(
        self, prices
    ) -> None:
        report = SessionRunner(
            prices, config=config(), reviewer=ModelThatGoesDown(answers=10**6)
        ).run()
        assert all(i.reviewer_error is None for i in report.intervals)
        assert report.reviewed_by_model_every_step

    def test_the_offline_stand_in_and_a_bypassed_run_are_not_model_reviews(
        self, prices
    ) -> None:
        offline = SessionRunner(prices, config=config()).run()
        bypassed = SessionRunner(prices, config=config(review_enabled=False)).run()
        assert not offline.reviewed_by_model_every_step
        assert not bypassed.reviewed_by_model_every_step
        assert all(i.model_responded is None for i in bypassed.intervals)

    def test_each_ablation_arm_carries_its_own_per_step_record(self, prices) -> None:
        result = compare_review_ablation(
            prices, config=config(), reviewer=ModelThatGoesDown(answers=3)
        )
        with_review = result["with_review"]
        assert [i.model_responded for i in with_review.intervals] == [
            True,
            True,
            True,
            False,
            False,
            False,
        ]
        assert all(i.model_responded is None for i in result["without_review"].intervals)

    def test_the_script_stops_on_a_transport_failure_and_writes_nothing(
        self, prices
    ) -> None:
        script = load_run_session_script()
        model = ModelThatGoesDown(answers=2)
        report = SessionRunner(prices, config=config(), reviewer=model).run()
        with pytest.raises(SystemExit) as stopped:
            script.stop_on_failed_reviews("main run", report, model)
        message = str(stopped.value)
        assert "returned no response at 4 of 6 steps" in message
        assert "RemoteDisconnected" in message
        assert "Nothing was written" in message

    def test_the_script_accepts_a_run_the_model_answered_in_full(self, prices) -> None:
        script = load_run_session_script()
        model = ModelThatGoesDown(answers=10**6)
        report = SessionRunner(prices, config=config(), reviewer=model).run()
        script.stop_on_failed_reviews("main run", report, model)

    def test_the_script_does_not_stop_the_labelled_offline_run(self, prices) -> None:
        script = load_run_session_script()
        reviewer = OfflineReviewer()
        report = SessionRunner(prices, config=config(), reviewer=reviewer).run()
        script.stop_on_failed_reviews("main run", report, reviewer)

    def test_the_logged_step_carries_the_review_evidence(self, prices) -> None:
        script = load_run_session_script()
        report = SessionRunner(
            prices, config=config(), reviewer=ModelThatGoesDown(answers=1)
        ).run()
        records = [script.interval_record(i) for i in report.intervals]
        json.dumps(records)
        assert records[0]["model_responded"] is True
        assert records[0]["reviewer_error"] is None
        assert records[0]["citations_verified"] is True
        assert records[1]["model_responded"] is False
        assert "RemoteDisconnected" in records[1]["reviewer_error"]
        assert records[1]["citations_verified"] is None
        assert records[1]["reviewer_observations"]


class TestScriptRun:
    """`run_session.main` end to end, with the data and the reviewer stubbed.

    The case that went wrong: Ollama passed the startup availability check, then
    every review failed with a connection error, and the script still wrote a log
    that called the run a language-model ablation.
    """

    @staticmethod
    def run_main(script, monkeypatch, tmp_path, prices, reviewer, argv=()) -> None:
        monkeypatch.setattr(script, "load_prices", lambda: (prices, "test frame"))
        monkeypatch.setattr(script, "resolve_reviewer", lambda: reviewer)
        monkeypatch.setattr(script, "REPORTS_DIR", tmp_path)
        monkeypatch.setattr(
            script,
            "SessionConfig",
            lambda **kw: SessionConfig(**{**kw, "max_intervals": 3, "train_sessions": 200}),
        )
        monkeypatch.setattr(sys, "argv", ["run_session.py", *argv])
        script.main()

    def test_every_call_failing_writes_no_log(self, prices, monkeypatch, tmp_path) -> None:
        script = load_run_session_script()
        model = ModelThatGoesDown(answers=0)
        with pytest.raises(SystemExit) as stopped:
            self.run_main(script, monkeypatch, tmp_path, prices, model)
        assert "returned no response at 3 of 3 steps" in str(stopped.value)
        assert "RemoteDisconnected" in str(stopped.value)
        assert not (tmp_path / "session_intervals.json").exists()

    def test_a_failure_in_a_later_run_writes_no_log(
        self, prices, monkeypatch, tmp_path
    ) -> None:
        script = load_run_session_script()
        model = ModelThatGoesDown(answers=3)
        with pytest.raises(SystemExit) as stopped:
            self.run_main(script, monkeypatch, tmp_path, prices, model)
        assert "qubo_annealing ablation" in str(stopped.value)
        assert not (tmp_path / "session_intervals.json").exists()

    def test_a_fully_answered_run_is_logged_as_model_reviewed(
        self, prices, monkeypatch, tmp_path
    ) -> None:
        script = load_run_session_script()
        model = ModelThatGoesDown(answers=10**6)
        self.run_main(script, monkeypatch, tmp_path, prices, model)
        log = json.loads((tmp_path / "session_intervals.json").read_text(encoding="utf-8"))
        assert log["reviewer"] == "flaky_model"
        assert log["reviewer_is_language_model"] is True
        assert len(log["intervals"]) == 3
        for arm in log["ablation"].values():
            assert len(arm["steps_with_review"]) == 3
            assert all(step["model_responded"] is True for step in arm["steps_with_review"])

    def test_the_labelled_offline_run_is_logged_as_not_model_reviewed(
        self, prices, monkeypatch, tmp_path
    ) -> None:
        script = load_run_session_script()
        self.run_main(
            script, monkeypatch, tmp_path, prices, OfflineReviewer(), argv=("--offline",)
        )
        log = json.loads((tmp_path / "session_intervals.json").read_text(encoding="utf-8"))
        assert log["reviewer"] == "offline_rules"
        assert log["reviewer_is_language_model"] is False


class TestBadPrices:
    """One bad bar must not switch off the drawdown and session-loss limits.

    A NaN close turned broker equity into NaN, drawdown then read 0.0, and the gate
    approved every later step.
    """

    def test_a_missing_row_is_carried_forward(self, prices) -> None:
        frame = prices.copy()
        frame.iloc[202] = np.nan
        report = SessionRunner(frame, config=config()).run()
        assert all(math.isfinite(i.equity_usd) for i in report.intervals)
        assert all(math.isfinite(i.drawdown_percent) for i in report.intervals)

    def test_zero_and_infinite_closes_are_treated_as_missing(self, prices) -> None:
        frame = prices.copy()
        frame.iloc[202, :4] = 0.0
        frame.iloc[203, 4:8] = np.inf
        report = SessionRunner(frame, config=config()).run()
        assert all(math.isfinite(i.equity_usd) for i in report.intervals)

    def test_a_symbol_with_no_usable_close_is_rejected_at_the_gate(self, prices) -> None:
        """No earlier close exists to carry forward, so the entry has no price.

        `run` never reaches this path, because `_optimize` leaves such a symbol out
        (see `TestLateSymbols`). A caller of `_execute` can still hand one in.
        """
        runner = SessionRunner(prices, config=config(review_enabled=False))
        train = runner.prices.iloc[:200].copy()
        train.iloc[:, 0] = np.nan
        forward = runner.prices.iloc[199:201]
        fills, reasons, submitted, label = runner._execute(
            SimpleNamespace(weights=np.array([1.0])),
            [train.columns[0]],
            train,
            forward,
            AccountState(equity_usd=10.0, peak_equity_usd=10.0),
            True,
            datetime(2026, 8, 3, 15, 0, tzinfo=timezone.utc),
        )
        assert (submitted, fills, label) == (0, [], "REJECT")
        assert "invalid_order_input" in reasons
        assert "no_stop_loss_attached" in reasons

    def test_non_finite_broker_equity_halts_every_later_step(self, prices) -> None:
        class BrokerWithNanEquity(SimulatedBroker):
            def __init__(self) -> None:
                super().__init__(starting_cash_usd=10.0)
                self.calls = 0

            def equity_usd(self, *args, **kwargs) -> float:
                self.calls += 1
                if self.calls > 2:
                    return math.nan
                return super().equity_usd(*args, **kwargs)

        report = SessionRunner(
            prices, config=config(review_enabled=False), broker=BrokerWithNanEquity()
        ).run()
        halted = report.intervals[3:]
        assert halted
        for interval in halted:
            assert interval.orders_submitted == 0
            assert interval.risk_decision == "REJECT"
            assert "non_finite_account_state" in interval.risk_reasons


class TestLateSymbols:
    """A symbol without a close on every row of the training window is left out.

    It used to be dropped by `dropna(axis=0, how="any")` for every symbol: the
    window shrank for the whole universe, and the run raised `InsufficientHistory`
    when the symbol's first close sat near the end of the window.
    """

    @pytest.mark.parametrize("first_close", [199, 200, 201, 203, 204])
    def test_a_symbol_that_starts_inside_the_replay_never_crashes_the_run(
        self, prices, first_close: int
    ) -> None:
        frame = prices.copy()
        frame.iloc[:first_close, 0] = np.nan
        report = SessionRunner(frame, config=config(review_enabled=False)).run()
        assert [i.symbols_considered for i in report.intervals] == [11] * 6
        traded = {f.symbol for i in report.intervals for f in i.fills}
        assert prices.columns[0] not in traded

    def test_a_late_symbol_joins_the_first_step_whose_window_it_fills(
        self, prices
    ) -> None:
        frame = prices.copy()
        frame.iloc[:3, 0] = np.nan
        report = SessionRunner(frame, config=config(review_enabled=False)).run()
        assert [i.symbols_considered for i in report.intervals] == [11, 11, 11, 12, 12, 12]

    def test_a_late_symbol_does_not_shrink_the_window_of_the_others(self, prices) -> None:
        """The step's problem equals the one built from the 11 full symbols alone."""
        frame = prices.copy()
        frame.iloc[:203, 0] = np.nan
        late = SessionRunner(frame, config=config(review_enabled=False))
        full = SessionRunner(
            prices.iloc[:, 1:], config=config(review_enabled=False)
        )
        window = late.prices.iloc[:200]
        _, late_problem, _ = late._optimize(window)
        _, full_problem, _ = full._optimize(window.iloc[:, 1:])
        assert late_problem.symbols == full_problem.symbols
        np.testing.assert_array_equal(
            late_problem.expected_returns, full_problem.expected_returns
        )
        np.testing.assert_array_equal(late_problem.covariance, full_problem.covariance)

    def test_a_symbol_with_no_close_at_all_is_left_out_of_every_step(self, prices) -> None:
        frame = prices.copy()
        frame.iloc[:, 0] = np.nan
        report = SessionRunner(frame, config=config(review_enabled=False)).run()
        assert [i.symbols_considered for i in report.intervals] == [11] * 6

    def test_no_symbol_filling_the_window_is_an_error_not_an_empty_step(
        self, prices
    ) -> None:
        frame = prices.copy()
        frame.iloc[:203, :] = np.nan
        with pytest.raises(InsufficientHistory, match="training window"):
            SessionRunner(frame, config=config(review_enabled=False)).run()

    def test_the_short_history_warning_is_not_hidden(self, prices) -> None:
        """Ten sessions give 9 returns for 12 assets: the covariance is singular."""
        with pytest.warns(RuntimeWarning, match="observations for"):
            SessionRunner(
                prices, config=config(train_sessions=10, max_intervals=1)
            ).run()


class TestBrokerGuard:
    def test_a_broker_that_is_not_simulated_is_refused(self, prices) -> None:
        """Equity came only from a `SimulatedBroker`, so any other broker read 0.0.

        With another broker the runner's equity never moved while the broker's did,
        which left the drawdown, session-loss and consecutive-loss limits unable
        to fire.
        """

        class RemoteBroker(Broker):
            name = "remote"

            def submit(self, client_order_id, symbol, side, notional_usd, reference_price):
                raise AssertionError("a refused broker must never receive an order")

            def equity_usd(self) -> float:
                return 12.0

        with pytest.raises(TypeError, match="SimulatedBroker"):
            SessionRunner(prices, config=config(), broker=RemoteBroker())

    def test_the_runner_reads_equity_from_its_broker_after_every_step(
        self, prices
    ) -> None:
        broker = SimulatedBroker(starting_cash_usd=10.0)
        report = SessionRunner(
            prices, config=config(review_enabled=False), broker=broker
        ).run()
        assert report.intervals[-1].equity_usd == pytest.approx(broker.equity_usd())


class TestRiskEnforcement:
    def test_never_exceeds_the_per_interval_order_cap(self, prices) -> None:
        report = SessionRunner(prices, config=config()).run()
        assert all(i.orders_submitted <= 2 for i in report.intervals)

    def test_respects_the_cardinality_cap(self, prices) -> None:
        report = SessionRunner(prices, config=config(cardinality_cap=3)).run()
        assert all(i.optimizer_positions <= 3 for i in report.intervals)

    def test_position_size_never_exceeds_the_policy_cap(self, prices) -> None:
        runner = SessionRunner(prices, config=config())
        report = runner.run()
        cap = runner.policy.max_single_position_usd
        for interval in report.intervals:
            for fill in interval.fills:
                assert fill.requested_notional_usd <= cap + 1e-9

    def test_positions_are_flattened_every_interval(self, prices) -> None:
        """Forced flatten: nothing is carried across an interval boundary."""
        broker = SimulatedBroker(starting_cash_usd=10.0)
        SessionRunner(prices, config=config(), broker=broker).run()
        assert broker.positions == {}

    def test_equity_stays_within_a_plausible_band(self, prices) -> None:
        """A $10 account trading $1.50 positions cannot move by tens of percent."""
        report = SessionRunner(prices, config=config()).run()
        assert 0.5 < report.ending_equity_usd / report.starting_equity_usd < 2.0


class TestAblation:
    def test_reports_a_change_rate(self, prices) -> None:
        result = compare_review_ablation(prices, config=config())
        assert result["intervals"] > 0
        assert 0.0 <= result["change_rate"] <= 1.0

    def test_offline_reviewer_changes_nothing_by_construction(self, prices) -> None:
        """Documents why the ablation needs a real model.

        OfflineReviewer abstains only on concentration >45%, risk >60%, or a
        single position. A 4-position allocation capped at 40% weight cannot
        breach any of those, so the rate is structurally 0% and measures the
        stand-in rather than LLM review. Lowering the thresholds to manufacture a
        non-zero rate would fabricate a graded result.
        """
        result = compare_review_ablation(prices, config=config())
        assert result["change_rate"] == 0.0

    def test_a_blocking_reviewer_produces_a_measurable_change(self, prices) -> None:
        """Proves the ablation *can* detect a difference when one exists."""
        result = compare_review_ablation(
            prices, config=config(), reviewer=AlwaysBlocks()
        )
        assert result["change_rate"] > 0.0
        assert result["traded_with_review"] < result["traded_without_review"]

    def test_both_arms_share_identical_inputs(self, prices) -> None:
        """Any difference must be attributable to review alone."""
        result = compare_review_ablation(prices, config=config())
        with_review = result["with_review"]
        without_review = result["without_review"]
        assert [i.candidate_id for i in with_review.intervals] == [
            i.candidate_id for i in without_review.intervals
        ]
        assert [i.optimizer_expected_return for i in with_review.intervals] == pytest.approx(
            [i.optimizer_expected_return for i in without_review.intervals]
        )


class TestConfiguration:
    def test_rejects_history_shorter_than_the_training_window(self, prices) -> None:
        with pytest.raises(ValueError, match="need at least"):
            SessionRunner(prices.tail(50), config=config(train_sessions=200))

    @pytest.mark.parametrize("optimizer", [QuboOptimizer, HeuristicOptimizer])
    def test_runs_with_each_optimizer(self, prices, optimizer) -> None:
        report = SessionRunner(prices, config=config(), optimizer=optimizer()).run()
        assert len(report.intervals) == 6
        assert all(i.method == optimizer.method for i in report.intervals)


class TestConsecutiveLossHalt:
    """The halt must be reachable, latched, and *visible*.

    The original counter incremented on any interval with negative P&L and reset
    only on positive P&L. A blocked interval has P&L of exactly 0.0, so once the
    halt engaged it blocked every order, every later interval scored 0.0, and the
    reset branch became unreachable. The run then looked like a strategy that
    stopped finding trades rather than one that had been switched off.
    """

    def test_untraded_intervals_do_not_count_as_losses(self, prices) -> None:
        """A blocked interval is not a losing trade and must not advance the halt."""
        report = SessionRunner(
            prices,
            config=config(max_intervals=12, review_enabled=True),
            reviewer=AlwaysBlocks(),
        ).run()
        # The reviewer vetoes everything, so nothing ever trades. Zero trades
        # cannot produce a consecutive-*loss* halt.
        assert all(i.orders_submitted == 0 for i in report.intervals)
        assert report.halted_at_interval is None

    def test_halt_is_recorded_when_it_engages(self, prices) -> None:
        report = SessionRunner(
            prices,
            config=config(max_intervals=60, review_enabled=False),
            optimizer=QuboOptimizer(),
        ).run()
        if report.halted_at_interval is not None:
            index = report.halted_at_interval
            assert 0 <= index < len(report.intervals)
            # Latched: nothing trades again after the halt engages.
            assert all(
                i.orders_submitted == 0 for i in report.intervals[index + 1 :]
            ), "halt must stay latched once engaged"

    def test_effective_intervals_excludes_the_halted_tail(self, prices) -> None:
        report = SessionRunner(
            prices,
            config=config(max_intervals=60, review_enabled=False),
            optimizer=QuboOptimizer(),
        ).run()
        if report.halted_at_interval is None:
            assert report.effective_intervals == len(report.intervals)
        else:
            assert report.effective_intervals == report.halted_at_interval + 1
            assert report.effective_intervals <= len(report.intervals)
