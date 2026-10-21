"""The decision cycle, end to end.

One step is: market data -> features -> optimizer -> fixed evidence set ->
reviewer -> deterministic risk gate -> paper order -> step report.

A step advances one row of the price frame. Every shipped caller passes daily
bars, so a step is one trading day: the entry uses the close of the last
training row and the forced exit uses the close of the next row. The
`interval_minutes` setting and the `opened_at` / `closed_at` stamps are labels
(a wall-clock start plus `interval_minutes` per step) and do not describe the
data. Feeding the runner 10-minute bars would make the labels true.

The ordering is the design. The reviewer sits *between* the optimizer and the
risk gate, and can only ever subtract: it may block a trade the optimizer wanted,
never authorize one the risk gate refused, and never alter a number. The size of
each entry is fixed: `SessionConfig.position_notional_usd`, capped at the
policy's single-position limit. `RiskGate` approves or rejects that order and
supplies its stop price; it does not size it. The forced-flatten sell at the end
of the step goes straight to the broker without a second gate evaluation, by
design, since an exit must never be blocked.

`quote_timestamp` and the gate's clock are both `opened_at` here, so the gate's
stale-quote check cannot fire in a replay. It is a control for a live feed.

`SessionRunner` is also the instrument for the proposal's required ablation --
"whether the LLM agent's review changes the final decision from the raw optimizer
output, and how often". Running the identical seed with `review_enabled` toggled
isolates the reviewer's contribution exactly, because nothing else in the loop
consumes randomness differently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from quantlab.data.estimators import ReturnEstimator, SampleMeanEstimator
from quantlab.data.features import InsufficientHistory, annualized_moments, to_returns
from quantlab.execution.broker import Broker, Fill, SimulatedBroker, new_client_order_id
from quantlab.knowledge.formulas import FORMULA_INDEX
from quantlab.knowledge.records import KnowledgeRecord
from quantlab.llm.reviewer import OfflineReviewer, ReviewerClient, ReviewRequest
from quantlab.optimizers import Constraints, Optimizer, PortfolioProblem, QuboOptimizer
from quantlab.risk.gate import AccountState, OrderIntent, RejectReason, RiskGate, RiskPolicy

#: Evidence offered to the reviewer each step. A fixed set of four formula
#: records, not the result of a retrieval, so the ablation measures the reviewer
#: and not retrieval variance.
REVIEW_EVIDENCE_IDS = (
    "formula:portfolio-variance",
    "formula:sharpe-ratio",
    "formula:transaction-costs",
    "formula:mean-variance-objective",
)


@dataclass
class IntervalReport:
    """One decision step (one price-frame row), fully accounted for."""

    index: int
    opened_at: datetime
    closed_at: datetime
    symbols_considered: int
    method: str
    estimator: str
    candidate_id: str
    optimizer_positions: int
    optimizer_expected_return: float
    optimizer_expected_risk: float
    optimizer_runtime_ms: float
    evidence_ids: tuple[str, ...]
    reviewer_source: str
    reviewer_verdict: str
    reviewer_permits: bool
    risk_decision: str
    risk_reasons: tuple[str, ...]
    orders_submitted: int
    fills: tuple[Fill, ...]
    realized_pnl_usd: float
    equity_usd: float
    drawdown_percent: float
    no_trade_reason: str = ""
    #: Why a review failed, or None. A genuine ABSTAIN and an ABSTAIN produced by
    #: an unreachable model carry the same verdict, so the error is kept here.
    #: None also when review was bypassed.
    reviewer_error: str | None = None
    #: True when every citation resolved, False when one named a source that does
    #: not exist, None when review was bypassed or no output was checked (the
    #: model was unreachable or its output failed the schema).
    citations_verified: bool | None = None
    #: Whether a model backend returned text for this step. None when review was
    #: bypassed. False for an unreachable model, so a run can tell a judgement
    #: abstention from one produced by a transport failure.
    model_responded: bool | None = None
    repaired_citations: tuple[str, ...] = ()
    #: The reviewer's stated risk observations, so its reasons can be read back
    #: from the log. Empty when review was bypassed.
    reviewer_observations: tuple[str, ...] = ()

    def traded(self) -> bool:
        return self.orders_submitted > 0


@dataclass
class SessionReport:
    """Aggregate over a run."""

    mode: str
    intervals: tuple[IntervalReport, ...]
    starting_equity_usd: float
    ending_equity_usd: float
    review_enabled: bool
    #: Index of the interval at which the consecutive-loss halt engaged, or None
    #: if it never did. Every interval at or after this index was blocked before
    #: the optimizer's output could reach the broker, so the *effective* sample
    #: is `intervals[:halted_at_interval]`. Reporting a win rate or a Sharpe over
    #: the full range without this would silently average in the flat tail of a
    #: switched-off system.
    halted_at_interval: int | None = None

    @property
    def effective_intervals(self) -> int:
        """Intervals in which trading was still permitted."""
        if self.halted_at_interval is None:
            return len(self.intervals)
        return self.halted_at_interval + 1

    @property
    def total_return(self) -> float:
        if self.starting_equity_usd <= 0:
            return 0.0
        return self.ending_equity_usd / self.starting_equity_usd - 1.0

    @property
    def intervals_traded(self) -> int:
        return sum(1 for i in self.intervals if i.traded())

    @property
    def abstentions(self) -> int:
        return sum(1 for i in self.intervals if not i.reviewer_permits)

    @property
    def reviewed_by_model_every_step(self) -> bool:
        """Whether a model backend answered at every step of a reviewed run.

        False for a bypassed run, an empty run, the offline stand-in, and any run
        in which a step's review failed in transit. The backend that was
        configured says nothing about this: an unreachable model abstains in
        exactly the same way as one that judged the candidate.
        """
        return (
            self.review_enabled
            and bool(self.intervals)
            and all(i.model_responded is True for i in self.intervals)
        )

    def equity_curve(self) -> np.ndarray:
        return np.array(
            [self.starting_equity_usd] + [i.equity_usd for i in self.intervals]
        )


@dataclass
class SessionConfig:
    """One run's parameters."""

    interval_minutes: int = 10
    max_intervals: int = 12          # steps; one price-frame row each (a day for daily bars)
    train_sessions: int = 300        # >= 120; 30 is singular above 28 assets
    cardinality_cap: int = 4
    max_weight: float = 0.4
    risk_aversion: float = 25.0
    review_enabled: bool = True
    position_notional_usd: float = 1.50
    starting_capital_usd: float = 10.0


class SessionRunner:
    """Runs decision intervals over a price history."""

    def __init__(
        self,
        prices: pd.DataFrame,
        config: SessionConfig | None = None,
        optimizer: Optimizer | None = None,
        estimator: ReturnEstimator | None = None,
        reviewer: ReviewerClient | None = None,
        broker: SimulatedBroker | None = None,
        policy: RiskPolicy | None = None,
    ) -> None:
        # The runner replays a price history and reads its own equity from the
        # broker after every step. A broker that places orders somewhere else
        # would be handed historical prices and report equity in other terms, so
        # the drawdown and loss limits would be measured on the wrong figures.
        if broker is not None and not isinstance(broker, SimulatedBroker):
            raise TypeError(
                f"SessionRunner replays history and needs a SimulatedBroker, "
                f"got {type(broker).__name__}"
            )

        # A missing, infinite or non-positive close is treated as missing and
        # carried forward from the last good close. One bad bar on the live-data
        # path would otherwise turn broker equity into NaN for every later step.
        # A symbol with no earlier good close keeps a NaN. `_optimize` leaves
        # such a symbol out of every step whose training window it does not
        # fill, so it never reaches the gate.
        ordered = prices.sort_index()
        self.prices = ordered.where(np.isfinite(ordered) & (ordered > 0)).ffill()
        self.config = config or SessionConfig()
        self.optimizer = optimizer or QuboOptimizer()
        self.estimator = estimator or SampleMeanEstimator()
        self.reviewer = reviewer or OfflineReviewer()
        self.broker = broker or SimulatedBroker(
            starting_cash_usd=self.config.starting_capital_usd
        )
        self.policy = policy or RiskPolicy(
            starting_capital_usd=self.config.starting_capital_usd,
            max_gross_usd=self.config.starting_capital_usd,
            interval_minutes=self.config.interval_minutes,
        )
        self.gate = RiskGate(self.policy)

        if len(self.prices) < self.config.train_sessions + 2:
            raise ValueError(
                f"need at least {self.config.train_sessions + 2} price rows, "
                f"got {len(self.prices)}"
            )

    # ------------------------------------------------------------------ evidence

    @staticmethod
    def _evidence() -> tuple[KnowledgeRecord, ...]:
        return tuple(FORMULA_INDEX[i] for i in REVIEW_EVIDENCE_IDS if i in FORMULA_INDEX)

    # --------------------------------------------------------------------- cycle

    def run(self) -> SessionReport:
        start_equity = self.config.starting_capital_usd
        peak_equity = start_equity
        equity = start_equity
        realized_total = 0.0
        consecutive_losses = 0
        halted_at_interval: int | None = None
        recent_ids: dict[str, datetime] = {}
        reports: list[IntervalReport] = []

        clock = datetime.now(timezone.utc).replace(microsecond=0)
        window_start = self.config.train_sessions

        for index in range(self.config.max_intervals):
            if window_start + 1 >= len(self.prices):
                break

            opened_at = clock + timedelta(minutes=self.config.interval_minutes * index)
            closed_at = opened_at + timedelta(minutes=self.config.interval_minutes)

            train = self.prices.iloc[window_start - self.config.train_sessions : window_start]
            forward = self.prices.iloc[window_start - 1 : window_start + 1]

            result, problem, symbols = self._optimize(train)
            evidence = self._evidence()

            # --- review -------------------------------------------------------
            if self.config.review_enabled:
                review = self.reviewer.review(
                    ReviewRequest(
                        candidate_id=result.candidate_id,
                        method=result.method,
                        symbols=tuple(symbols),
                        weights_pct=tuple(float(w) for w in result.weights),
                        expected_return=result.expected_return,
                        expected_risk=result.expected_risk,
                        n_positions=result.n_positions,
                        evidence=evidence,
                    )
                )
                reviewer_source = review.source
                reviewer_verdict = review.output.verdict.value
                reviewer_permits = review.permits_trading
                reviewer_error = review.error
                citations_verified: bool | None = review.citations_verified
                model_responded: bool | None = review.model_responded
                repaired_citations = tuple(review.repaired_citations)
                reviewer_observations = tuple(review.output.risk_observations)
            else:
                # Ablation arm: no reviewer at all. Recorded as "bypassed" rather
                # than as an approval, so the report never implies a verdict that
                # was never sought.
                reviewer_source = "none"
                reviewer_verdict = "BYPASSED"
                reviewer_permits = True
                reviewer_error = None
                citations_verified = None
                model_responded = None
                repaired_citations = ()
                reviewer_observations = ()

            # --- risk gate and execution --------------------------------------
            account = AccountState(
                equity_usd=equity,
                peak_equity_usd=peak_equity,
                gross_exposure_usd=0.0,
                open_positions=0,
                session_realized_pnl_usd=realized_total,
                consecutive_losses=consecutive_losses,
                orders_this_interval=0,
                recent_client_order_ids=dict(recent_ids),
            )

            fills, reasons, submitted, decision_label = self._execute(
                result, symbols, train, forward, account, reviewer_permits, opened_at
            )

            # Derive P&L from the equity change rather than from fill fields.
            # Equity is the broker's own accounting, so the two cannot disagree --
            # an earlier version computed P&L from fill notionals and reported
            # 0.0000 every interval while equity visibly moved.
            previous_equity = equity
            equity = self.broker.equity_usd()
            interval_pnl = equity - previous_equity
            realized_total += interval_pnl
            peak_equity = max(peak_equity, equity)

            # Count *traded* intervals only. An interval that submitted no order
            # has P&L of exactly 0.0, which is neither a win nor a loss -- and
            # counting it as "not a win" is what made this counter an absorbing
            # state: once the halt engaged, the gate blocked every order, every
            # subsequent interval scored 0.0, and the reset-on-a-win branch became
            # unreachable. The halt could never clear, and the run looked like a
            # strategy that stopped finding opportunities rather than one that was
            # switched off at interval 13.
            if submitted > 0:
                if interval_pnl < 0:
                    consecutive_losses += 1
                elif interval_pnl > 0:
                    consecutive_losses = 0

            # The halt itself stays latched for the rest of the session. That is
            # deliberate and unchanged: "stop after N consecutive losing trades"
            # is a risk control, and a control that silently resumes is not one.
            # What changes is that it is now recorded, so a truncated sample is
            # visible in the report instead of being mistaken for a flat strategy.
            if (
                halted_at_interval is None
                and consecutive_losses >= self.policy.stop_after_consecutive_losses
            ):
                halted_at_interval = index

            for fill in fills:
                recent_ids[fill.client_order_id] = opened_at

            drawdown = AccountState(
                equity_usd=equity, peak_equity_usd=peak_equity
            ).drawdown_percent

            reports.append(
                IntervalReport(
                    index=index,
                    opened_at=opened_at,
                    closed_at=closed_at,
                    symbols_considered=len(symbols),
                    method=result.method,
                    estimator=self.estimator.name,
                    candidate_id=result.candidate_id,
                    optimizer_positions=result.n_positions,
                    optimizer_expected_return=result.expected_return,
                    optimizer_expected_risk=result.expected_risk,
                    optimizer_runtime_ms=result.runtime_seconds * 1000.0,
                    evidence_ids=tuple(r.source_id for r in evidence),
                    reviewer_source=reviewer_source,
                    reviewer_verdict=reviewer_verdict,
                    reviewer_permits=reviewer_permits,
                    risk_decision=decision_label,
                    risk_reasons=reasons,
                    orders_submitted=submitted,
                    fills=tuple(fills),
                    realized_pnl_usd=interval_pnl,
                    equity_usd=equity,
                    drawdown_percent=drawdown,
                    no_trade_reason=(
                        "" if submitted else (reasons[0] if reasons else "no candidate")
                    ),
                    reviewer_error=reviewer_error,
                    citations_verified=citations_verified,
                    model_responded=model_responded,
                    repaired_citations=repaired_citations,
                    reviewer_observations=reviewer_observations,
                )
            )
            window_start += 1

        return SessionReport(
            mode="forward_paper",
            intervals=tuple(reports),
            starting_equity_usd=start_equity,
            ending_equity_usd=equity,
            review_enabled=self.config.review_enabled,
            halted_at_interval=halted_at_interval,
        )

    # ------------------------------------------------------------------ internals

    @staticmethod
    def _full_window(train: pd.DataFrame) -> pd.DataFrame:
        """The columns of `train` that hold a close on every row of the window.

        Prices were forward-filled in `__init__`, so a gap can only be a leading
        one: a symbol that had no good close yet. Estimating moments for such a
        symbol would mean dropping its missing rows for every symbol, which
        shrinks the window for the whole universe, or a crash when too few rows
        remain. The symbol is left out of this step instead, and the step report
        shows it in `symbols_considered`.
        """
        complete = train.loc[:, train.notna().all(axis=0)]
        if complete.shape[1] == 0:
            raise InsufficientHistory(
                f"no symbol has a close on all {len(train)} rows of the training window"
            )
        return complete

    def _optimize(self, train: pd.DataFrame):
        train = self._full_window(train)
        train_returns = to_returns(train)
        _, covariance, symbols, _ = annualized_moments(train_returns)
        expected = (
            self.estimator.estimate(train, train_returns).reindex(symbols).to_numpy()
        )
        problem = PortfolioProblem(
            expected_returns=expected,
            covariance=covariance,
            symbols=symbols,
            constraints=Constraints(
                max_positions=self.config.cardinality_cap,
                max_weight=self.config.max_weight,
                risk_aversion=self.config.risk_aversion,
            ),
        )
        return self.optimizer.solve(problem), problem, symbols

    def _execute(
        self,
        result,
        symbols: list[str],
        train: pd.DataFrame,
        forward: pd.DataFrame,
        account: AccountState,
        reviewer_permits: bool,
        opened_at: datetime,
    ) -> tuple[list[Fill], tuple[str, ...], int, str]:
        """Propose, gate, and submit. No number here comes from the reviewer.

        The notional is the configured fixed size, capped at the policy's
        single-position limit; the gate approves or rejects it.
        """
        fills: list[Fill] = []
        reasons: list[str] = []
        submitted = 0
        approved_any = False

        ranked = np.argsort(result.weights)[::-1]
        entries = [i for i in ranked if result.weights[i] > 1e-8]

        for rank, asset_index in enumerate(entries):
            if submitted >= self.policy.max_orders_per_interval:
                break

            symbol = symbols[asset_index]
            entry_price = float(train[symbol].iloc[-1])
            notional = min(
                self.config.position_notional_usd,
                self.policy.max_single_position_usd,
            )
            # A symbol with no usable close has no price to derive a stop from.
            # `run` never gets here with one, because `_optimize` leaves such a
            # symbol out. A direct caller can pass one: it goes to the gate with
            # no stop and the bad price, and the gate rejects it and logs why.
            has_price = math.isfinite(entry_price) and entry_price > 0
            stop = (
                self.gate.stop_price_for(entry_price, notional, is_long=True)
                if has_price
                else None
            )

            order = OrderIntent(
                client_order_id=new_client_order_id(),
                symbol=symbol,
                side="buy",
                notional_usd=notional,
                reference_price=entry_price,
                stop_loss_price=stop,
                quote_timestamp=opened_at,
                spread_percent=0.02,
            )
            state = AccountState(
                equity_usd=account.equity_usd,
                peak_equity_usd=account.peak_equity_usd,
                gross_exposure_usd=notional * submitted,
                open_positions=submitted,
                session_realized_pnl_usd=account.session_realized_pnl_usd,
                consecutive_losses=account.consecutive_losses,
                orders_this_interval=submitted,
                recent_client_order_ids=account.recent_client_order_ids,
            )

            decision = self.gate.evaluate(order, state, reviewer_permits, now=opened_at)
            if not decision.approved:
                reasons.extend(r.value for r in decision.reasons)
                continue

            approved_any = True
            buy = self.broker.submit(
                order.client_order_id, symbol, "buy", notional, entry_price
            )
            fills.append(buy)
            if not buy.succeeded:
                continue
            submitted += 1

            # Forced flatten at the end of the interval: exit unconditionally,
            # regardless of whether the position is winning.
            exit_price = float(forward[symbol].iloc[-1])
            breached = self.gate.stop_loss_breached(
                entry_price, exit_price, stop, is_long=True
            )
            realized_exit = stop if breached else exit_price
            fills.append(
                self.broker.submit(
                    new_client_order_id(), symbol, "sell", notional, realized_exit
                )
            )

        label = "APPROVE" if approved_any else "REJECT"
        return fills, tuple(dict.fromkeys(reasons)), submitted, label


def compare_review_ablation(
    prices: pd.DataFrame,
    config: SessionConfig | None = None,
    **runner_kwargs,
) -> dict[str, object]:
    """Run the identical session with and without the reviewer.

    Answers the proposal's question directly: how often does review change the
    decision? Both arms share the same prices, optimizer, estimator, and seed, so
    any difference is attributable to the review step alone.
    """
    base = config or SessionConfig()

    with_review = SessionRunner(
        prices,
        config=SessionConfig(**{**base.__dict__, "review_enabled": True}),
        broker=SimulatedBroker(starting_cash_usd=base.starting_capital_usd),
        **runner_kwargs,
    ).run()
    without_review = SessionRunner(
        prices,
        config=SessionConfig(**{**base.__dict__, "review_enabled": False}),
        broker=SimulatedBroker(starting_cash_usd=base.starting_capital_usd),
        **runner_kwargs,
    ).run()

    changed = sum(
        1
        for a, b in zip(with_review.intervals, without_review.intervals)
        if a.traded() != b.traded()
    )
    total = min(len(with_review.intervals), len(without_review.intervals))

    return {
        "intervals": total,
        "decisions_changed": changed,
        "change_rate": changed / total if total else 0.0,
        "traded_with_review": with_review.intervals_traded,
        "traded_without_review": without_review.intervals_traded,
        "abstentions": with_review.abstentions,
        "return_with_review": with_review.total_return,
        "return_without_review": without_review.total_return,
        "with_review": with_review,
        "without_review": without_review,
    }
