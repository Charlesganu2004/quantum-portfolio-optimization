"""Deterministic risk gate -- approves or rejects each entry order.

Everything here is plain arithmetic on explicit limits. No model output reaches
this code as a number: the reviewer contributes a verdict (approve / reject /
abstain) and nothing else. The gate checks exposure, position counts, losses and
stop levels against the account state and the policy, and derives the stop
price for an order. It does not choose the order size: the caller proposes a
notional (the session runner uses a fixed one) and `evaluate` accepts or rejects
that intent. Exit orders are not routed through `evaluate`; the session runner
sends its end-of-step flattening sell straight to the broker by design.

The design rule is **default-deny**. A missing input, a stale quote, an
unavailable reviewer, an unparseable verdict -- every one of them blocks. A
number that is not finite blocks too: every comparison with NaN is false, so a
limit that is only ever tested with `>` or `<=` would pass it. The only path to
an approved order is for every check to pass affirmatively.

This is where the project's one honest 100% target lives. Profit cannot be
guaranteed by anything; *these controls firing every time* can be, and is tested.

Stop-losses are enforced **client-side**. Alpaca fractional and notional orders
cannot carry a bracket or OCO leg, so no exchange-side protective order exists to
rely on. If this module fails to flatten, nothing else will.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum


class Decision(str, Enum):
    APPROVE = "APPROVE"
    REJECT = "REJECT"


class RejectReason(str, Enum):
    KILL_SWITCH = "kill_switch_engaged"
    REVIEWER_BLOCKED = "reviewer_did_not_approve"
    STALE_DATA = "market_data_too_old"
    WIDE_SPREAD = "spread_above_threshold"
    DUPLICATE_ORDER = "duplicate_client_order_id"
    INTERVAL_ORDER_CAP = "interval_order_cap_reached"
    GROSS_EXPOSURE = "gross_exposure_cap"
    POSITION_SIZE = "single_position_cap"
    POSITION_COUNT = "open_position_cap"
    SESSION_LOSS = "session_loss_cap"
    DRAWDOWN = "max_drawdown_cap"
    CONSECUTIVE_LOSSES = "consecutive_loss_halt"
    SHORTING_DISABLED = "short_sales_disabled"
    LEVERAGE_DISABLED = "leverage_disabled"
    NON_POSITIVE_SIZE = "non_positive_order_size"
    MISSING_STOP = "no_stop_loss_attached"
    INVALID_ORDER = "invalid_order_input"
    NON_FINITE_ACCOUNT = "non_finite_account_state"


@dataclass(frozen=True)
class RiskPolicy:
    """Configured limits, set in code. There is no configuration file."""

    starting_capital_usd: float = 10.00
    max_gross_usd: float = 10.00
    max_single_position_usd: float = 3.00
    max_open_positions: int = 4
    leverage_allowed: bool = False
    options_allowed: bool = False
    short_sales_allowed: bool = False

    max_loss_per_trade_usd: float = 0.10
    max_session_loss_usd: float = 0.50
    max_drawdown_percent: float = 5.0
    stop_after_consecutive_losses: int = 3

    interval_minutes: int = 10
    max_orders_per_interval: int = 2
    reject_if_data_age_seconds_above: float = 5.0
    reject_if_spread_percent_above: float = 0.50
    duplicate_order_window_seconds: float = 120.0

    def __post_init__(self) -> None:
        if self.starting_capital_usd <= 0:
            raise ValueError("starting_capital_usd must be positive")
        if self.max_single_position_usd > self.max_gross_usd:
            raise ValueError(
                f"max_single_position_usd ({self.max_single_position_usd}) exceeds "
                f"max_gross_usd ({self.max_gross_usd})"
            )
        if self.max_open_positions < 1:
            raise ValueError("max_open_positions must be at least 1")


@dataclass
class AccountState:
    """Everything the gate needs to know about the account right now."""

    equity_usd: float
    peak_equity_usd: float
    gross_exposure_usd: float = 0.0
    open_positions: int = 0
    session_realized_pnl_usd: float = 0.0
    consecutive_losses: int = 0
    orders_this_interval: int = 0
    recent_client_order_ids: dict[str, datetime] = field(default_factory=dict)

    @property
    def drawdown_percent(self) -> float:
        """Percent below the peak, or NaN when either figure is not finite.

        NaN is the honest answer for a broken account. Returning 0.0 would read
        as "no drawdown" and switch the drawdown limit off, so `RiskGate.evaluate`
        halts on a non-finite account instead of consulting this value.
        """
        if not (math.isfinite(self.equity_usd) and math.isfinite(self.peak_equity_usd)):
            return math.nan
        if self.peak_equity_usd <= 0:
            return 0.0
        return max(0.0, (1.0 - self.equity_usd / self.peak_equity_usd) * 100.0)


@dataclass(frozen=True)
class OrderIntent:
    """A proposed order, fully specified by deterministic code."""

    client_order_id: str
    symbol: str
    side: str                     # "buy" | "sell"
    notional_usd: float
    reference_price: float
    stop_loss_price: float | None
    quote_timestamp: datetime
    spread_percent: float
    is_short: bool = False
    uses_leverage: bool = False


@dataclass(frozen=True)
class RiskDecision:
    """The gate's verdict, with every reason it blocked."""

    decision: Decision
    reasons: tuple[RejectReason, ...] = ()
    detail: tuple[str, ...] = ()

    @property
    def approved(self) -> bool:
        return self.decision is Decision.APPROVE


class RiskGate:
    """Evaluates an order intent against the policy and account state."""

    def __init__(self, policy: RiskPolicy | None = None) -> None:
        self.policy = policy or RiskPolicy()
        self._kill_switch = False

    # --------------------------------------------------------------- kill switch

    def engage_kill_switch(self) -> None:
        """Halt all new orders. Requires explicit manual reset.

        An available control: `evaluate` honours it, but nothing in
        `SessionRunner` engages it. A caller has to.
        """
        self._kill_switch = True

    def reset_kill_switch(self) -> None:
        self._kill_switch = False

    @property
    def kill_switch_engaged(self) -> bool:
        return self._kill_switch

    # ------------------------------------------------------------------ decision

    def evaluate(
        self,
        intent: OrderIntent,
        account: AccountState,
        reviewer_approved: bool,
        now: datetime | None = None,
    ) -> RiskDecision:
        """Approve only if every check passes. Any failure blocks.

        All failing checks are collected rather than short-circuiting, so the
        activity log records every reason an order was refused instead of only
        the first.
        """
        now = now or datetime.now(timezone.utc)
        reasons: list[RejectReason] = []
        detail: list[str] = []

        def block(reason: RejectReason, message: str) -> None:
            reasons.append(reason)
            detail.append(message)

        if self._kill_switch:
            block(RejectReason.KILL_SWITCH, "kill switch engaged; manual reset required")

        if not reviewer_approved:
            block(
                RejectReason.REVIEWER_BLOCKED,
                "reviewer did not return APPROVE with verified citations",
            )

        # --- account state -------------------------------------------------------
        # Non-finite account figures halt trading. With a NaN equity the drawdown
        # reads as unknown and the session-loss comparison is false, so both
        # limits would silently stop firing.
        non_finite = [
            name
            for name, value in (
                ("equity_usd", account.equity_usd),
                ("peak_equity_usd", account.peak_equity_usd),
                ("gross_exposure_usd", account.gross_exposure_usd),
                ("session_realized_pnl_usd", account.session_realized_pnl_usd),
            )
            if not math.isfinite(value)
        ]
        if non_finite:
            block(
                RejectReason.NON_FINITE_ACCOUNT,
                f"account state is not finite ({', '.join(non_finite)}); "
                f"the loss limits cannot be checked",
            )

        # --- order inputs --------------------------------------------------------
        if intent.side not in ("buy", "sell"):
            block(
                RejectReason.INVALID_ORDER,
                f"side {intent.side!r} is neither 'buy' nor 'sell'",
            )
        if not math.isfinite(intent.reference_price) or intent.reference_price <= 0:
            block(
                RejectReason.INVALID_ORDER,
                f"reference price {intent.reference_price} is not a positive "
                f"finite number",
            )
        if not math.isfinite(intent.spread_percent) or intent.spread_percent < 0:
            block(
                RejectReason.INVALID_ORDER,
                f"spread {intent.spread_percent} is not a finite, non-negative number",
            )

        # --- market data quality -------------------------------------------------
        age = (now - intent.quote_timestamp).total_seconds()
        if age < 0:
            # A quote stamped after the decision time cannot be checked for age,
            # and a future stamp would otherwise read as perfectly fresh.
            block(
                RejectReason.INVALID_ORDER,
                f"quote is stamped {-age:.1f}s after the decision time",
            )
        if age > self.policy.reject_if_data_age_seconds_above:
            block(
                RejectReason.STALE_DATA,
                f"quote is {age:.1f}s old, above the "
                f"{self.policy.reject_if_data_age_seconds_above:.1f}s limit",
            )
        if intent.spread_percent > self.policy.reject_if_spread_percent_above:
            block(
                RejectReason.WIDE_SPREAD,
                f"spread {intent.spread_percent:.2f}% above the "
                f"{self.policy.reject_if_spread_percent_above:.2f}% limit",
            )

        # --- order hygiene -------------------------------------------------------
        previous = account.recent_client_order_ids.get(intent.client_order_id)
        if previous is not None:
            elapsed = (now - previous).total_seconds()
            if elapsed <= self.policy.duplicate_order_window_seconds:
                block(
                    RejectReason.DUPLICATE_ORDER,
                    f"client_order_id {intent.client_order_id} was submitted "
                    f"{elapsed:.0f}s ago",
                )
        if account.orders_this_interval >= self.policy.max_orders_per_interval:
            block(
                RejectReason.INTERVAL_ORDER_CAP,
                f"{account.orders_this_interval} orders already placed this interval "
                f"(cap {self.policy.max_orders_per_interval})",
            )

        # --- instrument permissions ---------------------------------------------
        if intent.is_short and not self.policy.short_sales_allowed:
            block(RejectReason.SHORTING_DISABLED, "short sales are disabled")
        if intent.uses_leverage and not self.policy.leverage_allowed:
            block(RejectReason.LEVERAGE_DISABLED, "leverage is disabled")

        # --- sizing --------------------------------------------------------------
        if not math.isfinite(intent.notional_usd):
            block(
                RejectReason.INVALID_ORDER,
                f"order notional {intent.notional_usd} is not a finite number",
            )
        elif intent.notional_usd <= 0:
            block(
                RejectReason.NON_POSITIVE_SIZE,
                f"order notional {intent.notional_usd} is not positive",
            )
        if intent.notional_usd > self.policy.max_single_position_usd:
            block(
                RejectReason.POSITION_SIZE,
                f"${intent.notional_usd:.2f} exceeds the "
                f"${self.policy.max_single_position_usd:.2f} single-position cap",
            )

        if intent.side == "buy":
            projected = account.gross_exposure_usd + intent.notional_usd
            if projected > self.policy.max_gross_usd:
                block(
                    RejectReason.GROSS_EXPOSURE,
                    f"projected gross ${projected:.2f} exceeds the "
                    f"${self.policy.max_gross_usd:.2f} cap",
                )
            if account.open_positions >= self.policy.max_open_positions:
                block(
                    RejectReason.POSITION_COUNT,
                    f"{account.open_positions} positions open (cap "
                    f"{self.policy.max_open_positions})",
                )
            # Only entries need a protective stop; exits are themselves the exit.
            if intent.stop_loss_price is None:
                block(
                    RejectReason.MISSING_STOP,
                    "entry has no stop_loss_price; fractional orders cannot carry "
                    "an exchange-side bracket, so a client-side stop is mandatory",
                )
            elif (
                not math.isfinite(intent.stop_loss_price)
                or intent.stop_loss_price <= 0
                or intent.stop_loss_price >= intent.reference_price
            ):
                # A stop at or below zero can never trigger, so it protects nothing.
                block(
                    RejectReason.INVALID_ORDER,
                    f"stop {intent.stop_loss_price} is not a finite price between "
                    f"zero and the reference price {intent.reference_price} of a "
                    f"long entry",
                )

        # --- loss controls -------------------------------------------------------
        if account.session_realized_pnl_usd <= -self.policy.max_session_loss_usd:
            block(
                RejectReason.SESSION_LOSS,
                f"session loss ${-account.session_realized_pnl_usd:.2f} reached the "
                f"${self.policy.max_session_loss_usd:.2f} limit",
            )
        if account.drawdown_percent >= self.policy.max_drawdown_percent:
            block(
                RejectReason.DRAWDOWN,
                f"drawdown {account.drawdown_percent:.2f}% reached the "
                f"{self.policy.max_drawdown_percent:.2f}% limit",
            )
        if account.consecutive_losses >= self.policy.stop_after_consecutive_losses:
            block(
                RejectReason.CONSECUTIVE_LOSSES,
                f"{account.consecutive_losses} consecutive losses reached the halt "
                f"threshold of {self.policy.stop_after_consecutive_losses}",
            )

        if reasons:
            return RiskDecision(Decision.REJECT, tuple(reasons), tuple(detail))
        return RiskDecision(Decision.APPROVE)

    # ------------------------------------------------------------- forced exits

    def stop_loss_breached(
        self, entry_price: float, current_price: float, stop_price: float, is_long: bool
    ) -> bool:
        """Client-side stop check. Runs every tick; there is no exchange-side leg."""
        if is_long:
            return current_price <= stop_price
        return current_price >= stop_price

    def must_flatten(
        self, opened_at: datetime, now: datetime | None = None
    ) -> bool:
        """Whether a position has exceeded the configured holding interval.

        The forced-flatten rule: nothing is held past the interval, regardless of
        whether it is winning or losing. Removes the discretion that turns a small
        loss into a large one. An available control for callers with a live
        clock: `SessionRunner` does not call it and flattens every position at
        the end of each step instead.
        """
        now = now or datetime.now(timezone.utc)
        return now - opened_at >= timedelta(minutes=self.policy.interval_minutes)

    def stop_price_for(
        self, entry_price: float, notional_usd: float, is_long: bool
    ) -> float:
        """Stop level that caps the loss at `max_loss_per_trade_usd`.

        Derived from the configured dollar limit rather than a percentage guess,
        so the risk per trade is the number in the policy.
        """
        if entry_price <= 0:
            raise ValueError(f"entry_price must be positive, got {entry_price}")
        if notional_usd <= 0:
            raise ValueError(f"notional_usd must be positive, got {notional_usd}")
        move_fraction = self.policy.max_loss_per_trade_usd / notional_usd
        move_fraction = min(move_fraction, 1.0)
        if is_long:
            return entry_price * (1.0 - move_fraction)
        return entry_price * (1.0 + move_fraction)
