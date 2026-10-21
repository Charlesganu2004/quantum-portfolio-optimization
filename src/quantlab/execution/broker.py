"""Execution adapters. Paper only -- live execution is not implemented.

Three implementations behind one `Broker` interface:

* `SimulatedBroker`  -- local deterministic fills. No network, no credentials.
* `AlpacaPaperBroker` -- Alpaca's paper environment, real quotes, simulated fills.
* `LiveDisabledBroker` -- refuses every order with a constant.

`LiveDisabledBroker` exists so that "live trading is off" is a *tested property*
of the code rather than a claim in a document. A test asserts it refuses, and
another asserts that `AlpacaPaperBroker` accepts only the paper host.

Paper trading is not free of illusions and the report should say so: paper fills
exclude latency slippage, queue position, and market impact, all of which move in
the wrong direction for a real account.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from urllib.parse import urlsplit

import requests

from quantlab.config import Settings, load_settings

LIVE_TRADING_DISABLED = "LIVE_TRADING_DISABLED_REQUIRES_SEPARATE_APPROVAL"

#: The only host `AlpacaPaperBroker` will send orders to.
PAPER_TRADING_HOST = "paper-api.alpaca.markets"


def is_paper_endpoint(url: str) -> bool:
    """True only for an https URL whose host is exactly `PAPER_TRADING_HOST`.

    The URL is parsed rather than searched for a substring. A substring test
    accepts "https://paper-api@api.alpaca.markets", whose real host is the live
    one and whose "paper-api" is only the user name.
    """
    try:
        parts = urlsplit(url)
        parts.port  # noqa: B018 -- raises ValueError on a malformed port
    except ValueError:
        return False
    return (
        parts.scheme == "https"
        and parts.username is None
        and parts.password is None
        and parts.hostname == PAPER_TRADING_HOST
    )


class OrderStatus(str, Enum):
    FILLED = "filled"
    REJECTED = "rejected"
    PENDING = "pending"


class BrokerError(RuntimeError):
    """Raised when a broker cannot process a request."""


class LiveTradingDisabled(RuntimeError):
    """Raised on any attempt to place a real-money order."""


@dataclass(frozen=True)
class Fill:
    """The outcome of one submitted order."""

    client_order_id: str
    symbol: str
    side: str
    status: OrderStatus
    requested_notional_usd: float
    filled_notional_usd: float
    filled_price: float
    filled_at: datetime
    fee_usd: float = 0.0
    slippage_usd: float = 0.0
    venue: str = "simulated"
    note: str = ""

    @property
    def succeeded(self) -> bool:
        return self.status is OrderStatus.FILLED


@dataclass
class SimulatedPosition:
    symbol: str
    notional_usd: float
    entry_price: float
    opened_at: datetime


class Broker(ABC):
    name: str = "unnamed"
    is_live: bool = False

    @abstractmethod
    def submit(
        self,
        client_order_id: str,
        symbol: str,
        side: str,
        notional_usd: float,
        reference_price: float,
    ) -> Fill:
        ...


class SimulatedBroker(Broker):
    """Deterministic local fills with configurable cost assumptions.

    Defaults model Alpaca US equities: commission-free, so cost is the spread
    only. `slippage_bps` is charged against the trader on both sides -- optimistic
    fill assumptions are the most common way a backtest lies.
    """

    name = "simulated"
    is_live = False

    def __init__(
        self,
        starting_cash_usd: float = 10.0,
        commission_bps: float = 0.0,
        slippage_bps: float = 2.0,
    ) -> None:
        if starting_cash_usd <= 0:
            raise ValueError("starting_cash_usd must be positive")
        if commission_bps < 0 or slippage_bps < 0:
            raise ValueError("cost assumptions cannot be negative")
        self.cash_usd = starting_cash_usd
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.positions: dict[str, SimulatedPosition] = {}
        self.fills: list[Fill] = []

    def submit(
        self,
        client_order_id: str,
        symbol: str,
        side: str,
        notional_usd: float,
        reference_price: float,
    ) -> Fill:
        now = datetime.now(timezone.utc)
        if side not in ("buy", "sell"):
            raise BrokerError(f"side must be 'buy' or 'sell', got {side!r}")
        if reference_price <= 0:
            raise BrokerError(f"reference_price must be positive, got {reference_price}")

        # Slippage always works against the order.
        drift = self.slippage_bps / 10_000.0
        fill_price = reference_price * (1.0 + drift if side == "buy" else 1.0 - drift)
        fee = notional_usd * self.commission_bps / 10_000.0
        slippage_cost = notional_usd * drift

        if side == "buy":
            if notional_usd + fee > self.cash_usd + 1e-9:
                fill = Fill(
                    client_order_id=client_order_id,
                    symbol=symbol,
                    side=side,
                    status=OrderStatus.REJECTED,
                    requested_notional_usd=notional_usd,
                    filled_notional_usd=0.0,
                    filled_price=0.0,
                    filled_at=now,
                    note=f"insufficient cash: ${self.cash_usd:.2f} available",
                )
                self.fills.append(fill)
                return fill
            self.cash_usd -= notional_usd + fee
            existing = self.positions.get(symbol)
            if existing is None:
                self.positions[symbol] = SimulatedPosition(
                    symbol, notional_usd, fill_price, now
                )
            else:
                existing.notional_usd += notional_usd
        else:
            position = self.positions.get(symbol)
            if position is None:
                fill = Fill(
                    client_order_id=client_order_id,
                    symbol=symbol,
                    side=side,
                    status=OrderStatus.REJECTED,
                    requested_notional_usd=notional_usd,
                    filled_notional_usd=0.0,
                    filled_price=0.0,
                    filled_at=now,
                    note="no open position to sell",
                )
                self.fills.append(fill)
                return fill
            proceeds = position.notional_usd * (fill_price / position.entry_price)
            self.cash_usd += proceeds - fee
            del self.positions[symbol]
            # Keep the caller's request distinct from the proceeds. Overwriting
            # `notional_usd` here would make requested == filled on every sell,
            # so any P&L derived from their difference silently reads zero.
            filled_notional = proceeds

        if side == "buy":
            filled_notional = notional_usd

        fill = Fill(
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            status=OrderStatus.FILLED,
            requested_notional_usd=notional_usd,
            filled_notional_usd=filled_notional,
            filled_price=fill_price,
            filled_at=now,
            fee_usd=fee,
            slippage_usd=slippage_cost,
        )
        self.fills.append(fill)
        return fill

    def equity_usd(self, marks: dict[str, float] | None = None) -> float:
        marks = marks or {}
        held = sum(
            position.notional_usd
            * (marks.get(symbol, position.entry_price) / position.entry_price)
            for symbol, position in self.positions.items()
        )
        return self.cash_usd + held


class AlpacaPaperBroker(Broker):
    """Alpaca paper trading. Real quotes, simulated fills, no real money.

    Refuses to start against a non-paper base URL, so a mistyped endpoint cannot
    silently become live trading.
    """

    name = "alpaca_paper"
    is_live = False

    def __init__(self, settings: Settings | None = None, timeout: float = 30.0) -> None:
        self.settings = settings or load_settings()
        if not is_paper_endpoint(self.settings.trading_base_url):
            raise LiveTradingDisabled(
                f"trading_base_url is {self.settings.trading_base_url!r}, which is "
                f"not the paper endpoint. This project has no live execution path; "
                f"set ALPACA_TRADING_BASE_URL to https://{PAPER_TRADING_HOST}"
            )
        self.timeout = timeout

    def account(self) -> dict:
        """Read-only account snapshot. Places no order and moves no money.

        The paper/live distinction is enforced by the base-URL check in
        `__init__`, not by anything in this response -- Alpaca's account object
        has no "this is paper" field, so the host is the authority.
        """
        try:
            response = requests.get(
                f"{self.settings.trading_base_url}/v2/account",
                headers=self.settings.auth_headers(),
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise BrokerError(f"Alpaca unreachable: {exc}") from exc

        if response.status_code == 401 or response.status_code == 403:
            raise BrokerError(
                f"Alpaca rejected the credentials (HTTP {response.status_code}). "
                f"Check that .env holds PAPER keys, not live keys."
            )
        if response.status_code >= 400:
            raise BrokerError(f"HTTP {response.status_code}: {response.text[:200]}")
        return response.json()

    def equity_usd(self) -> float:
        return float(self.account().get("equity", 0.0))

    def submit(
        self,
        client_order_id: str,
        symbol: str,
        side: str,
        notional_usd: float,
        reference_price: float,
    ) -> Fill:
        self.settings.require_alpaca()
        payload = {
            "symbol": symbol,
            "notional": round(notional_usd, 2),
            "side": side,
            "type": "market",       # Fractional/notional orders are market-only.
            "time_in_force": "day",  # Fractional orders require DAY.
            "client_order_id": client_order_id,
        }
        try:
            response = requests.post(
                f"{self.settings.trading_base_url}/v2/orders",
                json=payload,
                headers=self.settings.auth_headers(),
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise BrokerError(f"Alpaca unreachable: {exc}") from exc

        if response.status_code >= 400:
            return Fill(
                client_order_id=client_order_id,
                symbol=symbol,
                side=side,
                status=OrderStatus.REJECTED,
                requested_notional_usd=notional_usd,
                filled_notional_usd=0.0,
                filled_price=0.0,
                filled_at=datetime.now(timezone.utc),
                venue=self.name,
                note=f"HTTP {response.status_code}: {response.text[:200]}",
            )

        body = response.json()
        filled_price = float(body.get("filled_avg_price") or reference_price)
        filled_qty = float(body.get("filled_qty") or 0.0)
        return Fill(
            client_order_id=client_order_id,
            symbol=symbol,
            side=side,
            status=OrderStatus.FILLED if filled_qty > 0 else OrderStatus.PENDING,
            requested_notional_usd=notional_usd,
            filled_notional_usd=filled_qty * filled_price,
            filled_price=filled_price,
            filled_at=datetime.now(timezone.utc),
            venue=self.name,
            note=str(body.get("status", "")),
        )


class LiveDisabledBroker(Broker):
    """Refuses everything. The live path is deliberately unimplemented."""

    name = "live_disabled"
    is_live = False

    def submit(
        self,
        client_order_id: str,
        symbol: str,
        side: str,
        notional_usd: float,
        reference_price: float,
    ) -> Fill:
        raise LiveTradingDisabled(LIVE_TRADING_DISABLED)


def new_client_order_id(prefix: str = "ql") -> str:
    """Unique id so the duplicate-order guard has something to key on."""
    return f"{prefix}-{uuid.uuid4().hex[:16]}"


def resolve_broker(prefer_alpaca: bool = True) -> Broker:
    """Alpaca paper when credentials exist, local simulator otherwise."""
    if prefer_alpaca and load_settings().has_alpaca_credentials:
        try:
            return AlpacaPaperBroker()
        except LiveTradingDisabled:
            raise
        except Exception:
            return SimulatedBroker()
    return SimulatedBroker()
