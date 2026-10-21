"""Preflight for the Alpaca paper connection. Reads only -- places no order.

Run this before `run_session.py` on live data. It answers, in order, the four
things that actually go wrong:

  1. Is the endpoint the paper endpoint, after URL normalization?
  2. Do the credentials authenticate?
  3. Does the data plan return bars on the configured feed?
  4. Does the requested lookback support the universe, or is the covariance
     singular?

Question 4 is the one people skip. A 30-calendar-day window yields about 22
sessions and 21 return observations, so the sample covariance has rank at most
20. That is full rank for 16 symbols but singular above 20, and far below the
120 sessions this project treats as the minimum for a stable estimate --
shrinkage makes the number *appear*, it does not make it *informative*. The
check prints the rank so the choice is made against a measurement.

Nothing here submits, cancels, or modifies anything. The only write this script
performs is to the local bar cache.
"""

from __future__ import annotations

import sys
import warnings
from datetime import date, timedelta

import requests

from quantlab.config import load_settings, normalize_base_url
from quantlab.data.features import annualized_moments, to_returns
from quantlab.data.sources import (
    DEFAULT_UNIVERSE,
    AlpacaBarSource,
    BarRequest,
    DataUnavailable,
)
from quantlab.execution.broker import (
    AlpacaPaperBroker,
    BrokerError,
    LiveTradingDisabled,
    is_paper_endpoint,
)

#: Minimum sessions before the sample covariance can carry the 16-asset
#: universe. Below this the optimizer is choosing between assets whose
#: correlations were never observed.
MIN_VIABLE_SESSIONS = 120

OK = "  ok  "
FAIL = " FAIL "
WARN = " warn "


def line(status: str, label: str, detail: str = "") -> None:
    print(f"[{status}] {label:<28} {detail}")


def check_endpoint(settings) -> bool:
    raw_trading = settings.trading_base_url
    line(OK, "trading endpoint", raw_trading)
    line(OK, "data endpoint", settings.data_base_url)
    line(OK, "data feed", settings.data_feed)

    if not is_paper_endpoint(raw_trading):
        line(
            FAIL,
            "paper endpoint",
            f"{raw_trading} is not the paper host; this project has no live path",
        )
        return False
    line(OK, "paper endpoint", "host is paper-api.alpaca.markets")
    return True


def check_credentials(settings) -> AlpacaPaperBroker | None:
    if not settings.has_alpaca_credentials:
        line(FAIL, "credentials", "ALPACA_API_KEY_ID / _SECRET_KEY not set in .env")
        return None
    line(OK, "credentials", f"key id ...{settings.alpaca_key_id[-4:]} loaded")

    try:
        broker = AlpacaPaperBroker(settings=settings)
    except LiveTradingDisabled as exc:
        line(FAIL, "broker construction", str(exc))
        return None

    try:
        account = broker.account()
    except BrokerError as exc:
        line(FAIL, "account read", str(exc))
        return None

    line(OK, "account status", str(account.get("status", "unknown")))
    line(OK, "account equity", f"${float(account.get('equity', 0.0)):,.2f}")
    line(OK, "buying power", f"${float(account.get('buying_power', 0.0)):,.2f}")
    if account.get("trading_blocked") or account.get("account_blocked"):
        line(WARN, "account flags", "trading or account is blocked; orders will reject")
    return broker


def check_clock(settings) -> None:
    try:
        response = requests.get(
            f"{settings.trading_base_url}/v2/clock",
            headers=settings.auth_headers(),
            timeout=15.0,
        )
        response.raise_for_status()
        clock = response.json()
    except (requests.RequestException, ValueError) as exc:
        line(WARN, "market clock", f"unavailable: {exc}")
        return
    state = "OPEN" if clock.get("is_open") else "closed"
    line(OK, "market clock", f"{state}, next open {clock.get('next_open', '?')}")


def check_data(settings, lookback_days: int) -> None:
    end = date.today() - timedelta(days=1)
    request = BarRequest(
        symbols=DEFAULT_UNIVERSE, start=end - timedelta(days=lookback_days), end=end
    )
    try:
        prices = AlpacaBarSource(settings=settings).fetch(request)
    except (DataUnavailable, requests.RequestException) as exc:
        line(FAIL, f"bars ({lookback_days}d)", str(exc)[:120])
        return

    line(
        OK,
        f"bars ({lookback_days}d)",
        f"{prices.shape[0]} sessions x {prices.shape[1]} symbols, "
        f"{request.start}..{request.end}",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        _, _, _, diagnostics = annualized_moments(to_returns(prices))

    status = OK if not diagnostics.is_singular_sample else WARN
    line(status, "covariance", diagnostics.summary())
    if diagnostics.is_singular_sample:
        line(
            WARN,
            "lookback verdict",
            f"{diagnostics.n_observations} observations cannot support "
            f"{prices.shape[1]} assets (rank {diagnostics.rank}); "
            f"shrinkage hides this. Use >= {MIN_VIABLE_SESSIONS} sessions.",
        )
    else:
        line(OK, "lookback verdict", "sample covariance is full rank")


def main() -> int:
    lookback_days = int(sys.argv[1]) if len(sys.argv) > 1 else 30

    print("Alpaca paper preflight -- read-only, places no orders\n")
    settings = load_settings()

    if not check_endpoint(settings):
        return 1
    broker = check_credentials(settings)
    if broker is None:
        return 1
    check_clock(settings)
    print()
    check_data(settings, lookback_days)
    if lookback_days < MIN_VIABLE_SESSIONS:
        print()
        check_data(settings, 365)

    print(
        f"\nNormalization note: a URL of "
        f"'https://paper-api.alpaca.markets/v2' in .env resolves to "
        f"'{normalize_base_url('https://paper-api.alpaca.markets/v2')}', because "
        f"every request builder appends its own /v2 path."
    )
    print("No order was submitted. `run_session.py` is the script that trades.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
