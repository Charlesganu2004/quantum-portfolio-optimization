"""Market data sources behind one interface.

Three implementations share `BarSource` so the rest of the system never learns
where prices came from:

* `AlpacaBarSource`  -- real bars, needs paper credentials.
* `CsvBarSource`     -- deterministic replay for regression tests.
* `SyntheticBarSource` -- factor-model prices, no credentials, so the full
  four-method benchmark runs on a machine that has never touched a broker.

Alpaca is called over plain REST with `requests` rather than through `alpaca-py`.
The endpoints are identical; this keeps the dependency surface at zero new
packages and makes the exact request visible in the methods section.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from quantlab.config import Settings, load_settings, project_root

#: 16 liquid US large caps spread across sectors. Sized to the proposal's 8-16
#: asset MVP; the scalability sweep samples subsets down to 4 and synthesizes up.
DEFAULT_UNIVERSE: tuple[str, ...] = (
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META",   # technology
    "JPM", "BAC",                                       # financials
    "JNJ", "UNH",                                       # healthcare
    "XOM", "CVX",                                       # energy
    "WMT", "PG",                                        # staples
    "HD",                                               # discretionary
    "CAT",                                              # industrials
)


class DataUnavailable(RuntimeError):
    """Raised when a source cannot return usable bars."""


@dataclass(frozen=True)
class BarRequest:
    symbols: tuple[str, ...]
    start: date
    end: date
    timeframe: str = "1Day"

    def cache_key(self) -> str:
        # hashlib, not the built-in hash(): hash() of a string is salted per
        # process, which gave every run a new key and made the cache never hit.
        joined = "-".join(sorted(self.symbols))
        digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:10]
        return f"{self.timeframe}_{self.start}_{self.end}_{len(self.symbols)}a_{digest}"


class BarSource(ABC):
    """Returns a wide close-price frame: index = timestamp, columns = symbols."""

    name: str = "unnamed"

    @abstractmethod
    def fetch(self, request: BarRequest) -> pd.DataFrame:
        ...


class AlpacaBarSource(BarSource):
    """Historical bars from Alpaca's market data API.

    The default feed is IEX, a single exchange. Set `ALPACA_DATA_FEED=sip` for
    full-market bars. Alpaca's market-data documentation lists real-time data on
    the free plan as IEX only and limits its historical data only by a 15-minute
    delay, so daily bars that end before today are not affected by it.
    """

    name = "alpaca"

    def __init__(
        self,
        settings: Settings | None = None,
        cache_dir: Path | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.settings = settings or load_settings()
        self.cache_dir = cache_dir or project_root() / "data_cache"
        self.timeout = timeout

    def fetch(self, request: BarRequest) -> pd.DataFrame:
        cached = self._read_cache(request)
        if cached is not None:
            return cached

        self.settings.require_alpaca()
        # Alpaca's page limit counts bars across all symbols, so a page can end
        # partway through a symbol and the same date can appear on several pages
        # (once per symbol). Rows are therefore gathered per symbol across every
        # page and the frame is built once at the end. Concatenating per-page
        # frames and dropping repeated dates would discard whole symbols.
        gathered: dict[str, list[dict]] = {}
        page_token: str | None = None
        url = f"{self.settings.data_base_url}/v2/stocks/bars"

        while True:
            params: dict[str, str | int] = {
                "symbols": ",".join(request.symbols),
                "timeframe": request.timeframe,
                "start": request.start.isoformat(),
                "end": request.end.isoformat(),
                "feed": self.settings.data_feed,
                "adjustment": "split",
                "limit": 10_000,
            }
            if page_token:
                params["page_token"] = page_token

            response = requests.get(
                url,
                params=params,
                headers=self.settings.auth_headers(),
                timeout=self.timeout,
            )
            if response.status_code == 401:
                raise DataUnavailable(
                    "Alpaca rejected the credentials (401). Check that .env holds "
                    "PAPER keys from https://alpaca.markets and that they are current."
                )
            if response.status_code == 429:
                raise DataUnavailable(
                    "Alpaca rate limit hit (429). The free plan allows 200 "
                    "requests/minute; wait and retry."
                )
            response.raise_for_status()
            payload = response.json()

            for symbol, rows in (payload.get("bars") or {}).items():
                gathered.setdefault(symbol, []).extend(rows or [])
            page_token = payload.get("next_page_token")
            if not page_token:
                break

        combined = self._payload_to_frame(gathered)
        if combined.empty:
            raise DataUnavailable(
                f"Alpaca returned no bars for {request.symbols} between "
                f"{request.start} and {request.end} on feed "
                f"{self.settings.data_feed!r}."
            )
        self._write_cache(request, combined)
        return combined

    @staticmethod
    def _payload_to_frame(bars: dict[str, list[dict]]) -> pd.DataFrame:
        series: dict[str, pd.Series] = {}
        for symbol, rows in bars.items():
            if not rows:
                continue
            stamps = pd.to_datetime([row["t"] for row in rows], utc=True)
            closes = pd.Series(
                [float(row["c"]) for row in rows], index=stamps, name=symbol
            )
            # A repeated timestamp within one symbol keeps its last bar.
            series[symbol] = closes[~closes.index.duplicated(keep="last")]
        if not series:
            return pd.DataFrame()
        return pd.DataFrame(series).sort_index()

    def _cache_path(self, request: BarRequest) -> Path:
        return self.cache_dir / f"alpaca_{request.cache_key()}.parquet"

    def _read_cache(self, request: BarRequest) -> pd.DataFrame | None:
        path = self._cache_path(request)
        if not path.is_file():
            return None
        try:
            return pd.read_parquet(path)
        except Exception:
            # A corrupt cache must never mask live data; refetch instead.
            return None

    def _write_cache(self, request: BarRequest, frame: pd.DataFrame) -> None:
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(self._cache_path(request))
        except Exception:
            pass  # Caching is an optimization, never a correctness requirement.


class CsvBarSource(BarSource):
    """Deterministic replay from a stored CSV. Used for regression tests."""

    name = "csv_replay"

    def __init__(self, path: Path) -> None:
        self.path = path

    def fetch(self, request: BarRequest) -> pd.DataFrame:
        if not self.path.is_file():
            raise DataUnavailable(f"replay file not found: {self.path}")
        frame = pd.read_csv(self.path, index_col=0, parse_dates=True)
        missing = [s for s in request.symbols if s not in frame.columns]
        if missing:
            raise DataUnavailable(f"replay file lacks symbols: {missing}")
        window = frame.loc[
            str(request.start) : str(request.end), list(request.symbols)
        ]
        if window.empty:
            raise DataUnavailable(
                f"replay file has no rows between {request.start} and {request.end}"
            )
        return window


class SyntheticBarSource(BarSource):
    """Factor-model prices requiring no credentials.

    Lets the full benchmark run offline. Prices are generated from a three-factor
    model with per-asset idiosyncratic noise, which produces the correlation
    structure a cardinality-constrained optimizer is meant to exploit -- unlike
    independent random walks, where asset selection would be trivial.
    """

    name = "synthetic"

    def __init__(self, seed: int = 20260803, n_factors: int = 3) -> None:
        self.seed = seed
        self.n_factors = n_factors

    def fetch(self, request: BarRequest) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        n = len(request.symbols)
        sessions = pd.bdate_range(request.start, request.end)
        if len(sessions) < 2:
            raise DataUnavailable(
                f"synthetic range {request.start}..{request.end} has "
                f"{len(sessions)} business days; need at least 2"
            )

        drift = rng.normal(0.0004, 0.0006, n)
        loadings = rng.normal(0, 1, (n, self.n_factors))
        factors = rng.normal(0, 0.008, (len(sessions), self.n_factors))
        idiosyncratic = rng.normal(0, 0.010, (len(sessions), n))

        returns = drift + factors @ loadings.T + idiosyncratic
        prices = 100.0 * np.exp(np.cumsum(returns, axis=0))
        return pd.DataFrame(
            prices, index=sessions.tz_localize(timezone.utc), columns=list(request.symbols)
        )


def default_request(
    symbols: tuple[str, ...] = DEFAULT_UNIVERSE, lookback_days: int = 30
) -> BarRequest:
    """A lookback window ending yesterday.

    `lookback_days` counts calendar days, not sessions: 30 calendar days is
    about 22 sessions and 21 return observations.

    Ends a day back because the current session's bar is incomplete while the
    market is open, and a partial bar would silently corrupt the return series.
    """
    end = datetime.now(timezone.utc).date() - timedelta(days=1)
    return BarRequest(symbols=symbols, start=end - timedelta(days=lookback_days), end=end)


def resolve_source(prefer_live: bool = True) -> BarSource:
    """Alpaca when credentials exist, synthetic otherwise.

    Never blocks work on missing credentials -- the offline path is a first-class
    source, not a degraded fallback.
    """
    if prefer_live and load_settings().has_alpaca_credentials:
        return AlpacaBarSource()
    return SyntheticBarSource()
