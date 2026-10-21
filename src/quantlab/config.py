"""Configuration and credential loading.

Credentials come from `.env` (gitignored) in the current directory, or in the
directory named by `QUANTLAB_HOME`, or from the process environment. They never
come from committed files or from anything the LLM layer can write. `Settings`
carries the secret but its `repr` redacts it, so a stray log line or exception
traceback cannot leak the key.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def project_root() -> Path:
    """Directory that holds `.env` and `data_cache/`.

    The `QUANTLAB_HOME` environment variable when it is set, otherwise the current
    directory. It is never derived from `__file__`: for a wheel installed into a
    virtual environment that path points into site-packages, so `.env` would be
    looked for beside the installed package instead of in the project.
    """
    override = os.environ.get("QUANTLAB_HOME")
    return Path(override) if override else Path.cwd()


class MissingCredentials(RuntimeError):
    """Raised when a live data source is requested without credentials."""


def load_env_file(path: Path | None = None, *, override: bool = False) -> dict[str, str]:
    """Read a `.env` file into a dict and populate `os.environ`.

    Deliberately dependency-free -- one less package between a credential and
    the process holding it. Missing file is not an error; the synthetic data
    source works without credentials.
    """
    path = path or project_root() / ".env"
    values: dict[str, str] = {}
    if not path.is_file():
        return values

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        values[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return values


def normalize_base_url(url: str) -> str:
    """Reduce an Alpaca URL to the origin the request builders expect.

    Every call site appends its own versioned path (`/v2/orders`,
    `/v2/stocks/bars`), so the stored base must not already carry `/v2`. Alpaca's
    dashboard and docs both display the endpoint *with* the version suffix --
    `https://paper-api.alpaca.markets/v2` -- so pasting the documented value into
    `.env` is the expected thing to do, and left alone it would produce
    `/v2/v2/orders` and a 404 that looks like a credential problem.

    Trailing slashes and a single trailing `/v2` are removed. Nothing else is
    rewritten: the paper-endpoint check in `AlpacaPaperBroker` still sees the
    host it was given.
    """
    cleaned = url.strip().rstrip("/")
    if cleaned.lower().endswith("/v2"):
        cleaned = cleaned[: -len("/v2")].rstrip("/")
    return cleaned


@dataclass
class Settings:
    """Resolved runtime configuration."""

    alpaca_key_id: str = ""
    alpaca_secret_key: str = field(default="", repr=False)
    trading_base_url: str = "https://paper-api.alpaca.markets"
    data_base_url: str = "https://data.alpaca.markets"
    data_feed: str = "iex"

    def __post_init__(self) -> None:
        # Normalized here rather than at the call sites so a `Settings` built
        # directly in a test or a script gets the same treatment as one loaded
        # from `.env`.
        self.trading_base_url = normalize_base_url(self.trading_base_url)
        self.data_base_url = normalize_base_url(self.data_base_url)

    @property
    def has_alpaca_credentials(self) -> bool:
        return bool(self.alpaca_key_id and self.alpaca_secret_key)

    def require_alpaca(self) -> None:
        if not self.has_alpaca_credentials:
            raise MissingCredentials(
                "Alpaca credentials not found. Copy .env.example to .env and fill "
                "in ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY from your paper "
                "trading dashboard at https://alpaca.markets"
            )

    def auth_headers(self) -> dict[str, str]:
        self.require_alpaca()
        return {
            "APCA-API-KEY-ID": self.alpaca_key_id,
            "APCA-API-SECRET-KEY": self.alpaca_secret_key,
        }

    def __repr__(self) -> str:
        state = "set" if self.has_alpaca_credentials else "missing"
        return (
            f"Settings(alpaca_credentials={state}, feed={self.data_feed!r}, "
            f"trading={self.trading_base_url!r})"
        )


def load_settings(env_path: Path | None = None) -> Settings:
    """Load settings from `.env` then the environment."""
    load_env_file(env_path)
    return Settings(
        alpaca_key_id=os.environ.get("ALPACA_API_KEY_ID", ""),
        alpaca_secret_key=os.environ.get("ALPACA_API_SECRET_KEY", ""),
        trading_base_url=os.environ.get(
            "ALPACA_TRADING_BASE_URL", "https://paper-api.alpaca.markets"
        ),
        data_base_url=os.environ.get(
            "ALPACA_DATA_BASE_URL", "https://data.alpaca.markets"
        ),
        data_feed=os.environ.get("ALPACA_DATA_FEED", "iex"),
    )
