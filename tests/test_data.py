"""Data layer: credential handling, feature construction, covariance estimation."""

from __future__ import annotations

import warnings
from datetime import date

import numpy as np
import pandas as pd
import pytest

from quantlab.config import MissingCredentials, Settings, load_env_file, load_settings
from quantlab.data.features import (
    InsufficientHistory,
    TRADING_DAYS,
    annualized_moments,
    ledoit_wolf_shrinkage,
    to_returns,
)
from quantlab.data.sources import (
    AlpacaBarSource,
    BarRequest,
    DataUnavailable,
    SyntheticBarSource,
)


class TestCredentials:
    def test_secret_is_redacted_from_repr(self) -> None:
        """A traceback or log line must never carry the key."""
        settings = Settings(alpaca_key_id="PK123", alpaca_secret_key="supersecret")
        assert "supersecret" not in repr(settings)

    def test_missing_credentials_raise_actionable_error(self) -> None:
        settings = Settings()
        with pytest.raises(MissingCredentials, match=".env.example"):
            settings.require_alpaca()

    def test_auth_headers_require_credentials(self) -> None:
        with pytest.raises(MissingCredentials):
            Settings().auth_headers()

    def test_absent_env_file_is_not_an_error(self, tmp_path) -> None:
        """Offline work must not depend on a credentials file existing."""
        assert load_env_file(tmp_path / "nope.env") == {}

    def test_env_file_parsing_ignores_comments_and_quotes(self, tmp_path) -> None:
        path = tmp_path / ".env"
        path.write_text(
            "# comment\n\nALPACA_API_KEY_ID='PKTEST'\nALPACA_API_SECRET_KEY=\"shh\"\n",
            encoding="utf-8",
        )
        values = load_env_file(path, override=True)
        assert values["ALPACA_API_KEY_ID"] == "PKTEST"
        assert values["ALPACA_API_SECRET_KEY"] == "shh"

    def test_settings_report_credential_presence(self) -> None:
        assert not Settings().has_alpaca_credentials
        assert Settings(alpaca_key_id="a", alpaca_secret_key="b").has_alpaca_credentials


class TestSyntheticSource:
    def test_is_deterministic(self) -> None:
        request = BarRequest(("A", "B", "C"), date(2026, 1, 1), date(2026, 3, 1))
        first = SyntheticBarSource().fetch(request)
        second = SyntheticBarSource().fetch(request)
        pd.testing.assert_frame_equal(first, second)

    def test_returns_requested_symbols(self) -> None:
        request = BarRequest(("X", "Y"), date(2026, 1, 1), date(2026, 2, 1))
        frame = SyntheticBarSource().fetch(request)
        assert list(frame.columns) == ["X", "Y"]
        assert (frame > 0).all().all()

    def test_rejects_window_too_short_to_use(self) -> None:
        request = BarRequest(("A",), date(2026, 1, 5), date(2026, 1, 5))
        with pytest.raises(DataUnavailable, match="business days"):
            SyntheticBarSource().fetch(request)

    def test_assets_are_correlated_not_independent(self) -> None:
        """A factor model, not independent walks -- otherwise selection is trivial."""
        request = BarRequest(tuple(f"S{i}" for i in range(8)), date(2025, 1, 1), date(2026, 1, 1))
        returns = to_returns(SyntheticBarSource().fetch(request))
        corr = returns.corr().to_numpy()
        off_diagonal = corr[~np.eye(len(corr), dtype=bool)]
        assert np.abs(off_diagonal).max() > 0.15


class TestBarRequestCacheKey:
    def test_key_is_stable_across_processes(self) -> None:
        """`hash()` of a string is salted per process; the key must not be.

        The digest below was computed with SHA-256 over the sorted, dash-joined
        symbols. A second interpreter must reproduce it exactly.
        """
        import subprocess
        import sys

        request = BarRequest(("MSFT", "AAPL"), date(2026, 1, 1), date(2026, 2, 1))
        assert request.cache_key() == "1Day_2026-01-01_2026-02-01_2a_a56f40ef14"

        script = (
            "from datetime import date; from quantlab.data.sources import BarRequest; "
            "print(BarRequest(('MSFT', 'AAPL'), date(2026, 1, 1), date(2026, 2, 1)).cache_key())"
        )
        import os

        env = {**os.environ, "PYTHONHASHSEED": "random"}
        outputs = {
            subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True,
                text=True,
                env=env,
                check=True,
            ).stdout.strip()
            for _ in range(2)
        }
        assert outputs == {request.cache_key()}

    def test_key_ignores_symbol_order_and_separates_symbol_sets(self) -> None:
        a = BarRequest(("AAPL", "MSFT"), date(2026, 1, 1), date(2026, 2, 1))
        b = BarRequest(("MSFT", "AAPL"), date(2026, 1, 1), date(2026, 2, 1))
        c = BarRequest(("AAPL", "NVDA"), date(2026, 1, 1), date(2026, 2, 1))
        assert a.cache_key() == b.cache_key()
        assert a.cache_key() != c.cache_key()


class TestAlpacaPagination:
    """Alpaca's page limit counts bars across all symbols, so pages share dates."""

    @staticmethod
    def _bar(day: int, close: float) -> dict:
        return {"t": f"2026-01-{day:02d}T05:00:00Z", "c": close}

    @staticmethod
    def _fake_get(pages: list[dict]):
        calls: list[dict] = []

        class Response:
            status_code = 200

            def __init__(self, payload: dict) -> None:
                self._payload = payload

            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return self._payload

        def get(url, params=None, headers=None, timeout=None):
            calls.append(dict(params))
            return Response(pages[len(calls) - 1])

        return get, calls

    def _source(self, tmp_path) -> AlpacaBarSource:
        settings = Settings(alpaca_key_id="k", alpaca_secret_key="s")
        return AlpacaBarSource(settings=settings, cache_dir=tmp_path)

    def test_rows_sharing_a_date_across_pages_are_all_kept(
        self, monkeypatch, tmp_path
    ) -> None:
        """Page 1 is all of AAPL, page 2 all of MSFT: every date appears twice."""
        import quantlab.data.sources as sources

        pages = [
            {
                "bars": {
                    "AAPL": [self._bar(2, 100.0), self._bar(3, 101.0), self._bar(4, 102.0)]
                },
                "next_page_token": "p2",
            },
            {
                "bars": {
                    "MSFT": [self._bar(2, 200.0), self._bar(3, 201.0), self._bar(4, 202.0)]
                },
                "next_page_token": None,
            },
        ]
        get, calls = self._fake_get(pages)
        monkeypatch.setattr(sources.requests, "get", get)

        request = BarRequest(("AAPL", "MSFT"), date(2026, 1, 1), date(2026, 1, 31))
        frame = self._source(tmp_path).fetch(request)

        assert len(calls) == 2 and calls[1]["page_token"] == "p2"
        assert list(frame.columns) == ["AAPL", "MSFT"]
        assert frame.shape == (3, 2)
        assert not frame.isna().any().any()
        assert frame["AAPL"].tolist() == [100.0, 101.0, 102.0]
        assert frame["MSFT"].tolist() == [200.0, 201.0, 202.0]

    def test_a_symbol_split_across_a_page_boundary_is_joined(
        self, monkeypatch, tmp_path
    ) -> None:
        """Page 1 ends partway through MSFT; page 2 carries the rest of MSFT."""
        import quantlab.data.sources as sources

        pages = [
            {
                "bars": {
                    "AAPL": [self._bar(2, 100.0), self._bar(3, 101.0)],
                    "MSFT": [self._bar(2, 200.0)],
                },
                "next_page_token": "p2",
            },
            {
                "bars": {"MSFT": [self._bar(3, 201.0), self._bar(4, 202.0)]},
                "next_page_token": None,
            },
        ]
        get, _ = self._fake_get(pages)
        monkeypatch.setattr(sources.requests, "get", get)

        request = BarRequest(("AAPL", "MSFT"), date(2026, 1, 1), date(2026, 1, 31))
        frame = self._source(tmp_path).fetch(request)

        assert frame["MSFT"].tolist() == [200.0, 201.0, 202.0]
        assert frame["AAPL"].iloc[:2].tolist() == [100.0, 101.0]
        assert frame["AAPL"].isna().tolist() == [False, False, True]

    def test_a_repeated_timestamp_within_one_symbol_keeps_the_last_bar(
        self, monkeypatch, tmp_path
    ) -> None:
        import quantlab.data.sources as sources

        pages = [
            {
                "bars": {"AAPL": [self._bar(2, 100.0), self._bar(2, 105.0)]},
                "next_page_token": None,
            }
        ]
        get, _ = self._fake_get(pages)
        monkeypatch.setattr(sources.requests, "get", get)

        request = BarRequest(("AAPL",), date(2026, 1, 1), date(2026, 1, 31))
        frame = self._source(tmp_path).fetch(request)
        assert frame["AAPL"].tolist() == [105.0]

    def test_a_second_fetch_is_served_from_the_cache(
        self, monkeypatch, tmp_path
    ) -> None:
        import quantlab.data.sources as sources

        pages = [
            {"bars": {"AAPL": [self._bar(2, 100.0), self._bar(3, 101.0)]}, "next_page_token": None}
        ]
        get, calls = self._fake_get(pages)
        monkeypatch.setattr(sources.requests, "get", get)

        source = self._source(tmp_path)
        request = BarRequest(("AAPL",), date(2026, 1, 1), date(2026, 1, 31))
        first = source.fetch(request)
        second = source.fetch(request)

        assert len(calls) == 1
        pd.testing.assert_frame_equal(first, second, check_freq=False)

    def test_an_empty_response_raises(self, monkeypatch, tmp_path) -> None:
        import quantlab.data.sources as sources

        get, _ = self._fake_get([{"bars": {}, "next_page_token": None}])
        monkeypatch.setattr(sources.requests, "get", get)

        request = BarRequest(("AAPL",), date(2026, 1, 1), date(2026, 1, 31))
        with pytest.raises(DataUnavailable, match="no bars"):
            self._source(tmp_path).fetch(request)


class TestReturns:
    def test_requires_two_rows(self) -> None:
        frame = pd.DataFrame({"A": [100.0]}, index=pd.to_datetime(["2026-01-01"]))
        with pytest.raises(InsufficientHistory):
            to_returns(frame)

    def test_computes_percentage_change(self) -> None:
        frame = pd.DataFrame(
            {"A": [100.0, 110.0, 99.0]},
            index=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
        )
        returns = to_returns(frame)
        np.testing.assert_allclose(returns["A"].to_numpy(), [0.10, -0.10])


class TestShrinkage:
    def test_intensity_is_a_valid_proportion(self) -> None:
        rng = np.random.default_rng(0)
        _, intensity = ledoit_wolf_shrinkage(rng.normal(0, 0.01, (60, 10)))
        assert 0.0 <= intensity <= 1.0

    def test_output_is_symmetric_and_psd(self) -> None:
        rng = np.random.default_rng(1)
        shrunk, _ = ledoit_wolf_shrinkage(rng.normal(0, 0.01, (40, 12)))
        np.testing.assert_allclose(shrunk, shrunk.T, atol=1e-12)
        assert np.linalg.eigvalsh(shrunk).min() > 0

    def test_repairs_a_singular_sample(self) -> None:
        """The whole reason shrinkage is on by default: fewer rows than assets."""
        rng = np.random.default_rng(2)
        returns = rng.normal(0, 0.01, (10, 20))  # 10 observations, 20 assets
        sample = np.cov(returns, rowvar=False, bias=True)
        assert np.linalg.matrix_rank(sample) < 20

        shrunk, intensity = ledoit_wolf_shrinkage(returns)
        assert intensity > 0
        assert np.linalg.eigvalsh(shrunk).min() > 0

    def test_shrinks_harder_when_data_is_scarcer(self) -> None:
        rng = np.random.default_rng(3)
        _, scarce = ledoit_wolf_shrinkage(rng.normal(0, 0.01, (20, 15)))
        _, plentiful = ledoit_wolf_shrinkage(rng.normal(0, 0.01, (500, 15)))
        assert scarce > plentiful


class TestAnnualizedMoments:
    def _frame(self, sessions: int, assets: int, seed: int = 0) -> pd.DataFrame:
        request = BarRequest(
            tuple(f"S{i:02d}" for i in range(assets)), date(2024, 1, 1), date(2026, 6, 1)
        )
        return SyntheticBarSource(seed=seed).fetch(request).tail(sessions)

    def test_scales_by_trading_days(self) -> None:
        returns = to_returns(self._frame(200, 6))
        mu, _, _, _ = annualized_moments(returns)
        np.testing.assert_allclose(
            mu, returns.mean().to_numpy() * TRADING_DAYS, rtol=1e-9
        )

    def test_covariance_is_exactly_symmetric(self) -> None:
        """PortfolioProblem rejects asymmetry, so float drift must be cleaned here."""
        _, cov, _, _ = annualized_moments(to_returns(self._frame(150, 10)))
        assert np.array_equal(cov, cov.T)

    def test_warns_when_history_is_shorter_than_universe(self) -> None:
        returns = to_returns(self._frame(20, 30))
        with pytest.warns(RuntimeWarning, match="singular"):
            annualized_moments(returns)

    def test_flags_a_rank_deficient_sample(self) -> None:
        returns = to_returns(self._frame(20, 30))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            _, _, _, diagnostics = annualized_moments(returns)
        assert diagnostics.is_singular_sample
        assert diagnostics.rank < diagnostics.n_assets
        assert "rank-deficient" in diagnostics.summary()

    def test_thirty_day_window_cannot_support_a_large_universe(self) -> None:
        """Guards the lookback decision: ~30 sessions caps rank near 29.

        The scalability sweep to 64 assets must use a longer window; this test
        fails if someone quietly reverts it.
        """
        returns = to_returns(self._frame(30, 64))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            _, _, _, diagnostics = annualized_moments(returns)
        assert diagnostics.is_singular_sample
        assert diagnostics.rank <= 30

    def test_longer_window_supports_the_same_universe(self) -> None:
        _, _, _, diagnostics = annualized_moments(to_returns(self._frame(200, 64)))
        assert not diagnostics.is_singular_sample
        assert diagnostics.rank == 64

    def test_output_feeds_an_optimizer_unmodified(self) -> None:
        from quantlab.optimizers import Constraints, MeanVarianceOptimizer, PortfolioProblem

        mu, cov, symbols, _ = annualized_moments(to_returns(self._frame(150, 12)))
        problem = PortfolioProblem(
            expected_returns=mu,
            covariance=cov,
            symbols=symbols,
            constraints=Constraints(max_positions=4, max_weight=0.4, risk_aversion=25.0),
        )
        result = MeanVarianceOptimizer().solve(problem)
        assert result.n_positions <= 4
