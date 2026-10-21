"""Formula knowledge base -- definitions the reviewer may cite.

One honest caveat, stated up front because it is easy to assume otherwise:
**storing a formula here does not make the code that computes it more accurate.**
`evaluation.metrics` already computes Sharpe correctly; a definition of Sharpe in
a vector store changes nothing about that number.

What this actually provides:

* The LLM reviewer can explain *why* a metric matters and cite a definition,
  instead of paraphrasing one from memory and risking a subtly wrong statement.
* Every formula the system relies on is documented with provenance in one place,
  which the report can reference directly.
* Adding an indicator becomes a data change rather than a prompt change.

Each record's `source_id` is what appears in a citation, and the risk controller
rejects any answer citing an id that does not resolve here.
"""

from __future__ import annotations

from .records import Confidence, KnowledgeRecord, SourceType, utc

_PUBLISHED = utc(2026, 8, 3)


def _formula(source_id: str, title: str, content: str, citation: str) -> KnowledgeRecord:
    return KnowledgeRecord(
        source_id=source_id,
        source_type=SourceType.FORMULA,
        title=title,
        content=content.strip(),
        published_at=_PUBLISHED,
        retrieved_at=_PUBLISHED,
        license_or_permission="public domain (standard definition)",
        confidence=Confidence.DEFINITIONAL,
        citation=citation,
    )


FORMULA_RECORDS: tuple[KnowledgeRecord, ...] = (
    _formula(
        "formula:expected-return",
        "Portfolio expected return",
        """
        E[R_p] = w' mu, where w is the weight vector and mu the vector of per-asset
        expected returns. Annualized here by scaling daily means by 252 trading days.
        Linear in weights, so it carries no diversification benefit on its own.
        """,
        "Markowitz, H. (1952). Portfolio Selection. Journal of Finance, 7(1), 77-91.",
    ),
    _formula(
        "formula:portfolio-variance",
        "Portfolio variance and volatility",
        """
        Var(R_p) = w' Sigma w, volatility = sqrt(w' Sigma w). Unlike expected return,
        this is quadratic in weights: off-diagonal covariance terms are what make
        diversification reduce risk. Annualized by scaling the daily covariance by 252.
        """,
        "Markowitz, H. (1952). Portfolio Selection. Journal of Finance, 7(1), 77-91.",
    ),
    _formula(
        "formula:mean-variance-objective",
        "Mean-variance objective",
        """
        maximize w'mu - lambda * w' Sigma w subject to sum(w) = 1 and bounds on w.
        Lambda is risk aversion. Its scale must match the units of mu and Sigma: with
        raw daily returns (~8e-4) and daily variances (~1e-4), lambda near 1 makes the
        risk term negligible and the problem degenerates into return maximization.
        With annualized inputs, lambda in 1-10 is meaningful. Measured on this project:
        the solution holds 3 assets at lambda=1 and 29 at lambda=100 on a 32-asset
        universe, so any cardinality result is uninterpretable unless lambda is
        reported alongside it.
        """,
        "Markowitz (1952); scale sensitivity measured in this project, 2026-08-03.",
    ),
    _formula(
        "formula:sharpe-ratio",
        "Sharpe ratio",
        """
        S = (E[R_p] - R_f) / sigma_p, annualized by multiplying by sqrt(periods per
        year). Return per unit of total volatility. Penalizes upside and downside
        deviation equally, which is why Sortino is also reported.
        """,
        "Sharpe, W. F. (1966). Mutual Fund Performance. Journal of Business, 39(1).",
    ),
    _formula(
        "formula:sortino-ratio",
        "Sortino ratio",
        """
        Sortino = (E[R_p] - R_f) / sigma_downside, where sigma_downside uses only
        negative excess returns. Upside volatility is not a risk an investor wants
        removed, so this is the fairer ratio for asymmetric return profiles.
        """,
        "Sortino, F. A., & Price, L. N. (1994). Journal of Investing, 3(3), 59-64.",
    ),
    _formula(
        "formula:max-drawdown",
        "Maximum drawdown",
        """
        MDD = min over t of (V_t / max_{s <= t} V_s - 1), a negative fraction. Measures
        worst peak-to-trough loss. Unlike volatility it captures path dependence, which
        is what determines whether a stop-loss or margin call would have triggered.
        """,
        "Standard risk measure; see Magdon-Ismail & Atiya (2004) on drawdown analysis.",
    ),
    _formula(
        "formula:profit-factor",
        "Profit factor",
        """
        PF = sum of winning returns / absolute sum of losing returns. Above 1 means
        gross gains exceed gross losses. Undefined when there are no losses; reported
        as infinity only when gains exist and losses are genuinely zero.
        """,
        "Common trading-systems metric; see Pardo, R. (2008), Design and Testing.",
    ),
    _formula(
        "formula:qubo-portfolio",
        "Cardinality-constrained portfolio as a QUBO",
        """
        With binary x_i marking asset i as held and equal weights w = x/K:
          E(x) = -(1/K) mu'x + (lambda/K^2) x'Sigma x + P (sum(x) - K)^2
        Expanding with x_i^2 = x_i gives Q_ii = -mu_i/K + (lambda/K^2)Sigma_ii +
        P(1-2K) and Q_ij = (lambda/K^2)Sigma_ij + P.

        Two implementation facts measured in this project: (1) P scaled to the
        objective's absolute magnitude swamps the landscape and reduces annealing to
        random feasible selection -- it must be scaled to the objective's *variation*;
        (2) under cardinality-preserving swap moves, P contributes exactly zero to the
        energy delta, so the penalty formulation is preserved while the search sees the
        pure objective.
        """,
        "Lu et al., Quantum-Inspired Portfolio Optimization in the QUBO Framework.",
    ),
    _formula(
        "formula:simulated-annealing",
        "Metropolis acceptance in simulated annealing",
        """
        Accept a proposed move with probability 1 if dE <= 0, else exp(-dE / T), with
        T decreasing on a cooling schedule. Correctness requires tracking the best
        state *visited*, not the final state: with a finite schedule the walk may end
        above its own best. Omitting that made the annealer in this project miss the
        minimum-energy state in 9 of 9 diagnostic cases.
        """,
        "Kirkpatrick, S., Gelatt, C. D., & Vecchi, M. P. (1983). Science, 220(4598).",
    ),
    _formula(
        "formula:ledoit-wolf",
        "Ledoit-Wolf covariance shrinkage",
        """
        Sigma* = delta * F + (1 - delta) * S, with S the sample covariance and F a
        scaled-identity target. Delta is chosen to minimize expected squared error and
        rises as the sample gets noisier relative to the dispersion of S.

        Necessary, not optional, when observations T are near or below asset count N:
        S has rank at most T and is singular for T < N. Measured here: a 30-day window
        yields ~29 observations and caps rank at 28, so a 32- or 64-asset covariance is
        singular and the optimizer's QP returns nonsense without shrinkage.
        """,
        "Ledoit, O., & Wolf, M. (2004). Journal of Multivariate Analysis, 88(2).",
    ),
    _formula(
        "formula:james-stein",
        "James-Stein shrinkage of expected returns",
        """
        mu* = grand_mean + (1 - c)(mu_sample - grand_mean), shrinking per-asset means
        toward their common average. For three or more assets this dominates the sample
        mean in expected squared error. Relevant because mean-variance optimization
        amplifies estimation error in mu -- Michaud's "error maximization": the
        optimizer reads a noisily high estimate as an opportunity and concentrates into it.
        """,
        "James & Stein (1961); Michaud, R. (1989). Financial Analysts Journal, 45(1).",
    ),
    _formula(
        "formula:ema",
        "Exponential moving average",
        """
        EMA_t = alpha * P_t + (1 - alpha) * EMA_{t-1}, with alpha = 2/(span+1).
        Weights recent observations more heavily than a simple moving average.
        """,
        "Standard technical definition; see Murphy, J. (1999), Technical Analysis.",
    ),
    _formula(
        "formula:rsi",
        "Relative Strength Index (Wilder)",
        """
        RSI = 100 - 100/(1 + RS) where RS = average gain / average loss over the
        window, both smoothed with Wilder's alpha = 1/window. The SMA variant many
        libraries ship gives materially different values; this project uses Wilder's
        original smoothing.
        """,
        "Wilder, J. W. (1978). New Concepts in Technical Trading Systems.",
    ),
    _formula(
        "formula:momentum",
        "Cross-sectional momentum",
        """
        Total return over a lookback window, conventionally excluding the most recent
        period ("skip") because short-horizon reversal contaminates the signal. The
        classic specification is 12-month lookback skipping the most recent month.
        """,
        "Jegadeesh, N., & Titman, S. (1993). Journal of Finance, 48(1), 65-91.",
    ),
    _formula(
        "formula:trend-strength",
        "Trend strength as goodness of fit",
        """
        R-squared of an ordinary least-squares fit to log price over a window.
        Measures trend *consistency*, distinct from momentum's size and direction: a
        single large jump gives high momentum but low trend strength, while a steady
        drift gives the reverse.
        """,
        "Derived measure; standard OLS goodness-of-fit applied to log price.",
    ),
    _formula(
        "formula:transaction-costs",
        "Round-trip transaction cost",
        """
        cost = turnover * cost_rate, turnover = sum |w_new - w_old|. Reference rates
        measured 2026-08-03: Coinbase Advanced base tier 0.60% taker / 0.40% maker,
        so a taker round trip is 1.20%. Alpaca US equities are commission-free, leaving
        only the bid-ask spread (~0.01-0.05% on liquid large caps). A step is one
        entry and one exit: one trading day on daily bars. Over a step this short
        the cost floor is the hurdle a forecast has to clear before the trade can
        profit.
        """,
        "Venue fee schedules, retrieved 2026-08-03.",
    ),
)

FORMULA_INDEX: dict[str, KnowledgeRecord] = {r.source_id: r for r in FORMULA_RECORDS}


def get_formula(source_id: str) -> KnowledgeRecord:
    """Resolve a citation, or fail loudly.

    The reviewer's citations are checked against this; a fabricated id raises
    rather than silently returning nothing.
    """
    try:
        return FORMULA_INDEX[source_id]
    except KeyError:
        raise KeyError(
            f"unknown formula id {source_id!r}. Known ids: "
            f"{sorted(FORMULA_INDEX)}"
        ) from None


def search_formulas(query: str) -> list[KnowledgeRecord]:
    """Case-insensitive substring search over title and content."""
    needle = query.lower().strip()
    if not needle:
        return []
    return [
        record
        for record in FORMULA_RECORDS
        if needle in record.title.lower() or needle in record.content.lower()
    ]
