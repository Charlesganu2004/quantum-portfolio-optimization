# Measured Findings: Project II

**Generated:** 2026-08-03. **Updated:** 2026-10-21, after re-running the scripts marked in section 8 and correcting statements the reruns contradicted.
**Status:** numbers below were measured on one machine and can be reproduced from `scripts/`. Runtimes vary from run to run.
**Hardware:** Intel i7-11371H, 4 physical / 8 logical cores, 15.7 GB RAM
**Environment:** Python 3.11.5, NumPy 1.26.4, SciPy 1.12.0, Qiskit 1.4.6

Project I and the proposal are earlier documents of the same project and are not included in this repository. This file refers to them where the code follows a choice made there: the 30-session covariance window (section 5) and the required reviewer ablation (section 7).

> **Data caveat that applies to every number here.** All results run on the
> synthetic three-factor price generator, because no Alpaca credentials are
> configured. The generator has *constant per-asset drift*, which structurally
> favours the `sample_mean` estimator and handicaps momentum and trend estimators.
> The **optimizer** comparison is largely unaffected (all methods see identical
> inputs), but the **estimator** comparison in section 4 is not interpretable until
> it is re-run on real market data.

---

## 1. Headline: gap to brute-force optimum

Twenty problems: four cases (12 assets with cap 4, 16 assets with cap 4, 16 assets
with cap 6, 20 assets with cap 5) times five seeds. Ground truth is the
exhaustive enumeration of every C(n,k) subset. `scripts/validate_against_optimum.py`.
QAOA is not in this table: it was not run on this battery. Fifteen of the twenty
problems have more assets than its 14-asset cap, and the five 12-asset problems
were left out as well.

| method | mean gap | worst gap | optimal hit rate |
|---|---|---|---|
| heuristic (backward elimination + swap) | **0.007%** | 0.148% | **95%** |
| QUBO annealing (quantum-inspired) | 0.184% | **2.428%** | 70% |
| mean-variance (truncation) | 1.127% | **20.951%** | 80% |

**The `worst gap` column matters more than the mean.** Mean-variance looks
acceptable on average and reaches the optimum on 80% of problems, but when its
two-stage truncation fails it fails badly: 20.951% off optimal on one instance.
On this battery QUBO annealing never exceeds 2.428%.

**That bound belongs to this battery and not to the method.** The scalability
sweep (section 2) solved nine other problems with ground truth, and the worst
per-problem QUBO gaps there were 8.449% (8 assets), 12.427% (16 assets) and 9.839%
(32 assets). The heuristic also reached 9.839% at 32 assets, and mean-variance
truncation reached 29.493% there. QUBO annealing's worst case is still smaller than
truncation's in both problem sets (12.427% against 29.493% across the sweep,
2.428% against 20.951% on the battery), but at 8 assets truncation was the
better of the two (worst 2.204% against 8.449%). The earlier claim that QUBO
annealing offers "bounded tail risk" was drawn from the battery alone and is
withdrawn.

What the data supports: QUBO annealing is not better than the heuristic on average,
and its worst single-problem gap is 12.427% across the two problem sets. Its worst
case was smaller than truncation's in both sets, which makes it a more reliable
choice than truncation on these problems and not a guaranteed one.

---

## 2. Scalability (`scripts/scalability_sweep.py`)

Median runtime, milliseconds, 3 repeats, cap 4, 300-session training window:

| assets | mean-variance | heuristic | QUBO annealing | QAOA (simulated) |
|---|---|---|---|---|
| 4 | 1.12 | 0.86 | 0.94 | 0.86 |
| 8 | 2.69 | 12.71 | 36.58 | **979.80** |
| 16 | 3.21 | 37.24 | 76.52 | excluded |
| 32 | 6.70 | 108.59 | 147.99 | excluded |
| 64 | 21.78 | 458.70 | 284.52 | excluded |

Mean-variance is the cheapest at every size from 8 to 64 assets, since it solves one
convex QP. Runtimes are medians from one run and vary with machine load: a run made
while the machine was busy gave 902 ms for the heuristic at 64 assets against 459 ms
here. The heuristic costs more than QUBO annealing above 32 assets because
backward elimination runs `n - k` re-solves. QAOA is 27 times the next
slowest method at 8 assets and is excluded above 14 assets. The limit is runtime and
not memory: each added qubit doubles the cost of every circuit evaluation, while a
16-qubit statevector takes only 1 MiB and a 20-qubit one 16 MiB. `MAX_QUBITS = 14`
is a stated experimental cap.

All methods hold exactly the cap (4) at every size of 8 assets or more, so
cardinality is enforced uniformly and the runtime comparison is like-for-like.

Gap to the brute-force optimum in the same sweep (3 problems per size, ground truth
up to 32 assets). The mean hides single bad problems and the percentage is unstable
when an optimum is near zero, so the worst case is shown as a percentage and as an
absolute shortfall in objective units:

| assets | method | mean gap | worst gap | worst shortfall |
|---|---|---|---|---|
| 8 | heuristic | 0.000% | 0.000% | 0.000000 |
| 8 | mean-variance | 0.735% | 2.204% | 0.003864 |
| 8 | QUBO annealing | 2.816% | 8.449% | 0.014813 |
| 8 | QAOA (simulated) | 1830.013% | 5428.540% | 0.117126 |
| 16 | heuristic | 0.000% | 0.000% | 0.000000 |
| 16 | mean-variance | 7.672% | 19.853% | 0.028208 |
| 16 | QUBO annealing | 5.196% | 12.427% | 0.017656 |
| 32 | heuristic | 3.723% | 9.839% | 0.030929 |
| 32 | mean-variance | 17.897% | 29.493% | 0.091661 |
| 32 | QUBO annealing | 3.723% | 9.839% | 0.030929 |

---

## 3. QAOA: what actually happened

**QAOA performs badly on this problem, and it is 68 times slower than
classical annealing on identical inputs in the holdout run** (the ratio has ranged
from 68 to 79 across runs). That is the result. It
is consistent with Project I's conclusion (Project I is not in this repository) that
quantum methods are not yet practical here.

Holdout validation, 12 unseen problems of 8 to 12 assets
(`scripts/validate_qaoa_holdout.py`, 250 optimizer iterations):

| | QAOA (simulated) | QUBO annealing |
|---|---|---|
| mean gap | **53.256%** | 0.152% |
| worst gap | 257.641% | 1.820% |
| optimal hits | 5 / 12 | 11 / 12 |
| mean runtime | 3859 ms | 56 ms |

On the separate scalability problem set, QAOA's mean gap at 8 assets was **1830%**.
That figure comes from a single problem. In repeat 2 the optimum is -0.002158, so a
shortfall of 0.117126 objective units becomes a gap of 5428.54%. The other two
8-asset problems had gaps of 8.449% and 53.049%. Read the 1830% as a statement
about a near-zero denominator, not as a typical result. The absolute shortfall is
reported beside the percentage in section 2 for this reason, and
`scripts/debug_qaoa_case.py` prints both.

### The methodological finding is as important as the number

A grid search over depth and penalty strength (`scripts/tune_qaoa.py`) selected
`reps=2, penalty=5.0` because it scored 0.000% on all six tuning cases. Neighbouring
grid points scored 13% to 62%. That non-monotonicity was the warning sign, and
holdout validation confirmed it: the same configuration scored **53.256%** on
twelve problems it had not been selected against. `validate_qaoa_holdout.py` now
re-scores the tuning cases itself and measured 0.000% on them.

**Reported number: 53.256% (holdout). Not 0.000% (tuning).** The tuning
figure is an artifact of selecting a configuration against the cases used to
evaluate it.

The shipped defaults are the best available, not good. The library default is
`max_iterations=120`, while the tuning and holdout scripts used 250. The sweep and the
tests use the default, which has not been validated on
the holdout set. Test thresholds in `tests/test_qaoa.py` are deliberately loose
divergence guards, not quality bars: tightening them would only be achievable by
tuning against the test set, which is the exact error already caught once.

QAOA and the annealer do not solve the same QUBO. They build the same objective and
the same form of cardinality penalty, but QAOA uses a penalty multiplier of 5.0 and
the annealer 1.1 (penalty weights 1.3818 and 0.3040 on one test problem). A
difference in result is therefore attributable to the solver and the penalty
strength together. `tests/test_qaoa.py` checks that the two matrices differ by
exactly the penalty term.

### Two real bugs found and fixed along the way

1. **Penalty scaled to magnitude instead of variation.** Scaling the cardinality
   penalty off the objective's absolute size swamped the energy landscape and
   reduced annealing to picking a random feasible subset. Gap: 17.788% before the
   fix.
2. **Only the final state kept per restart.** Standard simulated annealing tracks
   the best state *visited*; keeping the terminal state made results depend on
   where the walk stopped. Combined with switching to cardinality-preserving swap
   moves, the search-attributable gap went **19.97% to 0.000%**, and the annealer
   now finds minimum energy in 9 of 9 diagnostic cases.

`scripts/diagnose_qubo.py` separates *model loss* (the equal-weight surrogate) from
*search loss* (annealing failing to minimize). Without that split the second bug
was invisible: the symptom looked like a penalty-tuning problem. Residual model
loss is 0.033%, and rank correlation between QUBO energy and true objective is
0.939.

---

## 4. Estimator vs optimizer (provisional, synthetic data)

Walk-forward, 120-session train, 10-session hold, all out-of-sample.
`scripts/estimator_vs_optimizer.py`.

| | mean Sharpe spread |
|---|---|
| changing the **estimator** | 0.675 |
| changing the **optimizer** | 0.712 |

Roughly equal. The hypothesis that expected-return estimation dominates optimizer
choice is **not** supported here, so the four-optimizer comparison is worth its
page count.

Two observations, both provisional pending real data:

- **The null hypothesis does well but does not win outright.** `grand_mean`,
  which deliberately holds no view on which asset outperforms, reached 2.078 Sharpe
  with the heuristic. That beats `sample_mean` (1.534) and `momentum` (1.387) with
  the same optimizer, but not `trend_quality` (2.227) or `james_stein` (2.332).
  Averaging the printed Sharpe values over the three optimizers, `grand_mean` leads
  (1.836, against 1.806 for `james_stein`, 1.540 for `sample_mean`, 1.525 for
  `trend_quality` and 1.335 for `momentum`), and it is the best estimator under
  mean-variance (1.562) and QUBO annealing (1.867). The lead over `james_stein` in
  that average is 0.030 on one synthetic history. This suggests that much of the
  performance comes from **risk structure and not return prediction**, which is
  consistent with the minimum-variance literature, but it does not show that
  `grand_mean` beats the other estimators in every pairing.
- **Best combination:** `james_stein` + `heuristic`, 2.332 Sharpe.

---

## 5. Covariance conditioning (affects the proposal)

`scripts/build_problem.py`. Sample covariance rank by lookback and universe size.
Lookbacks are counted in sessions:

| assets | 30 sessions | 120 sessions | 365 sessions |
|---|---|---|---|
| 8 | rank 8, ok | ok | ok |
| 16 | rank 16, ok | ok | ok |
| 32 | **rank 28, singular** | rank 32, ok | ok |
| 64 | **rank 28, singular** | rank 64, ok | ok |

A 30-session window yields 29 return observations, so sample covariance rank caps
at 28. **A 30-session window cannot support the scalability sweep to 32 or 64
assets**: the matrix is singular and the QP returns nonsense.

Calendar days and sessions differ. `default_request(lookback_days=30)` counts
calendar days, which is about 22 sessions and 21 return observations, so the rank
caps at 20 there. That is full rank for the 16-symbol default universe and singular
above 20 symbols.

Ledoit-Wolf shrinkage masks this: results still come out, which is precisely what
makes it dangerous. The minimum viable lookback is about 120 sessions, and the sweep
uses 300. **Resolved:** the proposal's window is amended to a 120-session floor.
The 30-session case is retained in `build_problem.py` as the printed evidence for
the amendment, not as a working configuration.

Risk-aversion scaling matters equally: position count moves from 3 to 29 as lambda
goes 1 to 100 on a 32-asset universe. Any cardinality result is uninterpretable
unless lambda is reported beside it. Inputs are annualized (x252) so lambda sits in
the conventional 1 to 10 range.

---

## 6. Trader-pattern data: what is actually obtainable

**No public record of individual day traders' trades exists.** No broker publishes
one, and no ranked leaderboard with trade history exists.

Legally available, named-individual, timestamped records:

| source | content | lag |
|---|---|---|
| SEC Form 4 | insider transactions | 2 business days |
| Congressional PTR | member trades | up to 45 days |
| 13F | institutional *holdings* | quarterly, 45 days |
| 13D/G | more than 5% stakes | 5 business days for 13D and passive 13G; up to 45 days after quarter end for institutional 13G (SEC amendments effective 5 February 2024, Release 33-11253) |

**None is fast enough to trade on.** `KnowledgeRecord.usable_as_live_signal()`
returns False for every filing and trade record whatever its age. An earlier
version measured age from the filing date and so passed a record filed an hour ago,
although a Form 4 filed today can describe a trade two days old. The check now
refuses those source types outright, and `tests/test_knowledge.py` tests a record
filed just now. A separate test asserts that no filing lag is under two days. The
records are a pattern-analysis corpus only, and derived patterns carry
`Confidence.HYPOTHESIS` because an observed trade is not a stated reason. No live
SEC ingestion is implemented.

---

## 7. Reviewer ablation, with a real model

Run 2026-10-21, `llama3.2:3b` via Ollama, 12 steps on the synthetic 16-symbol
universe, `SessionConfig(train_sessions=300, cardinality_cap=4, max_weight=0.4)`.
Each step advances one row of the price frame. The frame holds daily bars, so
the 12 steps cover 12 trading days, and the 10-minute spacing in the timestamps is
a label. The first run of this ablation (2026-08-06) was the first made
against an actual language model rather than the `offline_rules` stand-in, which
changes the decision 0% of the time by construction and measures nothing.

| optimizer | decisions changed | traded with review | traded without | return with | return without |
|---|---|---|---|---|---|
| QUBO annealing | 12/12 (**100%**) | 0 | 12 | +0.0000% | **+0.9526%** |
| heuristic | 11/12 (**91.7%**) | 0 | 11 | +0.0000% | **-1.6007%** |

**What the committed log shows.** The main run in `reports/session_intervals.json`
(QUBO candidates, reviewer `ollama`) abstained on all 12 steps. Each step records
the review error, the citation flag, any repaired citations and the reviewer's
stated observations. Eleven steps carry no error and a verdict from the model. In
step 5 the abstention came from the citation check and not from a judgement: the
model cited `source_id:transaction-costs`, which resolves to no record, so the
review failed closed (`reviewer cited 1 non-existent source(s)`). The
`citations_verified` field in this file reads true on every step, step 5 included,
because the version of `_abstain` that wrote it set true on every failure path.
Read `reviewer_error` for that step. The code now sets the field to false for a
fabricated citation and to null when no output was checked.

**What it does not show.** The two ablation arms make their own reviewer calls,
separate from the main run, and this file holds only their totals. It has no
per-step record of whether the model answered in those calls, and a failed call
abstains exactly like a declined one (`_abstain`), so the totals in the table
cannot tell a judgement from a transport failure. The heuristic arm in particular
has no logged step at all. The `reviewer_is_language_model: true` in the file was
computed from the backend name, not from the responses.

**What the script does now.** `scripts/run_session.py` stops without writing when a
model backend returns no response at any step of the main run or of either arm. It
records `model_responded`, `reviewer_error` and `citations_verified` for every step
of the main run and, as `steps_with_review`, for the reviewed run of each arm. It
sets `reviewer_is_language_model` true only when every reviewed step of all three
runs got a response from the model. A schema failure or a refused citation still
counts as a response, since the model answered and its answer was refused. The
committed file was written before this change and has none of these fields for the
arms, so the table above is a record of that earlier run. Rerun the script, with
an Ollama server that stays up for the whole run, before quoting the table as a
measurement of the model's decisions.

The reasons stated in the 11 steps with a verdict were varied. The model said the
evidence gave no clear basis for the optimizer's expected-risk estimate (10 steps),
for the Sharpe ratio estimate or its formula (4), for the transaction cost estimate
(4), for the risk-aversion parameter (4) and for the allocation percentages (3).
The earlier account of one consistent reason, that the Sharpe estimate excludes
transaction costs, is not what this run's log shows. Read together, the 3B model is
mostly saying that the four supplied formula records do not explain the
optimizer's numbers.

### The honest reading of the 100%

**A 100% change rate is not evidence the reviewer discriminates.** A reviewer that
never approves anything achieves 100% trivially. With review on, neither arm
traded. Removing review avoided the heuristic's loss and also forfeited QUBO's
gain, so review blocked a loser and a winner alike. That is what a uniformly
conservative reviewer looks like, and it is not selection. The main-run log, which
does record its reviews, shows the same pattern for the QUBO candidates: 12
abstentions, 11 of them verdicts from the model. The heuristic arm has no such
record. The report must not present the change rate as a positive result.

This is the designed failure direction: the architecture routes every uncertainty
to ABSTAIN. But "the safety property holds" and "the reviewer adds decision value"
are different claims, and only the first is supported here.

Two things would make the ablation discriminating rather than degenerate, and
neither is done. The first follows from the reasons the log records. In 10 of the
11 verdicts the model says the evidence gives no clear basis for the optimizer's
expected-risk estimate (7.44% to 7.50% in the logged steps). The prompt gives the
model the allocation, two estimates and four formula records, and no covariance
matrix or return history, so it cannot check a formula against the optimizer's
numbers. Supplying the figures behind the estimates would make that objection
answerable. The second is a larger model as a second arm, to separate *model
capacity* from *prompt content*.

### Citation formatting, reduced and not gone

Earlier runs of this model produced citations carrying square-bracket delimiters
copied from the evidence block (`[formula:sharpe-ratio]`) and citations with the
`formula:` namespace dropped. Both now resolve through `resolve_citation`, and the
evidence block presents ids on a labelled `source_id:` line with no delimiters to
copy. In this run the 11 steps with a verdict needed no repair. In step 5 the model
wrote `source_id:transaction-costs`, which keeps the label and drops the
`formula:` namespace, and the check refused it. Repairs remain recorded in
`ReviewResult.repaired_citations` rather than silently accepted, and ambiguous or
invented ids still fail closed to ABSTAIN.

### What a step is

A step feeds the optimizer a training window of 300 rows, enters at the last
training close, and exits at the next row's close (or at the stop if the close is
below it). With daily bars that is one trading day per step, so the ablation
returns above cover 12 trading days, and the 200-step trade analysis in
`reports/trade_analysis.json` covers about 200 trading days. The stale-quote check
in the risk gate cannot fire in this replay because the quote time and the
decision time are the same value. Entry size is a fixed 1.50 USD from
`SessionConfig`. The risk gate accepts or rejects that order and sets its stop, and
the end-of-step flattening sell goes to the broker without a second gate check.

---

## 8. Reproducing

```bash
PYTHONPATH=src python -m pytest tests -q
```

| script | produces |
|---|---|
| `validate_against_optimum.py` | section 1 gap table (`--json reports/gap_battery.json`) |
| `scalability_sweep.py` | section 2 tables and `reports/scalability.png` |
| `tune_qaoa.py` / `validate_qaoa_holdout.py` | section 3 overfitting result (`--json reports/qaoa_holdout.json`) |
| `diagnose_qubo.py` | section 3 model-versus-search attribution |
| `estimator_vs_optimizer.py` | section 4 walk-forward grid |
| `build_problem.py` | section 5 conditioning table |
| `run_session.py` | section 7 ablation and `reports/session_intervals.json` |
| `check_alpaca.py` | read-only Alpaca preflight, places no order |
| `make_figures.py` | `reports/figures/*.png`, read from the JSON files above |

Section 7 needs a local model. Without one, `run_session.py` stops with a message
and only runs the labelled `offline_rules` stand-in when given `--offline`. That
stand-in can approve trades and measures nothing about a language model. With a
model, the script also stops, and writes nothing, if a review gets no response at
any step. In the emitted JSON, `reviewer_is_language_model` is true only when every
reviewed step of the main run and of both arms got a response, and
`model_responded` records it per step. Check both before quoting any number:

```bash
ollama pull llama3.2:3b
```

**451 tests pass** on 2026-10-21, with no skips. Quality regressions are gated: `tests/test_ground_truth.py` holds each
classical optimizer to a measured mean-gap budget, so a change that degrades solution quality
fails the suite rather than being noticed by eye.
