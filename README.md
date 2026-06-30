# HonestEdge

An honest, lookahead-free next-day-direction prediction pipeline for a single
liquid instrument. Built as a weekend MVP whose **deliverable is the
methodology, not a money machine**: a walk-forward, leakage-resistant harness
that tells you the truth about whether a signal exists.

> The point isn't to find alpha in four hours. It's to build the machinery that
> *won't lie to you* about whether you found any. Most amateur quant projects
> "work" only because they leak the future. This one is engineered so that's
> structurally hard — and then reports what's actually left.

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# offline, deterministic (works anywhere, no network):
python -m honestedge.run --source synthetic --model lgbm

# real data on your machine:
python -m honestedge.run --source yfinance --ticker SPY --model lgbm

# run the tests (the most important part):
pytest -v
```

Key flags: `--model {lgbm,logistic}`, `--rolling`, `--threshold 0.55`,
`--embargo 5`, `--cost-bps 1`, `--no-control`, `--refresh`.

---

## Design (SOLID, modular)

Everything is wired through abstract interfaces in `interfaces.py`. The
high-level pipeline depends only on those abstractions; `run.py` is the single
composition root where concrete classes are chosen and injected.

| Module | Responsibility | Interface implemented |
|---|---|---|
| `data.py` | load OHLCV; cache; integrity-check | `PriceDataSource` (yfinance / synthetic / caching decorator) |
| `features.py` | backward features + forward target | `FeatureBuilder` |
| `splits.py` | walk-forward folds + embargo | `SplitStrategy` |
| `model.py` | fit / predict_proba_up | `ProbabilisticClassifier` (LightGBM / Logistic) |
| `pipeline.py` | walk-forward OOS prediction loop | depends only on abstractions |
| `backtest.py` | positions → returns → metrics | `PositionRule` + `Backtester` |
| `evaluation.py` | metrics, baselines, controls | — |
| `report.py` | plots + summary | — |
| `run.py` | composition root / CLI | — |

* **Single Responsibility** — each module does one thing; decision logic
  (`PositionRule`) is separate from P&L maths (`Backtester`).
* **Open/Closed** — add a data source, model, or splitter by writing a new
  class; no existing code changes.
* **Liskov** — `SyntheticDataSource` and `YFinanceDataSource` are
  interchangeable; the pipeline can't tell which it got.
* **Interface Segregation** — a model only implements `fit` + `predict_proba_up`.
* **Dependency Inversion** — `pipeline.walk_forward_predict` depends on the
  `ProbabilisticClassifier` protocol and a `SplitStrategy`, never on concretes.

---

## How leakage is prevented (the whole point)

1. **Walk-forward only**, never random K-fold; test always strictly after train.
2. **Embargo gap** between train and test kills rolling-window adjacency leakage.
3. **Features look backward, label looks forward** — the only forward `.shift()`
   in the codebase is the target, in `features.py`.
4. **Truncation-invariance test** (`tests/test_no_lookahead.py`) mechanically
   proves no feature value changes when the future is removed — and a meta-test
   proves that test can actually fail (it catches a deliberately leaky feature).
5. **Per-fold, train-only scaling** — the Logistic model's scaler is fit inside
   each training fold, so test statistics never leak into training.
6. **Execution lag** — a signal from the close of day *t* is acted on from *t+1*;
   the backtest cannot earn the same bar's return it predicted from.
7. **Explicit `freq=252`** everywhere annualised metrics are computed.

---

## Reading the output honestly

The run prints model metrics **beside naive baselines** and a **shuffled-label
control**. Interpret like this:

* **Edge over majority-class baseline ≈ 0 (or negative).** Normal and expected.
  Daily index direction is close to a coin flip; beating it out-of-sample after
  costs is genuinely hard. A small or negative edge is the *honest* result, not
  a bug.
* **Shuffled-label control ≈ baseline.** This is what you want. It means the
  harness isn't manufacturing a fake edge. If the control scored well *above*
  baseline, the pipeline would be leaking — stop and find the leak.
* **A suspiciously good real result (e.g. 56%+ accuracy) with a ~50% control**
  is the classic fingerprint of leakage, not discovery. Be more suspicious of
  good numbers than bad ones.

**Why even a good result here still wouldn't be a trading edge:** daily-bar
backtests ignore realistic fills, slippage, market impact, regime change, and
the multiple-comparisons problem (you'll have tried several configs). A positive
backtest is a *hypothesis*, not a strategy. Before risking money you would port
the signal to an event-driven engine with realistic execution, test across
regimes and assets, calibrate probabilities, and account for every config you
tried. This MVP deliberately stops at the honest-research line.

---

## Is this edge real? checklist

- [ ] Does it beat the **majority-class** baseline out-of-sample (not just 50%)?
- [ ] Does the **shuffled-label control** stay near baseline?
- [ ] Does it survive **transaction costs** (`--cost-bps`)?
- [ ] Is it **stable across thresholds / window sizes**, or one lucky config?
- [ ] Does it beat **buy-and-hold** on risk-adjusted terms (Sharpe/Calmar)?
- [ ] Would it survive realistic fills in an event-driven engine?

If you can't tick most of these, you have a learning project — which is exactly
what this is meant to be.

---

## Stretch backlog

Probability calibration (Brier/log-loss); triple-barrier labelling; a
cross-sectional multi-asset version (where `vectorbt`/Zipline-Reloaded earn
their keep); porting to Backtrader/Nautilus for realistic fills; a `vectorbt`
parameter-sweep triage harness.
