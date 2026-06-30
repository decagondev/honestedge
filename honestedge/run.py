"""Epic 6 / Story 6.1 — the composition root.

This is the ONLY module that names concrete implementations. It picks a data
source, a feature builder, a split strategy, a model factory, and a position
rule, then hands them to the abstract pipeline. Swapping any component
(yfinance -> synthetic, LightGBM -> logistic, expanding -> rolling) is a change
here and nowhere else.

Usage:
    python -m honestedge.run --ticker SPY --source synthetic
    python -m honestedge.run --ticker SPY --source yfinance --model lgbm
"""

from __future__ import annotations

import argparse

from .backtest import Backtester, ThresholdPositionRule
from .data import CachingDataSource, SyntheticDataSource, YFinanceDataSource
from .evaluation import baseline_metrics, classification_metrics, shuffled_label_control
from .features import DirectionFeatureBuilder
from .model import LightGBMClassifierModel, LogisticRegressionModel
from .pipeline import walk_forward_predict
from .report import print_summary, save_plots
from .splits import WalkForwardSplit


def build_source(name: str):
    inner = SyntheticDataSource() if name == "synthetic" else YFinanceDataSource()
    return CachingDataSource(inner)


def build_model_factory(name: str):
    if name == "logistic":
        return lambda: LogisticRegressionModel()
    return lambda: LightGBMClassifierModel()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="HonestEdge — honest next-day-direction MVP")
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--start", default="2015-01-01")
    p.add_argument("--end", default="2024-12-31")
    p.add_argument("--source", choices=["synthetic", "yfinance"], default="synthetic")
    p.add_argument("--model", choices=["lgbm", "logistic"], default="lgbm")
    p.add_argument("--initial-train", type=int, default=500)
    p.add_argument("--test-size", type=int, default=63)   # ~one quarter
    p.add_argument("--embargo", type=int, default=5)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--cost-bps", type=float, default=1.0)
    p.add_argument("--rolling", action="store_true", help="rolling instead of expanding window")
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--no-control", action="store_true", help="skip shuffled-label control")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args(argv)

    # --- wire concrete components (Dependency Injection happens here) --------
    source = build_source(args.source)
    feature_builder = DirectionFeatureBuilder()
    split = WalkForwardSplit(
        initial_train=args.initial_train,
        test_size=args.test_size,
        embargo=args.embargo,
        expanding=not args.rolling,
    )
    model_factory = build_model_factory(args.model)
    rule = ThresholdPositionRule(upper=args.threshold)

    # --- run -----------------------------------------------------------------
    print(f"[run] source={args.source} model={args.model} ticker={args.ticker}")
    prices = source.load(args.ticker, args.start, args.end, refresh=args.refresh)
    print(f"[run] loaded {len(prices)} bars {prices.index[0].date()}..{prices.index[-1].date()}")

    X, y = feature_builder.build(prices)
    print(f"[run] built {X.shape[1]} features, {len(X)} usable rows")
    split.print_folds(len(X))

    proba = walk_forward_predict(X, y, model_factory, split)
    clf = classification_metrics(y, proba, threshold=args.threshold)
    base = baseline_metrics(y.loc[proba.index])

    # asset next-day returns aligned to predictions, for the backtest
    asset_ret = prices["close"].pct_change().reindex(proba.index)
    positions = rule.positions(proba)
    bt = Backtester(cost_bps=args.cost_bps).run(asset_ret, positions)

    shuffled = None
    if not args.no_control:
        shuffled = shuffled_label_control(X, y, model_factory, split)

    print_summary(clf, base, bt, shuffled)

    if not args.no_plots:
        paths = save_plots(bt, tag=args.ticker.lower())
        print("[run] saved plots:", ", ".join(str(p) for p in paths))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
