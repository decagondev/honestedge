"""Tests for splits, backtest execution-lag, and metric sanity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from honestedge.backtest import Backtester, ThresholdPositionRule
from honestedge.splits import WalkForwardSplit


def test_walkforward_test_always_after_train():
    split = WalkForwardSplit(initial_train=100, test_size=20, embargo=5)
    for tr, te in split.split(400):
        assert tr.max() < te.min(), "test set must lie entirely after train set"
        # embargo gap respected
        assert te.min() - tr.max() - 1 >= 5 - 1


def test_walkforward_embargo_gap():
    split = WalkForwardSplit(initial_train=100, test_size=20, embargo=10)
    tr, te = next(split.split(400))
    gap = te.min() - tr.max() - 1
    assert gap == 10, f"embargo gap should be 10, got {gap}"


def test_rolling_window_fixed_length():
    split = WalkForwardSplit(initial_train=100, test_size=20, embargo=0, expanding=False)
    lengths = {len(tr) for tr, _ in split.split(500)}
    assert lengths == {100}, "rolling window train length must stay fixed"


def test_expanding_window_grows():
    split = WalkForwardSplit(initial_train=100, test_size=20, embargo=0, expanding=True)
    lengths = [len(tr) for tr, _ in split.split(500)]
    assert lengths == sorted(lengths) and lengths[0] < lengths[-1]


def test_execution_lag_prevents_same_bar_return():
    """A position formed at t must not earn the return realised at t."""
    idx = pd.date_range("2020-01-01", periods=10, freq="B")
    # asset return is +10% only on day 5
    ret = pd.Series(0.0, index=idx)
    ret.iloc[5] = 0.10
    # "perfect" same-bar position on day 5 (cheating) -> with lag it earns day 6
    pos = pd.Series(0.0, index=idx)
    pos.iloc[5] = 1.0

    bt = Backtester(cost_bps=0.0).run(ret, pos)
    # because of the lag, the +10% on day 5 is NOT captured by the day-5 signal
    assert abs(bt.strategy_returns.iloc[5]) < 1e-12, \
        "execution lag failed: strategy earned the same-bar return it predicted from"


def test_metrics_freq_explicit_sharpe_reasonable():
    idx = pd.date_range("2020-01-01", periods=252, freq="B")
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, size=252), index=idx)
    m = Backtester._metrics(r)
    # annualised vol of ~1% daily should be ~16%
    assert 0.10 < m["ann_vol"] < 0.22
    assert "sharpe" in m and "max_drawdown" in m


def test_threshold_rule_positions():
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    proba = pd.Series([0.2, 0.5, 0.55, 0.8, 0.49], index=idx)
    rule = ThresholdPositionRule(upper=0.5)
    pos = rule.positions(proba)
    assert list(pos.values) == [0.0, 0.0, 1.0, 1.0, 0.0]
