"""Epic 4 — turning probabilities into honest (paper) P&L.

Two responsibilities, kept separate:

* ``ThresholdPositionRule`` (a ``PositionRule``): decides the *position* from a
  probability. Pure decision logic, no returns maths.
* ``Backtester``: applies the execution lag, computes strategy returns, costs,
  the equity curve, and performance metrics.

The execution lag is the critical anti-leakage control here: a signal formed
from the close of day t can only be acted on from day t+1. We implement that by
shifting positions forward one bar before multiplying by the asset's *next-day*
return. There is no way, in this code path, to earn the return of the same bar
whose close produced the signal.

Annualisation is always ``freq``-explicit (252 trading days). Omitting this is
the most common source of silently-wrong Sharpe/Sortino numbers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .interfaces import PositionRule

TRADING_DAYS = 252


class ThresholdPositionRule(PositionRule):
    """Long when P(up) > upper; short (optional) when P(up) < lower; else flat."""

    def __init__(self, upper: float = 0.5, lower: float | None = None, allow_short: bool = False):
        self.upper = upper
        self.lower = lower if lower is not None else (1.0 - upper)
        self.allow_short = allow_short

    def positions(self, proba_up: pd.Series) -> pd.Series:
        pos = pd.Series(0.0, index=proba_up.index)
        pos[proba_up > self.upper] = 1.0
        if self.allow_short:
            pos[proba_up < self.lower] = -1.0
        return pos


@dataclass
class BacktestResult:
    equity: pd.Series
    strategy_returns: pd.Series
    benchmark_equity: pd.Series
    positions: pd.Series
    metrics: dict
    benchmark_metrics: dict


class Backtester:
    """Hand-rolled vectorized backtest. Small enough to fully trust, which is
    exactly why we don't hide it behind a framework for the MVP."""

    def __init__(self, cost_bps: float = 1.0):
        self.cost_bps = cost_bps  # per unit turnover, in basis points

    def run(
        self,
        asset_returns: pd.Series,
        positions: pd.Series,
        position_rule_applied: bool = True,
    ) -> BacktestResult:
        # Align
        df = pd.DataFrame({"ret": asset_returns, "pos": positions}).dropna()

        # --- EXECUTION LAG: act on tomorrow's bar, never today's -------------
        # position decided at close of t applies to the return from t -> t+1.
        lagged_pos = df["pos"].shift(1).fillna(0.0)

        # turnover & costs on position changes
        turnover = lagged_pos.diff().abs().fillna(lagged_pos.abs())
        cost = turnover * (self.cost_bps / 1e4)

        strat_ret = lagged_pos * df["ret"] - cost
        equity = (1.0 + strat_ret).cumprod()
        bench_equity = (1.0 + df["ret"]).cumprod()

        return BacktestResult(
            equity=equity,
            strategy_returns=strat_ret,
            benchmark_equity=bench_equity,
            positions=lagged_pos,
            metrics=self._metrics(strat_ret),
            benchmark_metrics=self._metrics(df["ret"]),
        )

    @staticmethod
    def _metrics(returns: pd.Series) -> dict:
        r = returns.dropna()
        if len(r) == 0:
            return {}
        total_return = float((1 + r).prod() - 1)
        ann_return = float((1 + r).prod() ** (TRADING_DAYS / len(r)) - 1)
        ann_vol = float(r.std() * np.sqrt(TRADING_DAYS))
        sharpe = float(r.mean() / r.std() * np.sqrt(TRADING_DAYS)) if r.std() > 0 else 0.0
        downside = r[r < 0].std()
        sortino = float(r.mean() / downside * np.sqrt(TRADING_DAYS)) if downside and downside > 0 else 0.0
        equity = (1 + r).cumprod()
        max_dd = float((equity / equity.cummax() - 1).min())
        calmar = float(ann_return / abs(max_dd)) if max_dd != 0 else 0.0
        hit = float((r > 0).mean())
        return {
            "total_return": total_return,
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_dd,
            "calmar": calmar,
            "hit_rate": hit,
            "n_periods": int(len(r)),
        }
