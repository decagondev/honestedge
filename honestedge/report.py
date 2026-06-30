"""Epic 6 / Story 6.2 — visual + textual reporting.

Pure presentation: takes already-computed results and renders them. No modelling
or backtest logic leaks in here (Single Responsibility). Saves figures to disk so
the pipeline is headless-friendly.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .backtest import BacktestResult  # noqa: E402


def _fmt(metrics: dict) -> str:
    order = ["accuracy", "roc_auc", "total_return", "ann_return", "sharpe",
             "sortino", "max_drawdown", "calmar", "hit_rate", "n", "n_periods"]
    parts = []
    for k in order:
        if k in metrics and metrics[k] == metrics[k]:  # skip NaN
            v = metrics[k]
            parts.append(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}")
    return "  ".join(parts)


def print_summary(
    clf_metrics: dict,
    baselines: dict,
    bt: BacktestResult | None = None,
    shuffled: dict | None = None,
) -> None:
    print("\n" + "=" * 70)
    print("HONESTEDGE — OUT-OF-SAMPLE RESULTS")
    print("=" * 70)
    print("\n[classification]")
    print("  model    :", _fmt(clf_metrics))
    print("  baselines:", "  ".join(f"{k}={v:.4f}" for k, v in baselines.items()))
    edge = clf_metrics.get("accuracy", 0) - baselines.get("majority_class_accuracy", 0.5)
    print(f"  edge over majority-class baseline: {edge:+.4f}")
    if shuffled is not None:
        print("\n[shuffled-label control]  (should be ~baseline; if not, harness leaks)")
        print("  shuffled :", _fmt(shuffled))
    if bt is not None:
        print("\n[backtest: strategy vs buy-and-hold]")
        print("  strategy :", _fmt(bt.metrics))
        print("  buy&hold :", _fmt(bt.benchmark_metrics))
    print("=" * 70 + "\n")


def save_plots(bt: BacktestResult, out_dir: str | Path = "./reports", tag: str = "spy") -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []

    # equity curve vs benchmark
    fig, ax = plt.subplots(figsize=(10, 5))
    bt.equity.plot(ax=ax, label="strategy")
    bt.benchmark_equity.plot(ax=ax, label="buy & hold", alpha=0.7)
    ax.set_title("Equity curve — strategy vs buy & hold (OOS)")
    ax.set_ylabel("growth of 1")
    ax.legend()
    ax.grid(alpha=0.3)
    p1 = out / f"equity_{tag}.png"
    fig.tight_layout(); fig.savefig(p1, dpi=120); plt.close(fig)
    paths.append(p1)

    # drawdown
    dd = bt.equity / bt.equity.cummax() - 1
    fig, ax = plt.subplots(figsize=(10, 3))
    dd.plot(ax=ax, color="crimson")
    ax.fill_between(dd.index, dd.values, 0, color="crimson", alpha=0.3)
    ax.set_title("Strategy drawdown (OOS)")
    ax.grid(alpha=0.3)
    p2 = out / f"drawdown_{tag}.png"
    fig.tight_layout(); fig.savefig(p2, dpi=120); plt.close(fig)
    paths.append(p2)

    return paths
