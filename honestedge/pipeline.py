"""Epic 3 / Story 3.2 — the walk-forward engine.

``walk_forward_predict`` is the high-level policy of the whole system, and it
depends on NOTHING concrete: it takes a model *factory* (a zero-arg callable
returning a fresh ``ProbabilisticClassifier``) and a ``SplitStrategy``. It loops
folds, fits a brand-new model on each train slice, predicts on the test slice,
and stitches the out-of-sample probabilities into one aligned series.

A fresh model per fold is essential: reusing a fitted model would let later
folds benefit from earlier test data via warm state.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

from .interfaces import ProbabilisticClassifier, SplitStrategy

ModelFactory = Callable[[], ProbabilisticClassifier]


def walk_forward_predict(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory: ModelFactory,
    split_strategy: SplitStrategy,
) -> pd.Series:
    if len(X) != len(y):
        raise ValueError("X and y length mismatch")

    n = len(X)
    proba = pd.Series(np.nan, index=X.index, name="proba_up")

    for train_idx, test_idx in split_strategy.split(n):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te = X.iloc[test_idx]

        # single-class training fold guard (rare on short data)
        if y_tr.nunique() < 2:
            proba.iloc[test_idx] = float(y_tr.iloc[0])
            continue

        model = model_factory().fit(X_tr, y_tr)
        proba.iloc[test_idx] = model.predict_proba_up(X_te)

    return proba.dropna()
