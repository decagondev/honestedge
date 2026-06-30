"""Epic 3 / Story 3.3 + Epic 5 — honest scoring and reality checks.

* ``classification_metrics`` + ``baseline_metrics``: never report a model number
  without the naive baselines beside it (majority class, always-up, coin-flip).
* ``shuffled_label_control``: the single most clarifying test in the project.
  Re-run the SAME pipeline on randomly shuffled targets. If accuracy barely
  drops, the model is fitting noise / structure that isn't predictive, and any
  apparent edge is an illusion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def classification_metrics(y_true: pd.Series, proba_up: pd.Series, threshold: float = 0.5) -> dict:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

    y_true = y_true.loc[proba_up.index]
    y_pred = (proba_up > threshold).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "n": int(len(y_true)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, proba_up))
    except ValueError:
        out["roc_auc"] = float("nan")  # single-class fold
    return out


def baseline_metrics(y_true: pd.Series) -> dict:
    """The bar any 'edge' must clear. Majority-class accuracy is the honest one."""
    p_up = float(y_true.mean())
    majority = max(p_up, 1 - p_up)
    return {
        "always_up_accuracy": p_up,
        "majority_class_accuracy": majority,
        "coin_flip_accuracy": 0.5,
        "base_rate_up": p_up,
    }


def shuffled_label_control(
    X: pd.DataFrame,
    y: pd.Series,
    model_factory,
    split_strategy,
    seed: int = 0,
) -> dict:
    """Train/eval walk-forward on SHUFFLED labels. Result should be ~baseline.
    A high score here is a red flag that the harness itself is leaking."""
    rng = np.random.default_rng(seed)
    y_shuf = pd.Series(rng.permutation(y.values), index=y.index, name=y.name)
    from .pipeline import walk_forward_predict  # local import to avoid cycle

    proba = walk_forward_predict(X, y_shuf, model_factory, split_strategy)
    return classification_metrics(y_shuf, proba)


def permutation_importance_oos(
    X: pd.DataFrame,
    y: pd.Series,
    proba_up: pd.Series,
    model_factory,
    split_strategy,
    n_repeats: int = 3,
    seed: int = 0,
) -> pd.Series:
    """Drop in predictive power when each feature is shuffled, measured on the
    concatenated OOS predictions. Higher = the model leans on it more."""
    from sklearn.metrics import accuracy_score
    from .pipeline import walk_forward_predict

    rng = np.random.default_rng(seed)
    base = accuracy_score(y.loc[proba_up.index], (proba_up > 0.5).astype(int))
    importances = {}
    for col in X.columns:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = rng.permutation(Xp[col].values)
            proba_p = walk_forward_predict(Xp, y, model_factory, split_strategy)
            acc = accuracy_score(y.loc[proba_p.index], (proba_p > 0.5).astype(int))
            drops.append(base - acc)
        importances[col] = float(np.mean(drops))
    return pd.Series(importances).sort_values(ascending=False)
