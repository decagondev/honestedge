"""Epic 3 / Story 3.1 — models behind a single tiny interface.

Both wrappers expose ``fit`` and ``predict_proba_up`` (the
``ProbabilisticClassifier`` protocol). The pipeline never imports LightGBM or
sklearn directly — it receives one of these objects, so swapping models is a
one-line change at the composition root (Dependency Inversion).

The Logistic wrapper folds its StandardScaler *inside* fit/predict, which means
scaling is always fit on the training fold only. Leakage-by-scaling is therefore
impossible by construction, not by remembering to do it right each time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class LightGBMClassifierModel:
    """Gradient-boosted trees. Strong tabular baseline; gives feature
    importances for free (used in Epic 5)."""

    def __init__(self, **params):
        defaults = dict(
            n_estimators=200,
            num_leaves=15,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            min_child_samples=30,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        defaults.update(params)
        self._params = defaults
        self._model = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMClassifierModel":
        from lightgbm import LGBMClassifier

        self._model = LGBMClassifier(**self._params)
        self._model.fit(X, y)
        return self

    def predict_proba_up(self, X: pd.DataFrame) -> np.ndarray:
        proba = self._model.predict_proba(X)
        # column index of the positive (up==1) class
        up_col = list(self._model.classes_).index(1)
        return proba[:, up_col]

    @property
    def feature_importances_(self) -> np.ndarray:
        return self._model.feature_importances_


class LogisticRegressionModel:
    """Interpretable, hard-to-overfit floor. Scaling is fit-on-train-only
    because the scaler lives inside this object's fit()."""

    def __init__(self, C: float = 1.0):
        self._C = C
        self._scaler = None
        self._model = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticRegressionModel":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X)        # fit on THIS fold's train only
        self._model = LogisticRegression(C=self._C, max_iter=1000)
        self._model.fit(Xs, y)
        return self

    def predict_proba_up(self, X: pd.DataFrame) -> np.ndarray:
        Xs = self._scaler.transform(X)            # transform with train stats
        proba = self._model.predict_proba(Xs)
        up_col = list(self._model.classes_).index(1)
        return proba[:, up_col]
