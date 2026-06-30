"""Epic 2 — features (backward-looking) and target (forward-looking).

The contract, restated because it is the whole game:
    features look backward, the label looks forward, and they meet at row t.

Every feature is computed from columns that are themselves lagged or from
rolling windows that end at t. The target is the ONLY forward shift in the
codebase, and it lives here, clearly marked. The truncation-invariance test in
``tests/test_no_lookahead.py`` mechanically proves no feature peeks ahead.

Indicators (RSI, MA-distance) are hand-computed so the project has zero hard
dependency on pandas-ta / TA-Lib; if you install pandas-ta you can swap in its
versions behind the same FeatureBuilder interface.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .interfaces import FeatureBuilder


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Classic Wilder RSI, computed only from past closes."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


class DirectionFeatureBuilder(FeatureBuilder):
    """Builds the standard MVP feature set + next-day-direction target."""

    def __init__(
        self,
        return_lags: tuple[int, ...] = (1, 2, 3, 5, 10),
        vol_window: int = 10,
        z_window: int = 20,
        rsi_window: int = 14,
        ma_window: int = 20,
    ):
        self.return_lags = return_lags
        self.vol_window = vol_window
        self.z_window = z_window
        self.rsi_window = rsi_window
        self.ma_window = ma_window
        self._feature_names: list[str] = []

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names)

    def build(self, prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        close = prices["close"]
        ret = close.pct_change()  # ret at t = (close_t / close_{t-1}) - 1, known at t

        feats: dict[str, pd.Series] = {}

        # --- lagged returns: ret at t-k, all strictly past -------------------
        for k in self.return_lags:
            feats[f"ret_lag_{k}"] = ret.shift(k - 1) if k == 1 else ret.shift(k - 1)
        # (ret itself is already "known at t"; ret_lag_1 = today's realised return)

        # --- rolling volatility of past returns ------------------------------
        feats[f"vol_{self.vol_window}"] = ret.rolling(self.vol_window).std()

        # --- z-score of return vs its trailing window ------------------------
        roll_mean = ret.rolling(self.z_window).mean()
        roll_std = ret.rolling(self.z_window).std()
        feats[f"ret_z_{self.z_window}"] = (ret - roll_mean) / roll_std

        # --- RSI (past closes only) ------------------------------------------
        feats[f"rsi_{self.rsi_window}"] = _rsi(close, self.rsi_window)

        # --- distance of close from its trailing MA --------------------------
        ma = close.rolling(self.ma_window).mean()
        feats[f"ma_dist_{self.ma_window}"] = (close - ma) / ma

        # --- calendar freebie -------------------------------------------------
        feats["dow"] = pd.Series(prices.index.dayofweek, index=prices.index).astype(float)

        X = pd.DataFrame(feats, index=prices.index)

        # --- TARGET: the one and only forward shift in the codebase ----------
        # y_t = 1 if close_{t+1} > close_t else 0
        future_close = close.shift(-1)
        y = (future_close > close).astype("float")
        y.name = "target_up"

        # Drop warm-up NaNs (from rolling windows) and the final row (no t+1).
        data = X.copy()
        data["__y__"] = y
        data = data.dropna()
        y_clean = data.pop("__y__").astype(int)
        X_clean = data

        self._feature_names = list(X_clean.columns)
        return X_clean, y_clean
