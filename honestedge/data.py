"""Epic 1 / Story 1.1 — data acquisition.

Three concrete pieces, all cooperating through the ``PriceDataSource`` seam:

* ``YFinanceDataSource``   — real adjusted daily OHLCV (works on a networked box).
* ``SyntheticDataSource``  — deterministic GBM-style series, fully offline, so
                             the pipeline and tests run anywhere with identical
                             numbers given a seed.
* ``CachingDataSource``    — a *decorator* that adds disk caching to ANY source
                             without that source knowing about caching
                             (Single Responsibility + Open/Closed).

Integrity checks live in one place (``_validate``) and are applied to whatever a
source returns, so a misbehaving source is caught at the boundary.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .interfaces import PriceDataSource

_REQUIRED_COLS = ("open", "high", "low", "close", "volume")


def _validate(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce the data contract every source must satisfy. Fails loudly rather
    than letting a silent gap or duplicate corrupt everything downstream."""
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    missing = [c for c in _REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"data source missing required columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("price index must be a DatetimeIndex")
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
    if df.index.has_duplicates:
        raise ValueError("price index has duplicate timestamps")
    n_nan = int(df[list(_REQUIRED_COLS)].isna().sum().sum())
    if n_nan:
        # Report, then drop rows with any NaN in required cols. We never
        # forward-fill prices silently — that manufactures fake bars.
        print(f"[data] NaN report: {n_nan} NaN cells in OHLCV; dropping affected rows")
        df = df.dropna(subset=list(_REQUIRED_COLS))
    return df[list(_REQUIRED_COLS)]


class YFinanceDataSource(PriceDataSource):
    """Real adjusted daily data via yfinance. Used on your machine."""

    def load(self, ticker: str, start: str, end: str, *, refresh: bool = False) -> pd.DataFrame:
        import yfinance as yf  # imported lazily so offline runs never need it

        raw = yf.download(
            ticker, start=start, end=end, progress=False, auto_adjust=True
        )
        if raw is None or len(raw) == 0:
            raise RuntimeError(
                f"yfinance returned no rows for {ticker}. "
                "Check connectivity or use SyntheticDataSource offline."
            )
        # yfinance may return a column MultiIndex for a single ticker; flatten it.
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw.rename(columns=str.lower)
        return _validate(raw)


class SyntheticDataSource(PriceDataSource):
    """Deterministic offline price generator.

    Produces a geometric-random-walk close with a mild, *learnable but weak*
    autocorrelation injected, so the modelling pipeline has something faintly
    real to find — without ever being a money machine. Seeded by ticker name so
    different tickers give different (but reproducible) series.
    """

    def __init__(self, seed: int = 7, ar_coef: float = 0.04, vol: float = 0.01):
        self._seed = seed
        self._ar_coef = ar_coef          # tiny momentum signal to detect
        self._vol = vol

    def _ticker_seed(self, ticker: str) -> int:
        h = hashlib.sha256(f"{ticker}:{self._seed}".encode()).hexdigest()
        return int(h[:8], 16)

    def load(self, ticker: str, start: str, end: str, *, refresh: bool = False) -> pd.DataFrame:
        rng = np.random.default_rng(self._ticker_seed(ticker))
        dates = pd.bdate_range(start=start, end=end)
        n = len(dates)
        if n < 50:
            raise ValueError("synthetic range too short; widen start/end")

        rets = np.zeros(n)
        eps = rng.normal(0.0, self._vol, size=n)
        rets[0] = eps[0]
        for t in range(1, n):
            # weak AR(1) momentum + noise: detectable edge, not a free lunch
            rets[t] = self._ar_coef * rets[t - 1] + eps[t]
        close = 100.0 * np.exp(np.cumsum(rets))

        # Build plausible OHLCV around the close path.
        intraday = np.abs(rng.normal(0.0, self._vol, size=n))
        high = close * (1 + intraday)
        low = close * (1 - intraday)
        open_ = np.concatenate([[close[0]], close[:-1]])  # open = prev close
        volume = rng.integers(1_000_000, 5_000_000, size=n).astype(float)

        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=pd.DatetimeIndex(dates, name="date"),
        )
        return _validate(df)


class CachingDataSource(PriceDataSource):
    """Decorator: adds parquet/csv disk caching to any wrapped source.

    The wrapped source has no idea it is being cached. Swap the cache strategy
    or remove it entirely without touching YFinance/Synthetic code.
    """

    def __init__(self, inner: PriceDataSource, cache_dir: str | Path = "./cache"):
        self._inner = inner
        self._dir = Path(cache_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, ticker: str, start: str, end: str) -> Path:
        key = f"{type(self._inner).__name__}_{ticker}_{start}_{end}".replace(":", "-")
        return self._dir / f"{key}.parquet"

    def load(self, ticker: str, start: str, end: str, *, refresh: bool = False) -> pd.DataFrame:
        path = self._path(ticker, start, end)
        if path.exists() and not refresh:
            try:
                return _validate(pd.read_parquet(path))
            except Exception:
                pass  # corrupt cache -> fall through and refetch
        df = self._inner.load(ticker, start, end, refresh=refresh)
        try:
            df.to_parquet(path)
        except Exception as exc:  # parquet engine missing -> degrade to csv
            df.to_csv(path.with_suffix(".csv"))
            print(f"[data] parquet unavailable ({exc}); cached as csv")
        return df
