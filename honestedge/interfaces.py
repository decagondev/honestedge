"""Abstract interfaces for HonestEdge.

These are the *seams* of the system. Concrete classes implement them; the
pipeline depends only on these abstractions. This is what makes the design
satisfy the Dependency Inversion Principle (high-level ``run`` logic does not
depend on yfinance, LightGBM, or any specific splitter) and the Open/Closed
Principle (add a new data source / model / splitter by writing a new
implementation, never by editing existing code).

Each interface is deliberately tiny (Interface Segregation): a model only has
to know how to ``fit`` and ``predict_proba``; it is not forced to know about
data loading or plotting.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Protocol, runtime_checkable

import numpy as np
import pandas as pd


class PriceDataSource(ABC):
    """Anything that can produce a clean OHLCV frame for one instrument.

    Implementations: ``YFinanceDataSource`` (real), ``SyntheticDataSource``
    (deterministic, offline). Both are Liskov-substitutable: the pipeline
    cannot tell which it received.
    """

    @abstractmethod
    def load(self, ticker: str, start: str, end: str, *, refresh: bool = False) -> pd.DataFrame:
        """Return a DataFrame indexed by a sorted, unique DatetimeIndex with at
        least columns: ``open, high, low, close, volume`` (lowercased)."""
        raise NotImplementedError


class FeatureBuilder(ABC):
    """Turns a raw OHLCV frame into an (X, y) supervised-learning problem.

    The single most important contract in the whole system:
      * every column of X at row t uses information available *up to and
        including the close of day t* — never later.
      * y at row t is the forward-looking label (direction of t -> t+1).
    """

    @abstractmethod
    def build(self, prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        raise NotImplementedError

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        raise NotImplementedError


class SplitStrategy(ABC):
    """Yields (train_idx, test_idx) positional-index arrays.

    Implementations must NEVER let a test index precede its train indices in
    time, and should support an embargo gap. Random K-fold is intentionally
    *not* an implementation — it is invalid for time series.
    """

    @abstractmethod
    def split(self, n_samples: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        raise NotImplementedError


@runtime_checkable
class ProbabilisticClassifier(Protocol):
    """Minimal model interface: fit, then predict calibrated-ish probabilities.

    A Protocol (structural typing) rather than an ABC so that thin wrappers and
    even some sklearn estimators satisfy it without explicit inheritance.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ProbabilisticClassifier": ...

    def predict_proba_up(self, X: pd.DataFrame) -> np.ndarray:
        """Return P(next day is up) as a 1-D array in [0, 1]."""
        ...


class PositionRule(ABC):
    """Maps a probability series into a position series in {-1, 0, +1} (or
    fractional). Separated from the model so the *decision* layer is
    independent of the *prediction* layer (Single Responsibility)."""

    @abstractmethod
    def positions(self, proba_up: pd.Series) -> pd.Series:
        raise NotImplementedError
