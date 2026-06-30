"""Epic 1 / Story 1.2 — walk-forward evaluation splits.

This is the heart of the project. Get it right and leakage becomes structurally
hard; get it wrong and every downstream number is a lie.

``WalkForwardSplit`` yields (train_idx, test_idx) where:
  * test always lies strictly *after* its training data in time,
  * an ``embargo`` gap of rows is removed between the end of train and the
    start of test, killing adjacency leakage (a feature at the train/test
    boundary that peeks across via a rolling window),
  * windows are ``expanding`` (train grows) or rolling (fixed train length).

Random K-fold is deliberately absent: shuffling time-series rows lets the model
train on the future and test on the past, which is the cardinal sin.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np

from .interfaces import SplitStrategy


class WalkForwardSplit(SplitStrategy):
    def __init__(
        self,
        initial_train: int,
        test_size: int,
        step: int | None = None,
        embargo: int = 0,
        expanding: bool = True,
    ):
        if initial_train <= 0 or test_size <= 0:
            raise ValueError("initial_train and test_size must be positive")
        if embargo < 0:
            raise ValueError("embargo must be >= 0")
        self.initial_train = initial_train
        self.test_size = test_size
        self.step = step or test_size           # default: non-overlapping test windows
        self.embargo = embargo
        self.expanding = expanding

    def split(self, n_samples: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        train_end = self.initial_train
        while True:
            test_start = train_end + self.embargo
            test_end = test_start + self.test_size
            if test_end > n_samples:
                break

            if self.expanding:
                train_idx = np.arange(0, train_end)
            else:
                train_start = max(0, train_end - self.initial_train)
                train_idx = np.arange(train_start, train_end)

            test_idx = np.arange(test_start, test_end)
            yield train_idx, test_idx
            train_end += self.step

    def print_folds(self, n_samples: int) -> None:
        """Human sanity check — *see* the windows before trusting them."""
        print(f"[splits] n_samples={n_samples} embargo={self.embargo} "
              f"{'expanding' if self.expanding else 'rolling'}")
        for i, (tr, te) in enumerate(self.split(n_samples)):
            print(f"  fold {i:>2}: train[{tr[0]:>4}..{tr[-1]:>4}] "
                  f"(n={len(tr):>4})  ->  test[{te[0]:>4}..{te[-1]:>4}] (n={len(te)})")
