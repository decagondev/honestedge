"""HonestEdge — an honest, lookahead-free next-day-direction prediction pipeline.

The package is organised around small, single-responsibility modules wired
together through abstract interfaces (see ``interfaces.py``). Nothing depends on
a concrete data source, model, or split strategy directly — only on the
abstractions — so each piece is swappable (Dependency Inversion).
"""

__version__ = "0.1.0"
