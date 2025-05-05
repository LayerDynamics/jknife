# jknife/src/__init__.py
"""
jknife: Framework‑agnostic Jackknife estimation utilities.

Exports:
    - jackknife: high‑level function for generic jackknife estimation
    - JackknifeResult: dataclass holding aggregated statistics
"""

from __future__ import annotations

from .core import jackknife, JackknifeResult

__all__ = ["jackknife", "JackknifeResult"]

__version__: str = "0.1.0"
