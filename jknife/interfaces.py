# jknife/interfaces.py
"""
Public‑facing type aliases & helper factories.
Kept minimal to avoid heavyweight typing dependencies.
"""

from __future__ import annotations
from typing import Callable, Any

ArrayLike = Any
FitFn = Callable[..., Any]
CoefFn = Callable[[Any], ArrayLike]

__all__ = ["ArrayLike", "FitFn", "CoefFn"]
