# jackknife_regression/contrib/__init__.py
"""
Optional contributed helpers (e.g. sklearn, statsmodels) loaded lazily.
"""

from importlib import import_module
from types import ModuleType
from typing import TYPE_CHECKING

__all__ = ["sklearn_adapters", "statsmodels_adapters"]


def __getattr__(name: str) -> ModuleType:  # noqa: D401  (one‑liner OK)
    if name in ("sklearn_adapters", "statsmodels_adapters"):
        modname = f"jknife.contrib.{name}"
        try:
            return import_module(modname)
        except ModuleNotFoundError as exc:
            missing = "scikit‑learn" if name == "sklearn_adapters" else "statsmodels"
            raise ImportError(
                f"{missing} is required for '{name}'. "
                f"Install via 'pip install {missing}'."
            ) from exc
    raise AttributeError(name)


if TYPE_CHECKING:  # pragma: no cover
    from . import sklearn_adapters as sklearn_adapters  # type: ignore
    from . import statsmodels_adapters as statsmodels_adapters  # type: ignore
