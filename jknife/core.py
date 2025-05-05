# jknife/core.py
"""
Core leave‑one‑out jackknife implementation.

The algorithm:
    1. For i in 0..n‑1:
         a. Fit model on data with sample i removed  -> θ_i
    2. θ̄ = average(θ_i)
    3. For each parameter j:
         bias_j = (n - 1) * (θ̄_j - θ_full_j)
         var_j  = (n - 1) / n * Σ(θ_i_j - θ̄_j)^2
         se_j   = sqrt(var_j)
    4. Return JackknifeResult carrying per‑parameter stats & helpers.
"""

from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Any, Sequence, List

import numpy as np

try:
    from joblib import Parallel, delayed
    _HAVE_JOBLIB = True
except ImportError:  # graceful fallback
    _HAVE_JOBLIB = False

from .utils import check_shapes, confidence_interval
from .exceptions import JackknifeError

ArrayLike = Any  # minimal typing to avoid heavy deps


@dataclass
class JackknifeResult:
    """Container for aggregated jackknife statistics."""
    point_estimate: np.ndarray
    bias: np.ndarray
    se: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    theta_i: np.ndarray
    # Raw θ_i for downstream diagnostics

    def summary(self) -> str:
        """Pretty text table summarising results."""
        header = (
            f"{'param':>6} | {'estimate':>12} | {'bias':>12} | "
            f"{'se':>12} | {'ci_low':>12} | {'ci_high':>12}"
        )
        lines = [header, "-" * len(header)]
        for idx, (est, b, se, lo, hi) in enumerate(
            zip(self.point_estimate, self.bias, self.se, self.ci_low, self.ci_high)
        ):
            lines.append(
                f"{idx:>6d} | {est:12.6f} | {b:12.6f} | {se:12.6f} | {lo:12.6f} | {hi:12.6f}"
            )
        return "\n".join(lines)

    # Convenience accessors
    @property
    def variance(self) -> np.ndarray:
        return self.se ** 2


def _default_parallel_backend(n_jobs: int):
    """Return Parallel object respecting joblib availability."""
    if n_jobs == 1 or not _HAVE_JOBLIB:
        def run_serial(funcs):
            return [f() for f in funcs]
        return run_serial, False
    
    # For joblib, return a function that actually executes the tasks
    def run_joblib(tasks):
        return Parallel(n_jobs=n_jobs, backend="loky")([delayed(task)() for task in tasks])
    
    return run_joblib, True


def _validate_callable(name: str, fn: Callable):
    if not callable(fn):
        raise TypeError(f"{name} must be callable.")
    # Removed the overly strict signature validation that caused "expects arguments (X, y, ...)" warning


def jackknife(
    X: ArrayLike,
    y: ArrayLike,
    *,
    fit_fn: Callable[..., Any],
    coef_fn: Callable[[Any], ArrayLike],
    alpha: float = 0.05,
    n_jobs: int = 1,
    fit_kwargs: Dict[str, Any] | None = None,
    return_models: bool = False,
) -> JackknifeResult | tuple[JackknifeResult, List[Any]]:
    """
    Generic jackknife estimator.

    Parameters
    ----------
    X, y : array‑likes
        Explanatory variables and target.
    fit_fn : callable
        Signature `(X, y, **fit_kwargs) -> model`. Run on each leave‑one‑out subset **and** on full data.
    coef_fn : callable
        Signature `(model) -> 1‑D numpy array` of statistics to aggregate.
    alpha : float
        Confidence‑interval level (two‑sided).
    n_jobs : int
        Parallel workers (requires joblib).
    fit_kwargs : dict, optional
        Extra keyword args forwarded to `fit_fn`.
    return_models : bool
        Whether to return the fitted full model and the list of jackknife models.

    Returns
    -------
    JackknifeResult or (JackknifeResult, [models])
    """
    fit_kwargs = fit_kwargs or {}
    _validate_callable("fit_fn", fit_fn)
    _validate_callable("coef_fn", coef_fn)

    X = np.asarray(X)
    y = np.asarray(y)
    n = X.shape[0]
    check_shapes(X, y)

    # Full‑sample fit
    full_model = fit_fn(X, y, **fit_kwargs)
    theta_full = np.asarray(coef_fn(full_model), dtype=float)
    p = theta_full.size

    # Prepare leave‑one‑out jobs
    tasks: List[Callable[[], np.ndarray]] = []
    fitted_models = []  # To store fitted models if return_models=True
    
    for i in range(n):
        # Closure to capture current i
        def make_task(idx=i):
            def task():
                X_minus_i = np.delete(X, idx, axis=0)
                y_minus_i = np.delete(y, idx, axis=0)
                model = fit_fn(X_minus_i, y_minus_i, **fit_kwargs)
                if return_models:
                    fitted_models.append(model)
                return np.asarray(coef_fn(model), dtype=float)
            return task
        
        tasks.append(make_task())

    # Execute tasks
    runner, is_joblib = _default_parallel_backend(n_jobs)
    theta_i_list = runner(tasks)

    # Convert results to array
    theta_i = np.vstack(theta_i_list)  # shape (n, p)
    theta_bar = theta_i.mean(axis=0)

    # Bias, variance, standard error
    bias = (n - 1) * (theta_bar - theta_full)
    var = (n - 1) / n * ((theta_i - theta_bar) ** 2).sum(axis=0)
    se = np.sqrt(var)

    ci_low, ci_high = np.empty(p), np.empty(p)
    for j in range(p):
        ci_low[j], ci_high[j] = confidence_interval(
            theta_full[j], se[j], dof=n - 1, alpha=alpha
        )

    result = JackknifeResult(
        point_estimate=theta_full,
        bias=bias,
        se=se,
        ci_low=ci_low,
        ci_high=ci_high,
        theta_i=theta_i,
    )

    if return_models:
        return result, [full_model] + fitted_models
    return result
