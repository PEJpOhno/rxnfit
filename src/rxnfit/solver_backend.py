# Copyright (c) 2025 Mitsuru Ohno
# Use of this source code is governed by a BSD-3-style
# license that can be found in the LICENSE file.

# 05/04/2026, M. Ohno  

"""Central ODE driver: SciPy ``solve_ivp`` with optional numbalsoda for LSODA.

Implements the integration policy described in ``docs/spec_docs/numbalsoda2.md``.

Notes:
    * Numbalsoda is considered only for ``method="LSODA"``, non-time-dependent
      rate laws, and a non-empty ``t_eval`` grid.
    * Import failures, unsupported RHS, or integrator errors trigger a
      one-time :class:`UserWarning` (per reason) and fall back to RK45 via
      :func:`scipy.integrate.solve_ivp`.
    * If ``time_dependent`` is true, LSODA requests are coerced to RK45
      without calling numbalsoda.
"""

from __future__ import annotations

from enum import Enum, auto
from types import SimpleNamespace
import warnings

import numpy as np
from scipy.integrate import solve_ivp


_NUMBALSODA_IMPORT_ERROR: Exception | None = None
_NUMBALSODA_IMPORTED = False
_NUMBALSODA_MODULE = None
_WARNED_KEYS: set[str] = set()

# §7: numbalsoda ``lsoda`` always receives a ``data`` array. When the RHS does
# not read ``p`` (``n_p == 0``), we still pass a length-1 dummy float64 array
# so the native wrapper always sees a valid pointer (same spirit as the
# library default ``np.array([0.0])``).


class LsodaImplicitTEvalMode(Enum):
    """Implicit ``t_eval`` construction mode for LSODA (names are part of the API).

    Each member selects the grid recipe documented in numbalsoda2.md section 5.5.
    """

    SOLVE_SYSTEM = auto()
    COMPUTE_RSS = auto()
    INTEGRATE_DATASETS = auto()
    EVAL_ODE_FIT = auto()


def build_t_eval_when_none_for_lsoda(
    mode: LsodaImplicitTEvalMode,
    *,
    t_span=None,
    t_start=None,
    t_exp_col=None,
    t_exp=None,
    t_points=None,
) -> np.ndarray:
    """Build a sorted unique ``float64`` time grid for LSODA when ``t_eval`` is omitted.

    Args:
        mode: Which recipe to apply (``SOLVE_SYSTEM``, ``COMPUTE_RSS``, etc.).
        t_span: Required for ``SOLVE_SYSTEM`` and ``INTEGRATE_DATASETS`` as ``(t0, t1)``.
        t_start: Dataset start time for ``INTEGRATE_DATASETS``.
        t_exp_col: Experimental times for ``COMPUTE_RSS``.
        t_exp: Concatenated experimental times for ``INTEGRATE_DATASETS``.
        t_points: Observation times for ``EVAL_ODE_FIT``.

    Returns:
        One-dimensional ``float64`` array in non-decreasing order with duplicates removed.

    Raises:
        ValueError: If required keyword arguments for ``mode`` are missing or invalid.
    """
    match mode:
        case LsodaImplicitTEvalMode.SOLVE_SYSTEM:
            if t_span is None or len(t_span) != 2:
                raise ValueError("SOLVE_SYSTEM requires t_span=(t0, t1).")
            t0, t1 = float(t_span[0]), float(t_span[1])
            n_dense = max(100, 1)
            out = np.linspace(t0, t1, n_dense, dtype=np.float64)
        case LsodaImplicitTEvalMode.COMPUTE_RSS:
            if t_exp_col is None:
                raise ValueError("COMPUTE_RSS requires t_exp_col.")
            out = np.sort(np.unique(np.asarray(t_exp_col, dtype=np.float64)))
        case LsodaImplicitTEvalMode.INTEGRATE_DATASETS:
            if t_span is None or len(t_span) != 2:
                raise ValueError("INTEGRATE_DATASETS requires t_span=(t0, t1).")
            if t_start is None or t_exp is None:
                raise ValueError("INTEGRATE_DATASETS requires t_start and t_exp.")
            t_exp = np.asarray(t_exp, dtype=np.float64).ravel()
            n_dense = max(100, len(t_exp) * 10)
            t_dense = np.linspace(float(t_start), float(t_span[1]), n_dense)
            out = np.sort(
                np.unique(np.concatenate([t_exp, t_dense]).astype(np.float64))
            )
        case LsodaImplicitTEvalMode.EVAL_ODE_FIT:
            if t_points is None:
                raise ValueError("EVAL_ODE_FIT requires t_points.")
            out = np.sort(
                np.unique(np.asarray(t_points, dtype=np.float64).ravel())
            )
        case _:
            raise ValueError(f"Unsupported mode: {mode!r}")
    return out.astype(np.float64, copy=False)


def _warn_once(key: str, message: str) -> None:
    if key in _WARNED_KEYS:
        return
    _WARNED_KEYS.add(key)
    warnings.warn(message, UserWarning, stacklevel=3)


def _load_numbalsoda():
    """Import numbalsoda once per process and cache success or failure."""
    global _NUMBALSODA_IMPORT_ERROR
    global _NUMBALSODA_IMPORTED
    global _NUMBALSODA_MODULE

    if _NUMBALSODA_IMPORTED:
        return _NUMBALSODA_MODULE

    try:
        import numbalsoda as _numbalsoda  # type: ignore

        _NUMBALSODA_MODULE = _numbalsoda
    except Exception as exc:  # pragma: no cover - environment dependent
        _NUMBALSODA_IMPORT_ERROR = exc
        _NUMBALSODA_MODULE = None
    finally:
        _NUMBALSODA_IMPORTED = True

    return _NUMBALSODA_MODULE


def _fallback_rk45(system_rhs, t_span, y0, t_eval, rtol, atol=None):
    kwargs = {}
    if atol is not None:
        kwargs["atol"] = atol
    return solve_ivp(
        system_rhs,
        t_span,
        y0,
        t_eval=t_eval,
        method="RK45",
        rtol=rtol,
        **kwargs,
    )


def _effective_t_span_from_t_eval(t_eval_arr: np.ndarray) -> tuple[float, float]:
    if t_eval_arr.size == 0:
        raise ValueError("t_eval is empty.")
    ts = np.sort(np.unique(t_eval_arr.astype(np.float64)))
    return float(ts[0]), float(ts[-1])


def solve_ode(
    system_rhs,
    t_span,
    y0,
    t_eval=None,
    method="RK45",
    rtol=1e-6,
    atol=None,
    time_dependent=False,
):
    """Integrate ``dy/dt = system_rhs(t, y)`` via SciPy or numbalsoda.

    Non-LSODA methods delegate to :func:`scipy.integrate.solve_ivp`. For
    ``method="LSODA"``, see ``docs/spec_docs/numbalsoda2.md``: effective
    ``t_span`` follows ``t_eval`` when rates are not time-dependent; callers
    must supply ``t_eval`` or build it with :func:`build_t_eval_when_none_for_lsoda`.

    Args:
        system_rhs: Callable ``(t, y)`` returning ``dy/dt`` (SciPy convention).
        t_span: Interval ``(t0, t1)``. For LSODA with ``t_eval`` supplied, the
            solver uses the min/max of ``t_eval`` internally (spec section 5.2).
        y0: Initial state vector.
        t_eval: Times where the solution is sampled. Required for LSODA (non-empty
            unless falling back to RK45 for an empty grid).
        method: ``solve_ivp`` method name, or ``"LSODA"`` to try numbalsoda.
        rtol: Relative integration tolerance.
        atol: Absolute tolerance; ``None`` selects library defaults.
        time_dependent: If True, LSODA is coerced to RK45 (internal callers set
            this from the kinetic model).

    Returns:
        Object compatible with ``scipy.integrate.OdeResult`` (``t``, ``y``,
        ``success``, ``message``, ...).

    Note:
        Direct callers must set ``time_dependent=True`` only when rates depend
        on time (callable dict branch or ``has_time_dependent_rates``); otherwise
        behavior is undefined (see numbalsoda2.md section 14).
    """
    method_name = str(method).upper()

    if method_name != "LSODA":
        kwargs = {}
        if atol is not None:
            kwargs["atol"] = atol
        return solve_ivp(
            system_rhs,
            t_span,
            y0,
            t_eval=t_eval,
            method=method,
            rtol=rtol,
            **kwargs,
        )

    # --- LSODA ---
    if time_dependent:
        _warn_once(
            "lsoda_time_dependent_coerce",
            "method=LSODA in rxnfit does not support time\u2011dependent rate "
            "constants, the solver should be switched to RK45.",
        )
        kwargs = {}
        if atol is not None:
            kwargs["atol"] = atol
        return solve_ivp(
            system_rhs,
            t_span,
            y0,
            t_eval=t_eval,
            method="RK45",
            rtol=rtol,
            **kwargs,
        )

    if t_eval is None:
        raise ValueError(
            "method='LSODA' requires an explicit t_eval array. For automatic "
            "grids use build_t_eval_when_none_for_lsoda(...) before solve_ode."
        )

    t_eval_arr = np.sort(np.unique(np.asarray(t_eval, dtype=np.float64).ravel()))
    if t_eval_arr.size == 0:
        kwargs = {}
        if atol is not None:
            kwargs["atol"] = atol
        return solve_ivp(
            system_rhs,
            t_span,
            y0,
            t_eval=t_eval_arr,
            method="RK45",
            rtol=rtol,
            **kwargs,
        )

    t_eff = _effective_t_span_from_t_eval(t_eval_arr)
    y0_arr = np.asarray(y0, dtype=float)

    mod = _load_numbalsoda()
    if mod is None:
        _warn_once(
            "lsoda_unavailable",
            "LSODA is not available in your environment. Falling back to RK45.",
        )
        return _fallback_rk45(
            system_rhs, t_eff, y0_arr, t_eval_arr, rtol, atol=atol
        )

    rhs_obj = getattr(system_rhs, "numbalsoda_rhs", None)
    rhs_address = getattr(rhs_obj, "address", None) if rhs_obj is not None else None
    if rhs_address is None:
        _warn_once(
            "lsoda_rhs_unsupported",
            "Though the import numbalsoda succeeded, RHS is not supported.",
        )
        return _fallback_rk45(
            system_rhs, t_eff, y0_arr, t_eval_arr, rtol, atol=atol
        )

    from .lsoda_p_vector import build_lsoda_parameter_vector_p

    n_p_attr = getattr(system_rhs, "_rxnfit_lsoda_n_p", 0)
    try:
        n_p = int(n_p_attr)
    except (TypeError, ValueError):
        n_p = 0
    if n_p > 0:
        rd = getattr(system_rhs, "_rxnfit_rate_const_dict", None)
        sk = getattr(system_rhs, "_rxnfit_symbolic_rate_const_keys", None)
        if rd is None or sk is None:
            _warn_once(
                "lsoda_rhs_unsupported",
                "Though the import numbalsoda succeeded, RHS is not supported.",
            )
            return _fallback_rk45(
                system_rhs, t_eff, y0_arr, t_eval_arr, rtol, atol=atol
            )
        p = build_lsoda_parameter_vector_p(rd, sk)
    else:
        p = np.zeros(1, dtype=np.float64)

    lsoda_kw: dict = {"rtol": rtol, "data": p}
    if atol is not None:
        lsoda_kw["atol"] = atol

    try:
        usol, success = mod.lsoda(
            rhs_address,
            y0_arr,
            t_eval_arr,
            **lsoda_kw,
        )
        if not bool(success):
            raise RuntimeError("numbalsoda lsoda() returned unsuccessful status.")
    except Exception:
        _warn_once(
            "lsoda_run_failed",
            "LSODA is not available in your environment. Falling back to RK45.",
        )
        return _fallback_rk45(
            system_rhs, t_eff, y0_arr, t_eval_arr, rtol, atol=atol
        )

    y_arr = np.asarray(usol, dtype=float)
    if y_arr.ndim != 2:
        _warn_once(
            "lsoda_run_failed",
            "LSODA is not available in your environment. Falling back to RK45.",
        )
        return _fallback_rk45(
            system_rhs, t_eff, y0_arr, t_eval_arr, rtol, atol=atol
        )
    y_arr = y_arr.T

    return SimpleNamespace(
        t=t_eval_arr,
        y=y_arr,
        success=True,
        message="",
    )
