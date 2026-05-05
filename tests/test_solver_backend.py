"""Tests for solver_backend (numbalsoda2.md behaviour)."""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from rxnfit.solver_backend import (
    LsodaImplicitTEvalMode,
    build_t_eval_when_none_for_lsoda,
    solve_ode,
)


def test_build_t_eval_solve_system():
    t = build_t_eval_when_none_for_lsoda(
        LsodaImplicitTEvalMode.SOLVE_SYSTEM,
        t_span=(0.0, 1.0),
    )
    assert t.shape[0] >= 100
    assert np.allclose(t[0], 0.0) and np.allclose(t[-1], 1.0)
    assert np.all(np.diff(t) > 0)


def test_build_t_eval_compute_rss():
    raw = np.array([0.3, 0.1, 0.3, 0.2])
    t = build_t_eval_when_none_for_lsoda(
        LsodaImplicitTEvalMode.COMPUTE_RSS,
        t_exp_col=raw,
    )
    assert list(t) == [0.1, 0.2, 0.3]


def test_build_t_eval_integrate_datasets():
    t = build_t_eval_when_none_for_lsoda(
        LsodaImplicitTEvalMode.INTEGRATE_DATASETS,
        t_span=(0.0, 10.0),
        t_start=0.5,
        t_exp=np.array([1.0, 2.0]),
    )
    assert t[0] >= 0.5
    assert np.all(np.diff(t) > 0)
    assert 1.0 in t and 2.0 in t


def test_solve_ode_rk45_unchanged():
    def rhs(t, y):
        return [-y[0]]

    sol = solve_ivp(rhs, (0.0, 1.0), [1.0], method="RK45", rtol=1e-6)
    sol2 = solve_ode(rhs, (0.0, 1.0), [1.0], method="RK45", rtol=1e-6)
    assert sol2.success
    assert np.allclose(sol.t, sol2.t)
    assert np.allclose(sol.y, sol2.y)


def test_lsoda_time_dependent_coerce():
    def rhs(t, y):
        return [-y[0]]

    with pytest.warns(UserWarning, match="LSODA"):
        sol = solve_ode(
            rhs,
            (0.0, 1.0),
            [1.0],
            t_eval=np.linspace(0, 1, 11),
            method="LSODA",
            rtol=1e-6,
            time_dependent=True,
        )
    assert sol.success


@pytest.mark.skipif(
    importlib.util.find_spec("numbalsoda") is None,
    reason="numbalsoda not installed",
)
def test_lsoda_with_numbalsoda_rhs_smoke():
    from numba import cfunc, carray
    from numbalsoda import lsoda_sig

    @cfunc(lsoda_sig)
    def rhs_cfunc(t, u, du, p):
        uu = carray(u, (1,))
        du[0] = -uu[0]

    def system_rhs(t, y):
        return (-y[0],)

    system_rhs.numbalsoda_rhs = rhs_cfunc
    system_rhs._rxnfit_lsoda_n_p = 0

    t_eval = np.linspace(0.0, 1.0, 20)
    sol = solve_ode(
        system_rhs,
        (0.0, 2.0),
        [1.0],
        t_eval=t_eval,
        method="LSODA",
        rtol=1e-6,
        time_dependent=False,
    )
    assert sol.success
    assert sol.y.shape[0] == 1
    assert np.allclose(sol.y[0, 0], 1.0, atol=0.01)
    assert np.allclose(sol.y[0, -1], np.exp(-1.0), rtol=0.05)
