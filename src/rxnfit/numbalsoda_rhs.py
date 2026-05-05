# Copyright (c) 2025 Mitsuru Ohno
# Use of this source code is governed by a BSD-3-style
# license that can be found in the LICENSE file.  
# 01/03/2026, M. Ohno  

"""Compile SymPy-derived ODE right-hand sides into numbalsoda LSODA kernels.

Generates Numba ``cfunc`` callables with the LSODA signature
``rhs(t, u, du, p)``. The parameter layout in ``p`` must stay synchronized with
:func:`rxnfit.lsoda_p_vector.build_lsoda_parameter_vector_p`.

Note:
    Do not import :mod:`rxnfit.build_ode` from this module; ``build_ode`` loads
    it lazily to keep the dependency graph acyclic.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple


def compile_numbalsoda_rhs_for_builder(
    builder: Any,
    rate_const_values: Any,
    symbolic_rate_const_keys: Any,
) -> Optional[Tuple[Any, int]]:
    """Return ``(numba cfunc, n_p)`` for numbalsoda, or ``None`` if unavailable.

    Skips compilation for time-dependent systems (callable ``rate_const_values``
    or ``t`` in rate expressions) or when optional deps fail.

    Note:
        SymPy 1.13+ no longer registers ``lambdify(..., modules="numba")``. We use
        ``modules="numpy"`` then compile with ``numba.njit(...)`` (see SymPy
        ``test_issue_20070``).
    """
    try:
        from sympy import lambdify
        from numba import cfunc, carray, njit
        from numbalsoda import lsoda_sig
    except Exception:
        return None

    from .rate_const_ft_eval import has_time_dependent_rates

    if callable(rate_const_values):
        return None
    if has_time_dependent_rates(builder.rate_consts_dict):
        return None

    try:
        spec = builder.get_numbalsoda_sympy_spec(
            rate_const_values,
            symbolic_rate_const_keys,
        )
    except Exception:
        return None

    expr_tuple = spec["expr_tuple"]
    lambdify_args = spec["lambdify_args"]
    n_state = int(spec["n_state"])
    n_p = int(spec["n_p"])

    try:
        np_rhs = lambdify(lambdify_args, expr_tuple, modules="numpy")
        vec_fn = njit(np_rhs)
    except Exception:
        return None

    probe_args = [0.0] + [1.0] * n_state + [0.25] * n_p
    try:
        tpl = vec_fn(*probe_args[: len(lambdify_args)])
    except Exception:
        return None

    import numpy as np

    use_flat = isinstance(tpl, np.ndarray)

    u_args = ", ".join(f"uu[{i}]" for i in range(n_state))
    if n_p > 0:
        p_args = ", ".join(f"pp[{i}]" for i in range(n_p))
        call_expr = f"_vec_fn(t, {u_args}, {p_args})"
        prelude = (
            f"    uu = carray(u, ({n_state},))\n"
            f"    pp = carray(p, ({n_p},))\n"
            f"    tpl = {call_expr}\n"
        )
    else:
        call_expr = f"_vec_fn(t, {u_args})"
        prelude = f"    uu = carray(u, ({n_state},))\n    tpl = {call_expr}\n"

    if use_flat:
        loop = f"""    for _i in range({n_state}):
        du[_i] = tpl.flat[_i]
"""
    else:
        loop = f"""    for _i in range({n_state}):
        du[_i] = tpl[_i]
"""

    src = (
        "from numba import cfunc, carray\n"
        "from numbalsoda import lsoda_sig\n"
        "@cfunc(lsoda_sig)\n"
        "def rhs(t, u, du, p):\n"
        f"{prelude}"
        f"{loop}"
    )
    ns: dict[str, Any] = {
        "cfunc": cfunc,
        "carray": carray,
        "lsoda_sig": lsoda_sig,
        "_vec_fn": vec_fn,
    }
    try:
        exec(src, ns, ns)
    except Exception:
        return None
    rhs = ns.get("rhs")
    if rhs is None or not hasattr(rhs, "address"):
        return None
    return rhs, n_p
