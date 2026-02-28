# Copyright (c) 2025 Mitsuru Ohno
# Use of this source code is governed by a BSD-3-style
# license that can be found in the LICENSE file.

"""Rate constant evaluator for time-dependent k(t).

"ft" in the module name stands for f(t), i.e. rate constants as functions of time.
Builds a callable (t, params) -> dict of rate constant values for use in ODE
integration and fitting when rate constants depend on time t and/or other parameters.
"""

from sympy import Symbol
from sympy.core.symbol import Symbol as SympySymbol


def has_time_dependent_rates(rate_consts_dict):
    """Return True if any rate constant expression contains the symbol t.

    Args:
        rate_consts_dict (dict): Rate constant key -> float, Symbol, or Expr.

    Returns:
        bool: True if any value has 't' in its free_symbols.
    """
    t_name = 't'
    for val in rate_consts_dict.values():
        if isinstance(val, (int, float)):
            continue
        free = getattr(val, 'free_symbols', None) or set()
        if any(str(s) == t_name for s in free):
            return True
    return False


def build_evaluator(rate_consts_dict, symbolic_rate_const_keys, fixed_rate_consts=None):
    """Build a callable (t, params) -> dict of rate constant values.

    Given rate_consts_dict (keys -> float, Symbol, or SymPy Expr possibly
    containing t and/or other symbols), returns a function that, for a given
    t and a list of parameter values (in symbolic_rate_const_keys order),
    returns a dict mapping every rate constant key to a numeric value.
    Used when rate constants are time-dependent or when building SolverConfig
    for re-integration after fitting.

    Args:
        rate_consts_dict (dict): Rate constant key -> float, sympy.Symbol,
            or SymPy Expr (e.g. a*exp(t), k1/4). May contain symbol t.
        symbolic_rate_const_keys (list): Order of free parameter names
            (e.g. ['k1', 'a', 'b']). Must not include 't'.
        fixed_rate_consts (dict, optional): Key -> float for constants already
            resolved. If None, only numeric entries in rate_consts_dict are
            treated as fixed.

    Returns:
        callable: evaluator(t, params) where t is float and params is a
            sequence of floats in symbolic_rate_const_keys order. Returns
            dict mapping each key in rate_consts_dict to float.
    """
    if fixed_rate_consts is None:
        fixed_rate_consts = {
            k: float(v) for k, v in rate_consts_dict.items()
            if isinstance(v, (int, float))
        }

    t_sym = Symbol('t')
    param_symbols = {name: Symbol(name) for name in symbolic_rate_const_keys}

    def evaluator(t_val, params):
        if len(params) != len(symbolic_rate_const_keys):
            raise ValueError(
                f"params length ({len(params)}) must match "
                f"symbolic_rate_const_keys length ({len(symbolic_rate_const_keys)})."
            )
        subs = {t_sym: t_val}
        for i, name in enumerate(symbolic_rate_const_keys):
            subs[param_symbols[name]] = float(params[i])
        result = dict(fixed_rate_consts)
        for k, v in result.items():
            subs[Symbol(k)] = v

        changed = True
        while changed:
            changed = False
            for key, val in rate_consts_dict.items():
                if key in result:
                    continue
                if isinstance(val, (int, float)):
                    result[key] = float(val)
                    subs[Symbol(key)] = result[key]
                    changed = True
                    continue
                if isinstance(val, (Symbol, SympySymbol)):
                    if val.name in result:
                        result[key] = result[val.name]
                    elif val.name in symbolic_rate_const_keys:
                        idx = symbolic_rate_const_keys.index(val.name)
                        result[key] = float(params[idx])
                    else:
                        continue
                    subs[Symbol(key)] = result[key]
                    changed = True
                    continue
                free_syms = getattr(val, 'free_symbols', None) or set()
                if not free_syms:
                    try:
                        result[key] = float(val)
                        subs[Symbol(key)] = result[key]
                        changed = True
                    except (TypeError, ValueError):
                        pass
                    continue
                try:
                    sub_val = val.subs(subs)
                    remaining = getattr(sub_val, 'free_symbols', None) or set()
                    if not remaining:
                        result[key] = float(sub_val)
                        subs[Symbol(key)] = result[key]
                        changed = True
                except (TypeError, ValueError):
                    pass

        for key, val in rate_consts_dict.items():
            if key in result:
                continue
            try:
                sub_val = val.subs(subs)
                result[key] = float(sub_val)
            except (TypeError, ValueError):
                result[key] = 0.0
        return result

    return evaluator
