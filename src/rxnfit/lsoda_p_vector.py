# Copyright (c) 2025 Mitsuru Ohno
# Use of this source code is governed by a BSD-3-style
# license that can be found in the LICENSE file.  

# 05/04/2026, M. Ohno  

"""Construct the numbalsoda parameter vector ``p`` for LSODA integration.

The ordering rules here are part of the public ABI and must mirror the Numba
RHS compiled in :mod:`rxnfit.numbalsoda_rhs`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def build_lsoda_parameter_vector_p(
    rate_const_values: Mapping[str, float],
    symbolic_rate_const_keys: Sequence[str],
) -> np.ndarray:
    """Assemble the numbalsoda parameter vector ``p`` from rate-constant values.

    Parameter layout for numbalsoda RHS (must match ``numbalsoda_rhs`` cfunc):

    - Let ``n_fitted = len(symbolic_rate_const_keys)``.
    - ``p[0] … p[n_fitted-1]``: values for ``symbolic_rate_const_keys`` in that
      exact order (same order as ``create_system_rhs`` / fitters).
    - ``p[n_fitted] … p[n_p-1]``: every other key present in ``rate_const_values``,
      excluding those in ``symbolic_rate_const_keys``, ordered by **lexicographic**
      sort of the key strings (not natural sort).
    - ``n_p = len(p)`` is fixed for a given RHS / build.

    Example: ``symbolic_rate_const_keys = ("k2", "k1")`` and the only remaining
    keys in ``rate_const_values`` are ``k3`` and ``k4`` →
    ``p = [v_k2, v_k1, v_k3, v_k4]``.

    The concentration vector ``u`` (species order / ``function_names``) does **not**
    need to align with the ordering of entries in ``p``.

    Args:
        rate_const_values: All rate constants needed for the current
            integration (numeric).
        symbolic_rate_const_keys: Fitted (or explicit) keys in the order used
            by ``create_system_rhs`` for extra ODE arguments.

    Returns:
        One-dimensional ``float64`` ndarray of length ``n_p``.
    """
    sym_set = set(symbolic_rate_const_keys)
    first = [float(rate_const_values[k]) for k in symbolic_rate_const_keys]
    rest_keys = sorted(k for k in rate_const_values if k not in sym_set)
    rest = [float(rate_const_values[k]) for k in rest_keys]
    return np.asarray(first + rest, dtype=np.float64)
