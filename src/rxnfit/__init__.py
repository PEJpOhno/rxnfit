"""Reaction-kinetics ODE construction, integration, and fitting.

This package builds ordinary differential equation (ODE) models from
elementary reaction CSVs and integrates or fits them numerically.
:class:`rxnfit.solv_ode.SolverConfig` and :class:`rxnfit.solv_ode.RxnODEsolver`
are imported explicitly below; symbols from :mod:`rxnfit.rxn_reader` and
:mod:`rxnfit.build_ode` populate the default namespace via star imports.
"""

from ._version import __version__
from .rxn_reader import *
from .build_ode import *
from .solv_ode import SolverConfig, RxnODEsolver

# from src.rxnfit.rxn_reader import *
# from src.rxnfit.build_ode import *
