.. rxnfit documentation master file, created by
   sphinx-quickstart on Tue Mar 10 20:22:10 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

rxnfit documentation
====================

Current version: |version|, release: |release|

Overviews
--------------------------------
"rxnfit" is a Python module that builds ordinary differential equations (ODEs) for reaction kinetics from elementary reaction formulas and rate constants given in CSV format, and computes concentration time courses numerically.

- When all rate constants are known, the time evolution of each species’ concentration is obtained by **numerical integration** of the ODEs (e.g. with `scipy.integrate.solve_ivp`).
- When some rate constants are unknown, they can be **estimated by fitting** the ODE solution to experimental concentration time-course data (e.g. with `scipy.optimize.minimize`).
- A rate constant may be given as an **expression** in other rate constants (e.g. k₂ = 2·k₁), or as a **function of time** *t* via an override CSV.

**Project Link :** https://github.com/PEJpOhno/rxnfit\

**Tutorial :** in preparation

**How to cite :** Ohno, M. rxnfit. GitHub. https://github.com/PEJpOhno/rxnfit (2023).  

This module and its accompanying documentation were developed with the support of Cursor’s AI-assisted tools.  

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   rxnfit.rxn_reader
   rxnfit.build_ode
   rxnfit.rate_const_ft_eval
   rxnfit.solv_ode
   rxnfit.expdata_reader
   rxnfit.expdata_fit_sci
   rxnfit.p0_opt_fit
   rxnfit.fit_metrics
   rxnfit.plot_results

   
   

