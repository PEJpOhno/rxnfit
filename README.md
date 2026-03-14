
## Overviews
"rxnfit" is a Python module that builds ordinary differential equations (ODEs) for reaction kinetics from elementary reaction formulas and rate constants given in CSV format, and computes concentration time courses numerically.

- When all rate constants are known, the time evolution of each species’ concentration is obtained by **numerical integration** of the ODEs (e.g. with `scipy.integrate.solve_ivp`).
- When some rate constants are unknown, they can be **estimated by fitting** the ODE solution to experimental concentration time-course data (e.g. with `scipy.optimize.minimize`).
- A rate constant may be given as an **expression** in other rate constants (e.g. k₂ = 2·k₁), or as a **function of time** *t* via an override CSV.

**Documentation :** https://pejpohno.github.io/rxnfit/  
**Tutorial :** in preparation  
**How to cite :** Ohno, M. rxnfit. GitHub. https://github.com/PEJpOhno/rxnfit (2023).  

## Current version and requirements
- current version = 0.3.0
- pyhon >=3.12

[dependencies]
- NumPy
- pandas
- SciPy
- SymPy
- Matplotlib  
- Optuna

## Copyright and license
Copyright (c) 2023-2026 Mitsuru Ohno
Released under the BSD-3 license, license that can be found in the LICENSE file.

## Installation
Create and activate a Python virtual environment, then  

```sh
$pip install rxnfit
```

If you want to give it a quick try, just copy or move the example directory from the cloned repo to wherever it's convenient. Then, activate your virtual environment, start Jupyter Notebook, and open one of the sample scripts inside "examples" directry to test it out.

## Usage
### Prepare csv file
1. Describe reaction formula in csv format. THe first row shoud be header. The first column and the second column should be reaction ID (named RID) and the rate constant (named k). Other column names can be omitted.
2. The reactants are putted in alternately with the coeficient and the chemical species. Each reaction should be described in one row.
3. To separate the reactants and the products, two ">" are setted with two columns. As with reaction SMARTS, you may write reaction conditions between the two ">"; in the current version of rxnfit these are unused.
4. The products are putted in alternately with the coeficient and the chemical species.
Note the reaction formula should be inputted by the left filling. To describe chemical species, it is better to avoid to use arithmetic symbols such as "+", "-" and "*" (e.g. use "**a**"(nion) in place of "-"). The coefficient '1' can be substituted for blank.  
5. Each elementary reaction’s rate constant must be specified in the format ‘k + RID’. If left empty, the system will automatically assign ‘k + RID’.  

The rate constant k_m can also be defined as a function of another elementary reaction rate constant k_n, or as a function of time t.  
The example of a csv format is as follows.  Attached "examples/sample_data/sample_rxn_ref1.csv" and "examples/sample_data/refX/sample_rxn_refX.csv" are also available for demonstration.

example of a csv format

    RID, k,,,,,,,,,,
    1,0.054,,AcOEt,,OHa1,>,>,,AcOa1,,EtOH
    2,1.4,,AcOiPr,,OHa1,>,>,,AcOa1,,iPrOH
    3,0.031,,EGOAc2,2,OHa1,>,>,2,AcOa1,,EG

## Modules and classes

The following items are those prominently used in the example notebooks.

### rxn_reader
Reads reaction CSV and builds symbolic ODE system.

| item | name | description |
|------|------|-------------|
| class | RxnToODE | Builds differential-type rate equations (strings) and SymPy ODE system from reaction CSV. Attributes: rate_consts_dict, sympy_symbol_dict, sys_odes_dict, function_names. |
| function | get_reactions | Parses reaction CSV and returns a list of reaction data (RID, k, reactants, products, conditions). |
| function | get_unique_species | Returns the list of unique species names from the reaction list. |
| function | rate_constants | Returns rate constant key and value (float or expression string) per reaction. |

### build_ode
Builds numerical ODE RHS from symbolic system; supports rate-constant overrides and time-dependent k(t).

| item | name | description |
|------|------|-------------|
| class | RxnODEbuild | Extends RxnToODE. Builds a callable ODE RHS for scipy.integrate.solve_ivp from reaction CSV. Optional rate_const_overrides (dict or CSV path). Used in examples as the builder for RxnODEsolver and ExpDataFitSci. get_ode_info() prints species list and count so you can set SolverConfig.y0 in the correct order (same as function_names). |

### expdata_reader
Loads and aligns experimental time-course data.

| item | name | description |
|------|------|-------------|
| function | expdata_read | Reads a list of DataFrames and returns, per dataset, (t_list, C_exp_list) for use in fitting or plotting. |
| function | get_y0_from_expdata | Returns initial concentrations per dataset in the order of function_names (species order). |

### expdata_fit_sci
Fits symbolic rate constants to experimental data (scipy.optimize.minimize).

| item | name | description |
|------|------|-------------|
| class | ExpDataFitSci | Fits symbolic rate constants to experimental time-course data (multi-dataset). run_fit(p0, ...) runs optimization and returns (result, param_info, fit_metrics); result has .fun (RSS), .tss, .r2; fit_metrics is a dict with keys 'rss', 'tss', 'r2'. plot_fitted_solution(expdata_df, ...) plots fitted time-courses with per-dataset y0 after run_fit. |

### solv_ode
Numerical integration and plotting of ODE solutions.

| item | name | description |
|------|------|-------------|
| class | SolverConfig | Dataclass holding integration settings: y0, t_span, t_eval, method, rtol. Optionally rate_const_values and symbolic_rate_const_keys for time-dependent or variable rate constants. |
| class | RxnODEsolver | Integrates ODEs with a builder and SolverConfig. solve_system() runs the integration. solution_plot() plots time-courses (optionally with experimental overlay). to_dataframe_list() returns a list of DataFrames (one per dataset). eval_fit_metrics(expdata_df, ...) returns a dict with 'rss', 'tss', 'r2'. |

### p0_opt_fit
Optimizes initial parameter values (p0) for rate constants using Optuna, then fits with ExpDataFitSci.

| item | name | description |
|------|------|-------------|
| class | P0OptFit | Optimize initial parameter values (p0) for rate constants using Optuna, then run ExpDataFitSci. |

## References  

- **Python**: Python Software Foundation. (2024). *Python* (Version 3.13) [Computer software]. https://www.python.org/  
- **NumPy**: Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). Array programming with NumPy. *Nature*, 585(7825), 357–362. https://doi.org/10.1038/s41586-020-2649-2
- **pandas**: McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proceedings of the 9th Python in Science Conference*, 56–61. https://doi.org/10.25080/Majora-92bf1922-00a
- **SciPy**: Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. *Nature Methods*, 17(3), 261–272. https://doi.org/10.1038/s41592-019-0686-2
- **SymPy**: Meurer, A., Smith, C. P., Paprocki, M., et al. (2017). SymPy: symbolic computing in Python. *PeerJ Computer Science*, 3, e103. https://doi.org/10.7717/peerj-cs.103
- **Matplotlib**: Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90–95. https://doi.org/10.1109/MCSE.2007.55
- **Optuna**: Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A next-generation hyperparameter optimization framework. *Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD '19)*, 2623–2631. https://doi.org/10.1145/3292500.3330701

## Acknowledgements
This module and its accompanying documentation were developed with the support of Cursor’s AI-assisted tools.  
