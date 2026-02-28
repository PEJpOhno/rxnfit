
## 1. Overviews
"rxnfit" is a Python module that builds ordinary differential equations (ODEs) for reaction kinetics from elementary reaction formulas and rate constants given in CSV format, and computes concentration time courses numerically.

- When all rate constants are known, the time evolution of each species’ concentration is obtained by **numerical integration** of the ODEs (e.g. with `scipy.integrate.solve_ivp`).
- When some rate constants are unknown, they can be **estimated by fitting** the ODE solution to experimental concentration time-course data (e.g. with `scipy.optimize.minimize`).
- A rate constant may be given as an **expression** in other rate constants (e.g. k₂ = 2·k₁), or as a **function of time** *t* via an override CSV.


## 2. Current version and requirements
- current version = 0.1.0
- pyhon >=3.13

[dependencies]
- NumPy
- pandas
- SciPy
- SymPy
- Matplotlib  

## 3. Copyright and license
Copyright (c) 2025 Mitsuru Ohno
Released under the BSD-3 license, license that can be found in the LICENSE file.

## 4. Installation
1. Clome repository on your computer.
https://github.com/PEJpOhno/rxnfit.git

2. Create and activate a Python virtual environment.

```sh
pip install ["PATH TO YOUR CLONED REPOSUTORY"]
```

If you want to give it a quick try, just copy or move the example directory from the cloned repo to wherever it's convenient. Then, activate your virtual environment, start Jupyter Notebook, and open one of the sample scripts inside the example folder to test it out.

## 5. Usage
### 5-1. Prepare csv file
1. Describe reaction formula in csv format. THe first row shoud be header. The first column and the second column should be reaction ID (named RID) and the rate constant (named k). Other column names can be omitted.
2. The reactants are putted in alternately with the coeficient and the chemical species. Each reaction should be described in one row.
3. To separate the reactants and the products, two ">" are setted with two columns. As with reaction SMARTS, you may write reaction conditions between the two ">"; in the current version of rxnfit these are unused.
4. The products are putted in alternately with the coeficient and the chemical species.
Note the reaction formula should be inputted by the left filling. To describe chemical species, it is better to avoid to use arithmetic symbols such as "+", "-" and "*" (e.g. use "**a**"(nion) in place of "-"). The coefficient '1' can be substituted for blank.  

The rate constant k_m can also be defined as a function of another elementary reaction rate constant k_n, or as a function of time t.  
The example of a csv format is as follows.  Attached "examples/sample_data/sample_rxn_ref1.csv" and "examples/sample_data/refX/sample_rxn_refX.csv" are also available for demonstration.

example of a csv format

    RID, k,,,,,,,,,,
    1,0.054,,AcOEt,,OHa1,>,>,,AcOa1,,EtOH
    2,1.4,,AcOiPr,,OHa1,>,>,,AcOa1,,iPrOH
    3,0.031,,EGOAc2,2,OHa1,>,>,2,AcOa1,,EG

## 6. Modules and classes

The following items are those prominently used in the example notebooks.

### 6-1. rxn_reader
Reads reaction CSV and builds symbolic ODE system.

| item | name | description |
|------|------|-------------|
| class | RxnToODE | Builds differential-type rate equations (strings) from reaction CSV, then ODE system expressions (SymPy). Attributes: rate_consts_dict, sympy_symbol_dict, sys_odes_dict, function_names. |
| function | get_reactions | Parse reaction CSV into list of reaction data (RID, k, reactants, products, conditions). |
| function | get_unique_species | Extract unique species names from reaction list. |
| function | rate_constants | Extract rate constant key and value (float or expression string) per reaction. |

### 6-2. build_ode
Builds numerical ODE RHS from symbolic system; supports rate-constant overrides and time-dependent k(t).

| item | name | description |
|------|------|-------------|
| class | RxnODEbuild | Extends RxnToODE. Builds callable ODE RHS for scipy.integrate.solve_ivp. Optional rate_const_overrides (dict or CSV path) and rate_const_overrides_encoding. Methods: create_ode_system(), create_ode_system_with_rate_consts(), get_ode_system(), get_ode_info(). |
| function | create_system_rhs | Build (t, y) RHS for solve_ivp from ODE functions dict. Optional rate_const_values (dict or callable(t)) and symbolic_rate_const_keys for variable or time-dependent rate constants. |

### 6-3. expdata_reader
Loads and aligns experimental time-course data.

| item | name | description |
|------|------|-------------|
| function | expdata_read | Read list of DataFrames into list of (t_list, C_exp_list) per dataset. |
| function | get_y0_from_expdata | Initial concentrations per dataset in function_names order. |

### 6-4. expdata_fit_sci
Fits symbolic rate constants to experimental data (scipy.optimize.minimize).

| item | name | description |
|------|------|-------------|
| class | ExpDataFitSci | Multi-dataset fitting. run_fit(p0, ...), get_solver_config_args(dataset_index), get_fitted_rate_const_dict(result). When model has k(t), get_solver_config_args() adds rate_const_values (callable) and symbolic_rate_const_keys after run_fit(). |

### 6-5. solv_ode
Numerical integration and plotting of ODE solutions.

| item | name | description |
|------|------|-------------|
| class | SolverConfig | Dataclass: y0, t_span, t_eval, method, rtol. Optional: rate_const_values (dict or callable(t)), symbolic_rate_const_keys (both or neither). |
| class | RxnODEsolver | Integrates ODE with builder and config. solve_system(), to_dataframe(), rsq(), solution_plot(). When config has rate_const_values and symbolic_rate_const_keys, uses rate-constants ODE path. |

## 7. References  

- **Python**: Python Software Foundation. (2024). *Python* (Version 3.13) [Computer software]. https://www.python.org/  
- **NumPy**: Harris, C. R., Millman, K. J., van der Walt, S. J., et al. (2020). Array programming with NumPy. *Nature*, 585(7825), 357–362. https://doi.org/10.1038/s41586-020-2649-2
- **pandas**: McKinney, W. (2010). Data Structures for Statistical Computing in Python. *Proceedings of the 9th Python in Science Conference*, 56–61. https://doi.org/10.25080/Majora-92bf1922-00a
- **SciPy**: Virtanen, P., Gommers, R., Oliphant, T. E., et al. (2020). SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. *Nature Methods*, 17(3), 261–272. https://doi.org/10.1038/s41592-019-0686-2
- **SymPy**: Meurer, A., Smith, C. P., Paprocki, M., et al. (2017). SymPy: symbolic computing in Python. *PeerJ Computer Science*, 3, e103. https://doi.org/10.7717/peerj-cs.103
- **Matplotlib**: Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. *Computing in Science & Engineering*, 9(3), 90–95. https://doi.org/10.1109/MCSE.2007.55

