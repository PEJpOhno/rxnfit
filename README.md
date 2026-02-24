
## 1. Overviews  
"rxnfit" is a Python module to set up theoretical reaction kinetics equations from reaction formulas and the rate constats provided as csv file.  

## 2. Current version and requirements  
- current version = 0.0.2   
- pyhon >=3.10  

[dependencies]  
- NumPy   
- pandas  
- SciPy  
- SymPy  
- Matplotlib  
- Optuna  

## 3. Copyright and license  
Copyright (c) 2025 Mitsuru Ohno  
Released under the BSD-3 license, license that can be found in the LICENSE file.  

## 4. Installation  
Download the directry "src" to a suitable directry on your computer.  
For quick start, put the sample script "rxnfit_sample_script.ipynb"  on the same directry described above. In this case, Jupyter Notebook is required.  

## 5. Usage  
### 5-1. Prepare csv file 
1. Describe reaction formula in csv format. THe first row shoud be header. The first column and the second column should be reaction ID (named RID) and the rate constant (named k). Other column names can be omitted.  
2. The reactants are putted in alternately with the coeficient and the chemical species. Each reaction should be described in one row.  
3. To separate the reactants and the products, two ">" are setted with two columns. 
4. The products are putted in alternately with the coeficient and the chemical species.  
Note the reaction formula should be inputted by the left filling. To describe chemical species, it is better to avoid to use arithmetic symbols such as "+", "-" and "*" (e.g. use "**a**"(nion) in place of "-"). The coefficient '1'　can be substituted for blank.  
The example of a csv format is as follows.  Attached "sample_data.csv" [1] and "sample_data2.csv" [2] are also available for demonstration.  

example of a csv format   

    RID, k,,,,,,,,,,  
    1,0.054,,AcOEt,,OHa1,>,>,,AcOa1,,EtOH  
    2,1.4,,AcOiPr,,OHa1,>,>,,AcOa1,,iPrOH  
    3,0.031,,EGOAc2,2,OHa1,>,>,2,AcOa1,,EG  

## Modules and classes   

| module | class | function |
|---|---|---|
| rxn_reader.py | RxnToODE | Generate differential-type rate equations as strings from reaction equations in CSV format. Then, create ODE system expressions for symbolic manipulation. |
| build_ode.py | RxnODEbuild | Convert symbolic  differential equations (ODEs; sympy) to numerical one (scipy). |
 solve_ode.py | solves numerical ODEs and plot the result. |






## References  
Future worls: 初期濃度を変化させる事例  
https://doi.org/10.1002/ange.201609757  
https://doi.org/10.1039/C8SC04698K  
https://doi.org/10.1039/D4DD00111G  


## THIRD-PARTY LICENSES

This software includes the following third-party libraries:

---

Library: Python
Website: https://www.python.org/
License: Python Software Foundation License Version 2
Copyright © 2001–2025 Python Software Foundation

---

Library: NumPy
Website: https://numpy.org/
License: BSD 3-Clause License
Copyright (c) 2005–2025, NumPy Developers

