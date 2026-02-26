# Copyright (c) 2025 Mitsuru Ohno
# Use of this source code is governed by a BSD-3-style
# license that can be found in the LICENSE file.

# 08/30/2025, M. Ohno

import csv
import io
from collections import defaultdict
from urllib.request import urlopen

from sympy import Symbol, symbols, Function, Eq, Derivative
from sympy.parsing.sympy_parser import parse_expr


def get_reactions(file_path, encoding=None):
    """Imports elementary reaction formulas from a CSV file into a Python list.

    The reaction formulas are written by reaction SMILES style.
    The CSV file is expected to have a header row and each subsequent row
    should represent a reaction with columns for Reaction ID, rate constant
    (k), followed by pairs of columns for reactant coefficient and reactant
    name, separated by ">", then conditions, separated by ">", and finally
    pairs of columns for product coefficient and product name. The "+"
    symbols in the CSV are ignored.

    file_path may be a local path or a URL (http:// or https://). If it is
    a URL, the content is fetched and parsed in memory.

    Rate constant column (k):
        - Empty: replaced by 'k' + RID (e.g. 'k2' for RID 2).
        - Numeric string (e.g. "0.04"): converted to float.
        - Other string: kept as-is (expression like "k1*2" or symbol name).

    Args:
        file_path (str): Path or URL to the CSV file containing the reaction
            formulas. If it starts with http:// or https://, the content
            is fetched from the URL.
        encoding (str, optional): The encoding of the CSV file. Defaults to
            'utf-8' if None.

    Returns:
        list: A list of lists, where each inner list represents a reaction
            and contains:
            - A list [reaction_id, rate_constant] where rate_constant is
              float, or str (e.g. 'k2' for empty, or expression like 'k1*2').
            - A list of [coefficient, name] for each reactant.
            - A list of [coefficient, name] for each product.
            - A list of condition strings.
    """
    if encoding is None:
        encoding = 'utf-8'
    reaction_equations = []  # 素反応を格納するリスト

    if file_path.strip().lower().startswith(('http://', 'https://')):
        with urlopen(file_path) as resp:
            text = resp.read().decode(encoding)
        file = io.StringIO(text)
    else:
        file = open(file_path, mode='r', encoding=encoding)

    try:
        reader = csv.reader(file)
        next(reader)  # ヘッダ行をスキップ

        for row in reader:
            # "+"を除外
            filtered_row = [e for e in row if e != "+" and e != "."]

            # すべての要素を文字列に変換
            filtered_row = [str(e) for e in filtered_row]

            # 分子の係数の"1"を""に置換
            filtered_row = [
                e if index < 2 or e != "1" else ""
                for index, e in enumerate(filtered_row)
            ]

            # 最初と二番目の">"の位置を取得
            i = filtered_row.index(">") if ">" in filtered_row else -1
            j = (
                filtered_row.index(">", i + 1)
                if i != -1 and ">" in filtered_row[i + 1:]
                else -1
            )

            # リストを作成
            if len(filtered_row) > 1:
                # reaction IDおよびkのリスト
                ID_k = filtered_row[:2]
                k_cell = (ID_k[1] or '').strip()
                if not k_cell:
                    ID_k[1] = 'k' + ID_k[0]
                else:
                    try:
                        ID_k[1] = float(k_cell)
                    except ValueError:
                        # 数値化できない文字列はそのまま（式またはシンボル名）
                        ID_k[1] = k_cell

                # reactantの分子数と分子のリスト
                reactants = (
                    [filtered_row[k:k + 2] for k in range(2, i, 2)]
                    if i > 0
                    else []
                )
                # 左づめ表記なっていない場合に対応
                reactants = [
                    sublist for sublist in reactants
                    if any(sublist) and
                    not any('>' in element for element in sublist)
                ]

                # 反応条件のリスト
                conditions = (
                    [filtered_row[k] for k in range(i + 1, j)]
                )

                # productの分子数と分子のリスト
                products = (
                    [filtered_row[k:k + 2]
                     for k in range(j + 1, len(filtered_row), 2)]
                    if j > 0
                    else []
                )
                # 反応条件の有無により行の長さが変わるため、
                # 全ての要素が空の化学種を削除
                products = [sublist for sublist in products if any(sublist)]

                # 反応式をネストしたリストとして構築
                reaction_equations.append(
                    [ID_k, reactants, products, conditions]
                )
    finally:
        if not isinstance(file, io.StringIO):
            file.close()

    return reaction_equations


def get_unique_species(reaction_equations):
    """Extract unique chemical species from elementary reactions.

    Args:
        reaction_equations (list): A list of lists representing elementary
            reactions.

    Returns:
        list: A list of unique chemical species sorted by their appearance
            order.
    """
    species = []
    flatten_species = []
    for item in reaction_equations:
        species.append([e[1] for e in item[1]])  # the list of the reactants
        species.append([e[1] for e in item[2]])  # the list of the products
    for elements in species:
        flatten_species.extend(elements)
    unique_species = sorted(set(flatten_species),
                            key=flatten_species.index)
    return list(unique_species)


def to_chempy_style(reaction):
    """Convert a reaction equation list to a ChemPy-style dictionary.

    Args:
        reaction (list): A list representing a reaction equation in the
            format [['ID', rate_constant], [[coeff, reactant1], ...],
            [[coeff, product1], ...]].

    Returns:
        list: A list containing the reaction ID and rate constant, followed
            by two dictionaries representing the reactants and products with
            their coefficients.
    """
    e_dict = [
        reaction[0],
        dict(
            map(
                lambda x: (x[1], float(x[0]) if x[0] != "" else 1),
                reaction[1]
            )
        ),
        dict(
            map(
                lambda x: (x[1], float(x[0]) if x[0] != "" else 1),
                reaction[2]
            )
        )
    ]
    return e_dict


def reactant_consumption(reaction_equation):
    """Generate rate law equations for reactant consumption.

    The rate law equations and coefficients are determined as follows.
    For a reaction aA + bB -> cC + dD:
    -d[A]/dt = k * A^a * B^b
    -(1/a)d[A]/dt = -(1/b)d[B]/dt = (1/c)d[C]/dt = (1/d)d[D]/dt
    That is,
    -d[A]/dt = -(a/b)d[B]/dt = (a/c)d[C]/dt = (a/d)d[D]/dt

    Args:
        reaction_equation (list): A list representing a reaction equation
            in the format [['ID', rate_constant], [[coeff, reactant1], ...],
            [[coeff, product1], ...]]. rate_constant may be float or str.

    Returns:
        list: The input reaction_equation list extended with:
            - A string representing the RHS of the rate law equation
            - A list of coefficients for reactants
            - A list of coefficients for products
    """
    reactant_eq = reaction_equation[:]
    rate_constant = 'k' + reaction_equation[0][0]
    terms = [rate_constant, ]
    for e in reaction_equation[1]:
        if e[0] != '':
            term = e[1] + '(t)' + '**' + e[0]
        else:
            term = e[1] + '(t)'
        terms.append(term)  # the list of the reactants
    reactant_equation = "*".join(terms)
    coef_kinetics = [reactant_equation, ]

    standard_val = reaction_equation[1][0][0]
    for e in reaction_equation[1:3]:
        coef = [e2[0] + '/' + standard_val for e2 in e]
        coef = [e[:-1] if e.endswith('/') else e for e in coef]
        coef = ['1' + e if e.startswith('/') else e for e in coef]
        coef_kinetics.append(coef)

    reactant_eq.append(coef_kinetics)
    return reactant_eq


def generate_ode(reaction):
    """Generate ODE expressions for a single reaction.

    This function is used internally by 'generate_sys_ode' to process
    individual reactions and generate differential equation expressions
    for each chemical species involved in the reaction.

    Args:
        reaction (list): A list representing a reaction equation in the
            format [['ID', rate_constant], reactants, products, conditions,
            [rate_law_string, [reactant_coeffs], [product_coeffs]]].
            rate_constant may be float or str. The reaction must have
            at least 5 elements.

    Returns:
        defaultdict: A dictionary mapping species names (str) to their ODE
            expression strings. Reactants contribute negative terms, products
            contribute positive terms.
    """
    dict_species = defaultdict(str)
    LHSs = []  # 常微分方程式左辺
    RHSs = []  # 常微分方程式右辺
    # equations = [] # 式全体 : 隠し機能

    # reaction[4]が存在するかチェック
    if len(reaction) < 5:
        print(f"Warning: reaction format is incomplete: {reaction}")
        return dict_species

    for i, half_reaction in enumerate(reaction[1:3]):
        for j, species in enumerate(half_reaction):
            # for human interface, str('d'+species[1]+'/dt')
            LHS_ODE = str(species[1])
            if i == 0:  # reactants
                RHS_ODE = str('-' + reaction[4][1][j] + '*' +
                              reaction[4][0])
            else:  # i==1, products
                RHS_ODE = str('+' + reaction[4][2][j] + '*' +
                              reaction[4][0])
            RHS_ODE = RHS_ODE.replace("-*", "-").replace("+*", "+")
            LHSs.append(LHS_ODE)
            RHSs.append(RHS_ODE)
            # equation = LHS_ODE + ' = ' + RHS_ODE
            # equation = equation.replace("= +", "= ")
            # equations.append(equation)

    for k, v in zip(LHSs, RHSs):
        dict_species[k] += v
    return dict_species


def generate_sys_ode(reactant_eq):
    """Generate system of ODEs from reactant equations.

    Args:
        reactant_eq (list): A list of reactant equations.

    Returns:
        dict: A dictionary containing the system of ODEs.
    """
    sys_odes_list = []
    for reaction in reactant_eq:
        sys_odes_list.append(generate_ode(reaction))
        sys_odes_dict = defaultdict(str)
        for d in sys_odes_list:
            for key, value in d.items():
                sys_odes_dict[key] += value
        sys_odes_dict = {
            key: value.lstrip('+') for key, value in sys_odes_dict.items()
        }
    return dict(sys_odes_dict)


def rate_constants(reactant_eq):
    """Extract rate constants from reactant equations.

    Each key is 'k' + reaction ID. Values are float when the rate constant
    cell was numeric; otherwise the raw string (e.g. 'k2', 'k1*2') is kept.

    Args:
        reactant_eq (list): A list of reactant equations (output of
            reactant_consumption for each reaction from get_reactions).

    Returns:
        dict: Mapping from rate constant key (e.g. 'k1', 'k2') to float or
            str. String values are used later for symbol or expression
            parsing in _build_rate_consts_sympy.
    """
    rate_consts_dict = {}
    for e in reactant_eq:
        key = 'k' + e[0][0]
        try:
            value = float(e[0][1])
        except (ValueError, TypeError):
            value = e[0][1]
        rate_consts_dict[key] = value
    return rate_consts_dict


def _build_rate_consts_sympy(raw_rate_consts):
    """Convert raw rate constants to SymPy values (float, Symbol, or Expr).

    - float: kept as is.
    - str: parsed as a SymPy expression (e.g. "k1*2"). If parsing yields
      a single symbol or parsing fails, the value is treated as a Symbol.
      All symbols appearing in expressions must be among the rate constant
      keys (allowed_keys); otherwise ValueError is raised.

    Args:
        raw_rate_consts (dict): Output of rate_constants(reactant_eq).
            Keys are rate constant names (e.g. 'k1', 'k2'); values are
            float or str.

    Returns:
        dict: Same keys as raw_rate_consts. Values are float, sympy.Symbol,
            or a SymPy expression (e.g. 2*k1 for "k1*2").

    Raises:
        ValueError: If an expression references a rate constant not in
            the reaction set (e.g. "k99*2" when only k1, k2 exist).
    """
    allowed_keys = set(raw_rate_consts.keys())
    sym_dict = {k: Symbol(k) for k in allowed_keys}
    result = {}
    for key, raw_val in raw_rate_consts.items():
        if isinstance(raw_val, (int, float)):
            result[key] = raw_val
            continue
        s = (raw_val if isinstance(raw_val, str) else str(raw_val)).strip()
        try:
            parsed = parse_expr(s, local_dict=sym_dict)
        except Exception:
            parsed = Symbol(s)
        free_names = {str(sym) for sym in getattr(parsed, 'free_symbols', [])}
        if isinstance(parsed, Symbol) and parsed.name not in allowed_keys:
            raise ValueError(
                "式の右辺に存在しない速度定数を指定しました: "
                f"'{parsed.name}'. 使用可能な速度定数: {sorted(allowed_keys)}"
            )
        if free_names and not free_names.issubset(allowed_keys):
            undefined = free_names - allowed_keys
            raise ValueError(
                "式の右辺に存在しない速度定数を指定しました: "
                f"{undefined}. 使用可能な速度定数: {sorted(allowed_keys)}"
            )
        result[key] = parsed
    return result


class RxnToODE:
    """Convert reaction equations from CSV files to ODE systems.

    Reads chemical reaction data from CSV and builds systems of ordinary
    differential equations using SymPy. Rate constants may be numeric,
    symbolic, or given by expressions (e.g. k2 = k1*2).

    Attributes:
        file_path (str): Path to the CSV file containing reaction data.
        reactant_eq (list): List of reactant equations from the file.
        t (Symbol): SymPy symbol for time.
        function_names (list): Chemical species names in appearance order.
        functions_dict (dict): Species names to SymPy Function objects.
        rate_consts_dict (dict): Rate constant key to float, Symbol, or
            SymPy Expr (e.g. 2*k1 for constraint k2=k1*2).
        sympy_symbol_dict (dict): Combined dict of t, Derivative,
            rate_consts_dict, and functions_dict for parsing ODEs.
        sys_odes_dict (dict): Species name to ODE right-hand side string.
    """

    def __init__(self, file_path, encoding=None):
        """Initialize from a reaction CSV file.

        Args:
            file_path (str): Path to the CSV file containing reaction data.
            encoding (str, optional): File encoding. Defaults to 'utf-8'
                if None.
        """
        self.file_path = file_path
        self.encoding = encoding if encoding else 'utf-8'

        # ファイルから反応式を読み込み、reactant_eqを生成
        reaction_equations = get_reactions(file_path, self.encoding)
        self.reactant_eq = [
            reactant_consumption(rxn) for rxn in reaction_equations
        ]

        self.t = symbols('t')
        self.function_names = get_unique_species(reaction_equations)
        self.functions_dict = dict(
            zip(self.function_names,
                [Function(name) for name in self.function_names])
        )
        raw_rate_consts = rate_constants(self.reactant_eq)
        self.rate_consts_dict = _build_rate_consts_sympy(raw_rate_consts)

        # 文字列内で使用するシンボル、関数、定数を統合
        self.sympy_symbol_dict = {'t': self.t, 'Derivative': Derivative}
        self.sympy_symbol_dict.update(self.rate_consts_dict)
        self.sympy_symbol_dict.update(self.functions_dict)

        # 方程式を構築
        self.sys_odes_dict = generate_sys_ode(self.reactant_eq)
    
    def get_equations(self):
        """Build the system of SymPy equations for the ODEs.

        Returns:
            list: List of sympy.Eq (lhs = Derivative(species(t), t), rhs expr).
        """
        system_of_equations = []
        for key in self.sys_odes_dict.keys():
            rhs_expr = parse_expr(
                self.sys_odes_dict[key],
                local_dict=self.sympy_symbol_dict
            )
            lhs = Derivative(self.functions_dict[key](self.t), self.t)
            eq = Eq(lhs, rhs_expr)
            system_of_equations.append(eq)
        return system_of_equations

    def create_ode_system(self):
        """Build ODE right-hand side expressions (SymPy) per species.

        Returns:
            dict: Species name -> SymPy expression for d(species)/dt.
        """
        ode_expressions = {}
        for key in self.sys_odes_dict.keys():
            rhs_expr = parse_expr(
                self.sys_odes_dict[key],
                local_dict=self.sympy_symbol_dict
            )
            ode_expressions[key] = rhs_expr
        return ode_expressions

