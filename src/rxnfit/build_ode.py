# Copyright (c) 2025 Mitsuru Ohno
# Use of this source code is governed by a BSD-3-style
# license that can be found in the LICENSE file.

# 08/30/2025, M. Ohno

import csv

from sympy import Function, Symbol, symbols, parse_expr, lambdify
from sympy.core.symbol import Symbol as SympySymbol
import inspect

from .rxn_reader import RxnToODE


def _load_rate_const_overrides_csv(file_path, encoding, allowed_keys):
    """Load rate constant overrides from a CSV with columns 'k' and 'f(t)'.

    Specification:
    - First row must be header exactly "k" and "f(t)".
    - Lines starting with # are not allowed (error).
    - Empty rows (both columns empty) are ignored.
    - Duplicate k in data rows raises error.
    - Row with empty f(t) is ignored (that k is not overridden).
    - k must be in allowed_keys (keys from reaction CSV), else error.
    - Only .csv path is supported.

    Returns:
        dict: k_name -> expression string. Keys are from data rows;
              rows with empty f(t) are skipped.
    """
    if not file_path.lower().endswith('.csv'):
        raise ValueError(
            "rate_const_overrides as path supports only CSV files. "
            f"Got: {file_path!r}"
        )
    enc = 'utf-8-sig' if encoding == 'utf-8' else encoding
    with open(file_path, mode='r', encoding=enc) as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError("Rate constant overrides CSV is empty.")
        header = [c.strip() for c in header]
        if header != ['k', 'f(t)']:
            raise ValueError(
                "Rate constant overrides CSV header must be exactly "
                "'k' and 'f(t)'. Got: {!r}".format(header)
            )
        seen_k = set()
        overrides = {}
        for row in reader:
            if len(row) < 2:
                if not any(c.strip() for c in row):
                    continue
                raise ValueError(
                    "Rate constant overrides CSV: row must have at least "
                    "two columns. Got: {!r}".format(row)
                )
            k_col = row[0].strip()
            ft_col = row[1].strip()
            if k_col.startswith('#'):
                raise ValueError(
                    "Comment lines (# ...) are not allowed in rate constant "
                    "overrides CSV."
                )
            if not k_col and not ft_col:
                continue
            if not k_col:
                continue
            if k_col not in allowed_keys:
                raise ValueError(
                    "Rate constant '{0}' in overrides CSV is not defined in "
                    "the reaction CSV. Allowed: {1}.".format(
                        k_col, sorted(allowed_keys)
                    )
                )
            if k_col in seen_k:
                raise ValueError(
                    "Duplicate rate constant '{}' in overrides CSV.".format(k_col)
                )
            seen_k.add(k_col)
            if not ft_col:
                continue
            overrides[k_col] = ft_col
    return overrides


def _resolve_rate_consts(rate_consts_dict):
    """Resolve rate constants to numeric values where possible.

    Used when building ODE functions so that expression-defined constants
    (e.g. k2 = 2*k1) are substituted to numbers when all referenced symbols
    have numeric values. Float values are copied; expressions are evaluated
    in dependency order; Symbol-only keys are left unresolved.

    Args:
        rate_consts_dict (dict): Rate constant key -> float, sympy.Symbol,
            or SymPy Expr (e.g. 2*k1).

    Returns:
        dict: Key -> float for every key that could be resolved to a number.
            Keys that remain Symbol or have unresolved symbols are omitted.
    """
    resolved = {
        k: v for k, v in rate_consts_dict.items()
        if isinstance(v, (int, float))
    }
    changed = True
    while changed:
        changed = False
        for k, v in rate_consts_dict.items():
            if k in resolved:
                continue
            free_syms = getattr(v, 'free_symbols', None)
            if not free_syms:
                continue
            if all(str(s) in resolved for s in free_syms):
                try:
                    subs = {s: resolved[str(s)] for s in free_syms}
                    resolved[k] = float(v.subs(subs))
                    changed = True
                except (TypeError, ValueError):
                    pass
    return resolved


class RxnODEbuild(RxnToODE):
    """Build ODE systems and numerical RHS functions from reaction definitions.

    Extends RxnToODE to produce callable ODE right-hand sides (e.g. for
    scipy.solve_ivp) and supports rate constants as expressions (e.g. k2=k1*2).
    Expression-defined constants are resolved to numbers when building
    create_ode_system(); for fitting, only free symbolic parameters appear
    in create_ode_system_with_rate_consts().

    Attributes:
        Inherits all attributes from RxnToODE.

    Methods:
        create_ode_system(): Numerical ODE functions (t, y) with rate
            constants resolved or embedded.
        create_ode_system_with_rate_consts(): ODE functions with rate
            constants as extra arguments (for fitting).
        get_ode_system(): Full ODE construction tuple for solvers.
        debug_ode_system(): Detailed debug info.
        get_ode_info(debug_info=False): Print summary and optional debug.
    """

    def __init__(self, file_path, encoding=None, rate_const_overrides=None,
                 rate_const_overrides_encoding=None):
        """Initialize from a reaction CSV file.

        Args:
            file_path (str): Path to the CSV file containing reaction data.
            encoding (str, optional): File encoding for the reaction CSV.
                Defaults to 'utf-8' if None.
            rate_const_overrides (dict, str, or None, optional): Optional
                overrides for rate constants. If None, no overrides are applied.
                If dict, keys are rate constant names and values are expression
                strings (e.g. "k1/4", "a*exp(-t)"). If str, path to a CSV file
                with columns "k" and "f(t)" (see module doc/spec). Defaults to
                None.
            rate_const_overrides_encoding (str, optional): Encoding for
                reading the overrides CSV when rate_const_overrides is a path.
                Only used when rate_const_overrides is str. If None, the
                reaction CSV encoding is used.
        """
        super().__init__(file_path, encoding)

        if rate_const_overrides is None:
            return

        allowed_keys = set(self.rate_consts_dict.keys())
        if isinstance(rate_const_overrides, str):
            enc = rate_const_overrides_encoding if rate_const_overrides_encoding is not None else self.encoding
            overrides_dict = _load_rate_const_overrides_csv(
                rate_const_overrides, enc, allowed_keys
            )
        elif isinstance(rate_const_overrides, dict):
            for k in rate_const_overrides:
                if k not in allowed_keys:
                    raise ValueError(
                        "Rate constant '{0}' in overrides is not defined in "
                        "the reaction CSV. Allowed: {1}.".format(
                            k, sorted(allowed_keys)
                        )
                    )
            overrides_dict = rate_const_overrides
        else:
            raise TypeError(
                "rate_const_overrides must be None, dict, or str (path). "
                "Got: {!r}".format(type(rate_const_overrides))
            )

        for k_name, expr_str in overrides_dict.items():
            expr_str = expr_str.strip()
            if not expr_str:
                continue
            try:
                parsed = parse_expr(expr_str, local_dict=self.sympy_symbol_dict)
            except Exception as e:
                raise ValueError(
                    "Failed to parse expression for '{0}': {1!r}. "
                    "Error: {2}".format(k_name, expr_str, e)
                ) from e
            self.rate_consts_dict[k_name] = parsed
            self.sympy_symbol_dict[k_name] = parsed

    def create_ode_system(self):
        """Build numerical ODE right-hand side functions for integration.

        Rate constants that are expressions (e.g. k2=k1*2) are resolved to
        numbers via _resolve_rate_consts when all referenced symbols have
        numeric values; otherwise the expression or Symbol is left in place.
        Functions take (t, y) with y in function_names order.

        Returns:
            dict: Species name -> callable(t, *y) returning d(species)/dt.
        """
        ode_functions = {}

        # 引数の順序を明示的に定義（文字列として）
        args = ['t'] + self.function_names

        # 式で定義された速度定数を数値に解決し、未解決は元の Symbol/Expr のまま
        resolved = _resolve_rate_consts(self.rate_consts_dict)
        local_dict = dict(self.sympy_symbol_dict)
        for k, v in self.rate_consts_dict.items():
            local_dict[k] = resolved.get(k, v)

        for key in self.sys_odes_dict.keys():
            rhs_expr = parse_expr(
                self.sys_odes_dict[key],
                local_dict=local_dict
            )

            # 式内の関数呼び出しを変数に置換
            # 例: AcOEt(t) -> AcOEt, OHa1(t) -> OHa1
            for func_name in self.function_names:
                func_sym = Function(func_name)
                rhs_expr = rhs_expr.subs(func_sym(self.t), symbols(func_name))

            # 関数を数値計算用に変換（引数順序を明示的に指定）
            try:
                ode_functions[key] = lambdify(
                    args, rhs_expr, modules='numpy'
                )
                print(f"Successfully created function for {key} "
                      f"with args: {args}")
            except Exception as e:
                print(f"Warning: Failed to create lambdify function "
                      f"for {key}: {e}")

        return ode_functions

    def get_symbolic_rate_const_keys(self):
        """Return the list of free parameter names (p0 order) for fitting.

        Names are t-excluding symbols from rate_consts_dict (Symbol or
        free_symbols of expressions), in sorted order. Same order as
        create_ode_system_with_rate_consts and run_fit(p0=...) expect.

        Returns:
            list[str]: Parameter names in order, e.g. ['a', 'km'].
        """
        free_param_names = set()
        for key, val in self.rate_consts_dict.items():
            if isinstance(val, (int, float)):
                continue
            if isinstance(val, (Symbol, SympySymbol)):
                if val.name != 't':
                    free_param_names.add(val.name)
            else:
                syms = getattr(val, 'free_symbols', [])
                for s in syms:
                    if str(s) != 't':
                        free_param_names.add(str(s))
        return sorted(free_param_names)

    def create_ode_system_with_rate_consts(self):
        """Build ODE functions with rate constants as explicit arguments.

        Used for fitting: rate constants are passed in at call time instead
        of being embedded. Only "free" parameters appear: keys that are
        Symbol or that appear inside expression-defined constants (e.g. k1
        in k2=2*k1). Keys defined only by expressions (e.g. k2) do not
        appear in symbolic_rate_const_keys.

        Returns:
            tuple: (ode_functions, symbolic_rate_const_keys)
                - ode_functions: Species name -> callable(t, *y, *rate_consts).
                - symbolic_rate_const_keys: List of free rate constant names
                  in order (e.g. ['k1', 'k3'] when k2=k1*2).
        """
        symbolic_rate_const_keys = self.get_symbolic_rate_const_keys()
        ode_functions = {}

        # 引数の順序を明示的に定義（速度定数も含む）
        args = ['t'] + self.function_names + symbolic_rate_const_keys

        for key in self.sys_odes_dict.keys():
            rhs_expr = parse_expr(
                self.sys_odes_dict[key],
                local_dict=self.sympy_symbol_dict
            )

            # 式内の関数呼び出しを変数に置換
            # 例: AcOEt(t) -> AcOEt, OHa1(t) -> OHa1
            for func_name in self.function_names:
                func_sym = Function(func_name)
                rhs_expr = rhs_expr.subs(
                    func_sym(self.t), symbols(func_name)
                )

            # 関数を数値計算用に変換（引数順序を明示的に指定、
            # 速度定数も含む）
            try:
                ode_functions[key] = lambdify(
                    args, rhs_expr, modules='numpy'
                )
            except Exception as e:
                print(f"Warning: Failed to create lambdify function "
                      f"for {key}: {e}")
                ode_functions[key] = None

        return ode_functions, symbolic_rate_const_keys

    def get_ode_system(self):
        """Return the full ODE construction for scipy.solve_ivp.

        Returns:
            tuple: (system_of_equations, sympy_symbol_dict, ode_system,
                function_names, rate_consts_dict). ode_system is from
                create_ode_system() (rate constants resolved or embedded).
        """
        system_of_equations = self.get_equations()
        ode_system = self.create_ode_system()

        return (
            system_of_equations, self.sympy_symbol_dict,
            ode_system, self.function_names, self.rate_consts_dict
        )

    def debug_ode_system(self):
        """Collect debug information for the ODE system.

        Returns:
            dict: function_names, rate_constants, ode_expressions,
                lambdify_args, ode_functions_info (and test results).
        """
        debug_info = {
            'function_names': self.function_names,
            'function_names_order': list(enumerate(self.function_names)),
            'rate_constants': self.rate_consts_dict,
            'ode_expressions': self.sys_odes_dict,
            'lambdify_args': ['t'] + self.function_names,
            'total_species': len(self.function_names)
        }

        # 各ODE関数の引数情報を確認
        ode_functions = self.create_ode_system()
        debug_info['ode_functions_info'] = {}

        for key, func in ode_functions.items():
            try:
                # 関数の引数情報を取得（可能な場合）
                debug_info['ode_functions_info'][key] = {
                    'function': func,
                    'expression': self.sys_odes_dict[key],
                    'function_type': str(type(func)),
                    'function_repr': repr(func)
                }

                # 関数の引数数を確認（可能な場合）
                try:
                    # テスト用の引数を作成
                    test_args = [0.0] + [1.0] * len(self.function_names)
                    test_result = func(*test_args)
                    debug_info['ode_functions_info'][key][
                        'test_successful'] = True
                    debug_info['ode_functions_info'][key][
                        'test_result'] = test_result
                except Exception as test_e:
                    debug_info['ode_functions_info'][key][
                        'test_successful'] = False
                    debug_info['ode_functions_info'][key][
                        'test_error'] = str(test_e)

            except Exception as e:
                debug_info['ode_functions_info'][key] = {
                    'error': str(e),
                    'expression': self.sys_odes_dict[key]
                }

        return debug_info

    def get_ode_info(self, debug_info: bool = False):
        """Print a short summary of the ODE system (and optionally debug).

        Args:
            debug_info (bool, optional): If True, also print detailed debug
                (e.g. lambdify args, ODE expressions). Defaults to False.
        """
        print(f"number of species: {len(self.function_names)}")
        print(f"unique species: {self.function_names}")
        print(f"rate constant: {self.rate_consts_dict}")

        if debug_info is True:
            # デバッグ情報を確認
            print("\n=== debug info ===")
            dbg = self.debug_ode_system()
            print(f"order of args: {dbg['lambdify_args']}")
            print(f"system of ODE: {dbg['ode_expressions']}")


def create_system_rhs(ode_functions_dict, function_names,
                      rate_const_values=None,
                      symbolic_rate_const_keys=None):
    """Build the RHS function (t, y) for scipy.solve_ivp.

    Creates a closure over the ODE functions and optional rate constant
    values. When rate_const_values and symbolic_rate_const_keys are given,
    the ODE functions are called with (t, y, *rate_consts) in the order
    of symbolic_rate_const_keys.

    Args:
        ode_functions_dict (dict): Species name -> ODE function (as from
            create_ode_system or create_ode_system_with_rate_consts).
        function_names (list): Species names in the same order as y.
        rate_const_values (dict or callable, optional): Rate constant key -> numeric
            value, or a callable (t) -> dict of rate constant values for
            time-dependent rates. Used when ODE functions expect rate constants
            as args. When callable, it is called with the current time t to
            get the dict.
        symbolic_rate_const_keys (list, optional): Order of rate constant
            keys matching the ODE functions. Required if rate_const_values
            is provided.

    Returns:
        callable: system_rhs(t, y) returning a list of d(species)/dt in
            function_names order.
    """
    # クロージャでパラメータをキャプチャ（solve_ivpは(t, y)シグネチャを要求）
    def system_rhs(t, y):
        """Compute d(species)/dt for the current (t, y).

        Args:
            t (float): Current time.
            y (array-like): Current concentrations in function_names order.

        Returns:
            list: Derivatives for each species in function_names order.
        """
        rhs_odesys = []
        for species_name in function_names:
            if species_name in ode_functions_dict:
                try:
                    func = ode_functions_dict[species_name]
                    if func is None:
                        rhs_odesys.append(0.0)
                        continue

                    # 速度定数を動的に渡す場合
                    if (rate_const_values is not None and
                            symbolic_rate_const_keys is not None):
                        if callable(rate_const_values):
                            rate_vals = rate_const_values(t)
                        else:
                            rate_vals = rate_const_values
                        rate_const_args = [
                            rate_vals[key]
                            for key in symbolic_rate_const_keys
                        ]
                        args = [t] + list(y) + rate_const_args
                        result = func(*args)
                    else:
                        # 速度定数が固定の場合（従来の方法）
                        expected_args = None
                        try:
                            sig = inspect.signature(func)
                            expected_args = len(sig.parameters)
                        except Exception:
                            try:
                                expected_args = func.__code__.co_argcount
                            except Exception:
                                expected_args = None

                        if expected_args is None:
                            args = [t] + list(y)
                        else:
                            n_y = max(0, expected_args - 1)
                            args = [t] + list(y[:n_y])

                        result = func(*args)

                    rhs_odesys.append(result)
                except Exception as e:
                    print(f"Error in {species_name}: {e}")
                    rhs_odesys.append(0.0)
            else:
                rhs_odesys.append(0.0)
        return rhs_odesys

    return system_rhs
