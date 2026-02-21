# Copyright (c) 2025 Mitsuru Ohno
# Use of this source code is governed by a BSD-3-style
# license that can be found in the LICENSE file.

# 08/30/2025, M. Ohno

from sympy import Function, Symbol, symbols, parse_expr, lambdify
from sympy.core.symbol import Symbol as SympySymbol
import inspect

from .rxn_reader import RxnToODE


def _resolve_rate_consts(rate_consts_dict):
    """Resolve rate constants to numbers where possible (for ODE solver).

    Values that are float are kept. Expressions (e.g. 2*k1) are evaluated
    once all their free symbols have numeric values in the dict. Symbols
    without values remain unresolved.

    Args:
        rate_consts_dict (dict): Maps key to float, Symbol, or SymPy Expr.

    Returns:
        dict: Subset of keys with numeric values (including from expressions).
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
                    resolved[k] = float(v.subs({s: resolved[str(s)] for s in free_syms}))
                    changed = True
                except (TypeError, ValueError):
                    pass
    return resolved


class RxnODEbuild(RxnToODE):
    """A class for building ODE systems from reaction definitions.

    This class extends RxnToODE to construct symbolic ODE expressions and
    generate numerical functions that can be used by external solvers
    (e.g., scipy.solve_ivp) for time integration.

    Attributes:
        Inherits all attributes from RxnToODE class.

    Methods:
        create_ode_system(): Creates numerical ODE functions for integration.
        get_ode_system(): Returns complete ODE system for numerical solvers.
        debug_ode_system(): Provides detailed debug information.
        get_ode_info(debug_info=False): Prints summary and optional debug info.
    """

    def __init__(self, file_path, encoding=None):
        """Initialize the RxnODEbuild class.

        Args:
            file_path (str): The path to the CSV file containing reaction
                data.
            encoding (str, optional): The encoding of the CSV file.
                Defaults to 'utf-8' if None.
        """
        # 親クラスの初期化を呼び出し
        super().__init__(file_path, encoding)

    def create_ode_system(self):
        """Create ODE system functions for numerical integration.

        Rate constants that are expressions (e.g. k2=k1*2) are resolved to
        numbers when all referenced symbols have numeric values.

        Returns:
            dict: Dictionary mapping species names to ODE functions.
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

    def create_ode_system_with_rate_consts(self):
        """Create ODE system functions with rate constants as arguments.

        This method creates lambdify functions that include rate constants
        as arguments, allowing them to be passed dynamically without
        relying on global scope.
        This method is used for the function 'solve_fit_model'

        Returns:
            tuple: A tuple containing:
                - dict: Dictionary mapping species names to ODE functions.
                  Functions accept arguments:
                  [t] + function_names + symbolic_rate_const_keys
                - list: List of symbolic rate constant keys in order
        """
        ode_functions = {}

        # フィッティングで変える速度定数＝式の右辺に現れるシンボルのみ（式で定義されたものは含めない）
        free_param_names = set()
        for key, val in self.rate_consts_dict.items():
            if isinstance(val, (int, float)):
                continue
            if isinstance(val, (Symbol, SympySymbol)):
                free_param_names.add(val.name)
            else:
                free_param_names.update(str(s) for s in getattr(val, 'free_symbols', []))
        symbolic_rate_const_keys = sorted(free_param_names)

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
        """Get ODE system objects for scipy.solve_ivp integration.

        Returns:
            tuple: A tuple containing:
                - system_of_equations: List of SymPy equations
                - sympy_symbol_dict: Dictionary of SymPy symbols
                - ode_system: ODE system functions
                - function_names: List of function names
                - rate_consts_dict: Dictionary of rate constants
        """
        system_of_equations = self.get_equations()
        ode_system = self.create_ode_system()

        return (
            system_of_equations, self.sympy_symbol_dict,
            ode_system, self.function_names, self.rate_consts_dict
        )

    def debug_ode_system(self):
        """Debug information for ODE system.

        Returns:
            dict: Debug information about the ODE system.
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
        """Print summary information about the ODE system.

        Args:
            debug_info (bool, optional): If True, also print detailed debug
                information. Defaults to False.
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
    """Create a function to compute the right-hand side of an ODE system.

    This function creates a closure that captures the ODE functions and
    parameters, returning a function with the signature (t, y) required
    by scipy.solve_ivp.

    Args:
        ode_functions_dict (dict): Dictionary mapping species names to
            their ODE functions.
        function_names (list): List of chemical species names in order.
        rate_const_values (dict, optional): Dictionary of rate constant
            values to be passed dynamically. If provided, rate constants
            are passed as function arguments rather than being embedded.
        symbolic_rate_const_keys (list, optional): List of rate constant
            keys in order. Required if rate_const_values is provided.

    Returns:
        function: A function system_rhs(t, y) that computes the right-hand
            side of the ODE system. The function accepts time t and state
            vector y, and returns a list of derivatives for each species.
    """
    # クロージャでパラメータをキャプチャ（solve_ivpは(t, y)シグネチャを要求）
    def system_rhs(t, y):
        """Compute the right-hand side of the ODE system.

        Args:
            t (float): Current time.
            y (array-like): Current state vector (concentrations).

        Returns:
            list: List of derivatives for each species in function_names order.
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
                        rate_const_args = [
                            rate_const_values[key]
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
