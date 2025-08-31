# Copyright (c) 2025 Mitsuru Ohno
# Use of this source code is governed by a BSD-3-style
# license that can be found in the LICENSE file.

# 08/30/2025, M. Ohno

from .rxn_reader import RxnToODE
from sympy import Function, symbols, parse_expr, lambdify

class RxnIVPsolv(RxnToODE):
    """A class for solving ODE systems using scipy.solve_ivp.
    
    This class extends RxnToODE to provide functionality for numerical
    integration of the ODE system. It automatically generates numerical
    functions from symbolic ODE expressions that can be used with
    scipy.solve_ivp for time integration.
    
    Attributes:
        Inherits all attributes from RxnToODE class.
        
    Methods:
        create_ode_system(): Creates numerical ODE functions for integration.
        get_ode_system(): Returns complete ODE system for scipy.solve_ivp.
        debug_ode_system(): Provides detailed debug information.
    """
    
    def __init__(self, file_path, encoding=None):
        """Initialize the RxnIVPsolv class.
        
        Args:
            file_path (str): The path to the CSV file containing reaction data.
            encoding (str, optional): The encoding of the CSV file. Defaults to 'utf-8' if None.
        """
        # 親クラスの初期化を呼び出し
        super().__init__(file_path, encoding)
    
    def create_ode_system(self):
        """Create ODE system functions for numerical integration.
        
        Returns:
            dict: Dictionary mapping species names to ODE functions.
        """
        ode_functions = {}
        
        # 引数の順序を明示的に定義（文字列として）
        args = ['t'] + self.function_names
        
        for key in self.sys_odes_dict.keys():
            rhs_expr = parse_expr(self.sys_odes_dict[key], local_dict=self.sympy_symbol_dict)

            # 式内の関数呼び出しを変数に置換
            # 例: AcOEt(t) -> AcOEt, OHa1(t) -> OHa1
            for func_name in self.function_names:
                func_sym = Function(func_name)
                rhs_expr = rhs_expr.subs(func_sym(self.t), symbols(func_name))

            # 関数を数値計算用に変換（引数順序を明示的に指定）
            try:
                # 引数順序を確実に制御
                ode_functions[key] = lambdify(args, rhs_expr, modules='numpy')
                print(f"Successfully created function for {key} with args: {args}")
            except Exception as e:
                print(f"Warning: Failed to create lambdify function for {key}: {e}")
                print(f"Expression: {rhs_expr}")
                print(f"Arguments: {args}")
                # フォールバック: より安全な方法でlambdifyを作成
                try:
                    ode_functions[key] = lambdify(args, rhs_expr, modules=['numpy', 'math'])
                    print(f"Fallback successful for {key}")
                except Exception as e2:
                    print(f"Fallback also failed for {key}: {e2}")
                    # 最後の手段: 引数を個別に指定
                    ode_functions[key] = lambdify((['t'] + self.function_names), rhs_expr, modules='numpy')
                    print(f"Individual args method successful for {key}")

        return ode_functions
    
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
        
        return (system_of_equations, self.sympy_symbol_dict, 
                ode_system, self.function_names, self.rate_consts_dict)
    
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
                    debug_info['ode_functions_info'][key]['test_successful'] = True
                    debug_info['ode_functions_info'][key]['test_result'] = test_result
                except Exception as test_e:
                    debug_info['ode_functions_info'][key]['test_successful'] = False
                    debug_info['ode_functions_info'][key]['test_error'] = str(test_e)
                    
            except Exception as e:
                debug_info['ode_functions_info'][key] = {
                    'error': str(e),
                    'expression': self.sys_odes_dict[key]
                }
        
        return debug_info


