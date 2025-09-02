# Copyright 2025 CEA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Polynomial Fitting Module
========================

Polynomial fitting functions for mechanical model calibration. The polynomials 
are of the form 1 + b*(x-1) + c*(x-1)² + ... which are suitable for mechanical 
models where functions must equal 1 at the undeformed state (x=1).

Created on Wed Nov 27 12:41:36 2024
@author: bouteillerp
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#%%Fonctions de fit

def shifted_polynomial_fixed_constant(x, *params):
    """
    Create polynomial of form: 1 + b*(x-1) + c*(x-1)^2 + d*(x-1)^3 + ...
    
    This polynomial form is used for functions that must equal 1
    when x=1 (reference undeformed state).
    
    Parameters
    ----------
    x : array-like Evaluation points for the polynomial
    *params : float  Polynomial coefficients (starting from degree 1)
    
    Returns
    -------
    array-like
        Polynomial values at points x
    """
    # print("Attention le premier terme de paramètres doit correspondre au terme linéaire")
    result = 1  # terme constant fixé à 1
    x_shifted = x - 1
    
    for i, param in enumerate(params, 1):
        result += param * (x_shifted ** i)
    
    return result

def fit_and_plot_shifted_polynomial_fixed(x_data, y_data, degree, plot, save_dict,
                                         xlabel=r'$J$', ylabel=r'$f$', title_prefix='', ):
    """
    Fit shifted polynomial to data with visualization.
    
    Parameters
    ----------
    x_data : array-like Independent variable data
    y_data : array-like Dependent variable data
    degree : int Polynomial degree
    plot : bool, optional Whether to create plot, by default True
    save_dict : dict, optional Save configuration with 'save' and 'name' keys, by default None
    xlabel : str, optional X-axis label, by default r'$J$'
    ylabel : str, optional Y-axis label, by default r'$f$'
    title_prefix : str, optional Plot title prefix, by default ''
    
    Returns
    -------
    tuple (coefficients, r_squared) fitted parameters and goodness of fit
    """
    # Valeurs initiales pour l'optimisation
    p0 = np.ones(degree)
    # Ajustement du polynôme
    params, covariance = curve_fit(shifted_polynomial_fixed_constant, x_data, y_data, p0=p0)
    
    # Génération des points pour le tracé de la courbe ajustée
    x_fit = np.linspace(min(x_data), max(x_data), 1000)
    y_fit = shifted_polynomial_fixed_constant(x_fit, *params)
    # Calcul des valeurs prédites aux points des données originales pour le calcul de R²
    y_pred = shifted_polynomial_fixed_constant(x_data, *params)
    
    # Calcul du coefficient de détermination R²
    r2 = 1 - np.sum((y_data - y_pred) ** 2) / np.sum((y_data - np.mean(y_data)) ** 2)
    if plot:
        plt.figure(figsize=(12, 7))
        plt.plot(x_data, y_data, 'b.', label='Original Data', alpha=0.5)
        plt.plot(x_fit, y_fit, 'r-', label=f'Polynomial degree {degree}')
        plt.xlabel(xlabel, fontsize = 20)
        plt.ylabel(ylabel, fontsize = 20)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.title(f'Polynomial fit (R² = {r2:.4f})', fontsize = 24)
        plt.legend(fontsize = 18)
        plt.grid(True)
        if save_dict.get("save"):
            file_name = save_dict.get("name")
            plt.savefig(f"{file_name}.pdf", bbox_inches = 'tight')
            plt.close()

    print("\nCoefficients du polynôme:")
    print("a (terme constant): 1.000000 (fixé)")
    for i, coef in enumerate(params, 1):
        print(f"Coefficient de (x-1)^{i}: {coef:.6e}")
        
    return params, r2
