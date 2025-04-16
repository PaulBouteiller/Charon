"""
Module de fonctions d'ajustement polynomial pour la calibration de modèles mécaniques

Ce module fournit des fonctions pour ajuster des polynômes de la forme 
1 + b*(x-1) + c*(x-1)² + ... aux données expérimentales. Ces polynômes 
sont particulièrement adaptés à la calibration de modèles mécaniques où 
les fonctions doivent valoir 1 à l'état non déformé (x=1).

Created on Wed Nov 27 12:41:36 2024
@author: bouteillerp
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#%%Fonctions de fit

def shifted_polynomial_fixed_constant(x, *params):
    """
    Crée un polynôme de la forme: 1 + b*(x-1) + c*(x-1)^2 + d*(x-1)^3 + ...
    
    Cette forme polynomiale est utilisée pour des fonctions qui doivent valoir 1
    lorsque x=1 (état de référence non déformé).
    
    Arguments: x (array-like): Points d'évaluation du polynôme
        *params: Coefficients du polynôme (à partir du degré 1)
    
    Returns: array-like: Valeurs du polynôme aux points x
    """
    # print("Attention le premier terme de paramètres doit correspondre au terme linéaire")
    result = 1  # terme constant fixé à 1
    x_shifted = x - 1
    
    for i, param in enumerate(params, 1):
        result += param * (x_shifted ** i)
    
    return result

def fit_and_plot_shifted_polynomial_fixed(x_data, y_data, degree, plot, plot_original=True,
                                         xlabel='J', ylabel='f', title_prefix='', ):
    """
    Ajuste un polynôme de la forme 1 + b*(x-1) + c*(x-1)^2 + ... sur les données
    et affiche les résultats graphiquement.
    
    Arguments:
        x_data (array-like): Les valeurs x des données expérimentales
        y_data (array-like): Les valeurs y des données expérimentales
        degree (int): Le degré du polynôme à ajuster
        plot_original (bool, optional): Si True, trace aussi les données originales
        xlabel (str, optional): Libellé de l'axe x (défaut: 'J')
        ylabel (str, optional): Libellé de l'axe y (défaut: 'f')
        title_prefix (str, optional): Préfixe pour le titre du graphique
    
    Returns:
        tuple: (params, r2) où params est un tableau des coefficients du polynôme
               et r2 est le coefficient de détermination
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
        if plot_original:
            plt.plot(x_data, y_data, 'b.', label='Données originales', alpha=0.5)
        plt.plot(x_fit, y_fit, 'r-', label=f'Polynôme (x-1) degré {degree}')
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f'Ajustement polynomial avec (x-1) et terme constant=1 (R² = {r2:.4f})')
        plt.legend()
        plt.grid(True)
        plt.show()

    print("\nCoefficients du polynôme:")
    print("a (terme constant): 1.000000 (fixé)")
    for i, coef in enumerate(params, 1):
        print(f"Coefficient de (x-1)^{i}: {coef:.6e}")
        
    return params, r2
