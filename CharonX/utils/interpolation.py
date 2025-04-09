#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 09:41:08 2025

@author: bouteillerp
"""
from dolfinx.fem import Function, Expression
from ..utils.petsc_operations import petsc_assign

def interpolate_to_function(target_function, value):
    """
    Interpole une valeur dans une fonction cible.
    
    Parameters
    ----------
    target_function : Function Fonction cible où stocker le résultat.
    value : float, Expression ou Function Valeur à interpoler.
        
    Raises
    ------
    ValueError Si le type de valeur n'est pas pris en charge.
    """
    if isinstance(value, float):
        target_function.x.array[:] = value
    elif isinstance(value, Expression):
        target_function.interpolate(value)
    elif isinstance(value, Function):
        petsc_assign(target_function, value)
    else:
        raise ValueError(f"Unsupported value type: {type(value)}")

def create_function_from_expression(V, expr):
    """
    Crée une fonction à partir d'une expression.
    
    Parameters
    ----------
    V : FunctionSpace Espace fonctionnel où créer la fonction.
    expr : Expression Expression à interpoler.
        
    Returns
    -------
    Function Nouvelle fonction contenant l'expression interpolée.
    """
    func = Function(V)
    func.interpolate(expr)
    return func

def interpolate_multiple(target_functions, values):
    """
    Interpole plusieurs valeurs dans plusieurs fonctions cibles.
    
    Parameters
    ----------
    target_functions : List[Function] Liste des fonctions cibles.
    values : List Liste des valeurs à interpoler (même longueur que target_functions).
        
    Raises
    ------
    ValueError
        Si les listes n'ont pas la même longueur.
    """
    if len(target_functions) != len(values):
        raise ValueError("Les listes de fonctions et de valeurs doivent avoir la même longueur")
    
    for func, value in zip(target_functions, values):
        interpolate_to_function(func, value)