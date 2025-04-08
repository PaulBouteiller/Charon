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
Created on Thu Apr 13 13:42:10 2023

@author: bouteillerp
"""

from ..utils.generic_functions import dt_update, petsc_assign
from dolfinx.fem import Function

def first_order_rk1(f, dot_f_expression, dot_f_function, dt, booleen = False, mesh = None, cells = None):
    """
    Schema de runge kutta d'ordre 1, pour les EDO d'ordre 1.
    Ce schéma est aussi appelé Euler-explicite.
    
    Parameters
    ----------
    f : Function, fonction à actualiser.
    dot_f_expression : Expression, dérivée temporelle de f.
    dot_f_function : Fonction, dérivée temporelle de f vivant dans le 
                                même espace fonctionnelle que f.
    dt : float, pas de temps.
    """
    dot_f_function.interpolate(dot_f_expression)
    if not booleen:
        dt_update(f, dot_f_function, dt)
        return 
    elif booleen:
        f_pred = f.copy()
        dt_update(f_pred, dot_f_function, dt)
        return f_pred
    
def first_order_rk2(f, dot_f_expression, dot_f_function, dt):
    """
    Schema de runge kutta d'ordre 2, pour les EDO d'ordre 1.
    
    Parameters
    ----------
    f : Function, fonction à actualiser.
    dot_f_expression : Expression, dérivée temporelle de f.
    dot_f_function : Fonction, dérivée temporelle de f vivant dans le 
                                même espace fonctionnelle que f.
    dt : float, pas de temps.
    """ 
    dot_f_function.interpolate(dot_f_expression)
    dt_update(f, dot_f_function, dt/2)
    dot_f_function.interpolate(dot_f_expression)
    dt_update(f, dot_f_function, dt/2)
    
def first_order_rk4(f, dot_f_expression, dot_f_function, dt):
    """
    Schema de runge kutta d'ordre 4, pour les EDO d'ordre 1.
    
    Parameters
    ----------
    f : Function, fonction à actualiser.
    dot_f_expression : Expression, dérivée temporelle de f.
    dot_f_function : Fonction, dérivée temporelle de f vivant dans le 
                                même espace fonctionnelle que f.
    dt : float, pas de temps.
    """
    prev_f = f.copy()
    V_f = f.function_space
    dot_f_1 = Function(V_f)
    dot_f_1.interpolate(dot_f_expression)
    petsc_assign(f, dt_update(prev_f, dot_f_1, dt/2, new_vec = True))
    dot_f_2 =  Function(V_f)
    dot_f_2.interpolate(dot_f_expression)
    petsc_assign(f, dt_update(prev_f, dot_f_2, dt/2, new_vec = True))
    dot_f_3 =  Function(V_f)
    dot_f_3.interpolate(dot_f_expression)
    petsc_assign(f, dt_update(prev_f, dot_f_3, dt, new_vec = True))
    dot_f_4 =  Function(V_f)
    dot_f_4.interpolate(dot_f_expression)
    dt_update(prev_f, dot_f_1, dt/6.)
    dt_update(prev_f, dot_f_2, dt/3.)
    dt_update(prev_f, dot_f_3, dt/3.)
    dt_update(prev_f, dot_f_4, dt/6.)
    petsc_assign(f, prev_f)
    
def second_order_rk1(f, dot_f, ddot_f_function, ddot_f_expression, dt):
    """
    Schema de runge kutta d'ordre 1, pour les EDO d'ordre 2.
    Ce schéma est aussi appelé Euler-explicite.
    
    Parameters
    ----------
    f : Function, fonction à actualiser.
    dot_f : Function, dérivée temporelle de f, vis dans le même espace fonctionnel que f.
    ddot_f_function : Function, dérivée temporelle seconde de f, 
                                vis dans le même espace fonctionnel que f.
    ddot_f_expression : Expression, expression de la dérivée temporelle seconde de f.
    dt : float, pas de temps.
    """
    ddot_f_function.interpolate(ddot_f_expression)
    dt_update(dot_f, ddot_f_function, dt)
    dt_update(f, dot_f, dt)  
    
def second_order_rk2(f, dot_f, ddot_f_function, ddot_f_expression, dt):
    """
    Schema de runge kutta d'ordre 2, pour les EDO d'ordre 2.
    
    Parameters
    ----------
    f : Function, fonction à actualiser.
    dot_f : Function, dérivée temporelle de f, vis dans le même espace fonctionnel que f.
    ddot_f_function : Function, dérivée temporelle seconde de f, 
                                vis dans le même espace fonctionnel que f.
    ddot_f_expression : Expression, expression de la dérivée temporelle seconde de f.
    dt : float, pas de temps.
    """  
    ddot_f_function.interpolate(ddot_f_expression)
    dt_update(f, dot_f, dt/2)
    dt_update(dot_f, ddot_f_function, dt/2)
    
    ddot_f_function.interpolate(ddot_f_expression)
    dt_update(f, dot_f, dt/2)
    dt_update(f, ddot_f_function, dt**2/4)
    dt_update(dot_f, ddot_f_function, dt/2)
    
def second_order_rk4(f, dot_f, ddot_f_function, ddot_f_expression, dt):
    """
    Schema de runge kutta d'ordre 4, pour les EDO d'ordre 2.
    
    Parameters
    ----------
    f : Function, fonction à actualiser.
    dot_f : Function, dérivée temporelle de f, vit dans le même espace fonctionnel que f.
    ddot_f_function : Function, dérivée temporelle seconde de f, 
                                vit dans le même espace fonctionnel que f.
    ddot_f_expression : Expression, expression de la dérivée temporelle seconde de f.
    dt : float, pas de temps.
    """
    prev_f = f.copy()
    prev_dot_f = dot_f.copy()
    V_f = f.function_space
    #Les ddot_f_function correpondent aux k de la notice
    ddot_f_function_1 = Function(V_f)
    ddot_f_function_2 = Function(V_f)
    ddot_f_function_3 = Function(V_f)
    ddot_f_function_4 = Function(V_f)
    
    ddot_f_function_1.interpolate(ddot_f_expression)
    dt_update(f, dot_f, dt/2)
    dt_update(dot_f, ddot_f_function_1, dt/2)
    
    ddot_f_function_2.interpolate(ddot_f_expression)
    dt_update(f, ddot_f_function_2, dt**2/4)
    petsc_assign(dot_f, dt_update(prev_dot_f, ddot_f_function_2, dt/2, new_vec = True))
    
    ddot_f_function_3.interpolate(ddot_f_expression)
    petsc_assign(f, dt_update(prev_f, prev_dot_f, dt, new_vec = True))
    dt_update(f, ddot_f_function_2, dt**2 / 2)
    petsc_assign(dot_f, dt_update(prev_dot_f, ddot_f_function_3, dt, new_vec = True))
    
    ddot_f_function_4.interpolate(ddot_f_expression)
    dt_update(prev_f, prev_dot_f, dt)
    petsc_assign(f, prev_f)
    dt_update(f, ddot_f_function_1, dt**2 / 6)
    dt_update(f, ddot_f_function_2, dt**2 / 6)
    dt_update(f, ddot_f_function_3, dt**2 / 6)    
    petsc_assign(dot_f, prev_dot_f)
    dt_update(dot_f, ddot_f_function_1, dt / 6)
    dt_update(dot_f, ddot_f_function_2, dt / 3)
    dt_update(dot_f, ddot_f_function_3, dt / 3)
    dt_update(dot_f, ddot_f_function_4, dt / 6)
    
    
# def second_order_rk4_GPT(f, dot_f, ddot_f_function, ddot_f_expression, dt):
#     """
#     Schema de runge kutta d'ordre 4, pour les EDO d'ordre 2, expression donée
#     par chatGPT
    
#     Parameters
#     ----------
#     f : Function, fonction à actualiser.
#     dot_f : Function, dérivée temporelle de f, vit dans le même espace fonctionnel que f.
#     ddot_f_function : Function, dérivée temporelle seconde de f, 
#                                 vit dans le même espace fonctionnel que f.
#     ddot_f_expression : Expression, expression de la dérivée temporelle seconde de f.
#     dt : float, pas de temps.
#     """

#     V_f = f.function_space
#     #Les ddot_f_function correpondent aux k de la notice
#     ddot_f_function_1 = Function(V_f)
#     ddot_f_function_2 = Function(V_f)
#     ddot_f_function_3 = Function(V_f)
#     ddot_f_function_4 = Function(V_f)
    
#     prev_f = f.copy()
#     prev_dot_f = dot_f.copy()
#     ddot_f_function_1.interpolate(ddot_f_expression)
    
#     dt_update(f, dot_f, dt/2)
#     dt_update(dot_f, ddot_f_function_1, dt/2)
#     dot_f_function_2 = dot_f.copy()
#     ddot_f_function_2.interpolate(ddot_f_expression)
    
#     dot_f_function_3 = dt_update(prev_dot_f, ddot_f_function_2, dt/2, new_vec = True)
#     petsc_assign(dot_f, dot_f_function_3)
#     petsc_assign(f, dt_update(prev_f, dot_f_function_2, dt/2, new_vec = True))    
#     ddot_f_function_3.interpolate(ddot_f_expression)
    
#     dot_f_function_4 = dt_update(prev_dot_f, ddot_f_function_3, dt, new_vec = True)
#     petsc_assign(dot_f, dot_f_function_4)
#     petsc_assign(f, dt_update(prev_f, dot_f_function_3, dt, new_vec = True))    
#     ddot_f_function_4.interpolate(ddot_f_expression)
    
#     #Phase de mise à jour
#     petsc_assign(f, prev_f)
#     dt_update(f, prev_dot_f, dt / 6)
#     dt_update(f, dot_f_function_2, dt / 3)
#     dt_update(f, dot_f_function_3, dt / 3)
#     dt_update(f, dot_f_function_4, dt / 6)    
#     petsc_assign(dot_f, prev_dot_f)
#     dt_update(dot_f, ddot_f_function_1, dt / 6)
#     dt_update(dot_f, ddot_f_function_2, dt / 3)
#     dt_update(dot_f, ddot_f_function_3, dt / 3)
#     dt_update(dot_f, ddot_f_function_4, dt / 6)   