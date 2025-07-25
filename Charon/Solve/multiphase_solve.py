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
Created on Wed Apr 12 13:57:33 2023

@author: bouteillerp
"""
from ..utils.generic_functions import Heav, ppart
from ..utils.petsc_operations import (dt_update, set_correction, 
                                 petsc_assign , higher_order_dt_update)
from ufl import tanh, sqrt, exp
from dolfinx.fem import Function, Expression
from math import pi
        
class MultiphaseSolver:
    def __init__(self, multiphase_object, dt, pressure, T, material, t):
        """
        Initialise le solveur faisant varier les concentrations.

        Parameters
        ----------
        multiphase_object : Objet de la classe multiphase.
        dt : Float, pas de temps.
        pressure : Expression, pression actuelle.
        T : Function, température actuelle.
        material : Objet de la classe material.
        """
        self.dt = dt
        self.mult = multiphase_object
        self.set_c_evolution(material)
        self.t = t

        self.material = material
        self.evol_type = multiphase_object.evolution_type
        if self.evol_type == "ForestFire":
            self.Forest_fire(pressure)
        elif self.evol_type == "Arhenius":
            self.Arhenius(T)
        elif self.evol_type == "wgt":
            self.wgt(T)
        elif self.evol_type == "KJMA":
            self.set_KJMA(self.mult, t)
        elif self.evol_type == "Smooth_Instantaneous":
            self.dot_c = None
        else:
            raise ValueError("Unknown law")
        self.init_dot_c_expression()
        if self.mult.explosive:
            self.set_c_old()
            
        
    def set_c_evolution(self, material):
        self.c_list = []
        self.mat_list = []
        for i in range(len(self.mult.c)):
            if self.mult.multiphase_evolution[i]:
                self.c_list.append(self.mult.c[i])
                self.mat_list.append(material[i])
                
        self.nb_evolution = len(self.c_list)
        
    def set_c_old(self):
        self.c_old_list = []
        for i in range(len(self.mult.c)):
            if self.mult.multiphase_evolution[i]:
                self.c_old_list.append(self.mult.c_old[i])
                
        self.nb_evolution = len(self.c_list)
        
    def Forest_fire(self, p, a=59467, b = 3.9e-5, cn = 2.5676, P0=3.5e9, Ps=1e8):
        """
        Loi d'évolution de type Forest Fire

        Parameters
        ----------
        p : Function, pression actuelle.
        a : Float, optional
            DESCRIPTION. The default is 59467.
        b : Float, optional
            DESCRIPTION. The default is 3.9e-5.
        cn : Float, optional
            DESCRIPTION. The default is 2.5676.
        P0 : Float, optional
            DESCRIPTION. The default is 3.5e3.
        Ps : Float, optional
            DESCRIPTION. The default is 1e2.
        """
        self.dot_c = (1 - self.c_list[1]) * a * ((abs(p/P0))**cn + b * (abs(p/P0))**7) * Heav(p - Ps)

    def Arhenius(self, T):
        """
        Loi d'évolution de type Arhénius à 4 matériaux. 

        Parameters
        ----------
        T : Function, champ de température actuelle.
        """
        n = len(self.c_list)
        R = 8.3144
        self.exp_list = []
        self.dot_c = [0 for i in range(n)]
        for i in range(n-1):
            self.exp_list.append(self.mat_list[i].kin_pref * exp(-self.mat_list[i].e_activation / (R * T)))
        self.exp_list.append(0)
        self.dot_c[0] = - self.c_list[0] * self.exp_list[0]
        for i in range(1, n):
            self.dot_c[i] = self.c_list[i-1] * self.exp_list[i-1] - \
                            self.c_list[i] * self.exp_list[i]
            
    def wgt(self, T, *args):
        """
        Loi d'évolution de type wgt

        Parameters
        ----------
        T : Function, champ de température actuelle.
        """
        SG1 = 25
        SG2 = 0.92
        TALL = 400
        TADIM = 1035
        KDG = 1.6e7
        NDG = 2
        KB = 2.8e5
        NB = 1
        
        sf2 = 1./2 * (1 - tanh(SG1 * (self.c_list[1] - SG2)))
        r_t = (T - TALL) / TADIM
        rate_2 = KDG * ppart(r_t)**NDG * (1 - self.c_list[1])
        rate_3 = KB * ppart(r_t) **NB * sqrt(1 - self.c_list[1])
        self.dot_c =  ppart(sf2 * rate_2 + (1-sf2) * rate_3)
        
    def set_KJMA(self, mult, t, n_lim = 100):
        """
        Définition des dérivées temporelles des fonctions intervenants
        dans le modèle de cinétique chimique KJMA
        """
        interp = mult.V_c.element.interpolation_points()
        #Transient alpha
        S = 2
        for i in range(1, n_lim):
            S += 2 * ((-1)**i * exp(-i**2 * t / mult.tau))
        mult.alpha *= S
        # mult.alpha_expr = Expression(mult.alpha, interp)
        
        #Création des fonctions et des expression des dérivées temporelles
        #successives de U
        self.dot_U = Function(mult.V_c)
        self.ddot_U = Function(mult.V_c)
        self.dddot_U = Function(mult.V_c)
        
        self.dot_U_expr = Expression(2 * mult.gamma * mult.G, interp)
        self.ddot_U_expr = Expression(2 * mult.gamma**2 * mult.J, interp)
        self.dddot_U_expr = Expression(mult.gamma**2 * mult.alpha, interp)
        
        #Création des fonctions et des expression des dérivées temporelles
        #successives de G   
        self.dot_G = Function(mult.V_c)
        self.ddot_G = Function(mult.V_c)
        
        self.dot_G_expr = Expression(mult.gamma * mult.J, interp)
        self.ddot_G_expr = Expression(mult.gamma * mult.alpha, interp)
        
        #Création de la fonction et de l'expression de la dérivée temporelle
        #première de J   
        self.dot_J = Function(mult.V_c)
        self.dot_J_expr = Expression(mult.alpha, interp)

        self.dot_c_expression = Expression(4 * pi * (1 - self.c_list[1])  * mult.gamma * mult.U, interp)
        self.dot_c = Function(mult.V_c)
        
    def init_dot_c_expression(self):
        """
        Initialisation des expressions qui seront utilisées pour actualiser les
        concentrations
        """
        V_c = self.mult.V_c
        if isinstance(self.dot_c, list):
            self.dot_c_list = [Function(V_c) for i in range(len(self.dot_c))]
            self.dot_c_expression_list = [Expression(self.dot_c[i], V_c.element.interpolation_points()) for i in range(len(self.dot_c))]
        else:
            pass
        ############## A CORRIGER
            # self.dot_c = Function(V_c)
            # self.dot_c_expression = Expression(self.dot_c, V_c.element.interpolation_points())
                
    def solve(self):
        """
        Actualisation des champs de concentrations et d'eventuels champs auxiliaires
        """
        if isinstance(self.dot_c, list):
            for i in range(len(self.dot_c)):
                self.dot_c_list[i].interpolate(self.dot_c_expression_list[i])
            for i in range(len(self.dot_c)):
                dt_update(self.c_list[i], self.dot_c_list[i], self.dt)
        elif self.dot_c is None:  
            self.instantaneous_evolution()            
        else:
            self.two_phase_evolution()
        for i in range(len(self.c_list)):
            set_correction(self.c_list[i], self.mult.inf_c, self.mult.max_c)
        self.auxiliary_field_evolution()
        
    def two_phase_evolution(self):
        """
        Mise à jour des concentrations dans un modèle à deux phases.
        """
        self.dot_c.interpolate(self.dot_c_expression)
        dt_update(self.c_list[0], self.dot_c, -self.dt)
        dt_update(self.c_list[1], self.dot_c, self.dt)
        
    def instantaneous_evolution(self):
        """
        Mise à jour des concentrations dans un modèle à deux phases.
        """
        self.c_list[0].interpolate(self.mult.c_expr)
        self.c_list[1].x.array[:] = 1 - self.c_list[0].x.array
        
        
    def auxiliary_field_evolution(self):
        """
        Mise à jour des fonctions intervenant dans le modèle de cinétique chimique KJMA
        """
        if self.evol_type == "KJMA":  
            self.dot_U.interpolate(self.dot_U_expr)
            self.ddot_U.interpolate(self.ddot_U_expr)
            self.dddot_U.interpolate(self.dddot_U_expr)
            self.dot_G.interpolate(self.dot_G_expr)
            self.ddot_G.interpolate(self.ddot_G_expr)
            self.dot_J.interpolate(self.dot_J_expr)
            higher_order_dt_update(self.mult.U, [self.dot_U, self.ddot_U, self.dddot_U], self.dt)
            higher_order_dt_update(self.mult.G, [self.dot_G, self.ddot_G], self.dt)
            dt_update(self.mult.J, self.dot_J, self.dt)
        else:
            pass
        
    def update_c_old(self):
        """
        Mise à jour des concentrations
        """
        for i in range(self.nb_evolution):
            petsc_assign(self.c_old_list[i], self.c_list[i])