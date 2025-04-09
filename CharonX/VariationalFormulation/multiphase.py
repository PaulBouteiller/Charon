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
Created on Thu Mar 24 09:54:52 2022

@author: bouteillerp
"""
from dolfinx.fem import Function, Expression
from ufl import exp
from ..utils.interpolation import interpolate_multiple

class Multiphase:
    def __init__(self, nb_phase, quadrature):
        """
        Définition d'un objet de la classe multiphase. Le nombre de phase à 
        l'étude est déterminé par la taille de la liste des matériaux donnée
        en arguments. Par défaut, les concentrations des différentes phases
        restent fixées et les matériaux sont considérés non réactif.

        Parameters
        ----------
        nb_phase : Int, nombre de phase à l'étude.
        """
        self.multiphase_evolution = [False] * nb_phase
        self.explosive = False
        self.nb_phase = nb_phase
        self.set_multiphase_function(quadrature)
        
    def set_multiphase_function(self, quadrature):  
        self.V_c = quadrature.quadrature_space(["Scalar"])
        self.inf_c = Function(self.V_c)
        self.max_c = Function(self.V_c)
        self.max_c.x.petsc_vec.set(1.)
        self.inf_c.x.petsc_vec.set(0.)
        self.c = [Function(self.V_c, name="Current_concentration") for i in range(self.nb_phase)]
    
    def set_multiphase(self, expression_list):
        """
        Définit les concentrations des différents composants.

        Parameters
        ----------
        expression_list : Liste de concentrations initiales pour chaque phase.
        """
        interpolate_multiple(self.c, expression_list)
            
    def set_two_phase_explosive(self, E_vol):
        """
        Défini la variation d'énergie à injecter dans l'équation de la chaleur
        consécutif à une variation de la concentration de la phase numérotée 1

        Parameters
        ----------
        E_vol : Float, Energie volumique libérée par l'explosif.
        """
        self.c_old = [c.copy() for c in self.c]   
        self.Delta_e_vol_chim = (self.c[1] - self.c_old[1]) * E_vol
        
    def set_evolution_parameters(self, params):
        """
        Méthode unifiée pour configurer l'évolution des phases.
        
        Parameters
        ----------
        params : dict, dictionnaire des paramètres d'évolution
        """
        if params.get("type") == "KJMA":
            self._set_KJMA_kinetic(
                params["rho"], 
                params["T"], 
                params["melt_param"], 
                params["gamma_param"], 
                params["alpha_param"],
                params["tau_param"]
            )
        elif params.get("type") == "smooth_instantaneous":
            self._set_smooth_instantaneous_evolution(
                params["rho"],
                params["rholim"],
                params["width"]
            )
        else:
            raise ValueError(f"Unknown evolution type: {params.get('type')}")
    
    def _set_KJMA_kinetic(self, rho, T, melt_param, gamma_param, alpha_param, tau_param):
        """
        Initialise les fonctions nécessaires à la définition du modèle de cinétique
        de KJMA.

        Parameters
        ----------
        rho : Float ou Function, champ de masse volumique actuelle.
        T : Function, champ de température actuelle.
        melt_param : List, liste contenant les deux flottants nécessaires
                            à la définition de la température de fusion.
        gamma_param : Float, vitesse d'interface liquide solide en fonction de la température.
        alpha_param : List, liste contenant les trois paramètres pour le champ alpha,
                            taux de germination.
        tau_param : List, liste contenant les trois paramètres pour le champ tau, temps d'induction.
        """
        T_fusion = melt_param[0] * rho ** melt_param[1]
        self.gamma = - gamma_param * (T - T_fusion)
        self.alpha = exp(alpha_param[0] + alpha_param[1] * rho + alpha_param[2] * T)
        self.tau = exp(tau_param[0] + tau_param[1] * rho + tau_param[2] * T)
        self.U = Function(self.V_c)
        self.G = Function(self.V_c)
        self.J = Function(self.V_c)
        
        
    def _set_smooth_instantaneous_evolution(self, rho, rholim, width):
        """
        Crée une fonction d'interpolation lisse entre 0 et 1 autour de x0,
        avec une largeur donnée pour passer de 0.01 à 0.99.
        
        Paramètres:
        x: float ou np.array - Point(s) où évaluer la fonction
        x0: float - Point central de la transition
        width: float - Largeur sur laquelle la fonction passe de 0.01 à 0.99
        
        Retourne:
        float ou np.array - Valeur(s) de la fonction entre 0 et 1
        """
        k = 9.19 / width  # Relation dérivée de 2*ln(99)/k = width
        c_expr = 1 / (1 + exp(-k * (rho - rholim)))
        self.c_expr = Expression(c_expr, self.V_c.element.interpolation_points())