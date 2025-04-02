"""
Created on Thu Mar 24 09:54:52 2022

@author: bouteillerp
"""
from dolfinx.fem import Function, Expression
from ufl import exp
from ..utils.generic_functions import petsc_assign

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
        self.multiphase_evolution = [False for i in range(nb_phase)]
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
        Définition des concentrations des différents composants

        Parameters
        ----------
    
        expression : Float ou Expression, concentration spatiale initiale de chacune des phases.
        """

        for i in range(self.nb_phase):
            if isinstance(expression_list[i], float):
                self.c[i].x.array[:] = expression_list[i]
            elif isinstance(expression_list[i], Expression):
                self.c[i].interpolate(expression_list[i])
            elif isinstance(expression_list[i], Function):
                petsc_assign(self.c[i], expression_list[i])
            else:
                raise ValueError("Concentration must be set")
            
    def set_two_phase_explosive(self, E_vol):
        """
        Défini la variation d'énergie à injecter dans l'équation de la chaleur
        consécutif à une variation de la concentration de la phase numérotée 1

        Parameters
        ----------
        E_vol : Float, Energie volumique libérée par l'explosif.
        """
        self.c_old = [self.c[i].copy() for i in range(self.nb_phase)]      
        self.Delta_e_vol_chim = (self.c[1] - self.c_old[1]) * E_vol
        
    def set_KJMA_kinetic(self, rho, T, melt_param, gamma_param, alpha_param, tau_param):
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
        
        
    def set_smooth_instantaneous_evolution(self, rho, rholim, width):
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