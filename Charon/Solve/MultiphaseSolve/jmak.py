#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 09:37:55 2025

@author: bouteillerp
"""
    def _set_KJMA_kinetic(self, rho, T, melt_param, gamma_param, alpha_param, tau_param):
        """
        Initialize functions needed for the KJMA kinetic model.
        
        This model describes phase transformations based on nucleation
        and growth processes, often used for crystallization phenomena.
        
        Parameters
        ----------
        rho : float or Function Current density field
        T : Function Current temperature field
        melt_param : list List containing two floats needed for the melting temperature definition
        gamma_param : float Speed of the liquid-solid interface as a function of temperature
        alpha_param : list List containing three parameters for the alpha field (nucleation rate)
        tau_param : list List containing three parameters for the tau field (induction time)
        """
        T_fusion = melt_param[0] * rho ** melt_param[1]
        self.gamma = - gamma_param * (T - T_fusion)
        self.alpha = exp(alpha_param[0] + alpha_param[1] * rho + alpha_param[2] * T)
        self.tau = exp(tau_param[0] + tau_param[1] * rho + tau_param[2] * T)
        self.U = Function(self.V_c)
        self.G = Function(self.V_c)
        self.J = Function(self.V_c)

    def set_evolution_parameters(self, params):
        """
        Unified method for configuring phase evolution.
        
        Parameters
        ----------
        params : dict Dictionary of evolution parameters
            
        Raises
        ------
        ValueError If an unknown evolution type is specified
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