#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 09:40:36 2025

@author: bouteillerp
"""
from ..utils.generic_functions import smooth_shifted_heaviside

    def _set_smooth_instantaneous_evolution(self, rho, rholim, width):
        """
        Create a smooth interpolation function between 0 and 1 around rholim.
        
        This function creates a smooth transition for phase concentration
        based on density, using a logistic function.
        
        Parameters
        ----------
        rho : float or numpy.array Density field
        rholim : float Central point of the transition
        width : float Width over which the function changes from 0.01 to 0.99
        """
        c_expr = smooth_shifted_heaviside(rho, rholim, width)
        self.c_expr = Expression(c_expr, self.V_c.element.interpolation_points())