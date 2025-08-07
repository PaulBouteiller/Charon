#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 09:42:56 2025

@author: bouteillerp
"""

WGT: A mesoscale-informed reactive burn model. Journal of Applied Physics
            
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
        
