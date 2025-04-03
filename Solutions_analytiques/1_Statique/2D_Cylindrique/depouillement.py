#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 15:38:30 2024

@author: bouteillerp
"""

from pandas import read_csv
import matplotlib.pyplot as plt
import sys
sys.path.append("../../")
from Generic_isotropic_material import *
from numpy import array
def validation_analytique(Pint, Pext, Rint, Rext):
    u_csv = read_csv("Cylindre_axi-results/U.csv")
    resultat = [u_csv[colonne].to_numpy() for colonne in u_csv.columns]
    r_result = resultat[0]
    solution_numerique = -resultat[-2]
    
    len_vec = len(r_result)
    A = (Pint * Rint**2 - Pext * Rext**2) / (Rext**2 - Rint**2)
    B = (Pint - Pext) / (Rext**2 - Rint**2) * Rint**2 * Rext**2
    C = 0
    a = (1-nu)/E * A - nu /E * C
    b = B / (2 * mu)
       
    def ur(r):
        return  a * r + b / r
    solution_analytique = array([ur(x) for x in r_result])
    # On calcule la différence entre les deux courbes
    diff_tot = solution_analytique - solution_numerique
    # Puis on réalise une sorte d'intégration discrète
    integrale_discrete = sum(abs(diff_tot[j]) for j in range(len_vec))/sum(abs(solution_analytique[j]) for j in range(len_vec))
    print("La difference est de", integrale_discrete)
    # assert integrale_discrete < 1e-3, "Cylindrical static compression fail"
    if __name__ == "__main__": 
        plt.plot(r_result, solution_analytique, linestyle = "--", color = "red")
        plt.scatter(r_result, solution_numerique, marker = "x", color = "blue")
        
        plt.xlim(Rint, Rext)
        plt.xlabel(r"$r$", size = 18)
        plt.ylabel(r"Déplacement radial", size = 18)
        plt.savefig("../../../Notice/fig/2D_Cylindric_compression.pdf", bbox_inches = 'tight')
        # plt.show()