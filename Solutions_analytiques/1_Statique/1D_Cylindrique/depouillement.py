"""
Created on Mon Sep  2 14:40:23 2024

@author: bouteillerp
"""

import matplotlib.pyplot as plt
from numpy import linspace, array
from pandas import read_csv
from materiau import *

def validation_analytique(Pint, Pext, Rint, Rext):
    u_csv = read_csv("Compression_cylindrique_1D-results/U.csv")
    resultat = [u_csv[colonne].to_numpy() for colonne in u_csv.columns]
    r_result = resultat[0]
    solution_numerique = -resultat[-1]
    
    
    mu = E / 2. / (1 + nu)
    len_vec = len(solution_numerique)
    A = (Pint * Rint**2 - Pext * Rext**2) / (Rext**2 - Rint**2)
    B = (Pint - Pext) / (Rext**2 - Rint**2) * Rint**2 * Rext**2
    C=2 * nu * A
    a = (1-nu)/E * A - nu /E * C
    b = B / (2 * mu)
       
    def ur(r):
        return  a * r + b / r
    pas_espace = linspace(Rint, Rext, len_vec)
    solution_analytique = array([ur(x) for x in pas_espace])
    # On calcule la différence entre les deux courbes
    vecteur_difference = solution_analytique - solution_numerique
    # Puis on réalise une sorte d'intégration discrète
    integrale_discrete = sum(abs(vecteur_difference[j]) for j in range(len_vec)) / sum(abs(solution_analytique[j]) for j in range(len_vec))
    print("La difference est de", integrale_discrete)
    assert integrale_discrete < 1e-3, "Cylindrical static compression fail"
    if __name__ == "__main__": 
        plt.plot(pas_espace, solution_analytique, linestyle = "--", color = "red", label = "Analytique")
        plt.scatter(pas_espace, solution_numerique, marker = "x", color = "blue", label = "CHARON")
        
        plt.xlim(Rint, Rext)
        plt.xlabel(r"$r$ (mm)", size = 18)
        plt.ylabel(r"Déplacement radial (mm)", size = 18)
        plt.legend()
        plt.savefig("../../../Notice/fig/Cylindric_compression.pdf", bbox_inches = 'tight')
        plt.show()
    
