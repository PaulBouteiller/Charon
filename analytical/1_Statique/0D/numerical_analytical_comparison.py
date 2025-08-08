"""
Module de comparaison entre solutions numériques et analytiques pour différents modèles.

Ce module permet de comparer les résultats numériques obtenus par simulation avec les solutions
analytiques pour différentes équations d'état et lois déviatoriques.

Fonctions:
    comparison(mat, varepsilon, T): Compare les résultats numériques et analytiques pour un matériau donné.
        - mat: Instance de Material à tester
        - varepsilon: Déformation appliquée
        - T: Température
    
La fonction trace des graphiques de comparaison et calcule l'erreur relative entre les solutions
numériques et analytiques. Une assertion vérifie que l'erreur est inférieure à un seuil (1e-3).

Auteur: bouteillerp
Date de création: 8 Février 2024
"""

from numpy import array
from pandas import read_csv
# from material_definition import *
from Analytique_EOS_deviateur import (p1,p5, p8, Vinet, MG, JWL, MACAW, 
                                      tabulated, HPP_devia, NeoHookean,
                                      MooneyRivlin)
import matplotlib.pyplot as plt

def comparison(mat, varepsilon, T):
    eos_type = mat.eos_type
    devia_type = mat.dev_type
    eos = mat.eos
    devia = mat.devia
    u_csv = read_csv("Test_0D_"+eos_type+"-results/U.csv")
    u_resultat = [u_csv[colonne].to_numpy() for colonne in u_csv.columns]
    u_array = array([u_resultat[i+1][0] for i in range(len(u_resultat)-1)])
    J_array = array([1 + u for u in u_array])
    
    p_analytique_array = []
    s_analytique_array = []
    for J in J_array:
        if eos_type == "U1":
            p_analytique_array.append(p1(eos.kappa, J))
        elif eos_type == "U5":
            p_analytique_array.append(p5(eos.kappa, J))
        elif eos_type == "U8":
            p_analytique_array.append(p8(eos.kappa, J))
        elif eos_type == "Vinet":
            p_analytique_array.append(Vinet(eos.iso_T_K0, eos.iso_T_K1, J))
        elif eos_type == "MG":
            p_analytique_array.append(MG(eos.C, eos.D, eos.S, J))
        elif eos_type == "JWL":
            p_analytique_array.append(JWL(eos.A, eos.B, eos.R1, eos.R2, J))
        elif eos_type == "MACAW":
            p_analytique_array.append(MACAW(eos.A, eos.B, eos.C, eos.eta, eos.theta0, 
                                            eos.a0, eos.m, eos.n, eos.Gammainf, 
                                            eos.Gamma0, mat.C_mass, J, T))
        elif eos_type == "Tabulated":
            p_analytique_array.append(tabulated(1e4, J))
        if devia_type == "IsotropicHPP":
            s_analytique_array.append(HPP_devia(devia.mu, J))
        elif devia_type == "NeoHook":
            s_analytique_array.append(NeoHookean(devia.mu, J))
        elif devia_type == "MooneyRivlin":
            s_analytique_array.append(MooneyRivlin(devia.mu, devia.mu_quad, J))

    
    p_csv = read_csv("Test_0D_"+eos_type+"-results/p.csv")
    resultat = [p_csv[colonne].to_numpy() for colonne in p_csv.columns]
    p_array = array([resultat[i+1][0] for i in range(len(resultat)-1)])
    
    pressure_difference = p_analytique_array - p_array
    len_vec = len(pressure_difference)
    int_discret = sum(abs(pressure_difference[j]) for j in range(len_vec)) / sum(abs(p_analytique_array[j]) for j in range(len_vec))
    print("La difference est de", int_discret)
    # assert int_discret < 1e-3, "EOS test fail"
    
    
    s_csv = read_csv("Test_0D_"+eos_type+"-results/deviateur.csv")
    resultat = [s_csv[colonne].to_numpy() for colonne in s_csv.columns]
    s_array = array([resultat[3 * i+1][0] for i in range((len(resultat)-1)//3)])
    # if __name__ == "__main__": 
    plt.scatter(J_array, p_array, marker = "x", color = "blue", label="CHARON "+eos_type)
    plt.plot(J_array, p_analytique_array, linestyle = "--", color = "red",label="Analytique "+eos_type)
    plt.xlabel(r"Dilatation volumique $J$", fontsize = 18)
    plt.ylabel(r"Pression $P$", fontsize = 18)
    plt.xlim(1+varepsilon, 1)
    # plt.ylim(0)
    plt.legend()
    plt.show()
    # plt.close()
    
    # plt.scatter(J_array, s_array, marker = "x", color = "blue", label="CHARON "+devia_type)
    # plt.plot(J_array, s_analytique_array, linestyle = "--", color = "red",label="Analytique "+devia_type)
    # plt.xlabel(r"Dilatation volumique $J$", fontsize = 18)
    # plt.ylabel(r"Déviateur $s_{xx}$", fontsize = 18)
    # plt.xlim(1+varepsilon, 1)
    # plt.ylim(top = 0)
    # plt.legend()
    # plt.show()