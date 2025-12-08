"""
Module de définition des matériaux pour différentes équations d'état et lois de comportement.

Ce module permet de créer des instances de matériaux (Material) en définissant leurs propriétés
selon le type d'équation d'état et de loi déviatorique choisis.

Fonctions:
    set_material(eos_type, devia_type): Crée et retourne un objet Material configuré selon les paramètres spécifiés.
        - eos_type: Type d'équation d'état (U1, U5, U8, Tabulated, Vinet, MG, JWL, MACAW)
        - devia_type: Type de loi déviatorique (None, IsotropicHPP, NeoHook, MooneyRivlin)

Remarque: Les valeurs des paramètres sont définies dans la fonction pour chaque type d'équation
d'état et de loi déviatorique.

Auteur: bouteillerp
Date de création: 8 Février 2024
"""

from Charon import Material
from Analytique_EOS_deviateur import *
from pandas import read_csv

def set_material(eos_type, devia_type):
    if eos_type in ["U1", "U5", "U8"]:
        kappa = 10000
        dico_eos = {"kappa" : kappa, "alpha" : 1}
        
    elif eos_type == "Tabulated":
        kappa = 10000
        df = read_csv('Exemple_tabule.csv', index_col='T/J')
        dico_eos = {"c0": 1e2, "Dataframe" : df}
        
    elif eos_type=="Vinet":
        iso_T_K0 = 10000
        T_dep_K0 = 0
        iso_T_K1 = 2
        T_dep_K1 = 0
        dico_eos = {"iso_T_K0": iso_T_K0, "T_dep_K0" : T_dep_K0, "iso_T_K1": iso_T_K1, "T_dep_K1" : T_dep_K1}
        
    elif eos_type=="MG":
        C = 10000
        D = 10
        S = 100
        gamma0 = 0
        dico_eos = {"C": C, "D" : D, "S": S, "gamma0" : gamma0}
    elif eos_type=="JWL":
        A = 40000
        R1 = 1.5
        B = -13046
        R2 = 0.38
        w = 0
        dico_eos = {"A": A, "B" : B, "R1": R1, "R2" : R2, "w":w}
        
    elif eos_type == "MACAW":
        # PETN parameters from Table III (Lozano et al., J. Appl. Phys. 2023)
        
        # Cold curve parameters
        A = 0.10054        # [GPa]
        B = 0.66667        # [-]
        C = 10.466         # [-]
        
        # Reference state
        rho0 = 1.774       # [g/cm³] - Table II experimental ambient density
        
        # Thermal model parameters
        vinf = 0.43752     # [cm³/g]
        theta0 = 218.95    # [K]
        a0 = 1.9477        # [-]
        m = 3.1674         # [-]
        n = 3.1674         # [-]
        
        # Grüneisen parameters
        Gamma0 = 1./84     # [-] (~0.0119)
        Gammainf = 2./3    # [-] (~0.6667)
        
        # Specific heat capacity
        cvinf = 2.2881e-3  # [kJ/(g·K)]
        
        dico_eos = {
            "A": A, "B": B, "C": C,
            "rho0": rho0, "vinf": vinf,
            "theta0": theta0, "a0": a0,
            "m": m, "n": n,
            "Gamma0": Gamma0, "Gammainf": Gammainf,
            "cvinf": cvinf
        }
    if eos_type == "MACAW":
        C_mass = 1e-2
    else:
        rho0 = 1
        C_mass = 1
    
    
    if devia_type == None:
        dico_devia = {}
    elif devia_type == "IsotropicHPP":
        mu = 7e3
        dico_devia = {"mu" : mu}
    elif devia_type == "NeoHook":
        mu = 7e3
        dico_devia = {"mu" : mu}
        
    elif devia_type == "MooneyRivlin":
        mu = 6e3
        mu_quad = 1e3
        dico_devia = {"mu" : mu, "mu_quad" : mu_quad}
        
    return Material(rho0, C_mass, eos_type, devia_type, dico_eos, dico_devia)