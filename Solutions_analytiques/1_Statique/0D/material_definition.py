"""
Created on Thu Feb  8 14:04:05 2024

@author: bouteillerp
"""

from CharonX import Material
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
        
    elif eos_type=="MACAW":
        rho0 = 1.6
        v0 = 1/rho0
        A = 0.127
        B = 4.6
        C = 2.4
        vinf = 0.56
        eta = v0/vinf
        theta0 = 80.6

        a0 = 2.0
        n = -1.4

        gamma0 = 0.2
        gammainf = 0.37
        m = 7.1
        dico_eos = {"A": A, "B" : B, "C": C, "eta" : eta, "theta0" : theta0,
                    "a0": a0, "n" : n, "m" : m, "Gamma0" : gamma0, "Gammainf" : gammainf}
        
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