"""
Created on Thu Jun  6 10:00:48 2024

@author: bouteillerp
"""
from CharonX import *
from pandas import read_csv
try:
    from jax.numpy import searchsorted, clip, array
    from jax import vmap, jit
except Exception:
    print("JAX has not been loaded therefore tabulated law cannot be used")
eos_type = "U1"
# dev_type = "Hypoelastic"
dev_type = None

###### Modèle matériau ######
def set_material():
    rho = 7.8e-3
    C_mass = 500
    T0 = 0.
    if eos_type == "U1":
        kappa = 175e3
        alpha=12e-6
        rigi_eos = kappa
        dico_eos = {"kappa" : kappa, "alpha" : alpha}
        
    elif eos_type == "Tabulated":
        rigi_eos = 175e3
        df = read_csv('Exemple_tabule.csv', index_col='T/J')

        # T_list, J_list, P_list = Dataframe_to_array(df)

        def tabulated_P(J):
            return -rigi_eos * (J - 1)
        T_list = array([0., 1000])
        J_list = array([0.5, 1., 5.])
        P_list = array([[tabulated_P(J) for J in J_list] for T in T_list])
        
        dico_eos = {"c0": 4736, "T" : T_list, "J" : J_list, "P" : P_list}
        
        

    elif eos_type == "Vinet":
        iso_T_K0 = 200e3
        T_dep_K0 = 200e1
        iso_T_K1 = 8
        T_dep_K1 = 2
        dico_eos = {"iso_T_K0" : iso_T_K0, "T_dep_K0" : T_dep_K0, 
                    "iso_T_K1" : iso_T_K1, "T_dep_K1" : T_dep_K1}
        rigi_eos = iso_T_K0 + T_dep_K0 * T0
        
    elif eos_type == "JWL":
        def w_adjust(A, B, R1, R2, rho0, C):
            return -(A * exp(-R1) + B * exp(-R2)) / (rho0 * C * T0)
        A = 500e2
        B = 500e2
        R1 = 1
        R2 = 2
        w = w_adjust(A, B, R1, R2, rho, C_mass)
        dico_eos = {"A": A, "B": B, "R1" : R1, "R2" : R2, "w" : w}
        rigi_eos = A*R1*exp(-R1)+B*R2*exp(-R2)
        
    elif eos_type == "MACAW":
        A = 1e3
        B = 4.6
        C = 2.4
        
        a0 = 2
        vinf = 5e2
        v0 = 1/rho
        eta = v0/vinf
        n = -1.4
        theta0 = 80
        
        gamma0 = 0.2
        gammainf = 0.37
        m = 7.
        dico_eos = {"A": A, "B" : B, "C": C, "eta" : eta, "theta0" : theta0,
                    "a0": a0, "n" : n, "m" : m, "Gamma0" : gamma0, "Gammainf" : gammainf}

        rigi_eos = A * (B - 1./2 * C + (B + C)**2)
        
    if dev_type == None:
        dico_devia = {}
        rigi_dev = 0

    elif dev_type in ["IsotropicHPP", "Hypoelastic"]:
        mu = 80769
        rigi_dev = 4 * mu /3 
        dico_devia = {"mu": mu}
    
    mat = Material(rho, C_mass, eos_type, dev_type, dico_eos, dico_devia)
    wave_speed = ((rigi_eos+rigi_dev)/rho)**(1./2)
    isotherm = True
    print("La vitesse estimée des ondes élastiques est", wave_speed)
    return mat, wave_speed, isotherm, T0