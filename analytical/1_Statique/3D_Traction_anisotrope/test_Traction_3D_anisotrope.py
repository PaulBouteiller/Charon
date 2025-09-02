"""
Test de traction 3D sur un matériau orthotrope selon différentes directions.

Ce script simule des essais de traction uniaxiale sur un cube 3D composé d'un matériau
orthotrope selon trois directions principales (fibre, normale transverse, normale hors-plan)
et compare les résultats numériques avec les solutions analytiques.

Paramètres géométriques:
    - Dimensions du cube: 1 × 1 × 1
    - Discrétisation: maillage 5×5×5

Matériau orthotrope:
    - Modules d'Young: EL=12827, ET=633, EN=1344
    - Modules de cisaillement: muLT=766, muLN=703, muTN=337
    - Coefficients de Poisson: nuLT=0.466, nuLN=0.478, nuTN=0.371
    - Équation d'état: Vinet avec kappa_eq calculé à partir de la matrice de rigidité

Tests de traction:
    - Direction "longitudinal": parallèle à la direction des fibres (EL)
    - Direction "transverse": perpendiculaire aux fibres, dans le plan (ET)
    - Direction "normal": direction normale au plan (EN)

Le script trace les courbes force-déplacement pour les trois directions et
effectue une comparaison avec les solutions analytiques basées sur les modules d'Young.
Il trace également l'évolution de la pression en fonction de la densité pour vérifier
la cohérence de l'équation d'état.

Auteur: bouteillerp
Date de création: 11 Mars 2022
"""
from Charon import Material, create_box, Tridimensional, Solve, MeshManager, build_orthotropic_stiffness, compute_bulk_modulus, build_transverse_isotropic_stiffness
from mpi4py.MPI import COMM_WORLD
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import exp
from numpy import array, loadtxt

###### Modèle mécanique ######
rho0 = 1
C_mass = 1

# Paramètres du comportement déviatorique anisotrope
EL = 12827
ET = 633
EN = 1344

muLT = 766
muLN = 703
muTN = 337

nuLT = 0.466
nuLN = 0.478
nuTN = 0.371



C = build_orthotropic_stiffness(EL, ET, EN, nuLT, nuLN, nuTN, muLT, muLN, muTN)


dev_type = "Anisotropic"
deviator_params = {"C" : C}
iso_T_K0 = compute_bulk_modulus(C)
T_dep_K0 = 0
iso_T_K1 = 1
T_dep_K1 = 0
eos_type = "Vinet"
dico_eos = {"iso_T_K0": iso_T_K0, "T_dep_K0" : T_dep_K0, "iso_T_K1": iso_T_K1, "T_dep_K1" : T_dep_K1}

Oak = Material(rho0, C_mass, eos_type, dev_type, dico_eos, deviator_params)

#%%Maillage
Longueur, Largeur, hauteur = 1, 1, 1.
Nx, Ny, Nz = 1, 1, 1
mesh = create_box(COMM_WORLD, [np.array([0, 0, 0]), 
                                   np.array([Longueur, Largeur, hauteur])],
                                  [Nx, Ny, Nz])


dictionnaire_mesh = {"tags": [1, 2, 3, 4, 5, 6],
                     "coordinate": ["x", "x", "y", "y", "z", "z"], 
                     "positions": [0, Longueur, 0, Largeur, 0, hauteur]
                     }
mesh_manager = MeshManager(mesh, dictionnaire_mesh)

#%%Chargement et paramètre du problème
eps = 0.01
Umax = eps * hauteur
chargement = {"type" : "rampe", "pente" : Umax}

dictionnaire = {"material" : Oak,
                "mesh_manager" : mesh_manager,
                "boundary_conditions": 
                    [{"component": "Ux", "tag": 1},
                     {"component": "Uy", "tag": 3},
                     {"component": "Uz", "tag": 5}
                    ],
                "analysis" : "static",
                "isotherm" : True
                }

traction = "longitudinal"    
output_name = "Traction_3D_sens_"+traction

dictionnaire_solve = {
    "Prefix" : output_name,
    "csv_output" : {"p" : True, "rho" : True}
    }
    
if traction == "longitudinal":
    dictionnaire["boundary_conditions"].append({"component": "Ux", "tag": 2, "value": chargement})
    dictionnaire_solve["csv_output"]["reaction_force"] = {"flag" : 2, "component" : "x"}
    
elif traction == "transverse":
    dictionnaire["boundary_conditions"].append({"component": "Uy", "tag": 4, "value": chargement})
    dictionnaire_solve["csv_output"]["reaction_force"] = {"flag" : 4, "component" : "y"}
    
elif traction == "normal":
    dictionnaire["boundary_conditions"].append({"component": "Uz", "tag": 6, "value": chargement})
    dictionnaire_solve["csv_output"]["reaction_force"] = {"flag" : 6, "component" : "z"}
    
pb = Tridimensional(dictionnaire)
solve_instance = Solve(pb, dictionnaire_solve, compteur=1, npas=10)
solve_instance.solve()

def force_elast(eps, essai):
    if essai == "longitudinal":
        return EL * eps * Largeur * hauteur
    elif essai == "transverse":
        return ET * eps * Longueur * hauteur
    elif essai == "normal":
        return EN * eps * Longueur * Largeur

output_folder = output_name + "-results/"
temps = loadtxt(output_folder + "export_times.csv",  delimiter=',', skiprows=1)
numerical_results = loadtxt(output_folder + "reaction_force.csv",  delimiter=',', skiprows=1)
eps_list = [eps * t for t in temps]    

solution_analytique = array([force_elast(eps, traction) for eps in eps_list])
eps_list_percent = [100 * eps for eps in eps_list]

df_p = pd.read_csv(output_folder + "p.csv")
colonnes_numpy = [df_p[colonne].to_numpy() for colonne in df_p.columns]  
p_list = [p[0] for p in colonnes_numpy[3:]]

df = pd.read_csv(output_folder + "rho.csv")
colonnes_numpy = [df[colonne].to_numpy() for colonne in df.columns]  
rho_list = [rho[0] for rho in colonnes_numpy[3:]]

plt.scatter(eps_list_percent, numerical_results, marker = "x", color = "red", label="CHARON")
plt.plot(eps_list_percent, solution_analytique, linestyle = "-", color = "black", label = "Analytique")

def Vinet(K0, K1, rho):
    J = rho0/rho
    return 3 * K0 * J**(-2/3) * (1-J**(1/3)) * exp(3./2 * (K1-1)*(1 - J**(1./3)))

p_analytique = [Vinet(iso_T_K0, iso_T_K1, rho) for rho in rho_list]
# plt.plot(rho_list, p_analytique, linestyle = "-", color = "black", label = "Analyique")
# plt.scatter(rho_list, p_list, marker = "x", color = "green", label = "L")
