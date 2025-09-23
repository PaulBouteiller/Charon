"""
Simulation de compression dynamique d'une sphère creuse

Ce script implémente la simulation de la compression dynamique d'une sphère creuse
soumise à une pression externe. Il compare les résultats obtenus avec un modèle 1D
en coordonnées sphériques et un modèle 2D axisymétrique.

Cas test:
---------
- Sphère creuse avec rayon intérieur R_int = 8 mm et rayon extérieur R_ext = 11 mm
- Application d'une pression échelon sur la surface extérieure
- Comparaison des déplacements radiaux entre les modèles 1D et 2D

Auteur: bouteillerp
Créé le: Fri Mar 11 09:36:05 2022
"""
from Charon import (Solve, CylindricalUD, create_1D_mesh, 
                     Axisymmetric, axi_sphere, MeshManager)
from pandas import read_csv
import matplotlib.pyplot as plt
import sys
sys.path.append("../../")
from Generic_isotropic_material import Acier

###### Paramètre géométrique ######
Rint = 8
Rext = 11

###### Temps simulation ######
Tfin = 1e-3
print("le temps de fin de simulation est", Tfin )
pas_de_temps = Tfin / 3e3
largeur_creneau = (Rext - Rint) / 4
magnitude = 1e3
T_unload = Tfin/10

sortie = 500
pas_de_temps_sortie = sortie * pas_de_temps
n_sortie = int(Tfin/pas_de_temps_sortie)

#%% Probleme1D
Nr = 100
mesh1D = create_1D_mesh(Rint, Rext, Nr)
dictionnaire_mesh = {"tags": [1, 2], "coordinate": ["r", "r"], "positions": [Rint, Rext]}
mesh_manager1D = MeshManager(mesh1D, dictionnaire_mesh)
chargement = {"type" : "creneau", "t_crit": T_unload, "amplitude" : magnitude}
dictionnaire1D = {"material" : Acier, 
                  "mesh_manager" : mesh_manager1D,
                  "loading_conditions": 
                      [{"type": "surfacique", "component" : "F", "tag": 2, "value" : chargement}],
                  "isotherm" : True
                  }

pb1D = CylindricalUD(dictionnaire1D)

output_name = "Sphere"
dictionnaire1D_solve = {"Prefix" : output_name, "output" : {"U" : True}, "csv_output" : {"U" : True}}
solve_instance = Solve(pb1D, dictionnaire1D_solve, compteur=sortie, TFin=Tfin, scheme = "fixed", dt = pas_de_temps)
solve_instance.solve()


#%%Problème 2D
mesh2D, _, facets = axi_sphere(Rint, Rext, 20, 80, tol_dyn = 1e-3)

dictionnaire_mesh = {"facet_tag": facets}
mesh_manager2D = MeshManager(mesh2D, dictionnaire_mesh)
dictionnaire2D = {"material" : Acier, 
                  "mesh_manager" : mesh_manager2D,
                  "boundary_conditions": [{"component": "Uz", "tag": 1}, {"component": "Ur", "tag": 2}],
                  "loading_conditions": [{"type": "surfacique", "component" : "pressure", "tag": 3, "value" : chargement}],
                  "isotherm" : True
                  }
pb2D = Axisymmetric(dictionnaire2D)

output_name_2D = "Sphere_axi"
dictionnaire2D_solve = {"Prefix" : output_name_2D, "csv_output" : {"U" : ["Boundary", 1]}}
solve_instance = Solve(pb2D, dictionnaire2D_solve, compteur=sortie, TFin=Tfin, scheme = "fixed", dt = pas_de_temps)
solve_instance.solve()

df_1 = read_csv(output_name_2D+"-results/U.csv")
resultat_axi = [df_1[colonne].to_numpy() for colonne in df_1.columns]
ur_axi = [resultat_axi[2 * i + 2] for i in range((len(resultat_axi)-2)//2)]


df_2 = read_csv(output_name+"-results/U.csv")
resultat_1D = [df_2[colonne].to_numpy() for colonne in df_2.columns]
ur_1D = [resultat_1D[i + 2] for i in range((len(resultat_1D)-2))]

for j in range(len(ur_axi)-1):
    plt.plot(resultat_axi[0], ur_axi[j+1], linestyle = "--")
    
for j in range(len(ur_1D)-1):
    plt.scatter(resultat_1D[0], ur_1D[j+1], marker = "x")
    plt.plot(resultat_1D[0], ur_1D[j+1], linestyle = "-")