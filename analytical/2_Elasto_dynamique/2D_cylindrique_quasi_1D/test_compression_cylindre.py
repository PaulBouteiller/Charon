"""
Simulation de compression dynamique d'un cylindre creux

Ce script implémente la simulation de la compression dynamique d'un cylindre creux
soumis à une pression externe. Il compare les résultats obtenus avec un modèle 1D
en coordonnées cylindriques et un modèle 2D axisymétrique.

Cas test:
---------
- Cylindre creux avec rayon intérieur R_int = 9 mm et rayon extérieur R_ext = 11 mm
- Hauteur du cylindre réduite (rapport hauteur/épaisseur = 1/25) pour assurer
  un comportement quasi-1D dans le modèle axisymétrique
- Application d'une pression échelon sur la surface extérieure
- Comparaison des déplacements radiaux entre les modèles 1D et 2D

Auteur: bouteillerp
Créé le: Fri Mar 11 09:36:05 2022
"""
from Charon import (Solve, CylindricalUD, create_1D_mesh, 
                     create_rectangle, Axisymmetric, MeshManager)
from pandas import read_csv
from numpy import loadtxt
from mpi4py.MPI import COMM_WORLD
import matplotlib.pyplot as plt
import sys
sys.path.append("../../")
from Generic_isotropic_material import Acier

###### Paramètre géométrique ######
Rint = 9
Rext = 11
epaisseur = Rext - Rint
rapport = 25
hauteur = epaisseur / rapport

###### Maillage #######
Nr = 100
Nz = int(Nr / rapport)

###### Chargement ######
Pext = 10

###### Temps simulation ######
Tfin = 7e-4
T_unload = Tfin/10
print("le temps de fin de simulation est", Tfin )
pas_de_temps = Tfin / 7000
largeur_creneau = (Rext - Rint) / 4
magnitude = 1e3

sortie = 2000
pas_de_temps_sortie = sortie * pas_de_temps
n_sortie = int(Tfin/pas_de_temps_sortie)

#%% Probleme1D
mesh1D = create_1D_mesh(Rint, Rext, Nr)
dictionnaire_mesh = {"tags": [1, 2], "coordinate": ["r", "r"], "positions": [Rint, Rext]}
mesh_manager = MeshManager(mesh1D, dictionnaire_mesh)

chargement = {"type" : "creneau", "t_crit": T_unload, "amplitude" : magnitude}
dictionnaire1D = {"material" : Acier, 
                  "mesh_manager" : mesh_manager,
                "loading_conditions": 
                    [{"type": "surfacique", "component" : "F", "tag": 2, "value" : chargement}],
                "isotherm" : True
                }

pb1D = CylindricalUD(dictionnaire1D)
output_name_1D = "Cylindre"
dictionnaire1D_solve = {"Prefix" : output_name_1D, "csv_output" : {"U" : True}}
solve_instance = Solve(pb1D, dictionnaire1D_solve, compteur=sortie, TFin=Tfin, scheme = "fixed", dt = pas_de_temps)
solve_instance.solve()

#%%Problème 2D
mesh2D = create_rectangle(COMM_WORLD, [(Rint, 0), (Rext, hauteur)], [Nr, Nz])

dictionnaire_mesh = {"tags": [1, 2, 3, 4],
                     "coordinate": ["r", "r", "z", "z"], 
                     "positions": [Rint, Rext, 0, hauteur]
                     }
mesh_manager = MeshManager(mesh2D, dictionnaire_mesh)

dictionnaire2D = {"material" : Acier, 
                  "mesh_manager" : mesh_manager,
                "loading_conditions": 
                    [{"type": "surfacique", "component" : "Fr", "tag": 2, "value" : chargement}],
                "boundary_conditions": 
                    [{"component": "Uz", "tag": 3}, {"component": "Uz", "tag": 4}],
                "isotherm" : True
                }
pb2D = Axisymmetric(dictionnaire2D)

output_name_2D = "Cylindre_axi"
dictionnaire2D_solve = {"Prefix" : output_name_2D, "csv_output" : {"U" : ["Boundary", 3]}}
solve_instance = Solve(pb2D, dictionnaire2D_solve, compteur=sortie, TFin=Tfin, scheme = "fixed", dt = pas_de_temps)
solve_instance.solve()
        
df_1 = read_csv(output_name_2D+"-results/U.csv")
resultat_axi = [df_1[colonne].to_numpy() for colonne in df_1.columns]
ur_axi = [resultat_axi[2 * i + 4] for i in range((len(resultat_axi)-4)//2)]
length = len(ur_axi)
print("longueur axi", len(ur_axi))


df_2 = read_csv(output_name_1D+"-results/U.csv")
resultat_1D = [df_2[colonne].to_numpy() for colonne in df_2.columns]
ur_1D = [resultat_1D[i + 2] for i in range((len(resultat_1D)-2))]
print("longueur cyl", len(ur_1D))

temps = loadtxt("Cylindre_axi-results/export_times.csv", delimiter=',', skiprows=1)

for j in range(length):
    plt.plot(resultat_axi[0], ur_axi[j], linestyle = "--", label = f"Cylindrical t={temps[j+1]}s")
    plt.scatter(resultat_1D[0], ur_1D[j], marker = "x", label = f"Axisymmetric t={temps[j+1]}s")
    plt.xlim(Rint, Rext)
    plt.xlabel(r"$r$", size = 18)
    plt.ylabel(r"Déplacement radial", size = 18)
    plt.legend()