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
from CharonX import Solve, CylindricalUD, MyConstant, create_1D_mesh, create_rectangle, Axisymmetric
from pandas import read_csv
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
chargement1D = MyConstant(mesh1D, T_unload, magnitude, Type = "Creneau")
dictionnaire1D = {"mesh" : mesh1D,
                "boundary_setup": 
                    {"tags": [1, 2],
                     "coordinate": ["r", "r"], 
                     "positions": [Rint, Rext]
                     },
                "loading_conditions": 
                    [{"type": "surfacique", "component" : "F", "tag": 2, "value" : chargement1D}
                    ],
                "isotherm" : True
                }

pb1D = CylindricalUD(Acier, dictionnaire1D)
dictionnaire1D_solve = {
    "Prefix" : "Cylindre",
    "csv_output" : {"U" : True, "Sig" : True}
    }
solve_instance = Solve(pb1D, dictionnaire1D_solve, compteur=sortie, TFin=Tfin, scheme = "fixed", dt = pas_de_temps)
solve_instance.solve()

#%%Problème 2D
mesh2D =create_rectangle(COMM_WORLD, [(Rint, 0), (Rext, hauteur)], [Nr, Nz])
chargement2D = MyConstant(mesh2D, T_unload, magnitude, Type = "Creneau")
dictionnaire2D = {"mesh" : mesh2D,
                "boundary_setup": 
                    {"tags": [1, 2, 3, 4],
                     "coordinate": ["r", "r", "z", "z"], 
                     "positions": [Rint, Rext, 0, hauteur]
                     },
                "loading_conditions": 
                    [{"type": "surfacique", "component" : "Fr", "tag": 2, "value" : chargement2D}
                    ],
                "boundary_conditions": 
                    [{"component": "Uz", "tag": 3},
                     {"component": "Uz", "tag": 4}
                    ],
                "isotherm" : True
                }
pb2D = Axisymmetric(Acier, dictionnaire2D)

dictionnaire2D_solve = {
    "Prefix" : "Cylindre_axi",
    "csv_output" : {"U" : ["Boundary", 3], "Sig" : True}
    }
solve_instance = Solve(pb2D, dictionnaire2D_solve, compteur=sortie, TFin=Tfin, scheme = "fixed", dt = pas_de_temps)
solve_instance.solve()
        
df_1 = read_csv("Cylindre_axi-results/U.csv")
resultat_axi = [df_1[colonne].to_numpy() for colonne in df_1.columns]
ur_axi = [resultat_axi[2 * i + 4] for i in range((len(resultat_axi)-4)//2)]
length = len(ur_axi)
print("longueur axi", len(ur_axi))


df_2 = read_csv("Cylindre-results/U.csv")
resultat_1D = [df_2[colonne].to_numpy() for colonne in df_2.columns]
ur_1D = [resultat_1D[i + 2] for i in range((len(resultat_1D)-2))]
print("longueur cyl", len(ur_1D))


for j in range(length):
    plt.plot(resultat_axi[0], ur_axi[j], linestyle = "--")
    plt.scatter(resultat_1D[0], ur_1D[j], marker = "x")