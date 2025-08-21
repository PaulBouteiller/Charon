"""
Test de validation pour la diffusion thermique 1D en coordonnées cartésiennes

Ce script implémente et exécute un test de validation pour l'équation de diffusion
thermique en 1D dans un système de coordonnées cartésiennes. Il compare
la solution numérique obtenue avec CharonX à la solution analytique de type
fonction d'erreur (erf).

Cas test:
---------
- Barre unidimensionnelle de longueur L = 210 μm
- Température initiale discontinue: T = 800K dans un intervalle central de 10 μm,
  T = 300K ailleurs
- Conditions aux limites: aucune condition imposée (diffusion libre)
- Comparaison des profils de température numériques et analytiques à différents instants

Théorie:
--------
L'équation de diffusion thermique en 1D s'écrit:
    ∂T/∂t = D·∂²T/∂x²

où D = λ/(ρ·C) est le coefficient de diffusion thermique.

La solution analytique pour une discontinuité initiale utilise la fonction d'erreur:
    T(x,t) = (T_chaud-T_froid)·[erf((x+point_droit)/√(4Dt)) - erf((x+point_gauche)/√(4Dt))]/2 + T_froid

Paramètres matériaux:
--------------------
- Conductivité thermique: λ = 240 W/(m·K)
- Capacité thermique massique: C = 1000 J/(kg·K)
- Masse volumique: ρ = 2.785×10³ kg/m³
- Coefficient de diffusion: D = λ/(ρ·C) ≈ 8.62×10⁻⁵ m²/s

Auteur: bouteillerp
"""

from Charon import CartesianUD, LinearThermal, create_interval, MeshManager, Solve
from mpi4py.MPI import COMM_WORLD
from pandas import read_csv
import matplotlib.pyplot as plt
from scipy.special import erf
import numpy as np
from ufl import conditional, SpatialCoordinate, And, lt, gt
from dolfinx.fem import Expression, Function
import sys
sys.path.append("../../")
from Generic_isotropic_material import Acier, rho, C

###### Modèle thermique ######
lmbda_diff = 240e-3
AcierTherm = LinearThermal(lmbda_diff)

###### Paramètre géométrique ######
L = 210e-3
bord_gauche = -105e-3
bord_droit = bord_gauche + L
point_gauche = -5e-3
point_droit = 5e-3

###### Champ de température initial ######
Tfin = 1e-3
Tfroid = 300
TChaud = 800

###### Paramètre temporel ######
sortie = 25
pas_de_temps = 5e-6
pas_de_temps_sortie = sortie * pas_de_temps

n_sortie = int(Tfin/pas_de_temps_sortie)

Nx = 1000
mesh = create_interval(COMM_WORLD, Nx, [np.array(bord_gauche), np.array(bord_droit)])

dictionnaire_mesh = {"tags": [1, 2], "coordinate": ["x", "x"], "positions": [bord_gauche, bord_droit]}
mesh_manager = MeshManager(mesh, dictionnaire_mesh)

dictionnaire = {"mesh_manager" : mesh_manager,
                "Thermal_material" : AcierTherm, 
                "analysis" : "Pure_diffusion"
                }
    
pb = CartesianUD(Acier, dictionnaire)
x = SpatialCoordinate(mesh)
ufl_condition = conditional(And(lt(x[0], point_droit), gt(x[0], point_gauche)), TChaud, Tfroid)
T_expr = Expression(ufl_condition, pb.V_T.element.interpolation_points())
pb.T0 = Function(pb.V_T)
pb.T0.interpolate(T_expr)
pb.T.interpolate(T_expr)
pb.bcs_T = []

dictionnaire_solve = {}
dictionnaire_solve = {"Prefix" : "Diffusion_1D", "csv_output" : {"T" : True}}
solve_instance = Solve(pb, dictionnaire_solve, compteur = sortie, TFin=Tfin, scheme = "fixed", dt = pas_de_temps)
solve_instance.solve()

def analytical_f(x,t):
    D_act = lmbda_diff/(rho * C)
    denom = np.sqrt(4 * D_act * t)
    return 1./2 * (erf((x+point_droit)/denom)-erf((x+point_gauche)/denom))

def analytical_T(x,t, Tf, Tc):
    return (Tc-Tf) * analytical_f(x,t) + Tf

df = read_csv("Diffusion_1D-results/T.csv")
temps = np.loadtxt("Diffusion_1D-results/export_times.csv",  delimiter=',', skiprows=1)
resultat = [df[colonne].to_numpy() for colonne in df.columns]

len_vec = len(resultat[0])
pas_espace = np.linspace(bord_gauche, bord_droit, len_vec)

sortie_label = True
for i, t in enumerate(temps):
    list_T_erf = [analytical_T(x, t, Tfroid, TChaud) for x in pas_espace]
    diff_tot = list_T_erf - resultat[i+1]
    int_discret = sum(abs(diff_tot[j]) for j in range(len_vec))/sum(abs(list_T_erf[j]) for j in range(len_vec))
    print("La difference est de", int_discret)
    # assert int_discret < 0.002, "1D cartesian diffusion fails"
    # print("1D cartesian diffusion succeed")
    if __name__ == "__main__": 
        if sortie_label:
            label_analytical = "Analytique"
            label_CHARON = "CHARON"
            sortie_label = False
        else:
            label_analytical = None
            label_CHARON = None
        plt.plot(pas_espace, list_T_erf, linestyle = "-", color = "red", label = label_analytical)
        plt.plot(pas_espace, resultat[i+1], linestyle = "--", color = "blue", label=label_CHARON)
        print("cette sortie correspond au temps", (t))
plt.xlim(bord_gauche/4, bord_droit/4)
plt.ylim(Tfroid, TChaud*1.02)
plt.xlabel(r"$x$", size = 18)
plt.ylabel(r"Temperature (K)", size = 18)
plt.legend()
plt.show()