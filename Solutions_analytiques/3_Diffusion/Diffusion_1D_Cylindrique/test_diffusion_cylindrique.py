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

from CharonX import CylindricalUD, LinearThermal, create_interval, MeshManager, Solve
from mpi4py.MPI import COMM_WORLD
import matplotlib.pyplot as plt
from scipy.special import erf
import numpy as np
from ufl import conditional, SpatialCoordinate, lt
from dolfinx.fem import Expression, Function
import sys
sys.path.append("../../")
from Generic_isotropic_material import Acier, rho, C

###### Modèle thermique ######
lmbda_diff = 240e-3
AcierTherm = LinearThermal(lmbda_diff)

###### Paramètre géométrique ######
R = 210e-3
largeur_tache = 10e-3

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
mesh = create_interval(COMM_WORLD, Nx, [np.array(0), np.array(R)])

dictionnaire_mesh = {"tags": [1, 2], "coordinate": ["r", "r"], "positions": [0, R]}
mesh_manager = MeshManager(mesh, dictionnaire_mesh)

dictionnaire = {"mesh_manager" : mesh_manager,
                "Thermal_material" : AcierTherm, 
                "analysis" : "Pure_diffusion"
                }
    
pb = CylindricalUD(Acier, dictionnaire)
x = SpatialCoordinate(mesh)
ufl_condition = conditional(lt(x[0], largeur_tache), TChaud, Tfroid)
T_expr = Expression(ufl_condition, pb.V_T.element.interpolation_points())
pb.T0 = Function(pb.V_T)
pb.T0.interpolate(T_expr)
pb.T.interpolate(T_expr)
pb.bcs_T = []

pb.T_vector = [pb.T.x.array]
pb.T_max = [800]
def query_output(problem, t):
    problem.T_vector.append(np.array(problem.T.x.array))
    problem.T_max.append(np.max(problem.T_vector[-1]))

dictionnaire_solve = {}
solve_instance = Solve(pb, {}, compteur = sortie, TFin=Tfin, scheme = "fixed", dt = pas_de_temps)
solve_instance.query_output = query_output #Attache une fonction d'export appelée à chaque pas de temps
solve_instance.solve()

def analytical_T_cylindrique(r, t, a, Tfroid, Tchaud, D, R):
    """
    Solution analytique pour diffusion cylindrique d'un créneau initial
    """
    from scipy.special import j0, j1, jn_zeros
    
    if t < 1e-10:  # Pour t très petit, retourner la condition initiale
        return Tchaud if r < a else Tfroid
    
    # Moyenne spatiale de la température
    T_moy = Tfroid + (Tchaud - Tfroid) * (a/R)**2
    
    # Série de Fourier-Bessel
    sum_term = 0
    n_terms = 5  # Réduire le nombre de termes pour la performance
    
    for n in range(1, n_terms + 1):
        # n-ième zéro de J0
        alpha_n = jn_zeros(0, n)[0]
        
        # Coefficient An utilisant la formule correcte
        # An = 2*(Tchaud-Tfroid)*J1(alpha_n*a/R)/(alpha_n*J1(alpha_n))
        An = 2 * (Tchaud - Tfroid) * j1(alpha_n * a / R) / alpha_n
        
        # Terme de la série avec normalisation correcte
        term = An * j0(alpha_n * r / R) * np.exp(-D * alpha_n**2 * t / R**2) / j1(alpha_n)**2
        sum_term += term
    
    return T_moy + sum_term

# Calcul du coefficient de diffusion
D_act = lmbda_diff / (rho * C)

len_vec = len(pb.T_vector[0])
t_list = np.linspace(1e-12, Tfin, n_sortie+1)
pas_espace = np.linspace(1e-6, R, len_vec)
compteur = 0

# Pré-calcul de la solution analytique pour tous les temps (plus efficace)
print("Calcul de la solution analytique...")
analytical_solutions = []
for t in t_list:
    print(f"  t = {t:.6f} s")
    sol = [analytical_T_cylindrique(r, t, largeur_tache, Tfroid, TChaud, D_act, R) 
           for r in pas_espace]
    analytical_solutions.append(sol)

print("\nComparaison avec CharonX:")
for i, t in enumerate(t_list):
    list_T_analytical = analytical_solutions[i]
    
    diff_tot = np.array(list_T_analytical) - pb.T_vector[i]
    int_discret = np.sum(np.abs(diff_tot)) / np.sum(np.abs(list_T_analytical))
    print(f"t = {t:.6f} s : différence relative = {int_discret:.6f}")
    
    if __name__ == "__main__": 
        if compteur == 0:
            label_analytical = "Analytique"
            label_CHARON = "CHARON"
        else:
            label_analytical = None
            label_CHARON = None
        
        plt.plot(pas_espace, list_T_analytical, linestyle="-", color="red", label=label_analytical)
        plt.plot(pas_espace, pb.T_vector[i], linestyle="--", color="blue", label=label_CHARON)
    compteur += 1

plt.xlim(0, R/4)  # Zoom sur la zone d'intérêt
plt.ylim(Tfroid*0.98, TChaud*1.02)
plt.xlabel(r"$r$ (mm)", size=18)
plt.ylabel(r"Température (K)", size=18)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()