"""
Vérification du code éléments finis multiphase utilisant FEniCSx
avec des lois d'Arrhenius pour les transitions d'espèces chimiques.

Cas testés:
1. Une étape: 0 -> 1 (solution analytique triviale)
2. Deux étapes: 0 -> 1 -> 2 (résolution EDO avec scipy)

Lois d'évolution:
- dot{c0} = -c0 * k_a * exp(-E_a/(RT))
- dot{c1} = c0 * k_a * exp(-E_a/(RT)) - c1 * k_b * exp(-E_b/(RT))
"""

from Charon import create_interval, MeshManager, CartesianUD, Solve, Material
from mpi4py.MPI import COMM_WORLD
import numpy as np
from ufl import SpatialCoordinate
import matplotlib.pyplot as plt
from pandas import read_csv
from math import exp
from scipy.integrate import solve_ivp

# Paramètres matériau fictif
rho = 1
C = 1
dico_eos = {"E": 1, "nu": 0, "alpha": 0}
dico_devia = {}
eos_type = "IsotropicHPP"
devia_type = None
dummy_mat = Material(rho, C, eos_type, devia_type, dico_eos, dico_devia)

# Paramètres géométriques
mesh = create_interval(COMM_WORLD, 1, [np.array(0), np.array(1)])
dictionnaire_mesh = {"tags": [1, 2], "coordinate": ["x", "x"], "positions": [0, 1]}
mesh_manager = MeshManager(mesh, dictionnaire_mesh)

# Paramètres physiques
x = SpatialCoordinate(mesh)
temperature = 1.
R = 8.314
e_activation_a = R * temperature
e_activation_b = 2 * R * temperature
kinetic_prefactor_a = 1
kinetic_prefactor_b = 2

# Calcul des constantes d'Arrhenius
k_a = kinetic_prefactor_a * exp(-e_activation_a/(R * temperature))
k_b = kinetic_prefactor_b * exp(-e_activation_b/(R * temperature))

# Paramètres temporels
Tfin = 10
sortie = 100
pas_de_temps = 1e-3

#=== CAS 1: UNE ÉTAPE (0 -> 1) ===
print("=== Simulation une étape: 0 -> 1 ===")

two_phase_dictionnary = {
    "material" : [dummy_mat, dummy_mat],
    "mesh_manager": mesh_manager,
    "boundary_conditions": [{"component": "U", "tag": 1}, {"component": "U", "tag": 2}],
    "multiphase": {
        "conditions": [x[0]>=x[0], x[0]<x[0]],
        "evolution_laws": [
            {"type": "Arrhenius", "params": {"kin_pref": kinetic_prefactor_a, "e_activation": e_activation_a}}, 
            None
        ]
    },
}

one_stage_arrhenius = CartesianUD(two_phase_dictionnary)

one_stage_dictionnaire_solve = {
    "Prefix": "one_stage_arrhenius",
    "csv_output": {"c": True},
    "output": {"c": True},
    "initial_conditions" : {"T" : temperature}
    
}

solve_instance = Solve(one_stage_arrhenius, one_stage_dictionnaire_solve, 
                      compteur=sortie, TFin=Tfin, scheme="fixed", dt=pas_de_temps)
solve_instance.solve()

# Lecture des résultats
temps = np.loadtxt("one_stage_arrhenius-results/export_times.csv", delimiter=',', skiprows=1)
df0 = read_csv("one_stage_arrhenius-results/Concentration0.csv")
c0_result = [df0[colonne].to_numpy() for colonne in df0.columns]
c0_numerique = [x for x in c0_result[1:]]

# Solution analytique pour c0: c0(t) = exp(-k_a * t)
c0_analytique = [exp(-k_a * t) for t in temps]

# Visualisation cas 1
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(temps, c0_numerique, label=r"$c_0$ numérique")
plt.scatter(temps, c0_analytique, marker="x", label=r"$c_0$ analytique")
plt.xlim(0, Tfin)
plt.ylim(min(c0_numerique), 1)
plt.xlabel("Temps")
plt.ylabel("Concentration")
plt.title("Une étape: 0 → 1")
plt.legend()

#=== CAS 2: DEUX ÉTAPES (0 -> 1 -> 2) ===
print("=== Simulation deux étapes: 0 -> 1 -> 2 ===")

three_phase_dictionnary = {
    "material" : [dummy_mat, dummy_mat, dummy_mat],
    "mesh_manager": mesh_manager,
    "boundary_conditions": [
        {"component": "U", "tag": 1},
        {"component": "U", "tag": 2}
    ],
    "multiphase": {
        "conditions": [x[0]<=x[0], x[0]<x[0], x[0]<x[0]],
        "evolution_laws": [
            {"type": "Arrhenius", "params": {"kin_pref": kinetic_prefactor_a, "e_activation": e_activation_a}}, 
            {"type": "Arrhenius", "params": {"kin_pref": kinetic_prefactor_b, "e_activation": e_activation_b}},
            None
        ]
    },
}

two_stage_arrhenius = CartesianUD(three_phase_dictionnary)
two_stage_arrhenius.T.x.array[:] = np.array([temperature])

two_stage_dictionnaire_solve = {
    "Prefix": "two_stage_arrhenius",
    "csv_output": {"c": True},
    "output": {"c": True},
    "initial_conditions" : {"T" : temperature}
}

solve_instance = Solve(two_stage_arrhenius, two_stage_dictionnaire_solve, 
                      compteur=sortie, TFin=Tfin, scheme="fixed", dt=pas_de_temps)
solve_instance.solve()

# Lecture des résultats deux étapes
df0_2 = read_csv("two_stage_arrhenius-results/Concentration0.csv")
df1_2 = read_csv("two_stage_arrhenius-results/Concentration1.csv")
df2_2 = read_csv("two_stage_arrhenius-results/Concentration2.csv")
c0_2_numerique = [df0_2[col].to_numpy() for col in df0_2.columns][1:]
c1_2_numerique = [df1_2[col].to_numpy() for col in df1_2.columns][1:]
c2_2_numerique = [df2_2[col].to_numpy() for col in df2_2.columns][1:]

# Solution analytique avec scipy pour le système d'EDO
def system_ode(t, y):
    """
    Système d'EDO pour 0 -> 1 -> 2:
    dy[0]/dt = -k_a * y[0]                    (concentration c0)
    dy[1]/dt = k_a * y[0] - k_b * y[1]        (concentration c1)
    dy[2]/dt = k_b * y[1]                     (concentration c2)
    """
    c0, c1, c2 = y
    dc0_dt = -k_a * c0
    dc1_dt = k_a * c0 - k_b * c1
    dc2_dt = k_b * c1
    return [dc0_dt, dc1_dt, dc2_dt]

# Conditions initiales: c0(0) = 1, c1(0) = 0, c2(0) = 0
y0 = [1.0, 0.0, 0.0]
t_span = (0, Tfin)
t_eval = temps

# Résolution avec scipy
sol = solve_ivp(system_ode, t_span, y0, t_eval=t_eval, method='RK45')
c0_2_analytique = sol.y[0]
c1_2_analytique = sol.y[1]
c2_2_analytique = sol.y[2]

# Visualisation cas 2
plt.subplot(1, 2, 2)
plt.plot(temps, c0_2_numerique, label=r"$c_0$ numérique", linestyle='-')
plt.plot(temps, c1_2_numerique, label=r"$c_1$ numérique", linestyle='-')
plt.plot(temps, c2_2_numerique, label=r"$c_2$ numérique", linestyle='-')
plt.scatter(temps, c0_2_analytique, marker="x", label=r"$c_0$ analytique", alpha=0.7)
plt.scatter(temps, c1_2_analytique, marker="+", label=r"$c_1$ analytique", alpha=0.7)
plt.scatter(temps, c2_2_analytique, marker="o", label=r"$c_2$ analytique", alpha=0.7, s=20)
plt.xlim(0, Tfin)
plt.xlabel("Temps")
plt.ylabel("Concentration")
plt.title("Deux étapes: 0 → 1 → 2")
plt.legend()

plt.tight_layout()
plt.show()

# Calcul des erreurs relatives
erreur_c0_1etape = np.max(np.abs(np.array(c0_numerique).flatten() - c0_analytique))
erreur_c0_2etapes = np.max(np.abs(np.array(c0_2_numerique).flatten() - c0_2_analytique))
erreur_c1_2etapes = np.max(np.abs(np.array(c1_2_numerique).flatten() - c1_2_analytique))
erreur_c2_2etapes = np.max(np.abs(np.array(c2_2_numerique).flatten() - c2_2_analytique))

print("\n=== ERREURS MAXIMALES ===")
print(f"Erreur c0 (1 étape): {erreur_c0_1etape:.2e}")
print(f"Erreur c0 (2 étapes): {erreur_c0_2etapes:.2e}")
print(f"Erreur c1 (2 étapes): {erreur_c1_2etapes:.2e}")
print(f"Erreur c2 (2 étapes): {erreur_c2_2etapes:.2e}")
