"""
Created on Fri Mar 11 09:36:05 2022

@author: bouteillerp
Essai de dilatation homogène sur un cube 3D
La transformation est de la forme
\vec{x} = \vec{X}(1+alpha *t) donc \GT = (1+\alpha * t) * \TenUnit
et donc la dilatation volumique égalise J = (1+\alpha * t)**3
 """
from Charon import create_box, MeshManager, Material, CellType, Solve, Tridimensional
from mpi4py.MPI import COMM_WORLD
import matplotlib.pyplot as plt
import pytest
from pandas import read_csv
from math import exp
import numpy as np

#%%Modèle matériau
C11 = 52e9
C22 = 50e9
C33 = 16e9

C44 = 0.97e9
C55 = 0.24e9
C66 = 16.46e9
C13 = 14.07e9
C23 = 3.23e9
C12 = 4.08e9

Rigi = np.array([[C11, C12, C13, 0, 0, 0],
                 [C12, C22, C23, 0, 0, 0],
                 [C13, C23, C33, 0, 0, 0],
                 [0, 0, 0, C44, 0, 0],
                 [0, 0, 0, 0, C55, 0],
                 [0, 0, 0, 0, 0, C66]])

eos_type = "Vinet"
print("La rigidité est", Rigi)
kappa_eq = 1./9 * sum(Rigi[i,j] for i in range(3) for j in range(3))
print("Compressibilité équivalente", kappa_eq)
dico_eos = {"kappa": kappa_eq, "alpha" : 1}

iso_T_K0 = 17.5e9
T_dep_K0 = 0
iso_T_K1 = 7.6
T_dep_K1 = 0
dico_eos = {'iso_T_K0': iso_T_K0, 'T_dep_K0' : T_dep_K0, 'iso_T_K1' : iso_T_K1, 'T_dep_K1': T_dep_K1}

devia_type = "Anisotropic"
dico_devia = {"C" : Rigi}
TATB = Material(1, 1, eos_type, devia_type, dico_eos, dico_devia)

#%% Paramètres géométriques et de maillage
L, l, h = 1, 1, 1
Nx, Ny, Nz = 1, 1, 1

alpha = -0.12
N_pas = 100

mesh = create_box(COMM_WORLD, [np.array([0, 0, 0]),  np.array([L, l, h])],
                                  [Nx, Ny, Nz], cell_type = CellType.hexahedron)
mesh_manager = MeshManager(mesh, {})

#%% Paramètre du problème 
dictionnaire = {"material" : TATB,
                "mesh_manager" : mesh_manager,
                "analysis" : "User_driven",
                "isotherm" : True,
                }

pb = Tridimensional(dictionnaire)
pb.J_list = []

#%% Conditions initiales
SHEAR_MAGNITUDE = 0.01

def set_initial_conditions(problem, SHEAR_MAGNITUDE, test):
    pos = problem.V.tabulate_dof_coordinates()
    pos_ravel = pos.ravel()
    if test == "cisaillement_xy":
        for i in range(0, len(problem.u.x.array), 3):
            problem.u.x.array[i] = SHEAR_MAGNITUDE * pos_ravel[i+1]
            problem.u.x.array[i+1] = 0
            problem.u.x.array[i+2] = 0
    elif test == "cisaillement_xz":
        for i in range(0, len(problem.u.x.array), 3):
            problem.u.x.array[i] = SHEAR_MAGNITUDE * pos_ravel[i+2]
            problem.u.x.array[i+1] = 0
            problem.u.x.array[i+2] = 0
    elif test == "cisaillement_yz":
        for i in range(0, len(problem.u.x.array), 3):
            problem.u.x.array[i] = 0
            problem.u.x.array[i+1] = SHEAR_MAGNITUDE * pos_ravel[i+2]
            problem.u.x.array[i+2] = 0
    problem.shear_displacement = problem.u.x.array
    problem.initial_position = pos_ravel

test = "cisaillement_xy"
set_initial_conditions(pb, SHEAR_MAGNITUDE, test)

dt = 1/N_pas
def user_defined_displacement(problem, t):
    problem.J_list.append((1 +  alpha *  t)**3)
    for i in range(len(problem.u.x.array)):
        problem.u.x.array[i] += problem.initial_position[i] * dt * alpha

output_name = "Compression_spherique_" + test
dictionnaire_solve = {
    "Prefix" : output_name,
    # "output" : {"U" : True},
    "csv_output" : {"p" : True, "s" : True}
    }

solve_instance = Solve(pb, dictionnaire_solve, compteur=1, TFin = 1, scheme = "fixed", dt = dt)
solve_instance.user_defined_displacement = user_defined_displacement #Attache une fonction d'export appelée à chaque pas de temps
solve_instance.solve()
    
p = read_csv(output_name + "-results/p.csv")
pressure = [p[colonne].to_numpy() for colonne in p.columns]
pressure = [pressure[3:]]

s = read_csv(output_name + "-results/s.csv")
deviateur = [s[colonne].to_numpy() for colonne in s.columns]
deviateur = [s.item() for s in deviateur[3:]]
n = len(deviateur) // 9
s_xy = [deviateur[1 + 9 * i] for i in range(N_pas+1)]
s_yz = [deviateur[5 + 9 * i] for i in range(N_pas+1)]


def M_0():
    unit_C_1 = Rigi[0,0] + Rigi[0,1] + Rigi[0,2]
    unit_C_2 = Rigi[1,0] + Rigi[1,1] + Rigi[1,2]
    unit_C_3 = Rigi[2,0] + Rigi[2,1] + Rigi[2,2]
    return np.array([unit_C_1, unit_C_2, unit_C_3])

M22 = M_0()[1]

def analytical_linearized_shear(J):
    if test == "cisaillement_xy":
        first_order_contribution = (M22/3. * (J - 1) + C66) * SHEAR_MAGNITUDE / J**(4./3)
        third_order_contribution = (C22 / (2 * J**2) - M22 / (6 * J**(8./3))) * SHEAR_MAGNITUDE**3 
        return first_order_contribution + third_order_contribution
        # return C44 * SHEAR_MAGNITUDE / J**(4./3)
    elif test == "cisaillement_yz":
        # return (M22/3. * (J - 1) + C44) * SHEAR_MAGNITUDE / J**(4./3)
        return C44 * SHEAR_MAGNITUDE / J**(4./3)

J_analytical = np.linspace(min(pb.J_list), max(pb.J_list), 100)
analytical_shear_list = [analytical_linearized_shear(J) for J in J_analytical]

plt.plot(J_analytical, analytical_shear_list, linestyle = "-.", color = "black",label=r"Analytical")
if test == "cisaillement_xy":
    plt.scatter(pb.J_list, s_xy, marker = "x", color = "red", label="Present")
elif test == "cisaillement_yz":
   plt.scatter(pb.J_list, s_yz, marker = "x", color = "red", label="Present")   