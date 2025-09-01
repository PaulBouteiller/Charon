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
#Lafourcade
# C11 = 46.07e9
# C22 = 45.74e9
# C33 = 14.78e9

#Matthew
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

######## Paramètres géométriques et de maillage ########
L, l, h = 1, 1, 1
Nx, Ny, Nz = 1, 1, 1

alpha = -0.12
N_pas = 10

mesh = create_box(COMM_WORLD, [np.array([0, 0, 0]),  np.array([L, l, h])],
                                  [Nx, Ny, Nz], cell_type = CellType.hexahedron)
mesh_manager = MeshManager(mesh, {})

###### Paramètre du problème ######
dictionnaire = {"material" : TATB,
                "mesh_manager" : mesh_manager,
                "analysis" : "User_driven",
                "isotherm" : True,
                "polycristal" : True
                }

pb = Tridimensional(dictionnaire)
pb.J_list = []

def user_defined_displacement(problem, t):
    pos = problem.V.tabulate_dof_coordinates()
    pos_ravel = pos.ravel()
    problem.J_list.append((1 +  alpha *  t)**3)
    for i in range(len(problem.u.x.array)):
        problem.u.x.array[i] = alpha *  t * pos_ravel[i]

dictionnaire_solve = {
    "Prefix" : "Compression_spherique_0D",
    "csv_output" : {"p" : True, "s" : True}
    }

solve_instance = Solve(pb, dictionnaire_solve, compteur=1, TFin = 1, scheme = "fixed", dt = 1/N_pas)
solve_instance.user_defined_displacement = user_defined_displacement #Attache une fonction d'export appelée à chaque pas de temps
solve_instance.solve()
    
p = read_csv("Compression_spherique_0D-results/p.csv")
pressure = [p[colonne].to_numpy() for colonne in p.columns]
pressure = [pressure[3:]]

s = read_csv("Compression_spherique_0D-results/s.csv")
deviateur = [s[colonne].to_numpy() for colonne in s.columns]
deviateur = [s.item() for s in deviateur[3:]]
n = len(deviateur) // 9
s_xx_list = [deviateur[9 * i] for i in range(n)]
s_yy_list = [deviateur[9 * i  + 4] for i in range(n)]
s_zz_list = [deviateur[9 * i + 8] for i in range(n)]

print("La liste des dilatation volumique est", pb.J_list)

def M_0():
    unit_C_1 = Rigi[0,0] + Rigi[0,1] + Rigi[0,2]
    unit_C_2 = Rigi[1,0] + Rigi[1,1] + Rigi[1,2]
    unit_C_3 = Rigi[2,0] + Rigi[2,1] + Rigi[2,2]
    trace = unit_C_1 + unit_C_2 + unit_C_3
    return np.array([unit_C_1, unit_C_2, unit_C_3]), trace


def deviateur_analytique(J):
    M0, trace = M_0()
    dev_M0 = M0 - 1./3 * trace * np.array([1,1,1])
    dev_SPKo = 1./3 * (J - 1) * dev_M0
    s= J**(-1) * dev_SPKo
    # s = J**(1./3) * dev_SPKo
    return s

def sigma_HPP(J):
    M0, _ = M_0()
    sigma = (J**(1./3)-1) * M0
    trace = sum(sigma[i] for i in range(3))
    p = -1/3 * trace
    s = sigma - 1./3 * trace * np.array([1,1,1])
    return p, s

def Vinet(K0, K1, J):
    return 3 * K0 * J**(-2/3) * (1-J**(1/3)) * exp(3./2 * (K1-1)*(1 - J**(1./3)))

p_analytique = []
s_analytique = []

p_HPP = []
s_HPP = []

for J in pb.J_list:
    p_analytique.append(Vinet(iso_T_K0, iso_T_K1, J))
    s_analytique.append(deviateur_analytique(J))
    p_, s_ = sigma_HPP(J)
    p_HPP.append(p_)
    s_HPP.append(s_)
    
s_xx_analytique = [s[0] for s in s_analytique]
s_yy_analytique = [s[1] for s in s_analytique]
s_zz_analytique = [s[2] for s in s_analytique]
s_xx_HPP = [s[0] for s in s_HPP]
s_yy_HPP = [s[1] for s in s_HPP]
s_zz_HPP = [s[2] for s in s_HPP]

# Graphique pression (sauvegarde individuelle)
plt.scatter(pb.J_list, pressure, marker = "x", color = "red", label="Present")
plt.plot(pb.J_list, p_analytique, linestyle = "--", color = "green",label="Vinet")
plt.plot(pb.J_list, p_HPP, linestyle = "--", color = "black",label="HPP")
plt.xlabel(r"Volumetric compression $J$", fontsize = 18)
plt.ylabel(r"Pressure $p$ (Pa)", fontsize = 18)
plt.legend()
plt.xlim(0.98 * min(pb.J_list), 1)
plt.ylim(0, 1.1 * max(p_analytique))
plt.savefig("p.pdf", bbox_inches = 'tight')
plt.close()

# Graphique contraintes déviatoriques (sauvegarde individuelle)
plt.scatter(pb.J_list, s_xx_list, marker = "x", color ="red")
plt.plot(pb.J_list, s_xx_analytique, linestyle = "--", color = "green")
plt.plot(pb.J_list, s_xx_HPP, linestyle = "--", color = "black",label=r"$s_{xx}$")
plt.scatter(pb.J_list, s_yy_list, marker = "x", color ="red")
plt.plot(pb.J_list, s_yy_analytique, linestyle = "-", color = "green")
plt.plot(pb.J_list, s_yy_HPP, linestyle = "-", color = "black",label=r"$s_{yy}$")
plt.scatter(pb.J_list, s_zz_list, marker = "x", color ="red")
plt.plot(pb.J_list, s_zz_analytique, linestyle = "-.", color = "green")
plt.plot(pb.J_list, s_zz_HPP, linestyle = "-.", color = "black",label=r"$s_{zz}$")
plt.xlabel(r"Volumetric compression $J$", fontsize = 18)
plt.ylabel(r"Deviatoric stress (Pa)", fontsize = 18)
plt.legend()
plt.xlim(0.98 * min(pb.J_list), 1)
plt.savefig("s.pdf", bbox_inches = 'tight')
plt.close()

# Subplot combiné pour visualisation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Subplot 1 : Pression
ax1.scatter(pb.J_list, pressure, marker = "x", color = "red", label="Present")
ax1.plot(pb.J_list, p_analytique, linestyle = "--", color = "green",label="Vinet")
ax1.plot(pb.J_list, p_HPP, linestyle = "--", color = "black",label="HPP")
ax1.set_xlabel(r"Volumetric compression $J$", fontsize = 18)
ax1.set_ylabel(r"Pressure $p$ (Pa)", fontsize = 18)
ax1.legend()
ax1.set_xlim(0.98 * min(pb.J_list), 1)
ax1.set_ylim(0, 1.1 * max(p_analytique))

# Subplot 2 : Contraintes déviatoriques
ax2.scatter(pb.J_list, s_xx_list, marker = "x", color ="red")
ax2.plot(pb.J_list, s_xx_analytique, linestyle = "--", color = "green")
ax2.plot(pb.J_list, s_xx_HPP, linestyle = "--", color = "black",label=r"$s_{xx}$")
ax2.scatter(pb.J_list, s_yy_list, marker = "x", color ="red")
ax2.plot(pb.J_list, s_yy_analytique, linestyle = "-", color = "green")
ax2.plot(pb.J_list, s_yy_HPP, linestyle = "-", color = "black",label=r"$s_{yy}$")
ax2.scatter(pb.J_list, s_zz_list, marker = "x", color ="red")
ax2.plot(pb.J_list, s_zz_analytique, linestyle = "-.", color = "green")
ax2.plot(pb.J_list, s_zz_HPP, linestyle = "-.", color = "black",label=r"$s_{zz}$")
ax2.set_xlabel(r"Volumetric compression $J$", fontsize = 18)
ax2.set_ylabel(r"Deviatoric stress (Pa)", fontsize = 18)
ax2.legend()
ax2.set_xlim(0.98 * min(pb.J_list), 1)

plt.tight_layout()
plt.show()
