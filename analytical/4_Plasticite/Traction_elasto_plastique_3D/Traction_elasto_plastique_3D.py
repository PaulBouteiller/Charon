"""
Created on Mon Feb 27 10:12:53 2023

@author: bouteillerp
"""

from CharonX import create_box, MyConstant, Tridimensional, Solve, MeshManager
from mpi4py.MPI import COMM_WORLD
import numpy as np
import time
import matplotlib.pyplot as plt
import sys
sys.path.append("../../")
from Generic_isotropic_material import Acier, E, sig0, H

######## Paramètres géométriques et de maillage ########
Longueur, Largeur, hauteur = 0.5, 2., 2.
Nx, Ny, Nz = 2, 8, 8

eps = 0.005
Umax = eps * hauteur
mesh = create_box(COMM_WORLD, [np.array([0, 0, 0]), 
                                   np.array([Longueur, Largeur, hauteur])],
                                  [Nx, Ny, Nz])

dictionnaire_mesh = {"tags": [1, 2, 3, 4],
                     "coordinate": ["x", "y", "z", "z"], 
                     "positions": [0, 0, 0, hauteur]
                     }
mesh_manager = MeshManager(mesh, dictionnaire_mesh)

chargement = MyConstant(mesh, Umax, Type = "Rampe")
n_pas = 300

###### Paramètre du problème ######
dictionnaire = {"mesh_manager" : mesh_manager,
                "boundary_conditions": 
                    [{"component": "Ux", "tag": 1},
                     {"component": "Uy", "tag": 2},
                     {"component": "Uz", "tag": 3},
                     {"component": "Uz", "tag": 4, "value": chargement},
                    ],
                "analysis" : "static",
                "plasticity" : {"model" : "HPP_Plasticity", "sigY" : sig0, "Hardening" : "Isotropic", "Hardening_modulus" : H},
                "isotherm" : True
                }

pb = Tridimensional(Acier, dictionnaire)
pb.eps_list = [0]
pb.sig_list = [0]
pb.Force = pb.set_F(4, "z")

def query_output(problem, t):
    problem.eps_list.append(eps * t)
    problem.sig_list.append(problem.get_F(problem.Force)/ (Largeur * Longueur))
    
dictionnaire_solve = {}

solve_instance = Solve(pb, dictionnaire_solve, compteur=1, npas=n_pas)
solve_instance.query_output = query_output #Attache une fonction d'export appelée à chaque pas de temps
solve_instance.solve()

eps_c = sig0 / E
def sig_elast(eps):
    return E * eps

def sig_plas(eps):
    ET = E * H/(E + H)
    return (sig0 + ET * (eps - eps_c))
    
eps_elas_list = np.linspace(0, eps_c, 10)
sig_elas_list = [sig_elast(eps) for eps in eps_elas_list]
eps_plas_list = np.linspace(eps_c, eps, 10)
sig_plas_list = [sig_plas(eps) for eps in eps_plas_list]

if __name__ == "__main__": 
    plt.plot(pb.eps_list, pb.sig_list, linestyle = "--", color = "blue", label = "CHARON")
    plt.plot(eps_elas_list, sig_elas_list, linestyle = "-", color = "red")
    plt.plot(eps_plas_list, sig_plas_list, linestyle = "-", color = "red", label = "Analytical")
    plt.legend()
    plt.xlim(0, 1.05 * eps)
    plt.ylim(0, 1.1 * max(pb.sig_list))
    plt.xlabel(r"Déformation", size = 18)
    plt.ylabel(r"Force (N)" , size = 18)