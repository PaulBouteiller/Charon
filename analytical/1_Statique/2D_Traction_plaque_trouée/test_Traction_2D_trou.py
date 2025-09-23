"""
Solution analytique pour une plaque trouée soumise à une traction.

Ce module implémente la solution analytique de Kirsch pour le champ de déplacement
dans une plaque infinie avec un trou circulaire soumise à une contrainte de traction
uniforme à l'infini.

Fonctions:
    prefactor(nu, E, sig_infty): Calcule le préfacteur commun aux expressions de déplacement
    ur(a, r, nu, E, sig_infty, theta): Calcule le déplacement radial
    utheta(a, r, nu, E, sig_infty, theta): Calcule le déplacement tangentiel

Paramètres:
    - a: Rayon du trou
    - r: Distance radiale depuis le centre du trou
    - nu: Coefficient de Poisson
    - E: Module d'Young
    - sig_infty: Contrainte appliquée à l'infini
    - theta: Angle (en radians) par rapport à l'axe horizontal
"""

from Charon import PlaneStrain, Solve, MeshManager
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
import pytest
from numpy import pi
from mpi4py.MPI import COMM_WORLD
import gmsh
from dolfinx.io.gmshio import model_to_mesh
import sys
sys.path.append("../../")
from Generic_isotropic_material import Acier, E, nu

###### Paramètre géométrique ######
Largeur = 2.
Longueur = 1
R = 0.06
mesh_size = 0.02
        
###### Chargement ######
f_surf = 1e3

def quarter_perforated_plate(width, height, radius, h):
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("Model")
    geom = gmsh.model.geo
    center = geom.add_point(0, 0, 0, h)
    p1 = geom.add_point(radius, 0, 0, h)
    p2 = geom.add_point(width, 0, 0, h)
    p3 = geom.add_point(width, height, 0, h)
    p4 = geom.add_point(0, height, 0, h)
    p5 = geom.add_point(0, radius, 0, h)
    
    arc = geom.add_circle_arc(p1, center, p5)
    l1 = geom.add_line(p5, p4)
    l2 = geom.add_line(p4, p3)
    l3 = geom.add_line(p3, p2)
    l4 = geom.add_line(p2, p1)
    
    loop = geom.add_curve_loop([arc, l1, l2, l3, l4])
    surf = geom.add_plane_surface([loop])
    geom.synchronize()
    
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h * 2)
    gmsh.option.setNumber("Mesh.Algorithm", 1)
    gmsh.model.addPhysicalGroup(2, [surf], 1)
    gmsh.model.mesh.generate(2)
    
    domain, _, _ = model_to_mesh(gmsh.model, COMM_WORLD, 0, gdim=2)
    gmsh.finalize()
    return domain, _, _

mesh, _, _ = quarter_perforated_plate(Largeur, Longueur, R, mesh_size)
dictionnaire_mesh = {"tags": [1, 2, 3],
                     "coordinate": ["x", "y", "x"], 
                     "positions": [0, 0, Largeur]
                     }
mesh_manager = MeshManager(mesh, dictionnaire_mesh)

dictionnaire = {"material" : Acier,
                "mesh_manager" : mesh_manager,
                "boundary_conditions": 
                    [{"component": "Ux", "tag": 1},
                     {"component": "Uy", "tag": 2}
                     ],
                "loading_conditions": 
                    [{"type": "surfacique", "component" : "Fx", "tag": 3, "value" : f_surf}],
                "analysis" : "static",
                "isotherm" : True
                }
                     
pb = PlaneStrain(dictionnaire)

###### Paramètre de la résolution ######
output_name = "Plaque_2D"
dictionnaire_solve = {"Prefix" : output_name, "csv_output" : {"U" : ["Boundary", 1]},
                      "output" : {"U" : True}}

solve_instance = Solve(pb, dictionnaire_solve, compteur=1, npas=20)
solve_instance.solve()

u_csv = read_csv(output_name+"-results/U.csv")
resultat = [u_csv[colonne].to_numpy() for colonne in u_csv.columns]
x_result = resultat[1]
displacement = resultat[-1]

from math import cos, sin

def prefactor(nu, E, sig_infty):
    return sig_infty * (1 + nu) / (2 * E)

def ur(a, r, nu, E, sig_infty, theta):
    pref = prefactor(nu, E, sig_infty)
    u_r = pref * ((1 - 2 * nu) * r + a**2 / r + \
                  (r - a**4 / r**3 + 4 * a**2 / r * (1 - nu)) * cos(2 * theta))
    return u_r

def utheta(a, r, nu, E, sig_infty, theta):
    pref = prefactor(nu, E, sig_infty)
    u_thet = -pref * (r + a**4 / r**3 + 2 * a**2 / r * (1- 2 * nu)) * sin(2 * theta)
    return u_thet
    
sol_anat = np.array([ur(R, x, nu, E, f_surf, pi/2) for x in x_result])
plt.plot(x_result, sol_anat, linestyle = "--", color = "red")
plt.scatter(x_result, displacement, marker = "x", color = "blue")
plt.xlabel(r"$r$", size = 18)
plt.ylabel(r"Déplacement radial", size = 18)