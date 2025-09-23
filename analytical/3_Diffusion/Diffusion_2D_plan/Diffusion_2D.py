"""
Test de validation pour la diffusion thermique 2D en déformation plane

Ce script implémente et exécute un test de validation pour l'équation de diffusion
thermique en 2D. Il simule la diffusion d'une source de chaleur localisée
dans un domaine carré, avec deux configurations possibles: une source carrée ou circulaire.

Cas test:
---------
- Domaine carré de dimensions 210×210 μm²
- Température initiale: T = 800K dans une région centrale (carrée ou circulaire),
  T = 300K ailleurs
- Conditions aux limites: température fixée à 300K sur le bord gauche du domaine
- Évolution du champ de température pendant 1 μs

Théorie:
--------
L'équation de diffusion thermique en 2D s'écrit:
    ∂T/∂t = D·[∂²T/∂x² + ∂²T/∂y²]

où D = λ/(ρ·C) est le coefficient de diffusion thermique.

Pour une source circulaire, la solution analytique fait intervenir des fonctions
de Bessel. Ce test se limite à une vérification qualitative de la diffusion.

Paramètres matériaux:
--------------------
- Conductivité thermique: λ = 240 W/(m·K)
- Capacité thermique massique: C = 1000 J/(kg·K)
- Masse volumique: ρ = 2.785×10³ kg/m³
- Coefficient de diffusion: D = λ/(ρ·C) ≈ 8.62×10⁻⁵ m²/s

Auteur: bouteillerp
Créé le: Thu Feb 23 16:45:40 2023
"""
from Charon import LinearThermal, create_rectangle, MeshManager, Solve, PlaneStrain, CellType
from mpi4py.MPI import COMM_WORLD
# import matplotlib.pyplot as plt 
from ufl import conditional, lt, And, gt
import sys
sys.path.append("../../")
from Generic_isotropic_material import Acier
from ufl import SpatialCoordinate

from numpy import array
###### Modèle thermique ######
lmbda_diff = 45
AcierTherm = LinearThermal(lmbda_diff)

###### Paramètre géométrique ######
Largeur = 210e-3
bord_gauche = -105e-3
bord_droit = bord_gauche + Largeur
taille_tache = 40e-3
point_gauche = -taille_tache / 2
point_droit = taille_tache / 2

###### Champ de température initial ######
Tfin = 1e-3
Tfroid = 300.
TChaud = 800.

###### Paramètre temporel ######
sortie = 1
pas_de_temps = 5e-6
pas_de_temps_sortie = sortie * pas_de_temps

n_sortie = int(Tfin/pas_de_temps_sortie)

###### Maillage ######
Nx = 200
Ny = 200
test = "carre"


mesh = create_rectangle(COMM_WORLD, [(bord_gauche, bord_gauche), (bord_droit, bord_droit)], [Nx, Ny], CellType.quadrilateral)
dictionnaire_mesh = {"tags": [1, 2, 3, 4], "coordinate": ["x", "x", "y", "y"], "positions": [bord_gauche, bord_droit, bord_gauche, bord_droit]}
mesh_manager = MeshManager(mesh, dictionnaire_mesh)

dictionnaire = {"material" : Acier,
                "mesh_manager" : mesh_manager,
                "boundary_conditions": 
                    [{"component": "T", "tag": 1, "value" : Tfroid},
                     {"component": "T", "tag": 2, "value" : Tfroid},
                     {"component": "T", "tag": 3, "value" : Tfroid},
                     {"component": "T", "tag": 4, "value" : Tfroid},
                    ],
                "Thermal_material" : AcierTherm, 
                "analysis" : "Pure_diffusion"
                }
pb = PlaneStrain(dictionnaire)

#%%
x = SpatialCoordinate(mesh)
if test == "carre":
    ufl_condition = conditional(And(gt(x[1], point_gauche), And(lt(x[1], point_droit), \
                                    And(lt(x[0], point_droit), gt(x[0], point_gauche)))), TChaud, Tfroid)
elif test == "rond":
    centre = array([0., 0.])
    # absisse =x[0] - centre[0]
    # ufl_condition = conditional(lt(absisse, taille_tache), TChaud, Tfroid)     
    ufl_condition = conditional(lt((x[0]-centre[0])**2 + (x[1]-centre[1])**2, taille_tache**2), TChaud, Tfroid)  

output_name = "Diffusion_2D"
dictionnaire_solve = {"Prefix" : output_name, "output" : {"T" : True}, "initial_conditions" : {"T" : ufl_condition}}
solve_instance = Solve(pb, dictionnaire_solve, compteur = sortie, TFin=Tfin, scheme = "fixed", dt = pas_de_temps)
solve_instance.solve()