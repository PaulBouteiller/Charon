"""
Simulation élasto-plastique d'une sphère creuse en modèle 2D axisymétrique.

Ce script implémente et exécute une simulation élasto-plastique d'une sphère creuse
soumise à une pression interne en utilisant un modèle 2D axisymétrique . 
Il compare les résultats numériques aux solutions analytiques pour valider l'approche bidimensionnelle.

Modèle numérique:
    - Géométrie: Axisymmetric (modèle 2D de révolution)
    - Maillage: Maillage sphérique généré par axi_sphere()
        * 40 éléments circonférentiels
        * 10 éléments radiaux
    - Analyse: Statique avec plasticité HPP_Plasticity
    - Écrouissage: Cinématique

Paramètres de simulation:
    - Pression appliquée: 1.1 × q_lim
    - Nombre de pas: 4000 Attention un nombre très important
    de pas peut être requis pour la convergence à cause du caractère explicite
    - Chargement: Pression surfacique via add_pressure()

Conditions aux limites:
    - Axe de symétrie: Déplacements bloqués (Uz=0, Ur=0)
    - Surface interne: Pression imposée progressive
    - Surface externe: Libre

Différences avec le modèle 1D:
    - Maillage 2D complet vs maillage 1D radial
    - Écrouissage cinématique vs isotrope
    - Chargement par pression surfacique vs force nodale
    - Conditions aux limites axisymétriques

Validation:
    Compare les résultats 2D aux solutions analytiques 1D pour évaluer:
    - La cohérence entre approches 1D et 2D
    - L'influence de la modélisation géométrique
    - La précision du maillage 2D

Sortie:
    - Fichier CSV: déplacements et déformations plastiques
    - Graphique: courbe pression vs déplacement avec solutions analytiques

Auteur: bouteillerp
Date de création: 28 Mai 2025
"""

from Charon import axi_sphere, Axisymmetric, Solve, MeshManager
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../../")
from Generic_isotropic_material import Acier, mu, kappa, sig0, H
from deplacement_analytique import deplacement_ilyushin, deplacement_elastique

###### Paramètre géométrique ######
Re = 600
Ri = 300.0

###### Plasticity dictionnaire ######
plasticity_model = "Finite_Plasticity"
plasticity_dic = {"model" : plasticity_model}
if plasticity_model == "HPP_Plasticity" or plasticity_model == "Finite_Plasticity":
    plasticity_dic.update({"sigY" : sig0, "Hardening" : "Isotropic", "Hardening_modulus" : H})
elif plasticity_model == "J2_JAX":
    
    b = 10.0
    sigu = 750.0
    
    import jax.numpy as jnp
    def yield_function(p):
        return sig0 + (sigu - sig0) * (1 - jnp.exp(-b * p))
    plasticity_dic.update({"sigY" : sig0, "Hardening" : "NonLinear", "Hardening_func" : yield_function})

#Paramètre élasto-plastique
q_lim = float(2 * np.log(Re / Ri) * sig0)
p_applied = 1.1 * q_lim
p_mid = q_lim
npas = 4000

mesh, _, facets = axi_sphere(Ri, Re, 40, 10, tol_dyn = 1e-5, quad = False)
dictionnaire_mesh = {"facet_tag": facets}
mesh_manager = MeshManager(mesh, dictionnaire_mesh)

dictionnaire = {"mesh_manager" : mesh_manager,
                "boundary_conditions": 
                    [{"component": "Uz", "tag": 1},
                     {"component": "Ur", "tag": 2}
                     ],
                "loading_conditions": 
                    [{"type": "surfacique", "component" : "pressure", "tag": 4, "value" : p_applied}],
                "analysis" : "static",
                "plasticity" : plasticity_dic,
                "isotherm" : True
                }

pb = Axisymmetric(Acier, dictionnaire)

###### Paramètre de la résolution ######
dictionnaire_solve = {
    "Prefix" : "Sphere_axi",
    "csv_output" : {"U" : ["Boundary", 1]}
    }

solve_instance = Solve(pb, dictionnaire_solve, compteur=1, npas=npas)
solve_instance.solve()

u_csv = read_csv("Sphere_axi-results/U.csv")
resultat = [u_csv[colonne].to_numpy() for colonne in u_csv.columns]
r_result = resultat[0]
solution_numerique = -resultat[-2]
inner_displacement = [resultat[2 * (i+1)][0] for i in range(len(resultat)//2 - 1)]
q_list = [p_applied * t for t in np.linspace(0, 1, npas)]
plt.plot(inner_displacement, q_list, linestyle = "--", color = "red")

# Plage de pression interne (échelle similaire à la Figure 2: 0-800 MPa)
p_int_range = np.linspace(p_mid, p_applied, 200)

p_int_elastique_range = np.linspace(0, p_mid, 200)
p_ext = 0
lambda_val = 1 - H / (3 * mu)
w_a_elastique = deplacement_elastique(p_int_elastique_range, Ri, Re, mu, kappa, p_ext)
w_a = deplacement_ilyushin(p_int_range, Ri, Re, mu, kappa, sig0, lambda_val, p_ext)
plt.plot(w_a_elastique, p_int_elastique_range, color="black", linewidth=2, 
         label='Elastique', linestyle='-')
plt.plot(w_a, p_int_range, color="black", linewidth=2, 
         label='Plastique', linestyle='--')
plt.legend()
plt.xlim(0, 1.02 * max(inner_displacement))
plt.ylim(0, 1.02 * p_applied)