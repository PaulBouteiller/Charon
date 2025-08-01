"""
Simulation élasto-plastique d'une sphère creuse en coordonnées sphériques 1D.

Ce script implémente et exécute une simulation élasto-plastique d'une sphère creuse
soumise à une pression interne croissante en utilisant un modèle 1D en coordonnées
sphériques. Il compare les résultats numériques aux solutions analytiques élastique
et plastique.

Modèle numérique:
    - Géométrie: SphericalUD (coordonnées sphériques 1D)
    - Maillage: 100 éléments radiaux entre Ri et Re
    - Analyse: Statique avec plasticité HPP_Plasticity
    - Écrouissage: Isotrope

Paramètres de simulation:
    - Pression appliquée: 1.1 × q_lim (10% au-dessus de la limite élastique)
    - Nombre de pas: 20000 (chargement progressif) Attention un nombre très important
    de pas peut être requis pour la convergence à cause du caractère explicite
    du traitement de la plasticité
    - Fréquence de sortie: tous les 100 pas

Conditions aux limites:
    - Surface interne (r=Ri): Pression imposée progressive
    - Surface externe (r=Re): Libre

Validation:
    La simulation génère une courbe pression-déplacement comparée aux solutions
    analytiques pour valider:
    - Le comportement élastique linéaire (p < q_lim)
    - La transition élasto-plastique (p ≈ q_lim)
    - Le comportement plastique non-linéaire (p > q_lim)

Sortie:
    - Fichier CSV: déplacements de la surface interne
    - Graphique: courbe pression vs déplacement avec solutions analytiques

Auteur: bouteillerp
Date de création: 28 Mai 2025
"""

from Charon import  create_1D_mesh, MeshManager, SphericalUD, Solve
import matplotlib.pyplot as plt
from deplacement_analytique import deplacement_ilyushin, deplacement_elastique
from pandas import read_csv
import numpy as np

import sys
sys.path.append("../../")
from Generic_isotropic_material import Acier, mu, kappa, sig0, H

###### Paramètre géométrique ######
Re = 600
Ri = 300.0


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

###### Maillage ######
Nx=50
mesh = create_1D_mesh(Ri, Re, Nx)
dictionnaire_mesh = {"tags": [1, 2], "coordinate": ["r", "r"], "positions": [Ri, Re]}
mesh_manager = MeshManager(mesh, dictionnaire_mesh)


p_applied = 1.1 * q_lim

npas = 20000
compteur = 100

###### Paramètre du problème ######
dictionnaire = {"mesh_manager" : mesh_manager,
                "loading_conditions": 
                    [{"type": "surfacique", "component" : "F", "tag": 1, "value" : p_applied}
                    ],
                "analysis" : "static",
                "plasticity" : plasticity_dic,
                "isotherm" : True
                }
    
pb = SphericalUD(Acier, dictionnaire)

###### Paramètre de la résolution ######
dictionnaire_solve = {
    "Prefix" : "Dilatation_spherique_elastoplast",
    "csv_output" : {"U" : True}
    }

solve_instance = Solve(pb, dictionnaire_solve, compteur=compteur, npas=npas)
solve_instance.solve()

df_u = read_csv("Dilatation_spherique_elastoplast-results/U.csv")
colonnes_numpy = [df_u[colonne].to_numpy() for colonne in df_u.columns]
u_int = [colonnes_numpy[i+1][0] for i in range(len(colonnes_numpy)-1)]
p_list = [p_applied * t for t in np.linspace(0, 1, len(u_int))]
plt.plot(u_int, p_list, color='r', label = "Charon")

p_int_range = np.linspace(q_lim, p_applied, 200)

p_int_elastique_range = np.linspace(0, q_lim, 200)
p_ext = 0
lambda_val = 1 - H / (3 * mu)
w_a_elastique = deplacement_elastique(p_int_elastique_range, Ri, Re, mu, kappa, p_ext)
w_a = deplacement_ilyushin(p_int_range, Ri, Re, mu, kappa, sig0, lambda_val, p_ext)
plt.plot(w_a_elastique, p_int_elastique_range, color="black", linewidth=2, 
         label='Elastique', linestyle='-')
plt.plot(w_a, p_int_range, color="black", linewidth=2, 
         label='Plastique', linestyle='--')
plt.xlim(0, 1.2 * max(u_int))
plt.ylim(0, 1.05 * max(p_list))
plt.legend()
