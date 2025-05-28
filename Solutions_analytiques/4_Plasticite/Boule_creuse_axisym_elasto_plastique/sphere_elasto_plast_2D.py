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

from CharonX import *
import time
import matplotlib.pyplot as plt
from material import Re, Ri, E, nu, sig0, H, q_lim, K, G
from deplacement_analytique import deplacement_ilyushin, deplacement_elastique


model = Axisymmetric
dico_eos = {"E" : E, "nu" : nu, "alpha" : 1}
dico_devia = {"E" : E, "nu" : nu}
Acier = Material(1, 1, "IsotropicHPP", "IsotropicHPP", dico_eos, dico_devia)
name = "Sphere_axi"
p_applied = 1.1 * q_lim
p_mid = q_lim
npas = 4000
   
class CoquilleAxi(model):
    def __init__(self, material):
        model.__init__(self, material, analysis = "static", plastic = "HPP_Plasticity", isotherm = True)
          
    def define_mesh(self):
        mesh, _, facets = axi_sphere(Ri, Re, 40, 10, tol_dyn = 1e-5, quad = False)
        self.facet_tag = facets
        return mesh

    def prefix(self):
        return name
        
    def set_boundary(self):
        """
        Set up boundary tags.
    
        """
        self.mesh_manager.facet_tag = self.facet_tag
        
    def set_boundary_condition(self):
        self.bcs.add_Uz(region = 1)
        self.bcs.add_Ur(region = 2)
        
    def set_plastic(self):
        self.constitutive.plastic.set_plastic(sigY = sig0, H = H, hardening = "Kinematic") 
        
    def set_loading(self):
        self.loading.add_pressure(p_applied * self.load, self.u_, self.ds(4))
        
    def csv_output(self):
        return {"U" : ["Boundary", 1]}
        
    def set_output(self):
        return {"eps_p" : True}
            
pb = CoquilleAxi(Acier)
Solve(pb, compteur=1, npas=npas)

u_csv = read_csv(name + "-results/U.csv")
resultat = [u_csv[colonne].to_numpy() for colonne in u_csv.columns]
r_result = resultat[0]
solution_numerique = -resultat[-2]
inner_displacement = [resultat[2 * (i+1)][0] for i in range(len(resultat)//2 - 1)]
q_list = [p_applied * t for t in linspace(0, 1, npas)]
plt.plot(inner_displacement, q_list, linestyle = "--", color = "red")

# Plage de pression interne (échelle similaire à la Figure 2: 0-800 MPa)
p_int_range = np.linspace(p_mid, p_applied, 200)

p_int_elastique_range = np.linspace(0, p_mid, 200)
p_ext = 0
lambda_val = 1 - H / (3 * G)
w_a_elastique = deplacement_elastique(p_int_elastique_range, Ri, Re, G, K, p_ext)
w_a = deplacement_ilyushin(p_int_range, Ri, Re, G, K, sig0, lambda_val, p_ext)
plt.plot(w_a_elastique, p_int_elastique_range, color="black", linewidth=2, 
         label='Elastique', linestyle='-')
plt.plot(w_a, p_int_range, color="black", linewidth=2, 
         label='Plastique', linestyle='--')
plt.legend()
plt.xlim(0, 1.02 * max(inner_displacement))
plt.ylim(0, 1.02 * p_applied)