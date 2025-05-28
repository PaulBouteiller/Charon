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

from CharonX import *
import matplotlib.pyplot as plt
from material import Re, Ri, E, nu, sig0, H, q_lim, K, G
from deplacement_analytique import deplacement_ilyushin, deplacement_elastique
import pandas as pd
model = SphericalUD
dico_eos = {"E" : E, "nu" : nu, "alpha" : 1}
dico_devia = {"E" : E, "nu" : nu}
Acier = Material(1, 1, "IsotropicHPP", "IsotropicHPP", dico_eos, dico_devia)
Nx=100
p_applied = 1.1 * q_lim
npas = 20000
compteur = 100
   
class IsotropicBall(model):
    def __init__(self, material):
        model.__init__(self, material, analysis = "static", plastic = "HPP_Plasticity", isotherm = True)
          
    def define_mesh(self):
        return create_interval(MPI.COMM_WORLD, Nx, [np.array(Ri), np.array(Re)])
    
    def prefix(self):
        if __name__ == "__main__": 
            return "Compression_spherique_1D"
        else:
            return "Test"
    
    def set_boundary(self):
        self.mesh_manager.mark_boundary([1, 2], ["x", "x"], [Ri, Re])
        
    def set_loading(self):
        self.loading.add_F(p_applied * self.load, self.u_, self.ds(1))
        
    def set_plastic(self):
        self.constitutive.plastic.set_plastic(sigY = sig0, H = H, hardening = "Isotropic")
        
    def csv_output(self):
        return {"U" : ["Boundary", 1]}

pb = IsotropicBall(Acier)
Solve(pb, compteur=compteur, npas=npas)

df_u = pd.read_csv("Compression_spherique_1D-results/U.csv")
colonnes_numpy = [df_u[colonne].to_numpy() for colonne in df_u.columns]
u_int = [colonnes_numpy[i+1][0] for i in range(len(colonnes_numpy)-1)]
p_list = [p_applied * t for t in np.linspace(0, 1, len(u_int))]
plt.plot(u_int, p_list, color='r', label = "Charon")

# Plage de pression interne (échelle similaire à la Figure 2: 0-800 MPa)
p_int_range = np.linspace(q_lim, p_applied, 200)

p_int_elastique_range = np.linspace(0, q_lim, 200)
p_ext = 0
lambda_val = 1 - H / (3 * G)
w_a_elastique = deplacement_elastique(p_int_elastique_range, Ri, Re, G, K, p_ext)
w_a = deplacement_ilyushin(p_int_range, Ri, Re, G, K, sig0, lambda_val, p_ext)
plt.plot(w_a_elastique, p_int_elastique_range, color="black", linewidth=2, 
         label='Elastique', linestyle='-')
plt.plot(w_a, p_int_range, color="black", linewidth=2, 
         label='Plastique', linestyle='--')
plt.legend()
