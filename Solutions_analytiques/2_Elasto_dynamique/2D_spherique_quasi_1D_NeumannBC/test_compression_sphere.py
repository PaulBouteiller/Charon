"""
Test de validation pour la compression d'une sphère creuse

Ce script exécute la simulation de la compression dynamique d'une sphère creuse
et compare les résultats obtenus avec un modèle 1D en coordonnées sphériques
et un modèle 2D axisymétrique.

Le test vérifie que les deux approches de modélisation produisent des résultats
cohérents pour le déplacement radial, confirmant ainsi la validité de l'approche
1D pour ce type de problème.

Auteur: bouteillerp
Créé le: Thu Feb 8 15:18:29 2024
"""
from Sphere_compression import *

for model in [SphericalUD, Axisymetric]:
    run_test(model)
    
    
df_1 = read_csv("Sphere_axi-results/Displacement0.csv")
resultat_axi = [df_1[colonne].to_numpy() for colonne in df_1.columns]
ur_axi = [resultat_axi[2 * i + 2] for i in range((len(resultat_axi)-2)//2)]


df_2 = read_csv("Sphere-results/Displacement0.csv")
resultat_1D = [df_2[colonne].to_numpy() for colonne in df_2.columns]
ur_1D = [resultat_1D[i + 2] for i in range((len(resultat_1D)-2))]

for j in range(len(ur_axi)-1):
    plt.plot(resultat_axi[0], ur_axi[j+1], linestyle = "--")
    
for j in range(len(ur_1D)-1):
    plt.plot(resultat_1D[0], ur_1D[j+1], linestyle = "-")
