"""
Test de compression d'une sphère creuse en axisymétrique (pseudo-2D).

Ce script simule la compression d'une sphère creuse soumise à une pression externe
en utilisant un modèle axisymétrique, puis compare la solution numérique avec la
solution analytique.

Paramètres géométriques:
    - Rayon interne (Rint): 9
    - Rayon externe (Rext): 11
    - Maillage: 40 éléments circonférentiels, 10 éléments radiaux

Chargement:
    - Pression externe (Pext): 10

Conditions aux limites:
    - Déplacement vertical bloqué sur l'axe de symétrie
    - Déplacement radial bloqué sur l'axe de symétrie
    - Pression sur la face externe

Une comparaison est effectuée entre le champ de déplacement radial calculé
numériquement et la solution analytique pour un domaine sphérique creux.

Auteur: bouteillerp
Date de création: 11 Mars 2022
"""
from CharonX import *
import time
import matplotlib.pyplot as plt
import pytest


model = Axisymetric
###### Modèle mécanique ######
E = 1e5
nu = 0.3
dico_eos = {"E" : E, "nu" : nu, "alpha" : 1}
dico_devia = {"E" : E, "nu" : nu}
Acier = Material(1, 1, "IsotropicHPP", "IsotropicHPP", dico_eos, dico_devia)

###### Paramètre géométrique ######
Rint = 9
Rext = 11

###### Chargement ######
Pext = 10
   
class CoquilleAxi(model):
    def __init__(self, material):
        model.__init__(self, material, analysis = "static", isotherm = True)
          
    def define_mesh(self):
        mesh, _, facets = axi_sphere(Rint, Rext, 40, 10, tol_dyn = 1e-5, quad = False)
        self.facet_tag = facets
        return mesh

    def prefix(self):
        if __name__ == "__main__": 
            return "Sphere_axi"
        else:
            return "Test"
        
    def set_boundary_condition(self):
        self.bcs.add_Uz(region = 1)
        self.bcs.add_Ur(region = 2)
        
    def set_loading(self):
        self.loading.add_pressure(Pext * self.load, self.mesh, self.u_, self.ds(3))
        
    def csv_output(self):
        return {"U" : ["Boundary", 1]}
        
    def set_output(self):
        return {"U" : True}
        
    def final_output(self):
        u_csv = read_csv("Sphere_axi-results/Displacement0.csv")
        resultat = [u_csv[colonne].to_numpy() for colonne in u_csv.columns]
        r_result = resultat[0]
        solution_numerique = -resultat[-2]
        len_vec = len(r_result)
        p_applied = -Pext
        def ur(r):
            return  - Rext**3 / (Rext**3 - Rint**3) * ((1 - 2*nu) * r + (1 + nu) * Rint**3 / (2 * r**2)) * p_applied / E
        solution_analytique = np.array([ur(x) for x in r_result])
        # On calcule la différence entre les deux courbes
        diff_tot = solution_analytique - solution_numerique
        # Puis on réalise une sorte d'intégration discrète
        int_discret = sum(abs(diff_tot[j]) for j in range(len_vec))/sum(abs(solution_analytique[j]) for j in range(len_vec))
        print("La difference est de", int_discret)
        # assert int_discret < 1e-3, "Cylindrical static compression fail"
        if __name__ == "__main__": 
            plt.plot(r_result, solution_analytique, linestyle = "--", color = "red")
            plt.scatter(r_result, solution_numerique, marker = "x", color = "blue")
            
            plt.xlim(Rint, Rext)
            plt.xlabel(r"$r$", size = 18)
            plt.ylabel(r"Déplacement radial", size = 18)
            plt.savefig("../../../Notice/fig/1D_spherique_compression_pseudo_2D.pdf", bbox_inches = 'tight')
            plt.show()
            
pb = CoquilleAxi(Acier)
Solve(pb, compteur=1, npas=20)