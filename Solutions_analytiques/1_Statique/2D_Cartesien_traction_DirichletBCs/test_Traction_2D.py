"""
Test de traction 2D avec conditions aux limites de Dirichlet.

Ce script simule un essai de traction uniaxiale sur une plaque rectangulaire
en conditions de déformation plane avec des déplacements imposés sur les bords.

Paramètres géométriques:
    - Longueur: 1
    - Largeur: 0.5
    - Discrétisation: maillage 20×20 (quadrilatères)

Chargement:
    - Déplacement imposé (Umax): 0.002 (0.2% de déformation)

Conditions aux limites:
    - Déplacement horizontal bloqué sur le bord gauche
    - Déplacement vertical bloqué sur le bord inférieur
    - Déplacement horizontal imposé sur le bord droit

Le script calcule la force résultante et la compare avec la solution analytique
pour un problème de déformation plane (correction par le facteur 1/(1-nu²)).
Une assertion vérifie que l'erreur relative est inférieure à 1%.

Auteur: bouteillerp
Date de création: 11 Mars 2022
"""
from CharonX import *
import time
import matplotlib.pyplot as plt
import pytest
from numpy import pi
import sys
sys.path.append("../")
from Generic_isotropic_material import *

model = Plane_strain
###### Paramètre géométrique ######
Largeur = 0.5
Longueur = 1

###### Chargement ######
Umax = 0.002

class Plate(model):
    def __init__(self, material):
        model.__init__(self, material, analysis = "static", isotherm = True)
          
    def define_mesh(self):
        return create_rectangle(MPI.COMM_WORLD, [(0, 0), (Longueur, Largeur)], [20, 20], CellType.quadrilateral)

    def prefix(self):
        if __name__ == "__main__": 
            return "Plaque_2D"
        else:
            return "Test"
        
    def set_boundary(self):
        self.mesh_manager.mark_boundary([1, 2, 3], ["x", "y", "x"], [0, 0, Longueur])
        
    def set_boundary_condition(self):
        self.bcs.add_Ux(region=1)
        self.bcs.add_Uy(region=2)
        chargement = MyConstant(self.mesh, Umax, Type = "Rampe")
        self.bcs.add_Ux(value = chargement, region=3)
        
    def set_output(self):
        self.eps_list = []
        self.F_list = []
        self.Force = self.set_F(3, "x")
        return {}
    
    def csv_output(self):
        return {'U': True}
    
    def query_output(self, t):
        self.eps_list.append(Umax / Longueur * t)
        self.F_list.append(self.get_F(self.Force))
        
    def final_output(self):
        def force_elast(eps):
            return E * eps * Largeur /(1 - nu**2)
        
        solution_analytique = array([force_elast(eps) for eps in self.eps_list])
        eps_list_percent = [100 * eps for eps in self.eps_list]
        numerical_results = array(self.F_list)
        # On calcule la différence entre les deux courbes
        len_vec = len(solution_analytique)
        diff_tot = solution_analytique - numerical_results
        # Puis on réalise une sorte d'intégration discrète
        integrale_discrete = sum(abs(diff_tot[j]) for j in range(len_vec))/sum(abs(solution_analytique[j]) for j in range(len_vec))
        print("La difference est de", integrale_discrete)
        assert integrale_discrete < 0.01, "Static 1D traction fail"
        if __name__ == "__main__": 
            plt.scatter(eps_list_percent, self.F_list, marker = "x", color = "blue", label="CHARON")
            plt.plot(eps_list_percent, solution_analytique, linestyle = "--", color = "red", label = "Analytique")
            plt.xlim(0, 1.1 * eps_list_percent[-1])
            plt.ylim(0, 1.1 * self.F_list[-1])
            plt.xlabel(r"Déformation(%)", size = 18)
            plt.ylabel(r"Force (N)", size = 18)
            plt.legend()
            plt.show()
            
def test_Traction2D():
    pb = Plate(Acier)
    Solve(pb, compteur=1, npas=20)
test_Traction2D()