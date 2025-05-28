"""
Test de traction uniaxiale sur un rectangle en déformation plane (pseudo-1D).

Ce script simule un essai de traction uniaxiale sur une plaque rectangulaire
en conditions de déformation plane, avec des conditions aux limites imposant
un état de déformation homogène équivalent à un problème 1D.

Paramètres géométriques:
    - Longueur: 1
    - Largeur: 0.5
    - Discrétisation: maillage 20x20 (quadrilatères)

Chargement:
    - Déplacement imposé (Umax): 0.002 (0.2% de déformation)

Conditions aux limites:
    - Blocage latéral sur les c√¥tés gauche et droite
    - Blocage vertical en bas et en haut
    - Déplacement horizontal imposé sur le c√¥té droit

Une comparaison est effectuée entre la force calculée numériquement et la solution analytique
dérivée de la théorie 1D, corrigée pour tenir compte de l'état de déformation plane.

Auteur: bouteillerp
Date de création: 11 Mars 2022
"""

from CharonX import *
import matplotlib.pyplot as plt
import pytest
import sys
sys.path.append("../")
from Traction_1D_cartesien_solution_analytique import *
from Generic_isotropic_material import Acier, kappa, mu, eos_type, devia_type

model = Plane_strain

###### Paramètre géométrique ######
Largeur = 0.5
Longueur = 1

###### Chargement ######
Umax = 0.002

mesh = create_rectangle(MPI.COMM_WORLD, [(0, 0), (Longueur, Largeur)], [20, 20], CellType.quadrilateral)
   
class Plate(model):
    def __init__(self, material):
        model.__init__(self, material, analysis = "static", isotherm = True)
          
    def define_mesh(self):
        return create_rectangle(MPI.COMM_WORLD, [(0, 0), (Longueur, Largeur)], [20, 20], CellType.quadrilateral)

    def prefix(self):
        if __name__ == "__main__": 
            return "Plaque_2D_pseudo_1D"
        else:
            return "Test"
        
    def set_boundary(self):
        self.mesh_manager.mark_boundary([1, 2, 3, 4], ["x", "y", "x", "y"], [0, 0, Longueur, Largeur])
        
    def set_boundary_condition(self):
        self.bcs.add_Ux(region=1)
        self.bcs.add_Uy(region=2)
        self.bcs.add_Uy(region=4)
        chargement = MyConstant(self.mesh, Umax, Type = "Rampe")
        self.bcs.add_Ux(value = chargement, region=3)
        
    def set_output(self):
        self.eps_list = []
        self.F_list = []
        self.Force = self.set_F(3, "x")
        return {"U": True}
    
    def csv_output(self):
        return {'U': True}
    
    def query_output(self, t):
        self.eps_list.append(Umax / Longueur * t)
        self.F_list.append(2 * self.get_F(self.Force))
        
    def final_output(self):
            
        solution_analytique = array([sigma_xx(epsilon, kappa, mu, eos_type, devia_type) for epsilon in self.eps_list])
        eps_list_percent = [100 * eps for eps in self.eps_list]
        numerical_results = array(self.F_list)
        # On calcule la différence entre les deux courbes
        len_vec = len(solution_analytique)
        diff_tot = solution_analytique - numerical_results
        # Puis on réalise une sorte d'intégration discrète
        integrale_discrete = sum(abs(diff_tot[j]) for j in range(len_vec))/sum(abs(solution_analytique[j]) for j in range(len_vec))
        print("La difference est de", integrale_discrete)
        assert integrale_discrete < 0.001, "Static 1D traction fail"
        if __name__ == "__main__": 
            plt.scatter(eps_list_percent, numerical_results, marker = "x", color = "blue", label="CHARON")
            plt.plot(eps_list_percent, solution_analytique, linestyle = "--", color = "red", label = "Analytique")
            plt.xlim(0, 1.05 * eps_list_percent[-1])
            plt.ylim(0, 1.05 * numerical_results[-1])
            plt.xlabel(r"Déformation(%)", size = 18)
            plt.ylabel(r"Force (N)", size = 18)
            plt.legend()
            plt.show()
            
def test_Traction2D():
    pb = Plate(Acier)
    Solve(pb, compteur=1, npas=20)
test_Traction2D()