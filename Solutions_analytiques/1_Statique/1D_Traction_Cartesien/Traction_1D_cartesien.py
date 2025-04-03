"""
Test de traction simple en coordonnées cartésiennes 1D.

Ce script simule un essai de traction uniaxiale sur une barre 1D et compare
les résultats numériques avec la solution analytique.

Paramètres géométriques:
    - Longueur de la barre: 1
    - Discrétisation: 2 éléments

Chargement:
    - Déplacement imposé (Umax): 1e-3 (0.1% de déformation)

La solution analytique utilise les expressions développées dans le module
Traction_1D_cartesien_solution_analytique pour différents modèles constitutifs.
Une comparaison est effectuée entre la force calculée numériquement et analytiquement.

Auteur: bouteillerp
Date de création: 24 Juillet 2023
"""
from CharonX import *
import matplotlib.pyplot as plt
import numpy as np
import pytest
import sys
sys.path.append("../")
from Traction_1D_cartesien_solution_analytique import *
sys.path.append("../../")
from Generic_isotropic_material import *
model = CartesianUD

###### Paramètre géométrique ######
Longueur = 1

###### Chargement ######
Umax=1e-3   
class IsotropicBeam(model):
    def __init__(self, material):
        model.__init__(self, material, analysis = "static", isotherm = True)
          
    def define_mesh(self):
        return create_interval(MPI.COMM_WORLD, 2, [np.array(0), np.array(Longueur)])
    
    def prefix(self):
        if __name__ == "__main__": 
            return "Traction_1D"
        else:
            return "Test"
            
    def set_boundary(self):
        self.mesh_manager.mark_boundary([1, 2], ["x", "x"], [0, Longueur])

    def set_boundary_condition(self):
        self.bcs.add_U(region=1)
        chargement = MyConstant(self.mesh, Umax, Type = "Rampe")
        self.bcs.add_U(value = chargement, region = 2)
        
    def set_output(self):
        self.eps_list = [0]
        self.F_list = [0]
        self.Force = self.set_F(2, "x")
        return {'U':True}
    
    def query_output(self, t):
        self.eps_list.append(Umax / Longueur * t)
        self.F_list.append(self.get_F(self.Force))
    
    def final_output(self):
        kappa = E / (3. * (1 - 2 * nu))
        mu = E / (2. * (1 + nu))            
        solution_analytique = array([sigma_xx(epsilon, kappa, mu, eos_type, devia_type) for epsilon in self.eps_list])
        eps_list_percent = [100 * eps for eps in self.eps_list]
        numerical_results = array(self.F_list)
        
        # On calcule la différence entre les deux courbes
        len_vec = len(solution_analytique)
        diff_tot = solution_analytique - numerical_results
        # Puis on réalise une sorte d'intégration discrète
        integrale_discrete = sum(abs(diff_tot[j]) for j in range(len_vec))/sum(abs(solution_analytique[j]) for j in range(len_vec))
        print("La difference est de", integrale_discrete)
        # assert integrale_discrete < 0.001, "Static 1D traction fail"
        if __name__ == "__main__": 
            plt.plot(eps_list_percent, solution_analytique, linestyle = "--", color = "red", label = "Analytique")
            plt.scatter(eps_list_percent, numerical_results, marker = "x", color = "blue", label="CHARON")
            plt.xlim(0, 1.05 * eps_list_percent[-1])
            plt.ylim(0, 1.05 * numerical_results[-1])
            plt.xlabel(r"Déformation(%)", size = 18)
            plt.ylabel(r"Force (N)", size = 18)
            plt.legend()
            # plt.savefig("../../../Notice/fig/Traction_1D"+str(eos_type)+str(devia_type)+".pdf", bbox_inches = 'tight')
            plt.show()

def test_Traction_1D():
    pb = IsotropicBeam(Acier)
    Solve(pb, compteur=1, npas = 10)
test_Traction_1D()