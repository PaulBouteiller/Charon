"""
Created on Fri Mar 11 09:36:05 2022

@author: bouteillerp
Essai de traction uni-axial sur un cube 3D
"""
from CharonX import *
import time
import matplotlib.pyplot as plt
import pytest
import sys
sys.path.append("../")
from Traction_1D_cartesien_solution_analytique import *
from Generic_isotropic_material import *

######## Modèle mécanique ########
model = Tridimensionnal
######## Paramètres géométriques et de maillage ########
Longueur, Largeur, hauteur = 0.5, 2., 2.
Nx, Ny, Nz = 10, 10, 10

eps = 0.01

class Cube3D(model):
    def __init__(self, material):
        model.__init__(self, material, analysis="static", isotherm = True)
        
    def prefix(self):
        if __name__ == "__main__": 
            return "Traction_3D"
        else:
            return "Test"
          
    def define_mesh(self):
        return create_box(MPI.COMM_WORLD, [np.array([0, 0, 0]), 
                                           np.array([Longueur, Largeur, hauteur])],
                                          [Nx, Ny, Nz])
    
    def set_boundary(self):
        self.mark_boundary([1, 2, 3, 4, 5, 6], ["x", "x", "y", "y", "z", "z"], [0, Longueur, 0, Largeur, 0, hauteur])

    def set_boundary_condition(self):
        self.bcs.add_Ux(region=1)
        self.bcs.add_Ux(region=2)
        self.bcs.add_Uy(region=3)
        self.bcs.add_Uy(region=4)
        self.bcs.add_Uz(region=5)
        Umax = eps * hauteur
        chargement = MyConstant(self.mesh, Umax, Type = "Rampe")
        self.bcs.add_Uz(value = chargement, region = 6)
        
    def set_output(self):
        self.eps_list = []
        self.F_list = []
        self.Force = self.set_F(6, "z")
        return {"U": True}
    
    def query_output(self, t):
        self.eps_list.append(eps * t)
        self.F_list.append(2 * self.get_F(self.Force))
        
    def final_output(self):       
        solution_analytique = array([2 * sigma_xx(epsilon, kappa, mu, eos_type, devia_type) for epsilon in self.eps_list])
        eps_list_percent = [100 * eps for eps in self.eps_list]
        numerical_results = array(self.F_list)
        # On calcule la différence entre les deux courbes
        len_vec = len(solution_analytique)
        diff_tot = solution_analytique - numerical_results
        # Puis on réalise une sorte d'intégration discrète
        integrale_discrete = sum(abs(diff_tot[j]) for j in range(len_vec))/sum(abs(solution_analytique[j]) for j in range(len_vec))
        print("La difference est de", integrale_discrete)
        # assert integrale_discrete < 0.01, "Static 1D traction fail"
        if __name__ == "__main__": 
            plt.scatter(eps_list_percent, self.F_list, marker = "x", color = "blue", label="CHARON")
            plt.plot(eps_list_percent, solution_analytique, linestyle = "--", color = "red", label = "Analytique")
            plt.xlim(0, 1.1 * eps_list_percent[-1])
            plt.ylim(0, 1.1 * self.F_list[-1])
            plt.xlabel(r"Déformation(%)", size = 18)
            plt.ylabel(r"Force (N)", size = 18)
            plt.legend()
            plt.savefig("../../../Notice/fig/1D_Traction_pseudo_3D.pdf", bbox_inches = 'tight')
            plt.show()

def test_Traction_3D():
    pb = Cube3D(Acier)
    Solve(pb, compteur=1, npas=10)
test_Traction_3D()
