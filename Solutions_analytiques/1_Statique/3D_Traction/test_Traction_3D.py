"""
Test de traction uniaxiale sur un cube 3D avec mat�riau isotrope.

Ce script simule un essai de traction uniaxiale sur un cube 3D et compare
les r�sultats num�riques avec la solution analytique.

Param�tres g�om�triques:
    - Dimensions du cube: 0.5 × 2.0 × 2.0
    - Discr�tisation: maillage 10×10×10

Chargement:
    - D�formation impos�e (eps): 0.005 (0.5% de d�formation)

Conditions aux limites:
    - D�placement horizontal bloqu� sur la face gauche
    - D�placement vertical bloqu� sur la face inf�rieure
    - D�placement selon Z bloqu� sur la face arri�re
    - D�placement selon Z impos� sur la face avant

Le script calcule la force r�sultante et la compare avec la solution analytique
pour un probl�me de traction uniaxiale 3D. Une assertion v�rifie que l'erreur
relative est inf�rieure à 1%.

Auteur: bouteillerp
Date de cr�ation: 11 Mars 2022
"""
from CharonX import *
import matplotlib.pyplot as plt
import pytest
import sys
sys.path.append("../")
from Generic_isotropic_material import *

######## Mod�le m�canique ########
model = Tridimensionnal
######## Param�tres g�om�triques et de maillage ########
L, l, h = 0.5, 2., 2.
Nx, Ny, Nz = 10, 10, 10

eps = 0.005

class Cube3D(model):
    def __init__(self, material):
        model.__init__(self, material, analysis="static", isotherm = True)
        
    def prefix(self):
        if __name__ == "__main__": 
            return "Traction_3D"
        else:
            return "Test"
          
    def define_mesh(self):
        return create_box(MPI.COMM_WORLD, [np.array([0, 0, 0]),  np.array([L, l, h])],
                                          [Nx, Ny, Nz])
    
    def set_boundary(self):
        self.mesh_manager.mark_boundary([1, 2, 3, 4], ["x", "y", "z", "z"], [0, 0, 0, h])

    def set_boundary_condition(self):
        self.bcs.add_Ux(region=1)
        self.bcs.add_Uy(region=2)
        self.bcs.add_Uz(region=3)
        Umax = eps * h
        chargement = MyConstant(self.mesh, Umax, Type = "Rampe")
        self.bcs.add_Uz(value = chargement, region = 4)
        
    def set_output(self):
        self.eps_list = []
        self.F_list = []
        self.Force = self.set_F(4, "z")
        return {"U": True}
    
    def query_output(self, t):
        self.eps_list.append(eps * t)
        self.F_list.append(self.get_F(self.Force))
        
    def final_output(self):
        def force_elast(eps):
            return E * eps * l * L
        
        solution_analytique = array([force_elast(eps) for eps in self.eps_list])
        eps_list_percent = [100 * eps for eps in self.eps_list]
        numerical_results = array(self.F_list)
        # On calcule la diff�rence entre les deux courbes
        len_vec = len(solution_analytique)
        diff_tot = solution_analytique - numerical_results
        # Puis on r�alise une sorte d'int�gration discr�te
        integrale_discrete = sum(abs(diff_tot[j]) for j in range(len_vec))/sum(abs(solution_analytique[j]) for j in range(len_vec))
        print("La difference est de", integrale_discrete)
        assert integrale_discrete < 0.01, "Static 1D traction fail"
        if __name__ == "__main__": 
            plt.scatter(eps_list_percent, self.F_list, marker = "x", color = "blue", label="CHARON")
            plt.plot(eps_list_percent, solution_analytique, linestyle = "--", color = "red", label = "Analytique")
            plt.xlim(0, 1.1 * eps_list_percent[-1])
            plt.ylim(0, 1.1 * self.F_list[-1])
            plt.xlabel(r"D�formation(%)", size = 18)
            plt.ylabel(r"Force (N)", size = 18)
            plt.legend()
            plt.show()

def test_Traction_3D():
    pb = Cube3D(Acier)
    Solve(pb, compteur=1, npas=10)
test_Traction_3D()
