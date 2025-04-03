"""
Created on Fri Mar 11 09:36:05 2022

@author: bouteillerp
Traction uniaxiale sur une plaque en déformation plane, condition aux limites 
en efforts"""

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
f_surf = 1e3
Npas = 20
   
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
        self.mark_boundary([1, 2, 3], ["x", "y", "x"], [0, 0, Longueur])
        
    def set_boundary_condition(self):
        self.bcs.add_Ux(region=1)
        self.bcs.add_Uy(region=2)
        
    def set_loading(self):
        self.loading.add_Fx(f_surf * self.load, self.u_, self.ds(3))
        
    def set_output(self):
        self.sig_list = []
        self.Force = self.set_F(1, "x")
        return {'Sig':True, 'U': True}

    def query_output(self, t):
        self.sig_list.append(-self.get_F(self.Force)/Largeur)
    
    def final_output(self):
        virtual_t = np.linspace(0, 1, Npas)
        force_elast_list = np.linspace(0, f_surf, len(virtual_t))
        
        if __name__ == "__main__": 
            plt.scatter(virtual_t, self.sig_list, marker = "x", color = "blue", label="CHARON")
            plt.plot(virtual_t, force_elast_list, linestyle = "--", color = "red", label = "Analytical")
            plt.xlim(0, 1.1 * virtual_t[-1])
            plt.ylim(0, 1.1 * self.sig_list[-1])
            plt.xlabel(r"Temps virtuel", size = 18)
            plt.ylabel(r"Contrainte (MPa)", size = 18)
            plt.legend()
            plt.savefig("../../../Notice/fig/Traction_2D_Neumann.pdf", bbox_inches = 'tight')
            plt.show()
            
def test_Traction2D():
    pb = Plate(Acier)
    Solve(pb, compteur=1, npas = Npas)
test_Traction2D()