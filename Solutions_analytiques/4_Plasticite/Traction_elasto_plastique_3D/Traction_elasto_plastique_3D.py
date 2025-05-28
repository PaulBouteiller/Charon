"""
Created on Mon Feb 27 10:12:53 2023

@author: bouteillerp
"""

from CharonX import *
import time
import matplotlib.pyplot as plt
import sys
sys.path.append("../../")
from Generic_isotropic_material import E, Acier

# from mshr import *
from numpy import pi
######## Modèle mécanique ########
model = Tridimensional

###### Chargement ######
N_pas=300

###### Plasticité ######
H = E/10
sigY = 300


######## Paramètres géométriques et de maillage ########
L, l, h = 0.5, 2., 2.
Nx, Ny, Nz = 5, 5, 5

eps = 0.003
Umax = eps * h

class Cube3D(model):
    def __init__(self, material):
        model.__init__(self, material, analysis="static", plastic = "HPP_Plasticity", isotherm = True)
        
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

        chargement = MyConstant(self.mesh, Umax, Type = "Rampe")
        self.bcs.add_Uz(value = chargement, region = 4)
        
    def set_plastic(self):
        self.constitutive.plastic.set_plastic(sigY = sigY, H = H, hardening = "Isotropic") 
        
    def set_output(self):
        self.eps_list = []
        self.sig_list = []
        self.Force = self.set_F(4, "z")
        return {}
    
    def query_output(self, t):
        self.eps_list.append(eps * t)
        self.sig_list.append(self.get_F(self.Force) / (l * L))
    
    def final_output(self):
        eps_c = sigY / E
        def sig_elast(eps):
            return E * eps
        
        def sig_plas(eps):
            ET = E * H/(E + H)
            return (sigY + ET * (eps - eps_c))
            
        eps_elas_list = np.linspace(0, eps_c, 10)
        sig_elas_list = [sig_elast(eps) for eps in eps_elas_list]
        eps_plas_list = np.linspace(eps_c, eps, 10)
        sig_plas_list = [sig_plas(eps) for eps in eps_plas_list]
        
        if __name__ == "__main__": 
            plt.plot(self.eps_list, self.sig_list, linestyle = "--", color = "blue", label = "CHARON")
            plt.plot(eps_elas_list, sig_elas_list, linestyle = "-", color = "red")
            plt.plot(eps_plas_list, sig_plas_list, linestyle = "-", color = "red", label = "Analytical")
            plt.legend()
            plt.xlim(0, 1.05 * eps)
            plt.ylim(0, 1.1 * max(self.sig_list))
            plt.xlabel(r"Déformation", size = 18)
            plt.ylabel(r"Force (N)" , size = 18)
            
pb = Cube3D(Acier)
Solve(pb, compteur=1, npas=N_pas)