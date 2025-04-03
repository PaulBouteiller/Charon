"""
Created on Mon Jul 24 16:37:39 2023
@author: bouteillerp
Traction simple en cartesién 1D à la solution analytique"""
from CharonX import *
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.linalg import block_diag
# from Traction_1D_cartesien_solution_analytique import *
model = CartesianUD
###### Modèle mécanique ######
kappa = 175e3
rho = 1
eos_type = "U1"
dico_eos = {"kappa" : kappa, "alpha" : 1}

# devia_type = "Lu_Transverse"
devia_type = "Anisotropic"
anisotropie = "Isotrope_transverse"
if devia_type == "Lu_Transverse":
    k2 = 2e3
    k3 = 0
    k4 = 0
    c = 1
    dico_devia = {"k2" : k2, "k3" : k3, "k4" : k4, "c" : c}
    Fibre = Material(rho, 1, eos_type, devia_type, dico_eos, dico_devia)
    
elif devia_type == "Anisotropic":
    if anisotropie == "Isotrope":
        mu = 80769
        C_sph = mu * np.array([[4./3, -2./3, -2./3],
                               [-2./3, 4./3, -2./3],
                               [-2./3, -2./3, 4./3]])
        C_shear = mu * np.diag([1., 1., 1.])
        C_tot = block_diag(C_sph, C_shear)
    elif anisotropie == "Isotrope_2":
        E = 210e3
        nu = 0.3
        mat = Isotropic(E, nu)
        C_corr = block_diag(kappa * np.ones((3, 3)), np.diag([0, 0, 0]))
        C_tot = mat.C - C_corr
        
    elif anisotropie == "Isotrope_transverse":
        EL = 230e3
        ET = 15e3
        muL = 50e3
        nuLT = 0.2 
        nuTN = 0.4
        mat = TransverseIsotropic(ET, EL, nuTN, nuLT, muL)
        mat.C = mat.rotate(mat.C, 90)
        C_corr = block_diag(kappa * np.ones((3, 3)), np.diag([0, 0, 0]))
        C_tot = mat.C - C_corr
        
    dico_devia = {"C" : C_tot}
    Fibre = Material(rho, 1, eos_type, devia_type, dico_eos, dico_devia)

###### Paramètre géométrique ######
Longueur = 1

###### Chargement ######
Umax=1e-2
   
class IsotropicBeam(model):
    def __init__(self, material):
        model.__init__(self, material, analysis = "static", isotherm = True)
          
    def define_mesh(self):
        return create_interval(MPI.COMM_WORLD, 2, [np.array(0), np.array(Longueur)])
    
    # def set_anisotropic_direction(self):
    #     # self.n0 = as_vector([1, 0, 0])
    #     # self.n0 = as_vector([0, 1, 0])
    #     self.n0 = as_vector([0, 0, 1])
    
    def prefix(self):
        if __name__ == "__main__": 
            return "Traction_1D_anisotrope"
        else:
            return "Test"
            
    def set_boundary(self):
        self.mark_boundary([1, 2], ["x", "x"], [0, Longueur])

    def set_boundary_condition(self):
        self.bcs.add_U(region=1)
        chargement = MyConstant(self.mesh, Umax, Type = "Rampe")
        self.bcs.add_U(value = chargement, region = 2)
        
    def set_output(self):
        self.eps_list = [0]
        self.F_list = [0]
        self.Force = self.set_F(2, "x")
        return {'U':True, "Sig":True}
    
    def query_output(self, t):
        self.eps_list.append(Umax / Longueur * t)
        self.F_list.append(self.get_F(self.Force))
    
    def final_output(self):
        eps_list_percent = [100 * eps for eps in self.eps_list]
        if __name__ == "__main__": 
            plt.scatter(eps_list_percent, self.F_list, marker = "x", color = "blue", label="CHARON")
            # plt.xlim(0, 1.1 * eps_list_percent[-1])
            # plt.ylim(0, 1.1 * self.F_list[-1])
            plt.xlabel(r"Déformation(%)", size = 18)
            plt.ylabel(r"Force (N)", size = 18)
            plt.legend()
            plt.savefig("../../../Notice/fig/Traction_1D"+str(eos_type)+str(devia_type)+".pdf", bbox_inches = 'tight')
            plt.show()

def test_Traction_1D():
    pb = IsotropicBeam(Fibre)
    Solve(pb, compteur=1, npas = 10)
test_Traction_1D()