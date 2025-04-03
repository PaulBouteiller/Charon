#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 09:36:05 2022

@author: bouteillerp
Traction uniaxiale sur une plaque en déformation plane"""

from CharonX import *
import time
import matplotlib.pyplot as plt
import pytest
from numpy import pi

model = Plane_strain
###### Modèle mécanique ######
kappa = 175e3
rho = 1
eos_type = "U1"
dico_eos = {"kappa" : kappa, "alpha" : 1}

angle = 90
EL = 230e3
ET = 15e3
muL = 50e3
nuLT = 0.3
nuTN = 0.3
mat = TransverseIsotropic(ET, EL, nuTN, nuLT, muL)
mat.C = mat.rotate(mat.C, angle)
C_corr = block_diag(kappa * np.ones((3, 3)), np.diag([0, 0, 0]))
C_tot = mat.C - C_corr
    
devia_type = "Anisotropic"
dico_devia = {"C" : C_tot}
Fibre = Material(rho, 1, eos_type, devia_type, dico_eos, dico_devia)

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
            return "Plaque_2D"
        else:
            return "Test"
        
    def set_boundary(self):
        self.mark_boundary([1, 2, 3], ["x", "y", "x"], [0, 0, Longueur])
        
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
            if angle == 0:
                return EL * eps * Largeur /(1 - nuLT**2)
            elif angle == 90:
                return ET * eps * Largeur /(1 - nuLT**2)
        
        solution_analytique = array([force_elast(eps) for eps in self.eps_list])
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
            plt.scatter(eps_list_percent, self.F_list, marker = "x", color = "blue", label="CHARON")
            plt.plot(eps_list_percent, solution_analytique, linestyle = "--", color = "red", label = "Analytique")
            plt.xlim(0, 1.1 * eps_list_percent[-1])
            plt.ylim(0, 1.1 * self.F_list[-1])
            plt.xlabel(r"Déformation(%)", size = 18)
            plt.ylabel(r"Force (N)", size = 18)
            plt.legend()
            plt.savefig("../../../Notice/fig/Traction_2D_Dirichlet.pdf", bbox_inches = 'tight')
            plt.show()
            
def test_Traction2D():
    pb = Plate(Fibre)
    Solve(pb, compteur=1, npas=20)
test_Traction2D()