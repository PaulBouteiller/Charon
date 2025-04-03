"""
Created on Mon Jul 24 16:37:39 2023
@author: bouteillerp
Traction simple en cartesién 1D à la solution analytique"""
from CharonX import *
import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy import linspace
model = CartesianUD
###### Modèle mécanique ######
E = 210e3
nu = 0.3
mu = E / 2. / (1 + nu)
dico_eos = {"E" : E, "nu" : nu, "alpha" : 1}
dico_devia = {"E":E, "nu" : nu}
eos_type = "IsotropicHPP"
devia_type = "NeoHook"
Acier = Material(1, 1, eos_type, devia_type, dico_eos, dico_devia)

###### Modèle mécanique ######
ratio = 3
E_alu = E / ratio
nu_alu = nu
mu_alu = E_alu / 2. / (1 + nu_alu)
dico_eos_alu = {"E" : E_alu, "nu" : nu_alu, "alpha" : 1}
dico_devia_alu = {"E" : E_alu, "nu" : nu_alu}
eos_type_alu = "IsotropicHPP"
devia_type_alu = "NeoHook"
Alu = Material(1, 1, eos_type_alu, devia_type_alu, dico_eos_alu, dico_devia_alu)

Mat = [Acier, Alu]

###### Paramètre géométrique ######
Longueur = 1

###### Chargement ######
Umax=1e-2
   
class IsotropicBeam(model):
    def __init__(self, material):
        model.__init__(self, material, analysis = "static", isotherm = True)
          
    def define_mesh(self):
        return create_interval(MPI.COMM_WORLD, 20, [np.array(0), np.array(Longueur)])
    
    def prefix(self):
        if __name__ == "__main__": 
            return "Traction_1D"
        else:
            return "Test"
            
    def set_boundary(self):
        self.mark_boundary([1, 2], ["x", "x"], [0, Longueur])

    def set_boundary_condition(self):
        self.bcs.add_U(region=1)
        chargement = MyConstant(self.mesh, Umax, Type = "Rampe")
        self.bcs.add_U(value = chargement, region = 2)
        
    def set_multiphase(self):
        x = SpatialCoordinate(self.mesh)
        mult = self.multiphase
        interp = mult.V_c.element.interpolation_points()
        demi_long = Longueur / 2
        ufl_condition_1 = conditional(x[0]<demi_long, 1, 0)
        c1_expr = Expression(ufl_condition_1, interp)
        ufl_condition_2 = conditional(x[0]>=demi_long, 1, 0)
        c2_expr = Expression(ufl_condition_2, interp)
        mult.multiphase_evolution =  [False, False]
        mult.explosive = False
        mult.set_multiphase([c1_expr, c2_expr])
        
    def csv_output(self):
        return {'U':True} 
        
    def final_output(self):
        u_csv = read_csv("Traction_1D-results/Displacement0.csv")
        resultat = [u_csv[colonne].to_numpy() for colonne in u_csv.columns]
        x_result = resultat[0]
        half_n_node = len(x_result)//2
        eps_tot = Umax
        eps_acier = 2 * eps_tot / (1 + ratio)
        eps_alu = ratio * eps_acier 
        dep_acier = [eps_acier * x for x in linspace(0, 0.5, half_n_node+1)]
        dep_alu = [eps_acier * 0.5 + eps_alu * x for x in linspace(0, 0.5, half_n_node+1)]
        dep_acier.pop()
        dep_tot = dep_acier + dep_alu
        solution_numerique = resultat[-1]
        if __name__ == "__main__": 
            plt.scatter(x_result, solution_numerique, marker = "x", color = "red")
            plt.plot(x_result, dep_tot, linestyle = "--", color = "blue")            
            plt.xlim(0, 1)
            plt.xlabel(r"Position (mm)", size = 18)
            plt.ylabel(r"Déplacement (mm)", size = 18)
            plt.savefig("../../../Notice/fig/Traction_bimat.pdf", bbox_inches = 'tight')
            # plt.show()

def test_Traction_1D():
    pb = IsotropicBeam(Mat)
    Solve(pb, compteur=1, npas = 10)
test_Traction_1D()