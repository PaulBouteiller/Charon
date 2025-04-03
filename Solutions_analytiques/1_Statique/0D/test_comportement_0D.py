"""
Test de comportement mécanique 0D pour différents modèles constitutifs.

Ce script teste le comportement mécanique d'un matériau en imposant une déformation
volumétrique (traction/compression) et en comparant les résultats numériques aux
solutions analytiques.

La simulation utilise:
    - Modèle: CartesianUD (1D cartésien)
    - Matériau: défini par les paramètres eos_type et devia_type
    - Chargement: déformation imposée varepsilon
    - Analyse: statique, isotherme
    
Le test vérifie la cohérence entre les contraintes calculées numériquement et analytiquement.

Auteur: bouteillerp
Date de création: 24 Juillet 2023
"""
from CharonX import *

import pytest
from Analytique_EOS_deviateur import *
from material_definition import set_material
from numerical_analytical_comparison import comparison


###### Modèle mécanique ######
model = CartesianUD

###### Materiau ######
eos_type = "Tabulated"
devia_type = "IsotropicHPP"
Mat = set_material(eos_type, devia_type)

###### Chargement ######
varepsilon = -0.5
T0 = 1e3
   
class IsotropicBeam(model):
    def __init__(self, material):
        model.__init__(self, material, analysis = "static", isotherm = True)
          
    def define_mesh(self):
        return create_1D_mesh(0, 1, 1)
    
    def prefix(self):
        if __name__ == "__main__": 
            return "Test_0D_" + eos_type
        else:
            return "Test"
            
    def set_boundary(self):
        self.mesh_manager.mark_boundary([1, 2], ["x", "x"], [0, 1])

    def set_initial_temperature(self):
        self.T0 = Function(self.V_T)
        self.T.x.petsc_vec.set(T0)
        self.T0.x.petsc_vec.set(T0)

    def set_boundary_condition(self):
        self.bcs.add_U(region=1)
        chargement = MyConstant(self.mesh, varepsilon, Type = "Rampe")
        self.bcs.add_U(value = chargement, region = 2)
        
    def csv_output(self):
        return {"Pressure" : True, "U" : ["Boundary", 2], "deviateur" :  True}
    
    def final_output(self):
        comparison(Mat, varepsilon, T0)

pb = IsotropicBeam(Mat)
Solve(pb, compteur=1, npas = 20)