"""
Created on Fri Mar 11 09:36:05 2022

@author: bouteillerp
Experience de compression d'un cylindre comparaion à la solution analytique"""
from CharonX import *
import time
import pytest
from depouillement import validation_analytique
import sys
sys.path.append("../")
from Generic_isotropic_material import *

model = CylindricalUD

###### Paramètre géométrique ######
e = 2
Nx = 20
Rint = 9
Rext = Rint + e

###### Chargement ######
Pext = -10
Pint = -5
   
class IsotropicCyl(model):
    def __init__(self, material):
        model.__init__(self, material, analysis = "static", isotherm = True)
          
    def define_mesh(self):
        return create_interval(MPI.COMM_WORLD, Nx, [np.array(Rint), np.array(Rext)])
    
    def prefix(self):
        if __name__ == "__main__": 
            return "Compression_cylindrique_1D"
        else:
            return "Test"
            
    def set_boundary(self):
        self.mark_boundary([1, 2], ["x", "x"], [Rint, Rext])

    def set_loading(self):
        self.loading.add_F(-Pint * self.load, self.u_, self.ds(1))
        self.loading.add_F(Pext * self.load, self.u_, self.ds(2))
    
    def csv_output(self):
        return {'U':True}
           
def test_Compression_1D():
    pb = IsotropicCyl(Acier)
    Solve(pb, compteur=1, npas=5)
test_Compression_1D()
validation_analytique(Pint, Pext, Rint, Rext)