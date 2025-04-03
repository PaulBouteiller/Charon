"""
Created on Fri Mar 11 09:36:05 2022

@author: bouteillerp
Traction uniaxiale sur une plaque en déformation plane"""

from CharonX import *
import time
import pytest
from numpy import pi
import sys
from depouillement import validation_analytique
sys.path.append("../")
from Generic_isotropic_material import *



###### Paramètre géométrique ######
L = 2
Nx = 100
Rint = 9
Rext = Rint + L
hauteur = 1

###### Chargement ######
Pext = -10
Pint = -5

model = Axisymetric
   
class Cylindre_axi(model):
    def __init__(self, material):
        model.__init__(self, material, analysis = "static", isotherm = True)
          
    def define_mesh(self):
        #Ne fonctionne qu'avec des triangles voir pourquoi
        return create_rectangle(MPI.COMM_WORLD, [(Rint, 0), (Rext, hauteur)], [10, 5])

    def prefix(self):
        if __name__ == "__main__": 
            return "Cylindre_axi"
        else:
            return "Test"
        
    def set_boundary(self):
        self.mark_boundary([1, 2, 3, 4], ["z", "r", "r", "z"], [0, Rint, Rext, hauteur])
        
    def set_boundary_condition(self):
        self.bcs.add_Uz(region=1)
        
    def set_loading(self):
        self.loading.add_Fr(-Pint * self.load, self.u_, self.ds(2))
        self.loading.add_Fr(Pext * self.load, self.u_, self.ds(3))
        
    def csv_output(self):
        return {"U" : ["Boundary", 1]}
            
def test_Compression_Axi():
    pb = Cylindre_axi(Acier)
    Solve(pb, compteur=1, npas=20)
test_Compression_Axi()
validation_analytique(Pint, Pext, Rint, Rext)