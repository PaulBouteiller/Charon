"""
Test de compression d'un cylindre en coordonnées cylindriques 1D.

Ce script simule la compression d'un cylindre creux soumis �  des pressions interne et externe,
puis compare la solution numérique au champ de déplacement radial analytique.

Paramètres géométriques:
    - Rayon interne (Rint): 9
    - Rayon externe (Rext): Rint + e
    - Épaisseur (e): 2
    - Discrétisation (Nx): 20 éléments radiaux

Chargement:
    - Pression interne (Pint): -5
    - Pression externe (Pext): -10

La solution analytique utilise les équations de Lamé pour un cylindre �  paroi épaisse.
Une assertion vérifie que l'erreur relative entre les solutions est inférieure �  0.1%.

Auteur: bouteillerp
Date de création: 11 Mars 2022
"""
from CharonX import *
import time
import pytest
from depouillement import validation_analytique
import sys
sys.path.append("../../")
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
        self.mesh_manager.mark_boundary([1, 2], ["x", "x"], [Rint, Rext])

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