"""
Test de compression d'un cylindre creux en 2D axisymétrique.

Ce script simule la compression d'un cylindre creux soumis �  des pressions interne
et externe en utilisant un modèle axisymétrique, puis compare la solution numérique
avec la solution analytique de Lamé.

Paramètres géométriques:
    - Rayon interne (Rint): 9
    - Rayon externe (Rext): 11
    - Hauteur du cylindre: 1
    - Discrétisation: maillage 10×5 (triangles)

Chargement:
    - Pression interne (Pint): -5
    - Pression externe (Pext): -10

Conditions aux limites:
    - Déplacement vertical bloqué sur la face inférieure

Une comparaison est effectuée entre le champ de déplacement radial calculé
numériquement et la solution analytique via le module depouillement.py.
L'erreur relative entre les deux solutions est calculée pour vérifier la précision
du modèle numérique.

Auteur: bouteillerp
Date de création: 11 Mars 2022
"""
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

model = Axisymmetric
   
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
        self.mesh_manager.mark_boundary([1, 2, 3, 4], ["z", "r", "r", "z"], [0, Rint, Rext, hauteur])
        
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