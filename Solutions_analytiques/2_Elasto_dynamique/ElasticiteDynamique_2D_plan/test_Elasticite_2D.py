"""
Test de validation pour l'élasticité dynamique en déformation plane 2D

Ce script implémente et exécute un test de validation pour les équations
d'élasticité linéaire en déformation plane 2D.

Cas test:
---------
- Rectangle avec longueur 50 mm et largeur 1 mm
- Condition aux limites de déplacement vertical nul sur le bord inférieur
- Application d'un chargement en créneau sur le bord gauche
- Propagation d'onde dans le domaine 2D

Ce test sert principalement à vérifier le comportement du code en 2D
et à évaluer les performances de calcul.

Auteur: bouteillerp
"""

from CharonX import *
import matplotlib.pyplot as plt
import pytest
import time

###### Modèle géométrique ######
model = Plane_strain
materiau = "Linear_Hook"
###### Modèle matériau ######
E = 210e3
nu = 0.3 
lmbda = E * nu / (1 - 2 * nu) / (1 + nu)
mu = E / 2. / (1 + nu)
rho = 7.8e-3
C = 500
alpha=12e-6
rigi = lmbda + 2 * mu
wave_speed = (rigi/rho)**(1./2)
dico_eos = {"E" : 210e3, "nu" : 0.3, "alpha" : 12e-6}
dico_devia = {"mu": mu}
Acier = Material(rho, C, "IsotropicHPP", "Hypoelastic", dico_eos, dico_devia)

###### Paramètre géométrique ######
Largeur = 1
Longueur = 50

###### Temps simulation ######
Tfin = 7e-3
print("le temps de fin de simulation est", Tfin )
pas_de_temps = Tfin/8000
largeur_creneau = Longueur/4
magnitude = 1e4

sortie = 1000
pas_de_temps_sortie = sortie * pas_de_temps
n_sortie = int(Tfin/pas_de_temps_sortie)
   
class Isotropic_beam(model):
    def __init__(self, material):
        model.__init__(self, material)
          
    def define_mesh(self):
        return create_rectangle(MPI.COMM_WORLD, [(0, 0), (Longueur, Largeur)], [200, 1], CellType.quadrilateral)
    
    def prefix(self):
        if __name__ == "__main__": 
            return "Test_elasticite"
        else:
            return "Test"
        
    def set_boundary(self):
        self.mesh_manager.mark_boundary([1, 2, 3], ["x", "y", "y"], [0, 0, Largeur])
        
    def set_boundary_condition(self):
        self.bcs.add_Uy(region=2)
        
    def set_loading(self):
        T_unload = largeur_creneau/wave_speed
        chargement = MyConstant(self.mesh, T_unload, magnitude, Type = "Creneau")
        self.loading.add_Fx(chargement, self.u_, self.ds(1))
        
    def csv_output(self):
        return {'Sig': True}
    
    def set_output(self):
        return {'Sig': True}

def test_Elasticite():
    pb = Isotropic_beam(Acier)
    tps1 = time.perf_counter()
    Solve(pb, compteur = sortie, TFin=Tfin, scheme = "fixed", dt = pas_de_temps)
    tps2 = time.perf_counter()
    print("temps d'execution", tps2 - tps1)
test_Elasticite()