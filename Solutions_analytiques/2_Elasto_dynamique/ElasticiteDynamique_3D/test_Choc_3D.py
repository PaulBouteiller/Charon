"""
Test de validation pour l'élasticité dynamique en 3D

Ce script implémente et exécute un test de validation pour les équations
d'élasticité linéaire en 3D.

Cas test:
---------
- Parallélépipède rectangle avec dimensions L=50, b=4, h=4 mm
- Application d'un chargement en créneau sur la face x=0
- Propagation d'onde dans le domaine 3D

Ce test sert principalement à vérifier le comportement du code en 3D
et à évaluer les performances de calcul pour des problèmes tridimensionnels.

Auteur: bouteillerp
"""

from CharonX import *
import matplotlib.pyplot as plt
import pytest
import time

###### Modèle géométrique ######
model = Tridimensionnal
###### Modèle matériau ######
E = 210e3
nu = 0.3 
rho = 7.8e-3
C = 500
alpha=12e-6

lmbda = E * nu / (1 - 2 * nu) / (1 + nu)
mu = E / 2. / (1 + nu)
rigi = lmbda + 2 * mu

dico_eos = {"E": E, "nu" : nu, "alpha" : alpha}
dico_devia = {"E":E, "nu" : nu}
Acier = Material(rho, C, "IsotropicHPP", "IsotropicHPP", dico_eos, dico_devia)

######## Paramètres géométriques et de maillage ########
L, b, h = 50, 4, 4
Nx, Ny, Nz = 100, 8, 8

###### Temps simulation ######
Tfin = 2e-2
print("le temps de fin de simulation est", Tfin )
pas_de_temps = Tfin/8000
largeur_creneau = L/4
magnitude = 1e2

sortie = 200
pas_de_temps_sortie = sortie * pas_de_temps
n_sortie = int(Tfin/pas_de_temps_sortie)
   
class Isotropic_beam(model):
    def __init__(self, material):
        model.__init__(self, material)
          
    def define_mesh(self):
        return create_box(MPI.COMM_WORLD, [np.array([0,0,0]), np.array([L, b, h])],
                  [Nx, Ny, Nz], cell_type = CellType.hexahedron)
    
    def prefix(self):
        if __name__ == "__main__": 
            return "Test_elasticite"
        else:
            return "Test"
        
    def set_boundary(self):
        self.mesh_manager.mark_boundary([1, 2, 3, 4], ["x", "y", "z"], [0, 0, 0])

    # def set_boundary_condition(self):
    #     self.bcs.add_Uy(region=2)
    #     self.bcs.add_Uz(region=3)
        
    def set_loading(self):

        wave_speed = (rigi/rho)**(1./2)
        T_unload = largeur_creneau/wave_speed
        chargement = MyConstant(self.mesh, T_unload, magnitude, Type = "Creneau")
        self.loading.add_Fx(chargement, self.u_, self.ds(1))
        
    def set_output(self):
        return {'Sig': True}

def test_Elasticite():
    pb = Isotropic_beam(Acier)
    tps1 = time.perf_counter()
    Solve(pb, compteur = sortie, TFin=Tfin, scheme = "fixed", dt = pas_de_temps)
    tps2 = time.perf_counter()
    print("temps d'execution", tps2 - tps1)
test_Elasticite()