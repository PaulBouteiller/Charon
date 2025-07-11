"""
Test de traction 2D avec conditions aux limites de Neumann.

Ce script simule un essai de traction uniaxiale sur une plaque rectangulaire
en conditions de déformation plane avec des forces imposées sur les bords.

Paramètres géométriques:
    - Longueur: 1
    - Largeur: 0.5
    - Discrétisation: maillage 20×20 (quadrilatères)

Chargement:
    - Force surfacique imposée (f_surf): 1e3
    - Nombre de pas: 20

Conditions aux limites:
    - Déplacement horizontal bloqué sur le bord gauche
    - Déplacement vertical bloqué sur le bord inférieur
    - Force horizontale imposée sur le bord droit

Le script calcule la contrainte résultante et la compare avec la valeur
imposée (f_surf) pour vérifier la cohérence de l'implémentation des
conditions aux limites de Neumann.

Auteur: bouteillerp
Date de création: 11 Mars 2022
"""
from CharonX import *
import time
import matplotlib.pyplot as plt
import pytest
from numpy import pi
import sys
sys.path.append("../")
from Generic_isotropic_material import *

model = Plane_strain

###### Paramètre géométrique ######
Largeur = 0.5
Longueur = 1

###### Chargement ######
f_surf = 1e3
Npas = 20


mesh = create_rectangle(MPI.COMM_WORLD, [(0, 0), (Longueur, Largeur)], [20, 20], CellType.quadrilateral)
T_unload = largeur_creneau/wave_speed
chargement = MyConstant(mesh, T_unload, magnitude, Type = "Creneau")

dictionnaire = {"mesh" : mesh,
                "boundary_setup": 
                    {"tags": [1, 2, 3],
                     "coordinate": ["x", "y", "x"], 
                     "positions": [0, 0, Longueur]
                     },
                "boundary_conditions": 
                    [{"component": "Ux", "tag": 1},
                     {"component": "Uy", "tag": 2}
                    ],
                "loading_conditions": 
                    [{"type": "surfacique", "component" : "Fx", "tag": 3, "value" : f_surf}
                    ],
                "isotherm" : True,
                }
    
pb = Plane_strain(Acier, dictionnaire)

dictionnaire_solve = {
    "Prefix" : "Test_elasticite",
    "csv_output" : {"Sig" : True}
    }

solve_instance = Solve(pb, dictionnaire_solve, compteur=sortie, TFin=Tfin, scheme = "fixed", dt = pas_de_temps)
solve_instance.solve()
   
class Plate(model):
    def __init__(self, material):
        model.__init__(self, material, analysis = "static", isotherm = True)
          
    def define_mesh(self):
        return 

    def prefix(self):
        if __name__ == "__main__": 
            return "Plaque_2D"
        else:
            return "Test"
        
    def set_boundary(self):
        self.mesh_manager.mark_boundary([1, 2, 3], ["x", "y", "x"], [0, 0, Longueur])
        
    def set_boundary_condition(self):
        self.bcs.add_Ux(region=1)
        self.bcs.add_Uy(region=2)
        
    def set_loading(self):
        self.loading.add_Fx(f_surf * self.load, self.u_, self.ds(3))
        
    def set_output(self):
        self.sig_list = []
        self.Force = self.set_F(1, "x")
        return {'Sig':True, 'U': True}

    def query_output(self, t):
        self.sig_list.append(-self.get_F(self.Force)/Largeur)
    
    def final_output(self):
        virtual_t = np.linspace(0, 1, Npas)
        force_elast_list = np.linspace(0, f_surf, len(virtual_t))
        
        if __name__ == "__main__": 
            plt.scatter(virtual_t, self.sig_list, marker = "x", color = "blue", label="CHARON")
            plt.plot(virtual_t, force_elast_list, linestyle = "--", color = "red", label = "Analytical")
            plt.xlim(0, 1.1 * virtual_t[-1])
            plt.ylim(0, 1.1 * self.sig_list[-1])
            plt.xlabel(r"Temps virtuel", size = 18)
            plt.ylabel(r"Contrainte (MPa)", size = 18)
            plt.legend()
            plt.savefig("../../../Notice/fig/Traction_2D_Neumann.pdf", bbox_inches = 'tight')
            plt.show()
            
def test_Traction2D():
    pb = Plate(Acier)
    Solve(pb, compteur=1, npas = Npas)
test_Traction2D()