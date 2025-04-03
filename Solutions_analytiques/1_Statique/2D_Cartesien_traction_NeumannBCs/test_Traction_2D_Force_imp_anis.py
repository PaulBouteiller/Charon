"""
Test de traction 2D avec conditions de Neumann pour un matériau anisotrope.

Ce script simule un essai de traction uniaxiale sur une plaque rectangulaire
en déformation plane composée d'un matériau anisotrope (isotrope transverse)
avec des forces imposées sur les bords.

Paramètres géométriques:
    - Longueur: 1
    - Largeur: 0.5
    - Discrétisation: maillage 20×20 (quadrilatères)

Matériau:
    - Équation d'état: U1 (kappa = 175e3)
    - Comportement déviatorique: Anisotropic (isotrope transverse)
    - Orientation des fibres: angle = 90° (perpendiculaire à la direction de traction)
    - Propriétés: EL=230e3, ET=15e3, muL=50e3, nuLT=0.2, nuTN=0.4

Chargement:
    - Force surfacique imposée (f_surf): 1e3
    - Nombre de pas: 20

Conditions aux limites:
    - Déplacement horizontal bloqué sur le bord gauche
    - Déplacement vertical bloqué sur le bord inférieur
    - Force horizontale imposée sur le bord droit

Le script calcule la contrainte résultante et la compare avec la valeur
imposée (f_surf) pour vérifier la cohérence de l'implémentation pour
un matériau anisotrope.

Auteur: bouteillerp
Date de création: 11 Mars 2022
"""
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
nuLT = 0.2 
nuTN = 0.4
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
f_surf = 1e3
Npas = 20
   
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
            plt.show()
            
def test_Traction2D():
    pb = Plate(Fibre)
    Solve(pb, compteur=1, npas = Npas)
test_Traction2D()