#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 09:36:05 2022

@author: bouteillerp
Traction uniaxiale sur une plaque en déformation plane"""

from CharonX import *
import matplotlib.pyplot as plt
###### Modèle mécanique ######
E = 1e5
nu = 0.3
dico_eos = {"E":E, "nu" : nu, "alpha" : 1}
dico_devia = {"E":E, "nu" : nu}
Acier = Material(1, 1, "IsotropicHPP", "IsotropicHPP", dico_eos, dico_devia)

###### Paramètre géométrique ######
Rint = 9
Rext = 11
epaisseur = Rext - Rint
rapport = 25
hauteur = epaisseur / rapport

###### Maillage #######
Nr = 100
Nz = int(Nr / rapport)

###### Chargement ######
Pext = 10

###### Temps simulation ######
Tfin = 7e-3
print("le temps de fin de simulation est", Tfin )
pas_de_temps = Tfin / 7000
largeur_creneau = (Rext - Rint) / 4
magnitude = 1e3

sortie = 2000
pas_de_temps_sortie = sortie * pas_de_temps
n_sortie = int(Tfin/pas_de_temps_sortie)


def run_test(model):
    class CylindreAxi(model):
        def __init__(self, material):
            model.__init__(self, material, isotherm = True)
              
        def define_mesh(self):
            if model == Axisymetric:
                mesh = create_rectangle(MPI.COMM_WORLD, [(Rint, 0), (Rext, hauteur)], [Nr, Nz])
            elif model == CylindricalUD:
                mesh = create_1D_mesh(Rint, Rext, 100)
            return mesh
    
        def prefix(self):
            if model == Axisymetric:
                return "Cylindre_axi"
            elif model == CylindricalUD:
                return "Cylindre"
            
        def set_boundary(self):
            if model == CylindricalUD:
                self.mark_boundary([1, 2], ["x", "x"], [Rint, Rext])
            elif model == Axisymetric:
                self.mark_boundary([1, 2, 3, 4], ["r", "r", "z", "z"], [Rint, Rext, 0, hauteur])
                
        def set_loading(self):
            T_unload = Tfin/10
            chargement = MyConstant(self.mesh, T_unload, Pext, Type = "Creneau")
            if model == Axisymetric:
                self.loading.add_Fr(chargement, self.u_, self.ds(2))
            elif model == CylindricalUD:
                self.loading.add_F(chargement, self.u_, self.ds(2))
                
        def set_boundary_condition(self):
            if model == Axisymetric:
                self.bcs.add_Uz(region = 3)
                self.bcs.add_Uz(region = 4)
            elif model == CylindricalUD:
                pass
            
        def set_output(self):
            return {"Sig" : True}
            
        def csv_output(self):
            if model == Axisymetric:
                return {"U" : ["Boundary", 3], "Sig" : True}
            elif model == CylindricalUD:
                return {"U" : True, "Sig" : True}
    
    pb = CylindreAxi(Acier)
    Solve(pb, compteur = sortie, TFin=Tfin, scheme = "fixed", dt = pas_de_temps)
    
if __name__ == "__main__": 
    for model in [Axisymetric]:
        run_test(model)