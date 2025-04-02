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
Rint = 8
Rext = 11

###### Chargement ######
Pext = 10

###### Temps simulation ######
Tfin = 1e-2
print("le temps de fin de simulation est", Tfin )
pas_de_temps = Tfin / 3e3
largeur_creneau = (Rext - Rint) / 4
magnitude = 1e3

sortie = 100
pas_de_temps_sortie = sortie * pas_de_temps
n_sortie = int(Tfin/pas_de_temps_sortie)


def run_test(model):
    class CoquilleAxi(model):
        def __init__(self, material):
            model.__init__(self, material, isotherm = True)
              
        def define_mesh(self):
            if model == Axisymetric:
                mesh, _, facets = axi_sphere(Rint, Rext, 20, 80, tol_dyn = 1e-3)
                self.facet_tag = facets
            elif model == SphericalUD:
                mesh = create_1D_mesh(Rint, Rext, 100)
            return mesh
    
        def prefix(self):
            if model == Axisymetric:
                return "Sphere_axi"
            elif model == SphericalUD:
                return "Sphere"
            
        def set_boundary(self):
            if model == SphericalUD:
                self.mark_boundary([1, 2], ["x", "x"], [Rint, Rext])
            elif model == Axisymetric:
                pass
                
        def set_loading(self):
            T_unload = Tfin/10

            if model == Axisymetric:
                chargement = MyConstant(self.mesh, T_unload, Pext, Type = "Creneau")
                self.loading.add_pressure(chargement, self.mesh, self.u_, self.ds(3))
            elif model == SphericalUD:
                chargement = MyConstant(self.mesh, T_unload, -Pext, Type = "Creneau")
                self.loading.add_F(chargement, self.u_, self.ds(2))
                
        def set_boundary_condition(self):
            if model == Axisymetric:
                self.bcs.add_Uz(region = 1)
                self.bcs.add_Ur(region = 2)
                # self.bcs.add_axi(region = 2)
            elif model == SphericalUD:
                pass
            
        def set_output(self):
            return {"Sig" : True}
            
        def csv_output(self):
            if model == Axisymetric:
                return {"U" : ["Boundary", 1]}
            elif model == SphericalUD:
                return {"U" : True}
    
    pb = CoquilleAxi(Acier)
    Solve(pb, compteur = sortie, TFin=Tfin, scheme = "fixed", dt = pas_de_temps)
    # Solve(pb, compteur = sortie, TFin=Tfin)
    
run_test(Axisymetric)