#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 09:36:05 2022

@author: bouteillerp
Traction uniaxiale sur une plaque en déformation plane"""

from CharonX import *
import time
import matplotlib.pyplot as plt
import pytest
from numpy import pi
from analytique_plaque_trou import *
import sys
sys.path.append("../")
from Generic_isotropic_material import *

model = Plane_strain
###### Paramètre géométrique ######
Largeur = 2
Longueur = 1
R = 0.2
mesh_size = 0.01
        
###### Chargement ######
f_surf = 1e3
   
class Plate(model):
    def __init__(self, material):
        model.__init__(self, material, analysis = "static", isotherm = True)
          
    def define_mesh(self):
        mesh, _, facets = quarter_perforated_plate(Largeur, Longueur, R, mesh_size)
        return mesh

    def prefix(self):
        if __name__ == "__main__": 
            return "Plaque_2D"
        else:
            return "Test"
        
    def set_boundary(self):
        self.mark_boundary([1, 2, 3], ["x", "y", "x"], [0, 0, Largeur])
        
    def set_boundary_condition(self):
        self.bcs.add_Ux(region=1)
        self.bcs.add_Uy(region=2)
        
    def set_loading(self):
        self.loading.add_Fx(f_surf * self.load, self.u_, self.ds(3))
        
    def csv_output(self):
        return {"U" : ["Boundary", 1]}
    
    def set_output(self):
        return {'Sig':True}
    
    def final_output(self):
        u_csv = read_csv("Plaque_2D-results/U.csv")
        resultat = [u_csv[colonne].to_numpy() for colonne in u_csv.columns]
        x_result = resultat[1]
        displacement = resultat[-1]
        sol_anat = array([ur(R, x, nu, E, f_surf, pi/2) for x in x_result])

        
        if __name__ == "__main__": 
            plt.plot(x_result, sol_anat, linestyle = "--", color = "red")
            plt.scatter(x_result, displacement, marker = "x", color = "blue")
            plt.xlabel(r"$r$", size = 18)
            plt.ylabel(r"Déplacement radial", size = 18)
        

def test_Traction2D():
    pb = Plate(Acier)
    Solve(pb, compteur=1, npas = 10)
test_Traction2D()