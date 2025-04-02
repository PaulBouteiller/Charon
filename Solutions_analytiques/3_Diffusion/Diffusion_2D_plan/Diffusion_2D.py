#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 16:45:40 2023

@author: bouteillerp
"""
from CharonX import *
import matplotlib.pyplot as plt 
from ufl import conditional, lt, And, gt
import time
###### Modèle géométrique ######
model = Plane_strain
# model = Axisymetric

###### Modèle matériau ######
rho_0 = 2.785e3
Cv = 1000


dico_eos = {"E": 1, "nu" : 0, "alpha" : 1}
dico_devia = {}
Materiau = Material(rho_0, Cv, "IsotropicHPP", None, dico_eos, dico_devia)

###### Modèle thermique ######
lmbda_diff = 240
AcierTherm = LinearThermal(lmbda_diff)

###### Paramètre géométrique ######
Largeur = 210e-6
bord_gauche = -105e-6
bord_droit = bord_gauche + Largeur
taille_tache = 40e-6
point_gauche = -taille_tache / 2
point_droit = taille_tache / 2

###### Champ de température initial ######
Tfin = 1e-6
Tfroid = 300
TChaud = 800

###### Paramètre temporel ######
sortie = 1
pas_de_temps = 5e-9
pas_de_temps_sortie = sortie * pas_de_temps

n_sortie = int(Tfin/pas_de_temps_sortie)

###### Maillage ######
Nx = 200
Ny = 200
test = "rond"
class Plate(model):
    def __init__(self, material):
        model.__init__(self, material, analysis = "Pure_diffusion",  adiabatic=False, Thermal_material = AcierTherm)
          
    def define_mesh(self):
        return create_rectangle(MPI.COMM_WORLD, [(bord_gauche, bord_gauche), (bord_droit, bord_droit)], [Nx, Ny], CellType.quadrilateral)
    
    def prefix(self):
        if __name__ == "__main__": 
            return "Test_diffusion_2D"
        else:
            return "Test"
        
    def set_boundary(self):
        self.mark_boundary([1, 2], ["x", "x"], [bord_gauche, bord_droit])
        
    def set_initial_temperature(self):
        x = SpatialCoordinate(self.mesh)
        if test == "carre":
            ufl_condition = conditional(And(gt(x[1], point_gauche), And(lt(x[1], point_droit), \
                                            And(lt(x[0], point_droit), gt(x[0], point_gauche)))), TChaud, Tfroid)
        elif test == "rond":
            centre = np.array([0., 0.])
            # absisse =x[0] - centre[0]
            # ufl_condition = conditional(lt(absisse, taille_tache), TChaud, Tfroid)     
            ufl_condition = conditional(lt((x[0]-centre[0])**2 + (x[1]-centre[1])**2, taille_tache**2), TChaud, Tfroid)           
        T_expr = Expression(ufl_condition, self.V_T.element.interpolation_points())
        self.T0 = Function(self.V_T)
        self.T0.interpolate(T_expr)
        self.T.interpolate(T_expr)
        
    def set_boundary_condition(self):
        self.bcs_T = []
        self.bcs.add_T(self.V_T, self.bcs_T, region=1, value = Constant(self.mesh, ScalarType(300)))
        
    def set_output(self):
        self.T_vector = [self.T.x.array]
        self.T_max=[800]
        return {'T':True}
    
    def query_output(self, t):
        self.T_vector.append(np.array(self.T.x.array))
        self.T_max.append(np.max(self.T_vector[-1]))

def test_Diffusion():
    pb = Plate(Materiau)
    tps1 = time.perf_counter()
    Solve(pb, compteur = sortie, TFin=Tfin, scheme = "fixed", dt = pas_de_temps)
    tps2 = time.perf_counter()
    print("temps d'execution", tps2 - tps1)
test_Diffusion()