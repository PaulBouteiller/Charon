"""
Test de validation pour le modèle hypoélastique en cisaillement 2D

Ce script implémente et exécute un test de validation pour le comportement
hypoélastique en cisaillement 2D. Il simule un essai de cisaillement alterné
et exporte les déviateurs de contraintes normalisés pour comparaison avec
la solution analytique.

Cas test:
---------
- Élément quadrilatéral unique soumis à un chargement cyclique
- Le chargement consiste en 4 phases:
  1. Déplacement vertical imposé (0-1s)
  2. Déplacement horizontal imposé (1-2s)
  3. Déplacement vertical inversé (2-3s)
  4. Déplacement horizontal inversé (3-4s)
- Amplitude des déplacements imposés: variable (0.006, 0.06 ou 0.6)
- Export des déviateurs de contraintes normalisés par le module de cisaillement μ

Théorie:
--------
Le modèle hypoélastique relie le taux de déformation au taux de contrainte
déviatorique. Lors d'un chargement cyclique, ce modèle peut présenter une
dérive des contraintes (contraintes résiduelles) qui dépend de l'amplitude
des déformations imposées.

La solution analytique est implémentée dans le module trace_sol_analytique.py.

Auteur: bouteillerp
"""

from CharonX import *
import pytest

import numpy as np
import csv

###### Modèle géométrique ######
model = Plane_strain
###### Modèle matériau ######
eos_type = "U1"
kappa = 175e3
alpha = 1
dico_eos = {"kappa" : kappa, "alpha" : alpha}


dev_type = "Hypoelastic"
mu = 80769
dico_devia = {"mu": mu}
Acier = Material(1, 1, eos_type, dev_type, dico_eos, dico_devia)
pas_de_temps = 8e-4

amplitude = 0.006
class Isotropic_beam(model):
    def __init__(self, material):
        model.__init__(self, material, analysis = "User_driven", isotherm = True)
          
    def define_mesh(self):
        return create_rectangle(MPI.COMM_WORLD, [(0, 0), (1, 1)], [1, 1], CellType.quadrilateral)
    
    def prefix(self):
        if __name__ == "__main__": 
            return "Test_hypo_elasticite"
        else:
            return "Test"
        
    def set_boundary(self):
        self.mesh_manager.mark_boundary([1, 2, 3], ["x", "y", "y"], [0, 0, 1])
        self.evol_1 = True
        self.evol_2 = True
        self.evol_3 = True
        self.evol_4 = True
        self.evol_5 = True
        
    def user_defined_displacement(self, t):
        Uf = amplitude
        Sf = amplitude
        # print("Déplacement imposé à t=", t)
        # print("Le déplacement associé est", self.u.x.array)
        if t<1 and self.evol_1:
            self.v.x.array[:] = np.array([0, 0, 0, Uf, 0, 0, 0, Uf])
            self.evol_1 = False
        elif t>=1  and t<2 and self.evol_2:
            self.v.x.array[:] = np.array([0, 0, Sf, 0, 0, 0, Sf, 0])
            self.evol_2 = False
        elif t>=2  and t<3 and self.evol_3:
            self.v.x.array[:] = np.array([0, 0, 0, -Uf, 0, 0, 0, -Uf])
            self.evol_3 = False
        elif t>=3  and t<4. and self.evol_4:
            self.v.x.array[:] = np.array([0, 0, -Sf, 0, 0, 0, -Sf, 0])
            self.evol_4 = False
        elif t>=4  and t<4.2 and self.evol_5:
            self.v.x.array[:] = np.array([0, 0, 0, 0, 0, 0, 0, 0])
            self.evol_5 = False
        if t<4.2:
            self.u.x.array[:] += pas_de_temps * self.v.x.array

        
    def csv_output(self):
        return {'deviateur': True}
    
    def set_output(self):
        self.devia_list_xx = []
        self.devia_list_yy = []
        self.devia_list_xy = []
        self.t_list = []
        return {'U': True}
    
    def query_output(self, t):
        mu = self.material.devia.mu
        self.t_list.append(t)
        self.devia_list_xx.append(self.material.devia.s.x.array[0]/mu)
        self.devia_list_yy.append(self.material.devia.s.x.array[1]/mu)
        self.devia_list_xy.append(self.material.devia.s.x.array[2]/mu)
        
    def final_output(self):
        def numpy_to_csv(arr, filename):
            if arr.ndim != 1:
                raise ValueError("Le tableau doit être unidimensionnel (1D)")
            
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(arr)
                
        def name_csv(coord, essai, old):
            if old:
                return "csv/"+coord+str(essai)+"old.csv"
            else:
                return "csv/"+coord+str(essai)+".csv"
            
        
        numpy_to_csv(np.array(self.t_list), 't.csv')
        numpy_to_csv(np.array(self.devia_list_xx), name_csv("s_xx", amplitude, False))
        numpy_to_csv(np.array(self.devia_list_yy), name_csv("s_yy", amplitude, False))
        numpy_to_csv(np.array(self.devia_list_xy), name_csv("s_xy", amplitude, False))
        
def test_Elasticite():
    pb = Isotropic_beam(Acier)
    Solve(pb, sortie = 10, TFin=4.2, scheme = "fixed", dt = pas_de_temps)
test_Elasticite()