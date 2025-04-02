from CharonX import *
import matplotlib.pyplot as plt
import pytest
import time
from pandas import read_csv
import os
import sys
# Obtenir le chemin absolu du dossier parent
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from Analytical_wave_propagation import *

###### Modèle géométrique ######
model = Plane_strain
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
dico_devia = {"E":210e3, "nu" : 0.3}
Acier = Material(rho, C, "IsotropicHPP", "IsotropicHPP", dico_eos, dico_devia)

###### Paramètre géométrique ######
Nx = 1000
Longueur = 50
Largeur = Longueur / Nx
###### Temps simulation ######
Tfin = 7e-3
print("le temps de fin de simulation est", Tfin )
pas_de_temps = Tfin/8000
largeur_creneau = Longueur/4
magnitude = 1e3

sortie = 4000
pas_de_temps_sortie = sortie * pas_de_temps
n_sortie = int(Tfin/pas_de_temps_sortie)


class Isotropic_beam(model):
    def __init__(self, material):
        model.__init__(self, material)
          
    def define_mesh(self):
        return create_rectangle(MPI.COMM_WORLD, [(0, 0), (Longueur, Largeur)], [Nx, 1], CellType.quadrilateral)
    
    def prefix(self):
        if __name__ == "__main__": 
            return "Test_elasticite"
        else:
            return "Test"
        
    def set_boundary(self):
        self.mark_boundary([1, 2, 3], ["x", "y", "y"], [0, 0, Largeur])
        
    def set_boundary_condition(self):
        self.bcs.add_Uy(region=2)
        self.bcs.add_Uy(region=3)
        
    def set_loading(self):
        T_unload = largeur_creneau/wave_speed
        chargement = MyConstant(self.mesh, T_unload, magnitude, Type = "Creneau")
        self.loading.add_Fx(chargement, self.u_, self.ds(1))
        
    def set_damping(self):
        """
        Redéfini la viscosité
        """
        damp = {}
        damp.update({"damping" : True})
        damp.update({"linear_coeff" : 0.1})
        damp.update({"quad_coeff" : 0.1})
        damp.update({"correction" : True})
        return damp
        
    def csv_output(self):
        return {'Sig': True}
    
    def set_output(self):
        self.t_output_list = []
        return {'Sig': True}
    
    def query_output(self, t):
        self.t_output_list.append(t)
        
    def final_output(self):
        df = read_csv("Test_elasticite-results/Stress0.csv")
        resultat = [df[colonne].to_numpy() for colonne in df.columns]
        n_sortie = (len(resultat)-2)//3
        sigma_xx = [resultat[3 * i + 2] for i in range((len(resultat)-2)//3)]
        pas_espace = np.linspace(0, Longueur, len(sigma_xx[0]))
        for j, t in enumerate(self.t_output_list):
            plt.plot(resultat[0], sigma_xx[j+1], linestyle = "--")
            analytics = cartesian1D_progressive_wave(-magnitude, -largeur_creneau, 0, wave_speed, pas_espace, t)
            plt.plot(pas_espace, analytics)
        plt.xlim(0, Longueur)
        plt.ylim(-1.1 * magnitude, 100)
        plt.xlabel(r"Position (mm)", size = 18)
        plt.ylabel(r"Contrainte (MPa)", size = 18)
        plt.legend()
        plt.savefig(f"../../../Notice/fig/Elastodynamique_quasi_1D.pdf", bbox_inches = 'tight')

def test_Elasticite():
    pb = Isotropic_beam(Acier)
    tps1 = time.perf_counter()
    Solve(pb, compteur = sortie, TFin=Tfin, scheme = "fixed", dt = pas_de_temps)
    tps2 = time.perf_counter()
    print("temps d'execution", tps2 - tps1)
test_Elasticite()