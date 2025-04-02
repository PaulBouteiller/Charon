from CharonX import *
import matplotlib.pyplot as plt
import pytest
import time
import sys
from materiau import set_material
Acier, wave_speed, isotherm, T0  = set_material()

import os
import sys
# Obtenir le chemin absolu du dossier parent
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from Analytical_wave_propagation import *
###### Modèle géométrique ######
model = CartesianUD
###### Paramètre géométrique ######
L = 50
bord_gauche = 0
bord_droit = bord_gauche + L

###### Temps simulation ######
Tfin = 3./4 * L / wave_speed
print("le temps de fin de simulation est", Tfin )
pas_de_temps = Tfin/8000
largeur_creneau = L/4
magnitude = 1e3

sortie =4000
pas_de_temps_sortie = sortie * pas_de_temps
n_sortie = int(Tfin/pas_de_temps_sortie)
   
class Isotropic_beam(model):
    def __init__(self, material):
        model.__init__(self, material, isotherm = True)
          
    def define_mesh(self):
        Nx = 1000
        return create_interval(MPI.COMM_WORLD, Nx, [np.array(bord_gauche), np.array(bord_droit)])
    
    def prefix(self):
        if __name__ == "__main__": 
            return "Test_elasticite"
        else:
            return "Test"
        
    def set_boundary(self):
        self.mark_boundary([1, 2], ["x", "x"], [bord_gauche, bord_droit])
        
    def set_loading(self):
        T_unload = largeur_creneau/wave_speed
        chargement = MyConstant(self.mesh, T_unload, magnitude, Type = "Creneau")
        self.loading.add_F(chargement, self.u_, self.ds(1))
        
    def set_initial_temperature(self):
        self.T0 = Function(self.V_T)
        self.T0.x.array[:] = T0
        self.T.x.array[:] = T0
        
    def csv_output(self):
        return {'Sig': True}
    
    def set_output(self):
        self.t_output_list = []
        return {'Sig': True}
        
    def query_output(self, t):
        self.t_output_list.append(t)
        
    def final_output(self):
        df = read_csv("Test_elasticite-results/Sig.csv")
        resultat = [df[colonne].to_numpy() for colonne in df.columns]
        n_sortie = len(self.t_output_list)
        pas_espace = np.linspace(bord_gauche, bord_droit, len(resultat[-1]))
        for i, t in enumerate(self.t_output_list):
            plt.plot(resultat[0], resultat[i + 2], linestyle = "--")
            analytics = cartesian1D_progressive_wave(-magnitude, -largeur_creneau, 0, wave_speed, pas_espace, t)
            plt.plot(pas_espace, analytics)
        plt.xlim(0, L)
        plt.ylim(-1.1 * magnitude, 100)
        plt.xlabel(r"Position (mm)", size = 18)
        plt.ylabel(r"Contrainte (MPa)", size = 18)
        plt.legend()
        plt.savefig(f"../../../Notice/fig/Elasticite_1D_cartesien.pdf", bbox_inches = 'tight')

def test_Elasticite():
    pb = Isotropic_beam(Acier)
    tps1 = time.perf_counter()
    Solve(pb, compteur = sortie, TFin=Tfin, scheme = "fixed", dt = pas_de_temps)
    # Solve(pb, TFin=Tfin, scheme = "fixed", dt = pas_de_temps)
    tps2 = time.perf_counter()
    print("temps d'execution", tps2 - tps1)
test_Elasticite()