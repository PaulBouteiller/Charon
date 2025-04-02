from CharonX import *
import matplotlib.pyplot as plt
import pytest
import time
import pandas as pd
from Solution_analytique_spherique import main_analytique

"""Dans cet exemple nous calculons la propagation d'un créneau dans un milieu sphérique,
l'export est réalisé au format csv puis est relu pour afficher les résultats"""


###### Modèle géométrique ######
model = SphericalUD
###### Modèle matériau ######
E = 210e3
nu = 0.3 
lmbda = E * nu / (1 - 2 * nu) / (1 + nu)
mu = E / 2. / (1 + nu)
rho = 7.8e-3
rigi = lmbda + 2 * mu
wave_speed = (rigi/rho)**(1./2)
dico_eos = {"E" : 210e3, "nu" : 0.3, "alpha" : 1}
dico_devia = {"E" : 210e3, "nu" : 0.3}
Acier = Material(rho, 1, "IsotropicHPP", "IsotropicHPP", dico_eos, dico_devia)
    
###### Paramètre géométrique ######
e = 5
R_int = 5
R_ext = R_int + e

###### Temps simulation ######
Tfin = 3e-4
print("le temps de fin de simulation est", Tfin )
pas_de_temps = Tfin / 10000
largeur_creneau = e
magnitude = 1e4
T_unload = Tfin

sortie = 2500
pas_de_temps_sortie = sortie * pas_de_temps
n_sortie = int(Tfin/pas_de_temps_sortie)
   
class Isotropic_ball(model):
    def __init__(self, material):
        model.__init__(self, material, isotherm = True)
          
    def define_mesh(self):
        Nx = 2000
        return create_interval(MPI.COMM_WORLD, Nx, [np.array(R_int), np.array(R_ext)])
    
    def prefix(self):
        if __name__ == "__main__": 
            return "Onde_spherique"
        else:
            return "Test"
        
    def set_boundary(self):
        self.mark_boundary([1, 2], ["r", "r"], [R_int, R_ext])
        
    def set_boundary_condition(self):
        self.bcs.add_axi(region=1)
        
    def set_loading(self):

        chargement = MyConstant(self.mesh, T_unload, magnitude, Type = "Creneau")
        self.loading.add_F(chargement, self.u_, self.ds(2))
    
    def csv_output(self):
        self.t_output_list = []
        return {'Sig': True}
        
    def query_output(self, t):
        self.t_output_list.append(t)
               
    def final_output(self):
        # Lecture des données
        df = pd.read_csv("Onde_spherique-results/Sig.csv")
        plt.plot(df['r'], df.iloc[:, -3], 
                linestyle="--", label=f'CHARON t={Tfin:.2e}ms')

def test_Elasticite():
    pb = Isotropic_ball(Acier)
    tps1 = time.perf_counter()
    Solve(pb, compteur=sortie, TFin=Tfin, scheme="fixed", dt=pas_de_temps)
    tps2 = time.perf_counter()
    print("temps d'execution", tps2 - tps1)
    
# Exécution des deux solutions
test_Elasticite()
main_analytique(R_int, R_ext, lmbda, mu, rho, magnitude, Tfin)
plt.legend()
plt.xlabel('r (mm)')
plt.ylabel(r'$\sigma_{rr}$ (MPa)')
plt.xlim(5, 10)
plt.savefig(f"../../../Notice/fig/Compression_spherique_dynamique.pdf", bbox_inches = 'tight')
plt.show()