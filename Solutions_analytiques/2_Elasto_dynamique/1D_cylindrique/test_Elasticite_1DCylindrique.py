from CharonX import *
import matplotlib.pyplot as plt
import pytest
import time
from Solution_analytique_cylindrique import main_analytique

###### Modèle géométrique ######
model = CylindricalUD
###### Modèle matériau ######
E = 210e3
nu = 0.3 
lmbda = E * nu / (1 - 2 * nu) / (1 + nu)
mu = E / 2. / (1 + nu)
rho = 7.8e-3
# rho = 1e-3
C=500
alpha=12e-6
rigi = lmbda + 2 * mu
wave_speed = (rigi/rho)**(1./2)
dico_eos = {"E":E, "nu" : nu, "alpha" : 12e-6}
dico_devia = {"E":E, "nu" : nu}
eos_type = "IsotropicHPP"

iso_T_K0 = 175e3
T_dep_K0 = 0
iso_T_K1 = 0
T_dep_K1 = 0
dico_eos = {"iso_T_K0": iso_T_K0, "T_dep_K0" : T_dep_K0, "iso_T_K1": iso_T_K1, "T_dep_K1" : T_dep_K1}
eos_type = "Vinet"

Acier = Material(rho, C, eos_type, "IsotropicHPP", dico_eos, dico_devia)

###### Paramètre géométrique ######
e = 5
R_int = 5
R_ext = R_int + e


###### Temps simulation ######
Tfin = 6e-4
pas_de_temps = Tfin/6000
magnitude = -1e4
T_unload = Tfin

sortie = 1000
pas_de_temps_sortie = sortie * pas_de_temps
n_sortie = int(Tfin/pas_de_temps_sortie)

t_etude =6e-4
n = int(Tfin / t_etude)
   
class Isotropic_ball(model):
    def __init__(self, material):
        model.__init__(self, material, isotherm = True)
          
    def define_mesh(self):
        Nx = 1000
        return create_interval(MPI.COMM_WORLD, Nx, [np.array(R_int), np.array(R_ext)])
    
    def prefix(self):
        if __name__ == "__main__": 
            return "Onde_cylindrique"
        else:
            return "Test"
        
    def set_boundary(self):
        self.mark_boundary([1, 2], ["r", "r"], [R_int, R_ext])
        
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
        df = read_csv("Onde_cylindrique-results/Sig.csv")
        plt.plot(df['r'], df.iloc[:, -2], 
                linestyle="--", label=f'CHARON t={Tfin:.2e}ms')

def test_Elasticite():
    pb = Isotropic_ball(Acier)
    Solve(pb, compteur = sortie, TFin=Tfin, scheme = "fixed", dt = pas_de_temps)
test_Elasticite()
main_analytique(R_int, R_ext, lmbda, mu, rho, magnitude, t_etude, num_points= 4000)
plt.legend()
plt.xlabel('r (mm)')
plt.ylabel(r'$\sigma_{rr}$ (MPa)')
plt.xlim(R_int, R_ext)
plt.savefig(f"../../../Notice/fig/Compression_cylindrique_dynamique.pdf", bbox_inches = 'tight')
plt.show()