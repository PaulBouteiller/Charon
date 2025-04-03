"""
Test de validation pour l'élasticité 1D dans un bi-matériau en coordonnées cartésiennes

Ce script implémente et exécute un test de validation pour la propagation d'ondes 
élastiques à travers l'interface entre deux matériaux différents en coordonnées cartésiennes 1D.
Il compare la solution numérique obtenue avec CharonX à la solution analytique.

Cas test:
---------
- Barre élastique avec deux matériaux (acier et aluminium)
- Interface située au milieu de la barre (x = 25 mm)
- Chargement en créneau sur l'extrémité gauche
- Propagation, réflexion et transmission d'onde à l'interface
- Comparaison des contraintes numériques et analytiques à différents instants

Théorie:
--------
Lors de la rencontre d'une onde avec une interface entre deux matériaux d'impédances
acoustiques Z₁ et Z₂ différentes, une partie de l'onde est réfléchie et une partie est
transmise selon les coefficients:
    R = (Z₂ - Z₁)/(Z₁ + Z₂)    (coefficient de réflexion)
    T = 2·Z₂/(Z₁ + Z₂)         (coefficient de transmission)

La solution analytique complète est implémentée dans le module Solution_analytique.py.

Auteur: bouteillerp
"""

from CharonX import *
import matplotlib.pyplot as plt
import pytest
import time
from Solution_analytique import compute_sigma_tot

###### Modèle mécanique ######
E_acier = 210e3
nu_acier = 0.3
mu_acier = E_acier / 2. / (1 + nu_acier)
rho_acier = 7.8e-3
dico_eos = {"E" : E_acier, "nu" : nu_acier, "alpha" : 1}
dico_devia = {"E":E_acier, "nu" : nu_acier}
eos_type = "IsotropicHPP"
devia_type = "NeoHook"
Acier = Material(rho_acier, 1, eos_type, devia_type, dico_eos, dico_devia)

###### Modèle mécanique ######
E_alu = 70e3
nu_alu = 0.34
mu_alu = E_alu / 2. / (1 + nu_alu)
rho_alu = 2.7e-3
dico_eos_alu = {"E" : E_alu, "nu" : nu_alu, "alpha" : 1}
dico_devia_alu = {"E" : E_alu, "nu" : nu_alu}
eos_type_alu = "IsotropicHPP"
devia_type_alu = "NeoHook"
Alu = Material(rho_alu, 1, eos_type_alu, devia_type_alu, dico_eos_alu, dico_devia_alu)

Mat = [Acier, Alu]

###### Modèle géométrique ######
model = CartesianUD

###### Paramètre géométrique ######
L = 50
demi_longueur = L/2
bord_gauche = 0
bord_droit = bord_gauche + L

###### Temps simulation ######
wave_speed = Acier.celerity

Tfin = 3./4 * L / wave_speed
print("le temps de fin de simulation est", Tfin )
pas_de_temps = Tfin/8000
largeur_creneau = L/4
T_unload = largeur_creneau/wave_speed
magnitude = 1e3

sortie = 4000
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
        self.mesh_manager.mark_boundary([1, 2], ["x", "x"], [bord_gauche, bord_droit])
        
    def set_loading(self):

        chargement = MyConstant(self.mesh, T_unload, magnitude, Type = "Creneau")
        self.loading.add_F(chargement, self.u_, self.ds(1))
        
    def set_initial_temperature(self):
        self.T0 = Function(self.V_T)
        self.T0.x.array[:] = 293.15
        self.T.x.array[:] = 293.15
        
    def set_multiphase(self):
        x = SpatialCoordinate(self.mesh)
        mult = self.multiphase
        interp = mult.V_c.element.interpolation_points()

        ufl_condition_1 = conditional(x[0]<demi_longueur, 1, 0)
        c1_expr = Expression(ufl_condition_1, interp)
        ufl_condition_2 = conditional(x[0]>=demi_longueur, 1, 0)
        c2_expr = Expression(ufl_condition_2, interp)
        mult.multiphase_evolution =  [False, False]
        mult.explosive = False
        mult.set_multiphase([c1_expr, c2_expr])
        
    def csv_output(self):
        return {'Sig': True}
    
    def set_output(self):
        self.t_output_list = []

        return {}
        
    def query_output(self, t):
        self.t_output_list.append(t)
        
    def final_output(self):
        df = read_csv("Test_elasticite-results/Sig.csv")
        resultat = [df[colonne].to_numpy() for colonne in df.columns]
        n_sortie = len(self.t_output_list)

        x_vals = np.linspace(0, L, 1000)
        for i, t in enumerate(self.t_output_list):
            plt.plot(resultat[0], resultat[i + 2], linestyle = "--")
            sigma_vals = compute_sigma_tot(t, T_unload, L, demi_longueur, magnitude, rho_acier, rho_alu, E_acier, E_alu, nu_acier, nu_alu)
            plt.plot(x_vals, sigma_vals, color='r')
            
        plt.xlim(0, L)
        plt.ylim(-1.1 * magnitude, 1.1 * magnitude)
        plt.xlabel(r"Position (mm)", size = 18)
        plt.ylabel(r"Contrainte (MPa)", size = 18)
        plt.legend()
        

def test_Elasticite():
    pb = Isotropic_beam(Mat)
    tps1 = time.perf_counter()
    Solve(pb, compteur = sortie, TFin=Tfin, scheme = "fixed", dt = pas_de_temps)
    tps2 = time.perf_counter()
    print("temps d'execution", tps2 - tps1)
test_Elasticite()