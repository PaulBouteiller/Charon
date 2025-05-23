"""
Test de validation pour la diffusion thermique 1D en coordonnées cartésiennes

Ce script implémente et exécute un test de validation pour l'équation de diffusion
thermique en 1D dans un système de coordonnées cartésiennes. Il compare
la solution numérique obtenue avec CharonX à la solution analytique de type
fonction d'erreur (erf).

Cas test:
---------
- Barre unidimensionnelle de longueur L = 210 μm
- Température initiale discontinue: T = 800K dans un intervalle central de 10 μm,
  T = 300K ailleurs
- Conditions aux limites: aucune condition imposée (diffusion libre)
- Comparaison des profils de température numériques et analytiques à différents instants

Théorie:
--------
L'équation de diffusion thermique en 1D s'écrit:
    ∂T/∂t = D·∂²T/∂x²

où D = λ/(ρ·C) est le coefficient de diffusion thermique.

La solution analytique pour une discontinuité initiale utilise la fonction d'erreur:
    T(x,t) = (T_chaud-T_froid)·[erf((x+point_droit)/√(4Dt)) - erf((x+point_gauche)/√(4Dt))]/2 + T_froid

Paramètres matériaux:
--------------------
- Conductivité thermique: λ = 240 W/(m·K)
- Capacité thermique massique: C = 1000 J/(kg·K)
- Masse volumique: ρ = 2.785×10³ kg/m³
- Coefficient de diffusion: D = λ/(ρ·C) ≈ 8.62×10⁻⁵ m²/s

Auteur: bouteillerp
"""

from CharonX import *
import matplotlib.pyplot as plt
from scipy.special import erf
# from Solution_analytique_1D_diffusion import * 
###### Modèle géométrique ######
model = CartesianUD

###### Modèle mécanique ######
E = 210e3
nu = 0.3 
rho = 2.785e3
C = 1000
alpha = 12e-6
dico_eos = {"E" : E, "nu" : nu, "alpha" : alpha}
dico_devia = {"E":E, "nu" : nu}
Materiau = Material(rho, C, "IsotropicHPP", "IsotropicHPP", dico_eos, dico_devia)

###### Modèle thermique ######
lmbda_diff = 240
AcierTherm = LinearThermal(lmbda_diff)

###### Paramètre géométrique ######
L = 210e-6
bord_gauche = -105e-6
bord_droit = bord_gauche + L
point_gauche = -5e-6
point_droit = 5e-6

###### Champ de température initial ######
Tfin = 1e-6
Tfroid = 300
TChaud = 800

###### Paramètre temporel ######
sortie = 25
pas_de_temps = 5e-9
pas_de_temps_sortie = sortie * pas_de_temps

n_sortie = int(Tfin/pas_de_temps_sortie)

class Isotropic_beam(model):
    def __init__(self, material):
        model.__init__(self, material, analysis = "Pure_diffusion",  adiabatic = False, Thermal_material = AcierTherm)
          
    def define_mesh(self):
        Nx = 1000
        return create_interval(MPI.COMM_WORLD, Nx, [np.array(bord_gauche), np.array(bord_droit)])
    
    def prefix(self):
        if __name__ == "__main__": 
            return "Test_diffusion_1D"
        else:
            return "Test"
        
    def set_boundary(self):
        self.mesh_manager.mark_boundary([1, 2], ["x", "x"], [bord_gauche, bord_droit])
        
    def set_initial_temperature(self):
        x = SpatialCoordinate(self.mesh)
        ufl_condition = conditional(And(lt(x[0], point_droit), gt(x[0], point_gauche)), TChaud, Tfroid)
        T_expr = Expression(ufl_condition, self.V_T.element.interpolation_points())
        self.T0 = Function(self.V_T)
        self.T0.interpolate(T_expr)
        self.T.interpolate(T_expr)
        
    def set_boundary_condition(self):
        self.bcs_T = []
        
    def set_output(self):
        self.T_vector = [self.T.x.array]
        self.T_max = [800]
        return {'Sig': True, 'U':True, 'T':True}
    
    def query_output(self, t):
        self.T_vector.append(np.array(self.T.x.array))
        self.T_max.append(np.max(self.T_vector[-1]))
    
    def final_output(self):
        def analytical_f(x,t):
            D_act = lmbda_diff/(rho * C)
            denom = np.sqrt(4 * D_act * t)
            return 1./2 * (erf((x+point_droit)/denom)-erf((x+point_gauche)/denom))

        def analytical_T(x,t, Tf, Tc):
            return (Tc-Tf) * analytical_f(x,t) + Tf
        len_vec = len(self.T_vector[0])
        t_list = np.linspace(1e-12, Tfin, n_sortie+1)
        pas_espace = np.linspace(bord_gauche, bord_droit, len_vec)
        compteur = 0
        for i, t in enumerate(t_list):
            list_T_erf =[analytical_T(x, t, Tfroid, TChaud) for x in pas_espace]
            diff_tot = list_T_erf - self.T_vector[i]
            int_discret = sum(abs(diff_tot[j]) for j in range(len_vec))/sum(abs(list_T_erf[j]) for j in range(len_vec))
            print("La difference est de", int_discret)
            # assert int_discret < 0.002, "1D cartesian diffusion fails"
            # print("1D cartesian diffusion succeed")
            if __name__ == "__main__": 
                if compteur == 0:
                    label_analytical = "Analytique"
                    label_CHARON = "CHARON"
                else:
                    label_analytical = None
                    label_CHARON = None
                plt.plot(pas_espace, list_T_erf, linestyle = "-", color = "red", label = label_analytical)
                plt.plot(pas_espace, self.T_vector[i], linestyle = "--", color = "blue", label=label_CHARON)
                print("cette sortie correspond au temps", (t))
            compteur+=1
        plt.xlim(bord_gauche/4, bord_droit/4)
        plt.ylim(Tfroid, TChaud*1.02)
        plt.xlabel(r"$x$", size = 18)
        plt.ylabel(r"Temperature (K)", size = 18)
        plt.legend()
        plt.show()
        
def test_Diffusion():
    pb = Isotropic_beam(Materiau)
    Solve(pb, compteur = sortie, TFin=Tfin, scheme = "fixed", dt = pas_de_temps)
test_Diffusion()