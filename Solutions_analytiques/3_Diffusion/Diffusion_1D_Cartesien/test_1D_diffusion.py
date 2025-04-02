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
        Nx = 100000
        return create_interval(MPI.COMM_WORLD, Nx, [np.array(bord_gauche), np.array(bord_droit)])
    
    def prefix(self):
        if __name__ == "__main__": 
            return "Test_diffusion_1D"
        else:
            return "Test"
        
    def set_boundary(self):
        self.mark_boundary([1, 2], ["x", "x"], [bord_gauche, bord_droit])
        
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
        for i in range(len(t_list)):
            list_T_erf =[analytical_T(x, t_list[i], Tfroid, TChaud) for x in pas_espace]
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
                print("cette sortie correspond au temps", (t_list[i]))
            compteur+=1
        plt.xlim(bord_gauche/4, bord_droit/4)
        plt.ylim(Tfroid, TChaud*1.02)
        plt.xlabel(r"$x$", size = 18)
        plt.ylabel(r"Temperature (K)", size = 18)
        plt.legend()
        plt.savefig("../../../Notice/fig/diffusion_1D.pdf", bbox_inches = 'tight')
        plt.show()
        
def test_Diffusion():
    pb = Isotropic_beam(Materiau)
    Solve(pb, compteur = sortie, TFin=Tfin, scheme = "fixed", dt = pas_de_temps)
test_Diffusion()