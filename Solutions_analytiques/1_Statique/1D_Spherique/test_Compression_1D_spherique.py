"""
Created on Fri Mar 11 09:36:05 2022

@author: bouteillerp
Experience de compression d'une sphère les paramètres retenues sont identiques à ceux
utilisés par Jeremy Bleyer:
https://comet-fenics.readthedocs.io/en/latest/demo/elasticity/axisymmetric_elasticity.html """
from CharonX import *
import matplotlib.pyplot as plt
import pytest
import sys
sys.path.append("../")
from Generic_isotropic_material import *
model = SphericalUD
###### Paramètre géométrique ######
e = 2
Nx = 10
Rint = 9
Rext = Rint + e

###### Chargement ######
p_applied = 10
   
class IsotropicBall(model):
    def __init__(self, material):
        model.__init__(self, material, analysis = "static", isotherm = True)
          
    def define_mesh(self):
        return create_interval(MPI.COMM_WORLD, Nx, [np.array(Rint), np.array(Rext)])
    
    def prefix(self):
        if __name__ == "__main__": 
            return "Compression_spherique_1D"
        else:
            return "Test"
    
    def set_boundary(self):
        self.mark_boundary([1, 2], ["x", "x"], [Rint, Rext])
        
    def set_loading(self):
        self.loading.add_F(-p_applied * self.load, self.u_, self.ds(2))
        
    def set_output(self):
        return {'Sig':True, 'U':True, 'Pressure':True}
    
    def final_output(self):
        solution_numerique = self.u.x.array
        len_vec = len(solution_numerique)
        def ur(r, p_ext):
            return  - Rext**3 / (Rext**3 - Rint**3) * ((1 - 2*nu) * r + (1 + nu) * Rint**3 / (2 * r**2)) * p_ext / E
        pas_espace = np.linspace(Rint, Rext, len_vec)
        solution_analytique = np.array([ur(x, p_applied) for x in pas_espace])
        # On calcule la différence entre les deux courbes
        diff_tot = solution_analytique - solution_numerique
        # Puis on réalise une sorte d'intégration discrète
        integrale_discrete = sum(abs(diff_tot[j]) for j in range(len_vec))/sum(abs(solution_analytique[j]) for j in range(len_vec))
        print("La difference est de", integrale_discrete)
        assert integrale_discrete < 0.001, "Spheric static compression fail"
        if __name__ == "__main__": 
            plt.plot(pas_espace, solution_analytique, linestyle = "--", color = "red", label = "Analytique")
            plt.scatter(pas_espace, solution_numerique, marker = "x", color = "blue", label = "CHARON")
            
            plt.xlim(Rint, Rext)
            # plt.ylim(- 1.05 * magnitude, 0)

            plt.xlabel(r"$r$ (mm)", size = 18)
            plt.ylabel(r"Déplacement radial (mm)", size = 18)
            plt.legend()
            plt.savefig("../../../Notice/fig/Spheric_compression.pdf", bbox_inches = 'tight')
            plt.show()

def test_Compression_1D():
    pb = IsotropicBall(Acier)
    Solve(pb, compteur=1, npas=10)
test_Compression_1D()