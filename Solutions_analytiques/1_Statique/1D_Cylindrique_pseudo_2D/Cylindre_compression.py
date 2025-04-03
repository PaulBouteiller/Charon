"""
Created on Fri Mar 11 09:36:05 2022

@author: bouteillerp
Traction uniaxiale sur une plaque en déformation plane"""

from CharonX import *
import time
import matplotlib.pyplot as plt
import pytest
from numpy import pi

model = Axisymetric
###### Modèle mécanique ######
E = 210e3
nu = 0.3 
lmbda = E * nu / (1 - 2 * nu) / (1 + nu)
mu = E / 2. / (1 + nu)
dico_eos = {"E" : E, "nu" : nu, "alpha" : 1}
dico_devia = {"E" : E, "nu" : nu}
Acier = Material(1, 1, "IsotropicHPP", "IsotropicHPP", dico_eos, dico_devia)

###### Paramètre géométrique ######
L = 2
Nx = 100
Rint = 9
Rext = Rint + L
hauteur = 1

###### Chargement ######
Pext = -10
Pint = -5
   
class Cylindre_axi(model):
    def __init__(self, material):
        model.__init__(self, material, analysis = "static", isotherm = True)
          
    def define_mesh(self):
        #Ne fonctionne qu'avec des triangles voir pourquoi
        # return create_rectangle(MPI.COMM_WORLD, [(Rint, 0), (Rext, hauteur)], [20, 10])
        return create_rectangle(COMM_WORLD, [(Rint, 0), (Rext, hauteur)], [20, 10], CellType.quadrilateral)

    def prefix(self):
        if __name__ == "__main__": 
            return "Cylindre_axi"
        else:
            return "Test"
        
    def set_boundary(self):
        self.mark_boundary([1, 2, 3, 4], ["z", "r", "r", "z"], [0, Rint, Rext, hauteur])
        # self.mark_boundary([1], ["z"], [0])
        
    def set_boundary_condition(self):
        self.bcs.add_Uz(region=1)
        self.bcs.add_Uz(region=4)
        
    # def set_boundary_condition(self):
    #     self.bcs.add_clamped(region=1)
        
    def set_loading(self):
        self.loading.add_Fr(-Pint * self.load, self.u_, self.ds(2))
        self.loading.add_Fr(Pext * self.load, self.u_, self.ds(3))
        
    def csv_output(self):
        return {"U" : ["Boundary", 1]}
        # return {"U" : True}
        
    def set_output(self):
        return {'U': True}
        
    def final_output(self):
        u_csv = read_csv("Cylindre_axi-results/Displacement0.csv")
        resultat = [u_csv[colonne].to_numpy() for colonne in u_csv.columns]
        r_result = resultat[0]
        solution_numerique = -resultat[-2]
        
        len_vec = len(r_result)
        A = (Pint * Rint**2 - Pext * Rext**2) / (Rext**2 - Rint**2)
        B = (Pint - Pext) / (Rext**2 - Rint**2) * Rint**2 * Rext**2
        C = 2 * nu * A
        a = (1-nu)/E * A - nu /E * C
        b = B / (2 * mu)
       
        def ur(r):
            return  a * r + b / r
        solution_analytique = np.array([ur(x) for x in r_result])
        # On calcule la différence entre les deux courbes
        diff_tot = solution_analytique - solution_numerique
        # Puis on réalise une sorte d'intégration discrète
        integrale_discrete = sum(abs(diff_tot[j]) for j in range(len_vec))/sum(abs(solution_analytique[j]) for j in range(len_vec))
        print("La difference est de", integrale_discrete)
        # assert integrale_discrete < 1e-3, "Cylindrical static compression fail"
        if __name__ == "__main__": 
            plt.plot(r_result, solution_analytique, linestyle = "--", color = "red")
            plt.scatter(r_result, solution_numerique, marker = "x", color = "blue")
            
            plt.xlim(Rint, Rext)
            plt.xlabel(r"$r$", size = 18)
            plt.ylabel(r"Déplacement radial", size = 18)
            plt.savefig("../../../Notice/fig/1D_Cylindrique_compression_pseudo_2D.pdf", bbox_inches = 'tight')
            # plt.show()
            
def test_Compression_Axi():
    pb = Cylindre_axi(Acier)
    Solve(pb, compteur=1, npas=20)
test_Compression_Axi()