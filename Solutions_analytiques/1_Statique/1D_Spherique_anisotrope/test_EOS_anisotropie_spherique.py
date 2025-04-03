"""
Created on Fri Mar 11 09:36:05 2022

@author: bouteillerp
Experience de compression d'une sphère anisotrope"""
from CharonX import *
import matplotlib.pyplot as plt
import pytest
from pandas import read_csv
model = SphericalUD
rho_TATB = 1.9e-3

iso_T_K0 = 14e3
T_dep_K0 = 0
iso_T_K1 = 6
T_dep_K1 = 0
eos_type = "Vinet"
dico_eos = {"iso_T_K0": iso_T_K0, "T_dep_K0" : T_dep_K0, "iso_T_K1": iso_T_K1, "T_dep_K1" : T_dep_K1}


Rigi = np.zeros((6,6))
Rigi[0,0] = Rigi[2,2] =60e3
Rigi[1,1] = 25e3
Rigi[0,1] = Rigi[1,0] = 25e3
Rigi[0,2] = Rigi[1,2] = Rigi[2,0] = Rigi[2,1] =15e3
Rigi[3,3] = 25e3
Rigi[4,4] = 2e3
Rigi[5,5] = 2e3
print("La rigidité est", Rigi)

devia_type = "Anisotropic"
dico_devia = {"C" : Rigi}

TATB = Material(rho_TATB, 1, eos_type, devia_type, dico_eos, dico_devia)
###### Paramètre géométrique ######
Nx = 1
Rint = 0
Rext = 1

class AnisotropicBall(model):
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
        
    def set_boundary_condition(self):
        Umax = 1e-2
        self.bcs.add_U(region=1)
        chargement = MyConstant(self.mesh, Umax, Type = "Rampe")
        self.bcs.add_U(value = chargement, region=2)
        
    def csv_output(self):
        return {"Pressure" : True, "deviateur" : True, "U" : True, "Sig" : True}
    
    def final_output(self):
        p = read_csv("Compression_spherique_1D-results/Pressure.csv")
        pressure = [p[colonne].to_numpy() for colonne in p.columns]
        del pressure[0]
        
        s = read_csv("Compression_spherique_1D-results/deviateur.csv")
        deviateur = [s[colonne].to_numpy() for colonne in s.columns]
        del deviateur[0]
        deviateur = [s.item() for s in deviateur]
        n = len(deviateur) // 3
        s_rr_list = [deviateur[3 * i] for i in range(n)]
        s_tt_list = [deviateur[3 * i  + 1] for i in range(n)]
        s_phiphi_list = [deviateur[3 * i + 2] for i in range(n)]
        
        U = read_csv("Compression_spherique_1D-results/U.csv")
        dep = [U[colonne].to_numpy() for colonne in U.columns]
        u_bord = [u_tot[1] for u_tot in dep[1:]]
        J_list = [(1+u/Rext)**3 for u in u_bord]
        print("La liste des dilatation volumique est", J_list)
        
        def deviateur_analytique(J):
            unit_C_1 = Rigi[0,0] + Rigi[0,1] + Rigi[0,2]
            unit_C_2 = Rigi[1,0] + Rigi[1,1] + Rigi[1,2]
            unit_C_3 = Rigi[2,0] + Rigi[2,1] + Rigi[2,2]
            trace = unit_C_1 + unit_C_2 + unit_C_3
            dev_M0 = np.array([unit_C_1, unit_C_2, unit_C_3]) - 1./3 * trace * np.array([1,1,1])
            dev_SPKo = 1./3 * (J - 1) * dev_M0
                                     
            s = J**(1./3) * dev_SPKo
            return s
        
        def Vinet(K0, K1, J):
            return 3 * K0 * J**(-2/3) * (1-J**(1/3)) * exp(3./2 * (K1-1)*(1 - J**(1./3)))
        
        p_analytique = []
        s_analytique = []
        for J in J_list:
            p_analytique.append(Vinet(iso_T_K0, iso_T_K1, J))
            s_analytique.append(deviateur_analytique(J))
        s_rr_analytique = [s[0] for s in s_analytique]
        s_tt_analytique = [s[1] for s in s_analytique]
        s_phiphi_analytique = [s[2] for s in s_analytique]

        plt.scatter(J_list, pressure, marker = "x", color = "blue", label="CHARON ")
        plt.plot(J_list, p_analytique, linestyle = "--", color = "red",label="Analytique Vinet")
        plt.xlabel(r"Dilatation volumique $J$", fontsize = 18)
        plt.ylabel(r"Pression $P$", fontsize = 18)
        plt.legend()
        plt.show()
        
        plt.scatter(J_list, s_rr_list, marker = "x", color = "blue")
        plt.plot(J_list, s_rr_analytique, linestyle = "--", color = "green",label="s_rr")
        plt.xlabel(r"Dilatation volumique $J$", fontsize = 18)
        plt.ylabel(r"Deviateur $P$", fontsize = 18)
        plt.legend()
        plt.show()
            

def test_Compression_1D():
    pb = AnisotropicBall(TATB)
    Solve(pb, compteur=1, npas=10)
test_Compression_1D()