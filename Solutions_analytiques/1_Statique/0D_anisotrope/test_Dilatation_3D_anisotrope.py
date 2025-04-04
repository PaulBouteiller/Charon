"""
Created on Fri Mar 11 09:36:05 2022

@author: bouteillerp
Essai de dilatation homogène sur un cube 3D
La transformation est de la forme
\vec{x} = \vec{X}(1+alpha *t) donc \GT = (1+\alpha * t) * \TenUnit
et donc la dilatation volumique égalise J = (1+\alpha * t)**3
 """
from CharonX import *
import time
import matplotlib.pyplot as plt
import pytest
from pandas import read_csv
from math import exp

######## Modèle mécanique ########
model = Tridimensionnal

######## Modèle matériau ########
rho_TATB = 1.9e-3

# iso_T_K0 = 14e3
#Rigidité cohérente
iso_T_K0 = 28333.
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
kappa_eq = 1./9 * sum(Rigi[i,j] for i in range(3) for j in range(3))
print("Compressibilité équivalente", kappa_eq)




matrix_f_func = [[[1, 0] for _ in range(6)] for _ in range(6)]
print(matrix_f_func)
matrix_f_func[3][3] = [1, -5.617482e-01, 3.253835e+01, 5.509592e+01]

devia_type = "Anisotropic"
dico_devia = {"C" : Rigi, "f_func" : matrix_f_func}
TATB = Material(rho_TATB, 1, eos_type, devia_type, dico_eos, dico_devia)

######## Paramètres géométriques et de maillage ########
L, l, h = 1, 1, 1
Nx, Ny, Nz = 1, 1, 1

alpha = -0.12
N_pas = 10

class Cube3D(model):
    def __init__(self, material):
        model.__init__(self, material, analysis="User_driven", isotherm = True)
        
    def prefix(self):
        if __name__ == "__main__": 
            return "Compression_spherique_1D"
        else:
            return "Test"
          
    def define_mesh(self):
        return create_box(MPI.COMM_WORLD, [np.array([0, 0, 0]),  np.array([L, l, h])],
                                          [Nx, Ny, Nz], cell_type = CellType.hexahedron)
    
    def user_defined_displacement(self, t):
        pos = self.V.tabulate_dof_coordinates()
        pos_ravel = pos.ravel()
        self.J_list.append((1 +  alpha *  t)**3)
        for i in range(len(self.u.x.array)):
            self.u.x.array[i] = alpha *  t * pos_ravel[i]
        
    def set_output(self):
        self.J_list = []
        return {"Pressure" : True}
    
    def csv_output(self):
        return {"Pressure" : True, "deviateur" : True, "U" : True}
        # return {"Pressure" : True}
    
    def final_output(self):
        p = read_csv("Compression_spherique_1D-results/Pressure.csv")
        pressure = [p[colonne].to_numpy() for colonne in p.columns]
        pressure = [pressure[3:]]
        
        s = read_csv("Compression_spherique_1D-results/deviateur.csv")
        deviateur = [s[colonne].to_numpy() for colonne in s.columns]
        deviateur = [s.item() for s in deviateur[3:]]
        n = len(deviateur) // 9
        s_xx_list = [deviateur[9 * i] for i in range(n)]
        s_yy_list = [deviateur[9 * i  + 4] for i in range(n)]
        s_zz_list = [deviateur[9 * i + 8] for i in range(n)]

        print("La liste des dilatation volumique est", self.J_list)
        
        def M_0():
            unit_C_1 = Rigi[0,0] + Rigi[0,1] + Rigi[0,2]
            unit_C_2 = Rigi[1,0] + Rigi[1,1] + Rigi[1,2]
            unit_C_3 = Rigi[2,0] + Rigi[2,1] + Rigi[2,2]
            trace = unit_C_1 + unit_C_2 + unit_C_3
            return np.array([unit_C_1, unit_C_2, unit_C_3]), trace
        
        
        def deviateur_analytique(J):
            M0, trace = M_0()
            dev_M0 = M0 - 1./3 * trace * np.array([1,1,1])
            dev_SPKo = 1./3 * (J - 1) * dev_M0
            s= J**(-1) * dev_SPKo
            # s = J**(1./3) * dev_SPKo
            return s
        
        def sigma_HPP(J):
            M0, _ = M_0()
            sigma = (J**(1./3)-1) * M0
            trace = sum(sigma[i] for i in range(3))
            p = -1/3 * trace
            s = sigma - 1./3 * trace * np.array([1,1,1])
            return p, s
        
        def Vinet(K0, K1, J):
            return 3 * K0 * J**(-2/3) * (1-J**(1/3)) * exp(3./2 * (K1-1)*(1 - J**(1./3)))
        
        p_analytique = []
        s_analytique = []
        
        p_HPP = []
        s_HPP = []
        
        for J in self.J_list:
            p_analytique.append(Vinet(iso_T_K0, iso_T_K1, J))
            s_analytique.append(deviateur_analytique(J))
            p_, s_ = sigma_HPP(J)
            p_HPP.append(p_)
            s_HPP.append(s_)
            
        s_xx_analytique = [s[0] for s in s_analytique]
        s_yy_analytique = [s[1] for s in s_analytique]
        s_zz_analytique = [s[2] for s in s_analytique]
        
        s_xx_HPP = [s[0] for s in s_HPP]
        s_yy_HPP = [s[1] for s in s_HPP]
        s_zz_HPP = [s[2] for s in s_HPP]

        plt.scatter(self.J_list, pressure, marker = "x", color = "red", label="Present")
        plt.plot(self.J_list, p_analytique, linestyle = "--", color = "green",label="Vinet")
        plt.plot(self.J_list, p_HPP, linestyle = "--", color = "black",label="HPP")
        plt.xlabel(r"Volumetric compression $J$", fontsize = 18)
        plt.ylabel(r"Pressure $p$ (MPa)", fontsize = 18)
        plt.legend()
        plt.xlim(0.98 * min(self.J_list), 1)
        plt.ylim(0, 1.1 * max(p_analytique))
        plt.savefig("p.pdf", bbox_inches = 'tight')
        plt.close()
        
        # plt.scatter(self.J_list, s_xx_list, marker = "x", color ="red", label = r"$s_{xx}$")
        # plt.plot(self.J_list, s_xx_analytique, linestyle = "--", color = "green",label=r"$s_{xx}$ Analytic")
        # plt.plot(self.J_list, s_xx_HPP, linestyle = "--", color = "black",label=r"$s_{xx}$ SPH")
        
        # plt.scatter(self.J_list, s_yy_list, marker = "x", color ="red", label = r"$s_{yy}$")
        # plt.plot(self.J_list, s_yy_analytique, linestyle = "-", color = "green",label=r"$s_{yy}$ Analytic")
        # plt.plot(self.J_list, s_yy_HPP, linestyle = "-", color = "black",label=r"$s_{yy}$ SPH")
        
        # plt.scatter(self.J_list, s_zz_list, marker = "x", color ="red", label = r"$s_{zz}$")
        # plt.plot(self.J_list, s_zz_analytique, linestyle = "-.", color = "green",label=r"$s_{zz}$ Analytic")
        # plt.plot(self.J_list, s_zz_HPP, linestyle = "-.", color = "black",label=r"$s_{zz}$ SPH")
        
        
        plt.scatter(self.J_list, s_xx_list, marker = "x", color ="red")
        plt.plot(self.J_list, s_xx_analytique, linestyle = "--", color = "green")
        plt.plot(self.J_list, s_xx_HPP, linestyle = "--", color = "black",label=r"$s_{xx}$")
        
        plt.scatter(self.J_list, s_yy_list, marker = "x", color ="red")
        plt.plot(self.J_list, s_yy_analytique, linestyle = "-", color = "green")
        plt.plot(self.J_list, s_yy_HPP, linestyle = "-", color = "black",label=r"$s_{yy}$")
        
        plt.scatter(self.J_list, s_zz_list, marker = "x", color ="red")
        plt.plot(self.J_list, s_zz_analytique, linestyle = "-.", color = "green")
        plt.plot(self.J_list, s_zz_HPP, linestyle = "-.", color = "black",label=r"$s_{zz}$")
        
        plt.xlabel(r"Volumetric compression $J$", fontsize = 18)
        plt.ylabel(r"Deviatoric stress (MPa)", fontsize = 18)
        plt.legend()
        plt.xlim(0.98 * min(self.J_list), 1)
        # plt.ylim(0, 1.1 * max(p_analytique))
        plt.savefig("s.pdf", bbox_inches = 'tight')
        plt.close()
            

def test_Traction_3D():
    pb = Cube3D(TATB)
    Solve(pb, compteur=1, TFin = 1., scheme = "fixed", dt = 1/N_pas)
test_Traction_3D()
