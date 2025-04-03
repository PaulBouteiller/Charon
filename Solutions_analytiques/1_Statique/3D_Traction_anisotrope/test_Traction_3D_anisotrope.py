"""
Test de traction 3D sur un matériau orthotrope selon différentes directions.

Ce script simule des essais de traction uniaxiale sur un cube 3D composé d'un matériau
orthotrope selon trois directions principales (fibre, normale transverse, normale hors-plan)
et compare les résultats numériques avec les solutions analytiques.

Paramètres géométriques:
    - Dimensions du cube: 1 × 1 × 1
    - Discrétisation: maillage 5×5×5

Matériau orthotrope:
    - Modules d'Young: EL=12827, ET=633, EN=1344
    - Modules de cisaillement: muLT=766, muLN=703, muTN=337
    - Coefficients de Poisson: nuLT=0.466, nuLN=0.478, nuTN=0.371
    - Équation d'état: Vinet avec kappa_eq calculé à partir de la matrice de rigidité

Tests de traction:
    - Direction "Fibre": parallèle à la direction des fibres (EL)
    - Direction "maty": perpendiculaire aux fibres, dans le plan (ET)
    - Direction "matz": direction normale au plan (EN)

Le script trace les courbes force-déplacement pour les trois directions et
effectue une comparaison avec les solutions analytiques basées sur les modules d'Young.
Il trace également l'évolution de la pression en fonction de la densité pour vérifier
la cohérence de l'équation d'état.

Auteur: bouteillerp
Date de création: 11 Mars 2022
"""
from CharonX import *
import time
import matplotlib.pyplot as plt
import pytest
import csv
import pandas as pd

######## Modèle mécanique ########
model = Tridimensionnal

###### Modèle mécanique ######
EL = 12827
ET = 633
EN = 1344

muLT = 766
muLN = 703
muTN = 337

nuLT = 0.466
nuLN = 0.478
nuTN = 0.371
mat = Orthotropic(ET, EL, EN, nuLT, nuLN, nuTN, muLT, muLN, muTN)
C = mat.C
print("Matrice de rigidité", C)
kappa_eq = 1./9 * sum(C[i, j] for i in range(3) for j in range(3))
print("Coefficient de compressibilité équivalent", kappa_eq)
# eos_type = "U1"
# dico_eos = {"kappa" : kappa_eq, "alpha" : 1}

iso_T_K0 = kappa_eq
T_dep_K0 = 0
iso_T_K1 = 6
T_dep_K1 = 0
eos_type = "Vinet"
dico_eos = {"iso_T_K0": iso_T_K0, "T_dep_K0" : T_dep_K0, "iso_T_K1": iso_T_K1, "T_dep_K1" : T_dep_K1}


    
devia_type = "Anisotropic"
dico_devia = {"C" : C}
rho0 = 1
Fibre = Material(rho0, 1, eos_type, devia_type, dico_eos, dico_devia)



######## Paramètres géométriques et de maillage ########
Longueur, Largeur, hauteur = 1, 1, 1.
Nx, Ny, Nz = 5, 5, 5

eps = 0.1

F_max = EL * eps * Largeur * hauteur

class Cube3D(model):
    def __init__(self, material):
        model.__init__(self, material, analysis="static", isotherm = True)
        
    def prefix(self):
        if __name__ == "__main__": 
            return "Traction_3D"+traction
        else:
            return "Test"
          
    def define_mesh(self):
        return create_box(MPI.COMM_WORLD, [np.array([0, 0, 0]), 
                                           np.array([Longueur, Largeur, hauteur])],
                                          [Nx, Ny, Nz])
    
    def set_boundary(self):
        self.mesh_manager.mark_boundary([1, 2, 3, 4, 5, 6, ], ["x", "y", "z", "x", "y", "z"], [0, 0, 0, Longueur, Largeur, hauteur])

    def set_boundary_condition(self):
        self.bcs.add_Ux(region=1)
        self.bcs.add_Uy(region=2)
        self.bcs.add_Uz(region=3)
        Umax = eps * hauteur
        chargement = MyConstant(self.mesh, Umax, Type = "Rampe")
        if traction == "Fibre":
            self.bcs.add_Ux(value = chargement, region = 4)
        elif traction == "maty":
            self.bcs.add_Uy(value = chargement, region = 5)
        elif traction == "matz":
            self.bcs.add_Uz(value = chargement, region = 6)
            
    def csv_output(self):
        return {"Pressure" : True, "rho" : True}
        
    def set_output(self):
        self.eps_list = []
        self.F_list = []
        if traction == "Fibre":
            self.Force = self.set_F(4, "x")
        elif traction == "maty":
            self.Force = self.set_F(5, "y")
        elif traction == "matz":
            self.Force = self.set_F(6, "z")
        return {"U": True}
    
    def query_output(self, t):
        self.eps_list.append(eps * t)
        self.F_list.append(self.get_F(self.Force))
        
    def final_output(self):
        numerical_results = array(self.F_list)
        with open("Deformation", "w", newline = '') as fichier:
            writer = csv.writer(fichier)
            writer.writerow(self.eps_list)
        with open("Test" + traction, "w", newline = '') as fichier:
            writer = csv.writer(fichier)
            writer.writerow(numerical_results)

def test_Traction_3D():
    pb = Cube3D(Fibre)
    Solve(pb, compteur=1, npas=10)

for traction in ["Fibre", "maty", "matz"]:
    test_Traction_3D()
    
def lire_csv_en_numpy(nom_du_fichier):
    with open(nom_du_fichier, 'r') as fichier:
        reader=csv.reader(fichier)
        data = list(reader)
        return np.array(data[0], dtype = float)

def plot_result():
    def force_elast(eps, essai):
        if essai == "Fibre":
            return EL * eps * Largeur * hauteur
        elif essai == "maty":
            return ET * eps * Longueur * hauteur
        elif essai == "matz":
            return EN * eps * Longueur * Largeur
    
    Eps = lire_csv_en_numpy("Deformation")
    F_fibre = lire_csv_en_numpy("TestFibre")
    F_maty = lire_csv_en_numpy("Testmaty")
    F_matz = lire_csv_en_numpy("Testmatz")
    
    df = pd.read_csv("Traction_3DFibre-results/Pressure.csv")
    colonnes_numpy = [df[colonne].to_numpy() for colonne in df.columns]  
    p_fibre = [p_list[0] for p_list in colonnes_numpy[3:]]
    
    df = pd.read_csv("Traction_3DFibre-results/rho.csv")
    colonnes_numpy = [df[colonne].to_numpy() for colonne in df.columns]  
    rho_fibre = [rho_list[0] for rho_list in colonnes_numpy[3:]]
    
    df = pd.read_csv("Traction_3Dmaty-results/Pressure.csv")
    colonnes_numpy = [df[colonne].to_numpy() for colonne in df.columns]  
    p_maty = [p_list[0] for p_list in colonnes_numpy[3:]]
    
    df = pd.read_csv("Traction_3Dmaty-results/rho.csv")
    colonnes_numpy = [df[colonne].to_numpy() for colonne in df.columns]  
    rho_maty = [rho_list[0] for rho_list in colonnes_numpy[3:]]
    
    df = pd.read_csv("Traction_3Dmatz-results/Pressure.csv")
    colonnes_numpy = [df[colonne].to_numpy() for colonne in df.columns]  
    p_matz = [p_list[0] for p_list in colonnes_numpy[3:]]
    
    df = pd.read_csv("Traction_3Dmatz-results/rho.csv")
    colonnes_numpy = [df[colonne].to_numpy() for colonne in df.columns]  
    rho_matz = [rho_list[0] for rho_list in colonnes_numpy[3:]]
    
    F_fibre_analytique = array([force_elast(eps, "Fibre") for eps in Eps])
    F_maty_analytique = array([force_elast(eps, "maty") for eps in Eps])
    F_matz_analytique = array([force_elast(eps, "matz") for eps in Eps])
    eps_list_percent = [100 * eps for eps in Eps]


    plt.scatter(eps_list_percent, F_fibre, marker = "x", color = "red", label="CHARON")
    plt.plot(eps_list_percent, F_fibre_analytique, linestyle = "-", color = "black", label = "Analytique")
    
    plt.scatter(eps_list_percent, F_maty, marker = "x", color = "red")
    plt.plot(eps_list_percent, F_maty_analytique, linestyle = "-", color = "black")

    plt.scatter(eps_list_percent, F_matz, marker = "x", color = "red")
    plt.plot(eps_list_percent, F_matz_analytique, linestyle = "-", color = "black")
    plt.xlim(0, 1.1 * eps_list_percent[-1])
    plt.ylim(0, 1.1 * F_max)
    plt.xlabel(r"Déformation(%)", size = 18)
    plt.ylabel(r"Force (N)", size = 18)
    plt.legend()
    plt.savefig("Traction_3D_anisotrope.pdf", bbox_inches = 'tight')
    plt.close()
    
    plt.scatter(rho_fibre, p_fibre, marker = "x", color = "green", label = "L")
    plt.scatter(rho_maty, p_maty, marker = "x", color = "blue", label = "T")
    plt.scatter(rho_matz, p_matz, marker = "x", color = "red", label = "N")
    
    def Vinet(K0, K1, rho):
        J = rho0/rho
        return 3 * K0 * J**(-2/3) * (1-J**(1/3)) * exp(3./2 * (K1-1)*(1 - J**(1./3)))
    
    rho_list = np.linspace(0.9, 1.02)
    p_analytique = [Vinet(kappa_eq, iso_T_K1, rho) for rho in rho_list]
    plt.plot(rho_list, p_analytique, linestyle = "-", color = "black", label = "Analyique")
    
    
    # plt.xlim(0, 1.1 * eps_list_percent[-1])
    # plt.ylim(0, 1.1 * F_max)
    plt.xlabel(r"$\rho$", size = 18)
    plt.ylabel(r"Pressure (MPa)", size = 18)
    plt.legend()
    plt.savefig("Pressure.pdf", bbox_inches = 'tight')
    plt.close()
        
    
    
    
plot_result()