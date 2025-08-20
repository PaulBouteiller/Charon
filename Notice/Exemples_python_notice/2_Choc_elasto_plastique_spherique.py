from CharonX import *
import matplotlib.pyplot as plt
from pandas import read_csv

# ###### Modèle mécanique ######
model = SphericalUD

# ###### Modèle matériau ######
rho = 7.8e-3
dico_eos = {"kappa" : 175e3, "alpha" : 12e-6}
dico_devia = {"mu" : 80e3}
Acier = Material(rho, 1, "U8", "NeoHook", dico_eos, dico_devia)

###### Paramètre géométrique ######
Rint = 0.
Rext = 50

###### Temps simulation ######
Tfin = 7e-3
pas_de_temps = Tfin/8000
sortie = 1000
pas_de_temps_sortie = sortie * pas_de_temps
n_sortie = int(Tfin/pas_de_temps_sortie)
   
class Isotropic_ball(model):
    def __init__(self, material):
        model.__init__(self, material, plastic="HPP_Plasticity", isotherm = True)
          
    def define_mesh(self):
        Nx = 100
        return create_interval(COMM_WORLD, Nx, [np.array(Rint), np.array(Rext)])
    
    def prefix(self):
        return "Spherique_elasto_plastique"
        
    def set_boundary(self):
        self.mark_boundary([1, 2], ["r", "r"], [Rint, Rext])

    def set_boundary_condition(self):
        self.bcs.add_axi(region=1)
        
    def set_loading(self):
        T_unload = Tfin/4
        magnitude = 1e4
        chargement = MyConstant(self.mesh, T_unload, magnitude, Type = "Creneau")
        self.loading.add_F(chargement, self.u_, self.ds(2))

    def set_plastic(self):
        sig_Y=1e3
        H=0
        self.constitutive.plastic.set_plastic(sigY = sig_Y, H = H)
        
    def csv_output(self):
        return {"Sig":True, "U" : ["Boundary", 1]}
        
    def final_output(self):
        df = read_csv("Spherique_elasto_plastique-results/Sig.csv")
        resultat = [df[colonne].to_numpy() for colonne in df.columns]
        n_sortie = (len(resultat)-1)//2
        for i in range(n_sortie):
            plt.plot(resultat[0], resultat[2 * i + 1], linestyle = "--")
    
###### Définition du problème et résolution ######
pb = Isotropic_ball(Acier)
Solve(pb, compteur = sortie, TFin=Tfin, scheme = "fixed", dt = pas_de_temps)