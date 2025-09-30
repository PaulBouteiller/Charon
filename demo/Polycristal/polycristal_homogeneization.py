from Charon import Tridimensional, Solve, MeshManager, Material
from mpi4py.MPI import COMM_WORLD
from dolfinx.io import gmshio
from pandas import read_csv
from math import exp
import matplotlib.pyplot as plt
from ufl import SpatialCoordinate
from numpy import array, mean, linspace, unique

###### Modèle mécanique ######
rho0 = 8850
C_mass = 1
C11 = 168.4e9
C12 = 121.4e9
C44 = 76.19e9
C = array([[C11, C12, C12, 0, 0, 0],
           [C12, C11, C12, 0, 0, 0],
           [C12, C12, C11, 0, 0, 0],
           [0, 0, 0, C44, 0, 0],
           [0, 0, 0, 0, C44, 0],
           [0, 0, 0, 0, 0, C44]])

calibration = array([-4.913598e+00, -1.342533e+01, -4.051586e+01])

f_func = [[None, None, None, None, None, None],
          [None, None, None, None, None, None],
          [None, None, None, None, None, None],
          [None, None, None, calibration, None, None],
          [None, None, None, None, calibration, None],
          [None, None, None, None, None, calibration]]

dev_type = "Anisotropic"
deviator_params = {"C" : C, "f_func" : f_func}

iso_T_K0 = 133e9
T_dep_K0 = 0
iso_T_K1 = 5.3
T_dep_K1 = 0
eos_type = "Vinet"
dico_eos = {"iso_T_K0": iso_T_K0, "T_dep_K0" : T_dep_K0, "iso_T_K1": iso_T_K1, "T_dep_K1" : T_dep_K1}

copper = Material(rho0, C_mass, eos_type, dev_type, dico_eos, deviator_params)

#%%Mesh
mesh_name= "polycristal3D_min"
output_name = "Compression_cisaillement"+mesh_name
mesh, cell_tags, _ = gmshio.read_from_msh(mesh_name+".msh", COMM_WORLD, gdim=3, rank=0)


dictionnaire_mesh = {"tags": [1, 2, 3, 4, 5, 6],
                     "coordinate": ["x", "y", "z", "x", "y", "z"], 
                     "positions": [0, 0, 0, 1, 1, 1],
                     "cell_tags" : cell_tags,
                     "fem_parameters" : {"u_degree" : 1, "schema" : "default"},
                     }
mesh_manager = MeshManager(mesh, dictionnaire_mesh)

#%% Paramètres du problème

def extract_orientations(filename):
    """Extrait orientations du fichier .msh"""
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Trouve section $ElsetOrientations
    start = None
    for i, line in enumerate(lines):
        if '$ElsetOrientations' in line:
            start = i + 1
            break
    
    if start is None:
        return None
    
    orientations = []
    for i in range(start, len(lines)):
        line = lines[i].strip()
        if line.startswith('$') or not line:
            break
        parts = line.split()
        if len(parts) >= 4:
            orientations.append([float(x) for x in parts[1:4]])  # phi1, Phi, phi2
    
    orientations = array(orientations)
    print(f"Orientations: {len(orientations)} grains")
    return orientations

x = SpatialCoordinate(mesh)
gamma = 1e-3
J_final = 0.5
alpha_f = J_final**(1./3)-1
ux_compression = alpha_f * (x[0] + gamma * x[1])
uy_compression = alpha_f * x[1]
uz_compression = alpha_f * x[2]

chargement = {"type" : "rampe", "pente" : gamma}
orientations = extract_orientations(mesh_name+".msh")
dictionnaire = {"material" : copper,
                "mesh_manager" : mesh_manager,
                "boundary_conditions": 
                    [{"component": "Ux", "tag": 1, "value": {"type" : "ufl_expression", "expression" : ux_compression}},
                     {"component": "Uy", "tag": 1, "value": {"type" : "ufl_expression", "expression" : uy_compression}},
                     {"component": "Uz", "tag": 1, "value": {"type" : "ufl_expression", "expression" : uz_compression}},
                     {"component": "Ux", "tag": 2, "value": {"type" : "ufl_expression", "expression" : ux_compression}},
                     {"component": "Uy", "tag": 2, "value": {"type" : "ufl_expression", "expression" : uy_compression}},
                     {"component": "Uz", "tag": 2, "value": {"type" : "ufl_expression", "expression" : uz_compression}},
                     {"component": "Ux", "tag": 3, "value": {"type" : "ufl_expression", "expression" : ux_compression}},
                     {"component": "Uy", "tag": 3, "value": {"type" : "ufl_expression", "expression" : uy_compression}},
                     {"component": "Uz", "tag": 3, "value": {"type" : "ufl_expression", "expression" : uz_compression}},
                     {"component": "Ux", "tag": 4, "value": {"type" : "ufl_expression", "expression" : ux_compression}},
                     {"component": "Uy", "tag": 4, "value": {"type" : "ufl_expression", "expression" : uy_compression}},
                     {"component": "Uz", "tag": 4, "value": {"type" : "ufl_expression", "expression" : uz_compression}},
                     {"component": "Ux", "tag": 5, "value": {"type" : "ufl_expression", "expression" : ux_compression}},
                     {"component": "Uy", "tag": 5, "value": {"type" : "ufl_expression", "expression" : uy_compression}},
                     {"component": "Uz", "tag": 5, "value": {"type" : "ufl_expression", "expression" : uz_compression}},
                     {"component": "Ux", "tag": 6, "value": {"type" : "ufl_expression", "expression" : ux_compression}},
                     {"component": "Uy", "tag": 6, "value": {"type" : "ufl_expression", "expression" : uy_compression}},
                     {"component": "Uz", "tag": 6, "value": {"type" : "ufl_expression", "expression" : uz_compression}},
                    ],
                "analysis" : "static",
                "isotherm" : True,
                "polycristal" : {"tags" : unique(cell_tags.values).tolist(), "euler_angle" : orientations}
                }

pb = Tridimensional(dictionnaire)

#%%Résolution
def u_cisaillement(x):
    return gamma * x[1]

n_pas = 21
dico_solve = {"Prefix" : output_name, "output" : {"sig" : True},
                "csv_output": {"sig" : True, "p" : True}, 
                "initial_conditions" : {"Ux" : u_cisaillement}}
solve_instance = Solve(pb, dico_solve, npas=n_pas)
solve_instance.solve()

#%%Post traitement
# Désactiver LaTeX et utiliser mathtext
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'stix'

t_list = linspace(0, 1, n_pas)
J_list = [(1 + alpha_f * t)**3 for t in t_list]
rho_list = [rho0/J for J in J_list]
Pa_to_GPa = 1e9

fontsize = 20
axe_size = 14
label_size = 18 
decalage_x = 0.01

# Création de la figure avec subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

#%% Pressure plot (subplot 1)
df_p = read_csv(output_name + "-results/p.csv")
resultat_p = [df_p[colonne].to_numpy() for colonne in df_p.columns]
p_list = []
for i in range(int(n_pas)):
    p_moy = mean(resultat_p[i + 3])/Pa_to_GPa
    p_list.append(p_moy)

def vinet(K0, K1, J):
    return 3 * K0 * J**(-2/3) * (1-J**(1/3)) * exp(3./2 * (K1-1)*(1 - J**(1./3)))

p_analytique = [vinet(iso_T_K0, iso_T_K1, J)/Pa_to_GPa for J in J_list]

ax1.scatter(J_list, p_list, color="blue", marker="x", label="Simulation")
ax1.plot(J_list, p_analytique, color="green", linestyle='--', label="Vinet analytique")
ax1.set_xlim(min(J_list)-decalage_x, decalage_x + max(J_list))
ax1.tick_params(axis='both', labelsize=axe_size)
ax1.set_xlabel(r"Volumetric compression $J$", size=label_size)
ax1.set_ylabel(r"Pressure (GPa)", size=label_size)
ax1.grid(True)
ax1.legend(fontsize=12)

#%% Shear modulus plot (subplot 2)
df = read_csv(output_name + "-results/sig.csv")
resultat = [df[colonne].to_numpy() for colonne in df.columns]
mu_list = []
for i in range(int(n_pas)):
    sig_xy_moy = mean(resultat[9 * i + 4])
    mu_moy = sig_xy_moy/gamma/(1e9)
    print(f"Module de cisaillement (GPa) pour J={J_list[i]}:", mu_moy)
    mu_list.append(mu_moy)

J_list_esc = [1, 0.9, 0.8, 0.7, 0.6, 0.5]
mu_list_esc = [53.912, 81.901, 104.66, 168.57, 279.70, 499.25]

ax2.plot(J_list, mu_list, color="blue", label="Simulation")
ax2.scatter(J_list_esc, mu_list_esc, color="red", marker="x", label="Données ESC")
ax2.set_ylim(0, 1.01*max(max(mu_list_esc), max(mu_list)))
ax2.set_xlim(min(J_list)-decalage_x, decalage_x + max(J_list))
ax2.tick_params(axis='both', labelsize=axe_size)
ax2.set_xlabel(r"Volumetric compression $J$", size=label_size)
ax2.set_ylabel(r"Shear modulus (GPa)", size=label_size)
ax2.grid(True)
ax2.legend(fontsize=12)
plt.tight_layout()
