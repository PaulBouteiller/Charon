from Charon import Material, create_1D_mesh, CartesianUD, Solve, MeshManager
###### Modèle matériau Acier ######
E = 210e3
nu = 0.3
rho = 7.8e-3
C_mass = 500
alpha = 12e-6
mu = E / 2. / (1 + nu)
dico_eos = {"E" : E, "nu" : nu, "alpha" : alpha}
dico_devia = {"mu" : mu}
eos_type = "IsotropicHPP"
devia_type = "IsotropicHPP"
Acier = Material(rho, C_mass, eos_type, devia_type, dico_eos, dico_devia)

###### Paramètre géométrique ######
L = 1
Nx = 2
mesh = create_1D_mesh(0, L, Nx)
dico_mesh = {"tags": [1, 2], "coordinate": ["x", "x"], "positions": [0, L]}
mesh_manager = MeshManager(mesh, dico_mesh)

###### Chargement ######
Umax=1e-3   
dico_chargement = {"type" : "rampe", "pente" : Umax}

dico_problem = {"material" : Acier,
                "mesh_manager" : mesh_manager,
                "boundary_conditions": 
                    [{"component": "U", "tag": 1},
                     {"component": "U", "tag": 2, "value": dico_chargement}
                    ],
                "analysis" : "static",
                "isotherm" : True
                }

pb = CartesianUD(dico_problem)

def query_output(problem, t):
    pass
    
dico_solve = {"Prefix" : "Traction_1D", "output" : {"U" : True}, "csv_output" : {"reaction_force" : {"flag" : 2, "component" : "x"}}}
solve_instance = Solve(pb, dico_solve, compteur=1, npas=10)
solve_instance.query_output = query_output #Attache une fonction d'export appelée à chaque pas de temps
solve_instance.solve()