"""
Created on Thu Mar 10 16:17:24 2022

@author: bouteillerp
Paramètres par défaut
"""

def default_parameters():
    p = {}
    subset_list = ["dynamic", "fem", "damping", "damage", "porosity", "post_processing"]
    for subparset in subset_list:
        subparset_is = eval("default_" + subparset + "_parameters()")
        p.update({subparset:subparset_is})
    return p

def default_fem_degree():
    """
    Degré d'interpolation par défaut du champ de déplacement
    """
    return 1#Remplacer cela par 2 génère un bug si jamais on souhaite 
                # exporter une contrainte définie sur un maillage 1D

def default_T_fem_degree():
    """
    Degré d'interpolation par défaut du champ de température
    """
    return 1

def default_dynamic_parameters():
    """
    Paramètres par du pas de temps CFL et du schema explicit:
    scheme \in \{ "LeapFrop", "Yoshida"\}
    """
    dynamic = {}
    dynamic.update({"scheme" : "LeapFrog"})
    # dynamic.update({"scheme" : "Yoshida"})
    dynamic.update({"CFL_ratio" : 0.2 / default_fem_degree()})
    return dynamic

def default_fem_parameters():
    """
    Degré d'interpolation par défaut des champs.
    """
    fem={}
    fem.update({"u_degree" : default_fem_degree()})
    fem.update({"Tdeg" : default_T_fem_degree()})
    fem.update({"schema" : "default"})
    return fem

def default_damping_parameters():
    """
    Paramètres par défaut de la pseudo-viscosité.
    """
    damp = {}
    damp.update({"damping" : True})
    damp.update({"linear_coeff" : 0.05})
    damp.update({"quad_coeff" : 0.05})
    damp.update({"correction" : True})
    return damp

def default_damage_parameters():
    """
    Paramètres par défaut de l'endommagement.
    """
    dam = {}
    dam.update({"degree" : 1})
    dam.update({"residual_stiffness" : 1e-3})
    dam.update({"default_damage" : 1e-10})
    return dam

def default_porosity_parameters():
    """
    Paramètres par défaut de la porosité.
    """
    poro = {}
    poro.update({"initial_porosity" : 1e-3})
    poro.update({"Initial_pore_distance" : 1e-7})
    return poro

def default_Newton_displacement_solver_parameters():
    solver_u = {}
    solver_u.update({"linear_solver" : "mumps"})
    solver_u.update({"relative_tolerance" : 1e-8})
    solver_u.update({"absolute_tolerance" : 1e-8})
    solver_u.update({"convergence_criterion" : "incremental"})
    solver_u.update({"maximum_iterations" : 2000})
    return solver_u

def default_regularization_linear_solver_parameters():
    linear_solver = {}
    linear_solver.update({"Solver_type" : "default"})
    
def default_damage_solver_type():
    """
    Solveur utilisé pour le modèle de Johnson
    Returns
    -------
    str, nom du solveur parmi :
        - "Kvaerno3" Solveur implicite
        - "Tsit5", "Dopri5", "Euler" Solveurs explicites,
    """
    return "Euler"
    # return "Tsit5"

def default_post_processing_parameters():
    post_processing = {}
    # post_processing.update({"writer" : "xdmf"})
    post_processing.update({"writer" : "VTK"})    
    if post_processing["writer"] == "xdmf":
        post_processing.update({"file_results": "results.xdmf"})
    elif post_processing["writer"] == "VTK":
        post_processing.update({"file_results": "results.pvd"})
    post_processing.update({"file_log": "log.txt"})
    return post_processing

def default_PhaseField_solver_parameters():
    PFSolver = {}
    PFSolver.update({"type" : "TAO"})
    PFSolver.update({"tol" : 1e-6})
    return PFSolver