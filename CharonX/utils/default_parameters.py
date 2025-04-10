

# Copyright 2025 CEA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Default Parameters Module
=======================

This module provides default parameters for various aspects of the simulation,
including time integration, finite elements, damping, and post-processing.

Key components:
- General parameter collections
- Time integration parameters (dynamic analysis)
- Finite element discretization parameters
- Damping and stabilization parameters
- Damage and porosity modeling parameters
- Solver configuration parameters
- Post-processing and output parameters

These parameters establish reasonable defaults for simulations and can be
modified by the user to customize the behavior of specific simulations.
"""
def default_parameters():
    """
    Create a complete dictionary of default parameters.
    
    Aggregates parameters from various subsystems into a unified dictionary
    with nested structure.
    
    Returns
    -------
    dict
        Complete dictionary of default parameters for all subsystems
    """
    p = {}
    subset_list = ["dynamic", "fem", "damping", "damage", "porosity", "post_processing"]
    for subparset in subset_list:
        subparset_is = eval("default_" + subparset + "_parameters()")
        p.update({subparset:subparset_is})
    return p

def default_fem_degree():
    """
    Get the default interpolation degree for the displacement field.
    
    Returns
    -------
    int Default polynomial degree (1)
        
    Notes
    -----
    Using degree 2 may generate a bug when exporting a stress field defined on a 1D mesh.
    """
    return 1

def default_T_fem_degree():
    """
    Get the default interpolation degree for the temperature field.
    
    Returns
    -------
    int Default polynomial degree for temperature (1)
    """
    return 1

def default_dynamic_parameters():
    """
    Get default parameters for dynamic simulations.
    
    Returns a dictionary with parameters for time integration and CFL criteria.
    
    Available time integration schemes:
    
    - "LeapFrog" (default): 2nd order symplectic scheme, used for its efficiency
      and stability. Good compromise between accuracy and computational cost.
    
    - "Yoshida": 4th order symplectic scheme with 3 stages. Offers excellent 
      energy conservation for long simulations. More expensive but more 
      accurate than LeapFrog.
    
    Returns
    -------
    dict
        Dictionary containing default dynamic parameters
    """
    dynamic = {}
    
    # Default time integration scheme
    dynamic.update({"order": 1})
    # Safety factor for CFL criterion
    # CFL ratio is inversely proportional to the polynomial degree
    dynamic.update({"CFL_ratio": 0.2 / default_fem_degree()})
    
    return dynamic

def default_fem_parameters():
    """
    Get default parameters for finite element discretization.
    
    Sets up default interpolation degrees for fields and quadrature scheme.
    
    Returns
    -------
    dict Dictionary containing FEM parameters
    """
    fem = {}
    fem.update({"u_degree": default_fem_degree()})
    fem.update({"Tdeg": default_T_fem_degree()})
    fem.update({"schema": "default"})
    return fem

def default_damping_parameters():
    """
    Get default parameters for artificial viscosity (pseudo-viscosity).
    
    Sets up default coefficients for numerical damping to stabilize simulations.
    
    Returns
    -------
    dict
        Dictionary containing damping parameters:
        - damping: Whether to use artificial viscosity (True)
        - linear_coeff: Linear coefficient (0.1)
        - quad_coeff: Quadratic coefficient (1)
        - correction: Whether to apply correction (True)
    """
    damp = {}
    damp.update({"damping": True})
    damp.update({"linear_coeff": 0.1})
    damp.update({"quad_coeff": 1})
    damp.update({"correction": True})
    return damp

def default_damage_parameters():
    """
    Get default parameters for damage modeling.
    
    Sets up default parameters for damage evolution and regularization.
    
    Returns
    -------
    dict
        Dictionary containing damage model parameters:
        - degree: Polynomial degree for damage field (1)
        - residual_stiffness: Residual stiffness factor (1e-3)
        - default_damage: Initial damage level (1e-10)
    """
    dam = {}
    dam.update({"degree": 1})
    dam.update({"residual_stiffness": 1e-3})
    dam.update({"default_damage": 1e-10})
    return dam

def default_porosity_parameters():
    """
    Get default parameters for porosity modeling.
    
    Sets up default parameters for initial porosity and pore distribution.
    
    Returns
    -------
    dict
        Dictionary containing porosity model parameters:
        - initial_porosity: Initial porosity level (1e-3)
        - Initial_pore_distance: Initial distance between pores (1e-7)
    """
    poro = {}
    poro.update({"initial_porosity": 1e-3})
    poro.update({"Initial_pore_distance": 1e-7})
    return poro

def default_Newton_displacement_solver_parameters():
    """
    Get default parameters for Newton solver for displacement problems.
    
    Sets up default solver type, tolerances, and iteration limits.
    
    Returns
    -------
    dict
        Dictionary containing Newton solver parameters:
        - linear_solver: Linear solver type ("mumps")
        - relative_tolerance: Relative convergence criterion (1e-8)
        - absolute_tolerance: Absolute convergence criterion (1e-8)
        - convergence_criterion: Type of convergence criterion ("incremental")
        - maximum_iterations: Maximum number of iterations (2000)
    """
    solver_u = {}
    solver_u.update({"linear_solver": "mumps"})
    solver_u.update({"relative_tolerance": 1e-8})
    solver_u.update({"absolute_tolerance": 1e-8})
    solver_u.update({"convergence_criterion": "incremental"})
    solver_u.update({"maximum_iterations": 2000})
    return solver_u

def default_regularization_linear_solver_parameters():
    """
    Get default parameters for regularization linear solver.
    
    Returns
    -------
    dict
        Dictionary containing linear solver parameters
    """
    linear_solver = {}
    linear_solver.update({"Solver_type": "default"})
    return linear_solver
    
def default_damage_solver_type():
    """
    Get the default solver type for damage models.
    
    Returns
    -------
    str Name of the solver: "Euler" (explicit solver)
        
    Notes
    -----
    Available solvers include:
    - "Kvaerno3": Implicit solver
    - "Tsit5", "Dopri5", "Euler": Explicit solvers
    """
    return "Euler"

def default_energy_solver_order():
    """
    Get the default order for energy solvers.
    
    Returns
    -------
    int Default solver order
    """
    return 1

def default_post_processing_parameters():
    """
    Get default parameters for post-processing and output.
    
    Sets up default file formats and output configuration.
    
    Returns
    -------
    dict
        Dictionary containing post-processing parameters:
        - writer: Output format ("VTK" or "xdmf")
        - file_results: Path to results file
        - file_log: Path to log file
    """
    post_processing = {}
    post_processing.update({"writer": "VTK"})    
    if post_processing["writer"] == "xdmf":
        post_processing.update({"file_results": "results.xdmf"})
    elif post_processing["writer"] == "VTK":
        post_processing.update({"file_results": "results.pvd"})
    post_processing.update({"file_log": "log.txt"})
    return post_processing

def default_PhaseField_solver_parameters():
    """
    Get default parameters for phase field solvers.
    
    Sets up default solver type and tolerance for phase field problems.
    
    Returns
    -------
    dict
        Dictionary containing phase field solver parameters:
        - type: Solver type ("TAO")
        - tol: Convergence tolerance (1e-6)
    """
    PFSolver = {}
    PFSolver.update({"type": "TAO"})
    PFSolver.update({"tol": 1e-6})
    return PFSolver