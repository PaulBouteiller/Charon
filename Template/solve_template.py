"""
Solve Template - Complete Configuration Options
==============================================
This template shows all available solver configuration options and required parameters
for the Charon finite element framework Solve class.

The Solve class orchestrates time-stepping solution of coupled multi-physics problems
including mechanics, thermodynamics, damage, plasticity, and multiphase interactions.

"""

from Charon import Solve, Material, MeshManager, CartesianUD, PlaneStrain, Axisymmetric, Tridimensional
import numpy as np

# =============================================================================
# SOLVE CLASS INITIALIZATION
# =============================================================================

"""
Basic syntax:
solve_instance = Solve(problem, dictionnaire_solve, **kwargs)

Where:
- problem: Problem instance (CartesianUD, PlaneStrain, Axisymmetric, Tridimensional)
- dictionnaire_solve: Dictionary with solver configuration
- **kwargs: Additional time-stepping and output parameters
"""

# =============================================================================
# DICTIONNAIRE_SOLVE OPTIONS
# =============================================================================

# -----------------------------------------------------------------------------
# 1. OUTPUT AND EXPORT CONFIGURATION
# -----------------------------------------------------------------------------

dictionnaire_solve_full_output = {
    # Output file prefix
    "Prefix": "MySimulation",  # String: Base name for output files
    
    # VTK output configuration (3D visualization)
    "output": {
        "U": True,              # Export displacement field
        "V": True,              # Export velocity field (dynamics only)
        "A": True,              # Export acceleration field (dynamics only)
        "sig": True,            # Export stress tensor
        "eps": True,            # Export strain tensor
        "T": True,              # Export temperature field
        "p": True,              # Export pressure field
        "rho": True,            # Export density field
        "J": True,              # Export Jacobian (volume change)
        "s": True,              # Export deviatoric stress
        "eps_p": True,          # Export plastic strain (plasticity only)
        "p_plastic": True,      # Export cumulative plastic strain (plasticity only)
        "d": True,              # Export damage variable (damage only)
        "f": True,              # Export porosity (Johnson damage only)
        "c": True,              # Export concentrations (multiphase only)
        "Be_bar": True,         # Export elastic left Cauchy-Green tensor (finite plasticity)
        "eps_eq": True,         # Export equivalent strain
        "sig_eq": True,         # Export equivalent stress
        "all": True             # Export all available fields
    },
    
    # CSV output configuration (time series data)
    "csv_output": {
        # Field exports at all points
        "U": True,              # Displacement field
        "V": True,              # Velocity field
        "sig": True,            # Stress tensor
        "eps": True,            # Strain tensor
        "T": True,              # Temperature field
        "p": True,              # Pressure field
        "rho": True,            # Density field
        "s": True,              # Deviatoric stress
        "eps_p": True,          # Plastic strain
        "d": True,              # Damage variable
        "f": True,              # Porosity
        "c": True,              # Concentrations
        "deviateur": True,      # Deviatoric stress (alias)
        
        # Boundary-specific exports
        "U": ["Boundary", 1],               # Displacement on boundary tag 1
        "U": ["Boundary", 2],               # Displacement on boundary tag 2
        "V": ["Boundary", 1],               # Velocity on boundary tag 1
        "T": ["Boundary", 1],               # Temperature on boundary tag 1
        
        # Reaction force exports
        "reaction_force": {
            "flag": 1,                      # Boundary tag
            "component": "x"                # Component: "x", "y", "z", "r"
        },
        "reaction_force": {
            "flag": 2,
            "component": "y"
        }
    }
}

# -----------------------------------------------------------------------------
# 2. INITIAL CONDITIONS
# -----------------------------------------------------------------------------

from ufl import conditional, SpatialCoordinate, lt, gt, And
from dolfinx.fem import Expression

# Temperature initial conditions
T_initial = 500.0  # Uniform temperature
# OR spatial distribution
x = SpatialCoordinate(mesh)  # mesh should be defined in your problem
T_spatial = conditional(lt(x[0], 0.5), 800.0, 300.0)  # Piecewise temperature

dictionnaire_solve_initial = {
    "initial_conditions": {
        "T": T_initial,         # Uniform temperature [K]
        # OR
        "T": T_spatial          # Spatial temperature distribution (UFL expression)
    }
}

# -----------------------------------------------------------------------------
# 3. DAMPING/ARTIFICIAL VISCOSITY CONFIGURATION
# -----------------------------------------------------------------------------

dictionnaire_solve_damping = {
    "damping": {
        "linear_coeff": 0.1,        # Linear viscosity coefficient [-]
        "quad_coeff": 0.1,          # Quadratic viscosity coefficient [-]
        "correction": True          # Apply Jacobian correction [bool]
    }
}

# -----------------------------------------------------------------------------
# 4. SOLVER-SPECIFIC PARAMETERS
# -----------------------------------------------------------------------------

# Newton solver parameters (for static analysis)
dictionnaire_solve_newton = {
    "Newton_solver": {
        "absolute_tolerance": 1e-8,     # Absolute convergence tolerance
        "relative_tolerance": 1e-6,     # Relative convergence tolerance
        "maximum_iterations": 50,       # Maximum Newton iterations
        "convergence_criterion": "residual"  # "residual" or "incremental"
    }
}

# Linear solver parameters
dictionnaire_solve_linear = {
    "linear_solver": {
        "solver_type": "hybrid",        # "default", "hybrid"
        "direct_solver": {
            "solver": "mumps",          # "mumps", "superlu_dist", "petsc"
            "type": "lu",               # "lu", "cholesky"
            "blr": True                 # Block Low-Rank compression
        },
        "iterative_solver": {
            "solver": "cg",             # "cg", "gmres", "bicgstab"
            "maximum_iterations": 100,   # Maximum iterations
            "preconditioner": "ilu"     # "ilu", "sor", "jacobi"
        },
        "hybrid_parameters": {
            "iteration_switch": 10,     # Switch threshold
            "user_switch": True         # User-controlled switching
        }
    }
}

# Time integration parameters (for dynamics)
dictionnaire_solve_dynamics = {
    "dynamic_solver": {
        "order": 2,                     # Symplectic integrator order (1-5)
        "CFL_ratio": 0.8               # CFL stability factor
    }
}

# Energy solver parameters
dictionnaire_solve_energy = {
    "energy_solver": {
        "order": 4,                     # Runge-Kutta order for explicit energy
        "implicit_diffusion": True      # Use implicit scheme for diffusion
    }
}

# Damage solver parameters
dictionnaire_solve_damage = {
    "damage_solver": {
        "type": "Tsit5",               # ODE solver: "Tsit5", "Kvaerno3", "Dopri5", "Euler"
        "PhaseField_solver": {
            "type": "TAO",             # "TAO" or "SNES"
            "tol": 1e-8                # Convergence tolerance
        }
    }
}

# =============================================================================
# TIME-STEPPING KWARGS OPTIONS
# =============================================================================

# -----------------------------------------------------------------------------
# 1. TIME CONTROL PARAMETERS
# -----------------------------------------------------------------------------

time_kwargs_basic = {
    "TFin": 1.0,                    # Final simulation time [s]
    "TInit": 0.0,                   # Initial time [s] (default: 0.0)
    "dt": 1e-6,                     # Fixed time step [s]
    "time_step_scheme": "fixed"     # "fixed", "default", "adaptative"
}

time_kwargs_fixed = {
    "TFin": 1.0,
    "scheme": "fixed",              # Fixed time stepping
    "dt": 1e-6                      # Time step size
}

time_kwargs_adaptive = {
    "TFin": 1.0,
    "scheme": "adaptative",         # Adaptive time stepping (experimental)
    "dt_min": 1e-8,                # Minimum time step
    "CFL_ratio": 0.8               # CFL stability factor
}

# For static analysis (pseudo-time)
time_kwargs_static = {
    "TFin": 1.0,                   # Load factor final value
    "npas": 100,                   # Number of load steps
    "scheme": "default"            # Default static stepping
}

# Piecewise linear loading
time_kwargs_piecewise = {
    "TFin": 1.0,
    "scheme": "piecewise_linear",
    "discretisation_list": [
        [0.0, 0.5, 1.0],           # Time intervals
        [50, 50]                   # Steps in each interval
    ]
}

# -----------------------------------------------------------------------------
# 2. OUTPUT CONTROL PARAMETERS
# -----------------------------------------------------------------------------

output_kwargs = {
    "compteur": 10,                 # Output every N time steps (default: 1)
    "Prefix": "MySimulation"        # Output file prefix (overrides dictionnaire)
}

# =============================================================================
# COMPLETE SOLVER EXAMPLES BY ANALYSIS TYPE
# =============================================================================

# -----------------------------------------------------------------------------
# EXAMPLE 1: EXPLICIT DYNAMICS SIMULATION
# -----------------------------------------------------------------------------

def example_explicit_dynamics():
    """Complete example for explicit dynamics with plasticity and damage"""
    
    # Assume material, mesh_manager, and problem are already defined
    # pb = CartesianUD(material, dictionnaire_problem)
    
    dictionnaire_solve = {
        "Prefix": "ExplicitDynamics",
        "output": {
            "U": True,
            "V": True,
            "sig": True,
            "eps_p": True,
            "d": True
        },
        "csv_output": {
            "U": ["Boundary", 2],
            "reaction_force": {"flag": 1, "component": "x"}
        },
        "damping": {
            "linear_coeff": 0.05,
            "quad_coeff": 0.1,
            "correction": True
        }
    }
    
    kwargs = {
        "TFin": 1e-3,              # 1 ms simulation
        "scheme": "fixed",
        "dt": 1e-7,               # 0.1 μs time step
        "compteur": 100           # Output every 100 steps
    }
    
    # solve_instance = Solve(pb, dictionnaire_solve, **kwargs)
    # solve_instance.solve()

# -----------------------------------------------------------------------------
# EXAMPLE 2: STATIC NONLINEAR ANALYSIS
# -----------------------------------------------------------------------------

def example_static_nonlinear():
    """Complete example for static analysis with plasticity"""
    
    dictionnaire_solve = {
        "Prefix": "StaticNonlinear",
        "output": {
            "U": True,
            "sig": True,
            "eps_p": True
        },
        "csv_output": {
            "reaction_force": {"flag": 2, "component": "y"},
            "U": ["Boundary", 2]
        },
        "Newton_solver": {
            "absolute_tolerance": 1e-9,
            "relative_tolerance": 1e-7,
            "maximum_iterations": 100
        }
    }
    
    kwargs = {
        "TFin": 1.0,              # Load factor
        "npas": 200,              # 200 load steps
        "scheme": "default",
        "compteur": 1             # Output every step
    }
    
    # solve_instance = Solve(pb, dictionnaire_solve, **kwargs)
    # solve_instance.solve()

# -----------------------------------------------------------------------------
# EXAMPLE 3: THERMAL DIFFUSION PROBLEM
# -----------------------------------------------------------------------------

def example_thermal_diffusion():
    """Complete example for pure thermal diffusion"""
    
    from ufl import conditional, SpatialCoordinate, lt
    
    # Initial temperature distribution
    # x = SpatialCoordinate(pb.mesh)
    # T_init = conditional(lt(x[0], 0.1), 800.0, 300.0)
    
    dictionnaire_solve = {
        "Prefix": "ThermalDiffusion",
        "output": {"T": True},
        "csv_output": {"T": True},
        "initial_conditions": {
            # "T": T_init
        },
        "energy_solver": {
            "implicit_diffusion": True
        }
    }
    
    kwargs = {
        "TFin": 1e-3,             # 1 ms
        "scheme": "fixed",
        "dt": 1e-6,              # 1 μs
        "compteur": 50           # Output every 50 steps
    }
    
    # solve_instance = Solve(pb, dictionnaire_solve, **kwargs)
    # solve_instance.solve()

# -----------------------------------------------------------------------------
# EXAMPLE 4: MULTIPHASE REACTIVE SIMULATION
# -----------------------------------------------------------------------------

def example_multiphase_reactive():
    """Complete example for multiphase reactive materials"""
    
    dictionnaire_solve = {
        "Prefix": "MultiphaseReactive",
        "output": {
            "c": True,              # Concentrations
            "T": True,              # Temperature
            "p": True,              # Pressure
            "rho": True             # Density
        },
        "csv_output": {
            "c": True,              # All concentrations
            "T": True
        }
    }
    
    kwargs = {
        "TFin": 1e-4,             # 100 μs
        "scheme": "fixed",
        "dt": 1e-7,              # 0.1 μs
        "compteur": 100
    }
    
    # solve_instance = Solve(pb, dictionnaire_solve, **kwargs)
    # solve_instance.solve()

# -----------------------------------------------------------------------------
# EXAMPLE 5: PHASE-FIELD FRACTURE
# -----------------------------------------------------------------------------

def example_phase_field_fracture():
    """Complete example for phase-field fracture simulation"""
    
    dictionnaire_solve = {
        "Prefix": "PhaseFieldFracture",
        "output": {
            "U": True,
            "d": True,              # Damage field
            "sig": True
        },
        "csv_output": {
            "d": True,
            "U": ["Boundary", 2],
            "reaction_force": {"flag": 2, "component": "y"}
        },
        "damage_solver": {
            "PhaseField_solver": {
                "type": "TAO",
                "tol": 1e-9
            }
        },
        "Newton_solver": {
            "absolute_tolerance": 1e-8,
            "maximum_iterations": 200
        }
    }
    
    kwargs = {
        "TFin": 1.0,              # Load factor
        "npas": 500,              # Many small steps for stability
        "compteur": 5
    }
    
    # solve_instance = Solve(pb, dictionnaire_solve, **kwargs)
    # solve_instance.solve()

# -----------------------------------------------------------------------------
# EXAMPLE 6: USER-DRIVEN DISPLACEMENT
# -----------------------------------------------------------------------------

def example_user_driven():
    """Complete example for user-defined displacement evolution"""
    
    dictionnaire_solve = {
        "Prefix": "UserDriven",
        "output": {
            "U": True,
            "sig": True,
            "p": True
        },
        "csv_output": {
            "sig": True,
            "p": True
        }
    }
    
    kwargs = {
        "TFin": 1.0,
        "npas": 100,
        "compteur": 1
    }
    
    # Define user displacement function in problem class:
    # def user_defined_displacement(self, t):
    #     # Custom displacement evolution
    #     pass
    
    # solve_instance = Solve(pb, dictionnaire_solve, **kwargs)
    # solve_instance.solve()

# =============================================================================
# ADVANCED CUSTOMIZATION OPTIONS
# =============================================================================

# -----------------------------------------------------------------------------
# CUSTOM OUTPUT FUNCTIONS
# -----------------------------------------------------------------------------

def setup_custom_output():
    """Example of custom output function setup"""
    
    dictionnaire_solve = {
        "Prefix": "CustomOutput",
        "output": {"U": True, "sig": True}
    }
    
    # solve_instance = Solve(pb, dictionnaire_solve, TFin=1.0, npas=100)
    
    # Custom query output function (called at each time step)
    def custom_query_output(problem, t):
        """Custom output function called during time loop"""
        # Example: Compute and store custom quantities
        max_displacement = np.max(np.abs(problem.u.x.array))
        max_stress = np.max(np.abs(problem.constitutive.sig.x.array))
        
        # Store in problem attributes
        if not hasattr(problem, 'custom_data'):
            problem.custom_data = {'t': [], 'max_u': [], 'max_sig': []}
        
        problem.custom_data['t'].append(t)
        problem.custom_data['max_u'].append(max_displacement)
        problem.custom_data['max_sig'].append(max_stress)
        
        # Optional: Print progress
        if t % 0.1 < 1e-6:  # Every 0.1 time units
            print(f"Time: {t:.3f}, Max displacement: {max_displacement:.2e}")
    
    # Attach custom function
    # solve_instance.query_output = custom_query_output
    
    # Custom final output function (called at end)
    def custom_final_output(problem):
        """Custom final processing function"""
        import matplotlib.pyplot as plt
        
        # Plot custom data
        if hasattr(problem, 'custom_data'):
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(problem.custom_data['t'], problem.custom_data['max_u'])
            plt.xlabel('Time')
            plt.ylabel('Max Displacement')
            plt.title('Maximum Displacement Evolution')
            
            plt.subplot(1, 2, 2)
            plt.plot(problem.custom_data['t'], problem.custom_data['max_sig'])
            plt.xlabel('Time')
            plt.ylabel('Max Stress')
            plt.title('Maximum Stress Evolution')
            
            plt.tight_layout()
            plt.savefig('custom_analysis.png')
            plt.show()
            
            # Save data to CSV
            import pandas as pd
            df = pd.DataFrame(problem.custom_data)
            df.to_csv('custom_analysis.csv', index=False)
    
    # Attach custom functions
    # solve_instance.query_output = custom_query_output
    # solve_instance.final_output = custom_final_output

# =============================================================================
# COMPLETE CSV OUTPUT OPTIONS
# =============================================================================

# The CSV export system supports various field types and boundary specifications
csv_output_complete = {
    # -----------------------------------------------------------------------------
    # DISPLACEMENT AND VELOCITY FIELDS
    # -----------------------------------------------------------------------------
    "U": True,                          # All displacement points
    "U": ["Boundary", 1],               # Displacement on boundary tag 1
    "U": ["Boundary", 2],               # Displacement on boundary tag 2
    "v": True,                          # All velocity points (dynamics only)
    "v": ["Boundary", 1],               # Velocity on specific boundary
    
    # -----------------------------------------------------------------------------
    # STRESS AND STRAIN FIELDS
    # -----------------------------------------------------------------------------
    "sig": True,                        # Full stress tensor at all points
    "s": True,                          # Deviatoric stress at all points
    "eps": True,                        # Strain tensor (if available)
    "eps_p": True,                      # Plastic strain (plasticity only)
    
    # Components exported depend on problem type:
    # - CartesianUD: σ_xx (1 component)
    # - CylindricalUD: σ_rr, σ_θθ (2 components)  
    # - SphericalUD: σ_rr, σ_θθ, σ_φφ (3 components)
    # - PlaneStrain: σ_xx, σ_yy, σ_xy (3 components)
    # - Axisymmetric: σ_rr, σ_θθ, σ_zz, σ_rz (4 components)
    # - Tridimensional: Full 3×3 tensor (9 components)
    
    # -----------------------------------------------------------------------------
    # THERMODYNAMIC FIELDS
    # -----------------------------------------------------------------------------
    "T": True,                          # Temperature field
    "T": ["Boundary", 1],               # Temperature on specific boundary
    "p": True,                          # Pressure field
    "rho": True,                        # Density field
    "J": True,                          # Jacobian (volume change)
    
    # -----------------------------------------------------------------------------
    # DAMAGE AND FAILURE FIELDS
    # -----------------------------------------------------------------------------
    "d": True,                          # Damage variable (damage analysis only)
    "f": True,                          # Porosity (Johnson damage only)
    
    # -----------------------------------------------------------------------------
    # MULTIPHASE FIELDS
    # -----------------------------------------------------------------------------
    "c": True,                          # All phase concentrations
    # Exports concentrations for each material phase:
    # - Creates separate CSV files: Concentration0.csv, Concentration1.csv, etc.
    
    # -----------------------------------------------------------------------------
    # REACTION FORCES
    # -----------------------------------------------------------------------------
    "reaction_force": {
        "flag": 1,                      # Boundary tag
        "component": "x"                # Force component: "x", "y", "z", "r"
    },
    # Additional reaction forces on different boundaries
    "reaction_force": {
        "flag": 2,
        "component": "y"
    },
    
    # -----------------------------------------------------------------------------
    # SPECIALIZED EXPORTS
    # -----------------------------------------------------------------------------
    "FreeSurf_1D": ["Boundary", 2],    # Free surface tracking (1D problems)
    # For explosive/impact problems with moving boundaries
}

# =============================================================================
# VTK/XDMF OUTPUT OPTIONS (3D Visualization)
# =============================================================================

# Standard VTK output (most common)
output_vtk_standard = {
    "U": True,                          # Displacement field
    "v": True,                          # Velocity field (dynamics)
    "sig": True,                        # Stress tensor
    "T": True,                          # Temperature field
    "p": True,                          # Pressure field
    "rho": True,                        # Density field
}

# Complete VTK output (all available fields)
output_vtk_complete = {
    # Kinematic fields
    "U": True,                          # Displacement
    "v": True,                          # Velocity (dynamics only)
    "A": True,                          # Acceleration (dynamics only)
    
    # Stress/strain fields
    "sig": True,                        # Stress tensor
    "eps": True,                        # Strain tensor
    "s": True,                          # Deviatoric stress
    "D": True,                          # Rate of deformation tensor
    "eps_eq": True,                     # Equivalent strain
    "sig_eq": True,                     # Equivalent stress
    
    # Plasticity fields
    "eps_p": True,                      # Plastic strain (plasticity only)
    "p_plastic": True,                  # Cumulative plastic strain (plasticity only)
    "Be_bar": True,                     # Elastic left Cauchy-Green (finite plasticity)
    
    # Damage fields
    "d": True,                          # Damage variable (damage only)
    "f": True,                          # Porosity (Johnson damage only)
    
    # Thermodynamic fields
    "T": True,                          # Temperature
    "p": True,                          # Pressure
    "rho": True,                        # Density
    "J": True,                          # Jacobian (volume change)
    
    # Multiphase fields
    "c": True,                          # Phase concentrations (multiphase only)
    
    # Special option
    "all": True                         # Export all available fields
}

# Minimal output (performance-optimized)
output_vtk_minimal = {
    "U": True,                          # Only displacement
    "sig": True                         # Only stress
}

# =============================================================================
# EXPORT FILE FORMAT CONFIGURATION
# =============================================================================

# Default export parameters (can be customized)
export_parameters_custom = {
    "post_processing": {
        "writer": "VTK",                # "VTK" or "xdmf" 
        "file_results": "results.vtk"  # Output filename
    }
}

# For large simulations, XDMF format may be preferred
export_parameters_xdmf = {
    "post_processing": {
        "writer": "xdmf",
        "file_results": "results.xdmf"
    }
}

# =============================================================================
# PRACTICAL EXAMPLE COMBINATIONS
# =============================================================================

# -----------------------------------------------------------------------------
# EXAMPLE: HIGH-FREQUENCY OUTPUT (Every time step)
# -----------------------------------------------------------------------------
def example_high_frequency_output():
    """Example with detailed output every time step"""
    
    dictionnaire_solve = {
        "Prefix": "HighFrequency",
        "output": {
            "U": True,
            "v": True,
            "sig": True,
            "T": True
        },
        "csv_output": {
            "U": ["Boundary", 2],        # Track specific boundary
            "reaction_force": {"flag": 1, "component": "x"}
        }
    }
    
    kwargs = {
        "TFin": 1e-3,
        "dt": 1e-6,
        "compteur": 1                   # Output every step
    }
    
    # solve_instance = Solve(pb, dictionnaire_solve, **kwargs)

# -----------------------------------------------------------------------------
# EXAMPLE: LOW-FREQUENCY OUTPUT (Performance optimized)
# -----------------------------------------------------------------------------
def example_low_frequency_output():
    """Example with minimal output for performance"""
    
    dictionnaire_solve = {
        "Prefix": "LowFrequency", 
        "output": {"U": True},          # Only displacement for visualization
        "csv_output": {
            "reaction_force": {"flag": 2, "component": "y"}  # Only key result
        }
    }
    
    kwargs = {
        "TFin": 1.0,
        "npas": 1000,
        "compteur": 50                  # Output every 50 steps
    }

# -----------------------------------------------------------------------------
# EXAMPLE: MULTIPHYSICS OUTPUT
# -----------------------------------------------------------------------------
def example_multiphysics_output():
    """Example for coupled multiphysics problems"""
    
    dictionnaire_solve = {
        "Prefix": "Multiphysics",
        "output": {
            "U": True,                  # Displacement
            "T": True,                  # Temperature
            "c": True,                  # Concentrations (multiphase)
            "d": True,                  # Damage
            "eps_p": True               # Plastic strain
        },
        "csv_output": {
            "T": True,                  # Full temperature field
            "c": True,                  # All concentrations
            "U": ["Boundary", 1],       # Boundary displacement
            "reaction_force": {"flag": 1, "component": "x"}
        }
    }

# -----------------------------------------------------------------------------
# EXAMPLE: DEBUG/VALIDATION OUTPUT  
# -----------------------------------------------------------------------------
def example_debug_output():
    """Example for debugging and validation"""
    
    dictionnaire_solve = {
        "Prefix": "Debug",
        "output": {
            "all": True                 # Export everything available
        },
        "csv_output": {
            "U": True,                  # Full displacement field
            "sig": True,                # Full stress field
            "T": True,                  # Full temperature field
            "p": True,                  # Full pressure field
            "rho": True,                # Full density field
            "J": True                   # Jacobian field
        }
    }
    
    kwargs = {
        "compteur": 1                   # Output every step for analysis
    }

# =============================================================================
# SOLVER WORKFLOW SUMMARY
# =============================================================================

"""
COMPLETE SOLVER WORKFLOW:

1. PROBLEM SETUP:
   - Define Material(s) with constitutive models
   - Create mesh and MeshManager with boundary conditions
   - Initialize Problem class (CartesianUD, PlaneStrain, etc.)
   - Set boundary conditions, loading, and analysis type

2. SOLVER CONFIGURATION:
   - Create dictionnaire_solve with output options
   - Set time-stepping parameters in kwargs
   - Configure any custom solver parameters

3. SOLVER EXECUTION:
   solve_instance = Solve(problem, dictionnaire_solve, **kwargs)
   
   # Optional: Attach custom functions
   solve_instance.query_output = custom_query_function
   solve_instance.final_output = custom_final_function
   
   # Run simulation
   solve_instance.solve()

4. OUTPUT FILES GENERATED:
   - VTK/XDMF files: [Prefix]-results/results.vtk (3D visualization)
   - CSV files: [Prefix]-results/[field_name].csv (time series data)
   - Export times: [Prefix]-results/export_times.csv (time stamps)
   - Reaction forces: [Prefix]-results/reaction_force.csv (if requested)

PERFORMANCE CONSIDERATIONS:

- Use compteur > 1 for large simulations to reduce output frequency
- Limit CSV exports to essential data for memory efficiency
- Use VTK format for standard visualization, XDMF for large datasets
- Consider "minimal" output options for production runs
- Use "debug" output options only for validation and troubleshooting

UNITS AND CONVENTIONS:
- All physical quantities in SI units (m, kg, s, Pa, K)
- Time exports in simulation time units
- Spatial coordinates match problem coordinate system
- Stress/strain tensors in Voigt notation for 2D/3D problems
- Temperature in Kelvin, pressure in Pascal

TROUBLESHOOTING:
- Check boundary tag numbers match mesh definition
- Verify field availability for current analysis type
- Monitor CSV file sizes for large simulations
- Use MPI-compatible export options for parallel runs
"""