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
Main solver class for structural dynamics calculations with FEniCSx.

This module provides the central Solve class that orchestrates the time-stepping
solution of coupled problems including mechanics, thermodynamics, damage, and
plasticity using explicit or implicit schemes.

Created on Tue Mar  8 15:51:14 2022
@author: bouteillerp
"""
from .displacement_solve import ExplicitDisplacementSolver
from .energy_solve import ExplicitEnergySolver, DiffusionSolver
from .damping import Damping
from .PlasticSolve.hpp_plastic_solver import HPPPlasticSolver
from .PlasticSolve.finite_strain_plastic_solver import FiniteStrainPlasticSolver
from .PlasticSolve.jax_j2_plastic_solver import JAXJ2PlasticSolver
from .PlasticSolve.jax_gurson_plastic_solver import JAXGursonPlasticSolver
from .PlasticSolve.jax_gurson_plastic_solver_hpp import GTNSimpleJAXSolver

from .multiphase_solve import MultiphaseSolver



from .damage_solve import StaticJohnsonSolve, DynamicJohnsonSolve, InertialJohnsonSolve, PhaseFieldSolve
from .hypoelastic_solve import HypoElasticSolve
from .time_stepping import TimeStepping

from ..utils.default_parameters import default_Newton_displacement_solver_parameters
from ..Export.export_result import ExportResults

from dolfinx.nls.petsc import NewtonSolver
from dolfinx.fem import petsc
from tqdm import tqdm


class Solve:
    """
    Main solver class for time-dependent structural mechanics problems.
    
    This class orchestrates the solution of coupled multi-physics problems including:
    - Structural mechanics (explicit/implicit dynamics, statics)
    - Heat transfer (adiabatic/diffusion)
    - Material nonlinearities (plasticity, damage)
    - Multiphase interactions
    - Artificial viscosity stabilization
    
    The solver supports various analysis types:
    - "explicit_dynamic": Explicit time integration for dynamics
    - "static": Static equilibrium problems
    - "User_driven": User-defined displacement evolution
    
    Parameters
    ----------
    problem : Problem Problem object containing mesh, materials, boundary conditions, etc.
    dictionnaire : dict Configuration dictionary containing solver parameters, export settings,
                        damping parameters, and other simulation options
    **kwargs : dict
        Additional keyword arguments:
        - Tfin : float, optional Final simulation time
        - Scheme : str, optional Time integration scheme
        - compteur : int, optional Output frequency counter (default: 1)
        - Prefix : str, optional Output file prefix
    """
    
    def __init__(self, problem, dictionnaire, **kwargs):
        """
        Initialize the solver with problem setup and configuration.
        
        The initialization process:
        1. Sets up export system
        2. Configures time stepping
        3. Initializes stabilization (damping)
        4. Creates specialized solvers
        5. Sets up optimized solution routines
        """
        self.pb = problem
        self.t = 0        
        self.setup_export(dictionnaire)
        self.set_iterative_solver_parameters()
        self.set_time_step(**kwargs)
        # self.update_Pth()
        self.update_form_with_stabilization(dictionnaire)
        self.set_solver()
        self.pb.set_time_dependant_BCs(self.load_steps)
        self.compteur_output = kwargs.get("compteur", 1)
        self._create_time_and_bcs_update()
        self._create_problem_solve()
        self._create_output()
        
    def update_form_with_stabilization(self, dictionnaire):
        """
        Add artificial viscosity stabilization to the weak form.
        
        For explicit dynamic analysis, artificial viscosity (pseudo-viscosity)
        is added to improve numerical stability, especially in shock-dominated
        problems or when dealing with high-speed deformations.
        
        Parameters
        ----------
        dictionnaire : dict Configuration dictionary containing damping parameters
        """
        if self.pb.analysis == "explicit_dynamic":
            h = self.pb.mesh_manager.calculate_mesh_size(self.pb.mesh, self.pb.dim)
            self.damping = Damping(dictionnaire, self.pb.u, self.pb.u_, self.pb.v, 
                                   self.pb.J_transfo, self.pb.material, self.pb.kinematic, 
                                   self.pb.dx, h, self.pb.multiphase, self.pb.form, self.pb.name)
            self.pb.form -= self.damping.damping_form
        
    def _create_time_and_bcs_update(self):
        """
        Create optimized time and boundary condition update function.
        
        This method pre-computes references to avoid repeated attribute lookups
        during the time-stepping loop.
        The created function updates:
        - Current time
        - Load values
        - Displacement boundary conditions
        - Velocity boundary conditions
        - External loading
        """
        load_steps = self.load_steps
        pb_load = self.pb.load
        pb_update_bcs = self.pb.update_bcs
        bcs_refs = [(bc.constant, bc.value_array, bc.v_constant, bc.speed_array) 
                    for bc in self.pb.bcs.my_constant_list]
        loading_refs = [(load.constant, load.value_array) 
                        for load in self.pb.loading.my_constant_list]
        
        def update_time_and_bcs(j):
            t = load_steps[j]
            pb_load.value = t
            self.t = t
            pb_update_bcs(self.pb, j)
            for constant, values, v_constant, speeds in bcs_refs:
                constant.value = values[j]
                v_constant.value = speeds[j]
            for constant, values in loading_refs:
                constant.value = values[j]
        self.update_time_and_bcs = update_time_and_bcs
    
    def _create_output(self):
        """
        Create optimized output function based on export frequency.
        
        This method eliminates conditional checks during time-stepping by
        creating specialized output functions:
        - If compteur_output == 1: Always export
        - Otherwise: Export every compteur_output steps
        """
        if self.compteur_output == 1:
            def always_export(compteur_output):
                self.in_loop_export(self.t)
            self.output = always_export
        else:
            self.compteur = 0
            
            def output_with_counter(compteur_output):
                if self.compteur == compteur_output:
                    self.in_loop_export(self.t)
                    self.compteur = 0
                self.compteur += 1
            
            self.output = output_with_counter
    
    def _create_problem_solve(self):
        """
        Create optimized problem-solving function based on analysis type and options.
        
        This method eliminates all conditional checks during time-stepping by 
        pre-building a sequence of operations based on the problem configuration.
        The operations are ordered to respect physical coupling:
        1. Mechanical solution (displacement/velocity)
        2. Pressure updates (if tabulated EOS)
        3. Constitutive updates (hypoelastic, damage, plasticity)
        4. Multiphase evolution
        5. Energy/temperature solution
        """
        operations = []
        
        if self.pb.analysis == "explicit_dynamic":
            operations.append(lambda: self.explicit_displacement_solver.u_solve())
            
            if self.pb.is_tabulated:
                operations.append(lambda: self.update_pressure())
            
            if self.pb.is_hypoelastic:
                operations.append(lambda: self.hypo_elastic_solver.solve())
            
            if self.pb.damage_analysis:
                operations.append(lambda: self.damage_solver.explicit_damage())
            
            if self.pb.plastic_analysis:
                operations.append(lambda: self.plastic_solver.solve())
    
        elif self.pb.analysis == "static":
            if self.pb.damage_analysis or self.pb.plastic_analysis:
                operations.append(lambda: self.staggered_solve())
            else:
                operations.append(lambda: self.solver.solve(self.pb.u))
            
            if self.pb.is_tabulated:
                operations.append(lambda: self.update_pressure())
        
        elif self.pb.analysis == "User_driven":
            operations.append(lambda: self.pb.user_defined_displacement(self.t))
            
            if self.pb.is_tabulated:
                operations.append(lambda: self.update_pressure())
            
            if self.pb.is_hypoelastic:
                operations.append(lambda: self.hypo_elastic_solver.solve())
        
        # Common operations for all analysis types
        if self.pb.multiphase_analysis and self.pb.multiphase_evolution:
            operations.append(lambda: self.multiphase_solver.solve())
        
        if not self.pb.iso_T:
            operations.append(lambda: self.energy_solver.energy_solve())
        
        if self.pb.multiphase_analysis and hasattr(self.pb.multiphase, 'explosive') and self.pb.multiphase.explosive:
            operations.append(lambda: self.multiphase_solver.update_c_old())
            
        def problem_solve():
            for op in operations:
                op()
        self.problem_solve = problem_solve

    def setup_export(self, dictionnaire):
        """
        Configure the result export system.
        
        Sets up the export manager for VTK and CSV outputs based on the
        problem's export settings. Performs initial export at t=0.
        
        Parameters
        ----------
        dictionnaire : dict
            Configuration dictionary containing:
            - "Prefix" : str, optional Output file prefix (default: "Problem")
            - "output" : dict, optional VTK output configuration
            - "csv_output" : dict, optional CSV output configuration
        """
        self.export = ExportResults(
            self.pb, 
            dictionnaire.get("Prefix", "Problem"), 
            dictionnaire.get("output", {}),
            dictionnaire.get("csv_output", {})
        )
        self.export.export_results(0)   
        self.export.csv.csv_export(0)
        
    def set_solver(self):
        """
        Initialize all specialized solvers based on problem configuration.
        
        This method creates solver instances for different physical phenomena:
        - Displacement solver (explicit dynamics)
        - Energy solvers (explicit or diffusion)
        - Static displacement solver (Newton method)
        - Hypoelastic constitutive solver
        - Damage evolution solvers
        - Plasticity solvers
        - Multiphase evolution solver
        
        The choice of solvers depends on analysis type and material models.
        """
        # Explicit displacement solver
        if self.pb.analysis == "explicit_dynamic":
            self.explicit_displacement_solver = ExplicitDisplacementSolver(
                self.pb.u, self.pb.v, self.dt, self.pb.m_form, self.pb.form, self.pb.bcs
            )
            
        # Energy/temperature solvers
        if not self.pb.iso_T:
            if self.pb.adiabatic:
                self.energy_solver = ExplicitEnergySolver(
                    self.dt, self.pb.T, self.pb.therm.C_tan, self.pb.pint_vol
                )
            else:
                self.energy_solver = DiffusionSolver(
                    self.dt, self.pb.T, self.pb.T_, self.pb.dT,
                    self.pb.pint_vol, self.pb.therm.C_tan,
                    self.pb.bilinear_flux_form,
                    self.pb.bcs_T, self.pb.kinematic, self.pb.dx
                )
        
        # Static displacement solver    
        elif self.pb.analysis == "static":
            self.set_static_u_solver()
            self.Nitermax = 2000
                
        # Hypoelastic constitutive solver
        if self.pb.is_hypoelastic:
            self.hypo_elastic_solver = HypoElasticSolve(self.pb.material.devia, self.dt)
                
        # Damage evolution solver
        if self.pb.damage_analysis:
            damage_solver_mapper = {
                "PhaseField": PhaseFieldSolve, 
                "StaticJohnson": StaticJohnsonSolve,
                "DynamicJohnson": DynamicJohnsonSolve,
                "InertialJohnson": InertialJohnsonSolve
            }
            damage_class = damage_solver_mapper[self.pb.constitutive.damage_model]
            self.damage_solver = damage_class(
                self.pb.constitutive.damage, 
                self.pb.kinematic, self.pb.mesh.comm, 
                self.pb.dx, self.dt
            )
            
        # Plasticity solver
        if self.pb.plastic_analysis:
            plastic_solver_mapper = {
                "HPP_Plasticity": HPPPlasticSolver,
                "Finite_Plasticity": FiniteStrainPlasticSolver,
                "J2_JAX": JAXJ2PlasticSolver,
                "Gurson_JAX": JAXGursonPlasticSolver,
                "HPP_Gurson": GTNSimpleJAXSolver
            }
            plastic_class = plastic_solver_mapper[self.pb.constitutive.plastic.plastic_model]
            self.plastic_solver = plastic_class(self.pb, self.pb.constitutive.plastic, self.pb.u)
            
        # Multiphase evolution solver
        if self.pb.multiphase_analysis and self.pb.multiphase_evolution:
            self.multiphase_solver = MultiphaseSolver(self.pb.multiphase, self.dt)

    def set_time_step(self, **kwargs):
        """
        Initialize time-stepping parameters and load steps.

        Creates a TimeStepping object that handles time discretization,
        stability conditions, and load step generation based on the
        analysis type and material properties.

        Parameters
        ----------
        **kwargs : dict Time-stepping parameters:
            - Tfin : float, optional Final simulation time
            - Scheme : str, optional Time integration scheme
            - dt : float, optional Fixed time step (if not using adaptive stepping)
        """
        self.time_stepping = TimeStepping(self.pb.analysis, self.pb.mesh, self.pb.material, **kwargs)
        self.load_steps = self.time_stepping.load_steps
        self.Tfin = self.time_stepping.Tfin
        self.dt = self.time_stepping.dt
        
    def update_Pth(self):
        """
        Update thermodynamic pressure for explosive multiphase problems.
        
        For explosive multiphase materials, this method adds the chemical
        energy release rate to the internal volumetric power, affecting
        the thermodynamic state evolution.
        """
        if self.pb.multiphase_analysis and self.pb.multiphase.has_chemical_energy:
            self.pb.pint_vol += self.pb.multiphase.Delta_e_vol_chim / self.dt
         
    def set_static_u_solver(self):
        """
        Configure Newton solver for static displacement problems.
        
        Sets up a Newton-Raphson solver with default parameters for
        solving nonlinear equilibrium equations in static analysis.
        The solver uses PETSc's nonlinear problem interface.
        """
        param = default_Newton_displacement_solver_parameters()
        self.problem_u = petsc.NonlinearProblem(self.pb.form, self.pb.u, self.pb.bcs.bcs)
        self.solver = NewtonSolver(self.pb.mesh.comm, self.problem_u)
        self.solver.atol = param.get("absolute_tolerance")
        self.solver.rtol = param.get("relative_tolerance")
        self.solver.convergence_criterion = param.get("convergence_criterion")
        
    def set_iterative_solver_parameters(self):
        """
        Configure iterative solver parameters.
        
        Placeholder method for setting advanced iterative solver parameters.
        Can be overridden in derived classes for specific solver configurations.
        """
        pass
            
    def update_pressure(self):
        """
        Update pressure fields for tabulated equations of state.
        
        For materials using tabulated EOS, this method updates the pressure
        field based on current thermodynamic state (density, temperature).
        Handles both single-material and multiphase cases.
        """
        if not self.pb.multiphase_analysis:
            self.pb.constitutive.p.x.array[:] = self.pb.material.eos.update_pressure()
        if self.pb.multiphase_analysis:
            for i, mat in enumerate(self.pb.material):
                if mat.eos_type == "Tabulated":
                    self.pb.constitutive.p_list[i].x.array[:] = mat.eos.update_pressure()        

    def staggered_solve(self):
        """
        Staggered solver for coupled damage-plasticity problems in statics.
        
        This method implements a staggered solution approach where displacement,
        damage, and plasticity are solved iteratively until convergence.
        The algorithm:
        1. Solve displacement equilibrium
        2. Update damage state
        3. Update plastic state
        4. Check convergence and iterate if necessary
        
        Maximum iterations are limited by self.Nitermax to prevent infinite loops.
        """
        niter = 0
        evol_dam = self.pb.damage_analysis
        evol_plas = self.pb.plastic_analysis
        
        if evol_dam:   
            self.damage_solver.inf_damage()
            
        while niter < self.Nitermax and (evol_dam or evol_plas):
            # Displacement solve
            self.solver.solve(self.pb.u)
            
            # Damage evolution
            if evol_dam:                          
                evol_dam = self.damage_solver.damage_evolution()
                
            # Plasticity update
            if evol_plas:
                self.plastic_solver.solve()
                evol_plas = False
                
            niter += 1

    def solve(self):
        """
        Execute the main time-stepping solution loop.
        
        This method performs the complete time-dependent simulation:
        1. Time-stepping loop with progress bar
        2. At each time step:
           - Update time and boundary conditions
           - Solve the coupled problem
           - Export results (if needed)
        3. Finalize exports and cleanup
        
        The loop continues until the final time is reached, with progress
        displayed via tqdm progress bar.
        """
        num_time_steps = self.time_stepping.num_time_steps
        
        with tqdm(total=num_time_steps, desc="Progression", unit="pas") as pbar:
            j = 0
            while self.t < self.Tfin:
                self.update_time_and_bcs(j)
                self.problem_solve()
                j += 1
                self.output(self.compteur_output)
                pbar.update(1)
    
        self.export.csv.close_files()
        self.final_output(self.pb)

    def in_loop_export(self, t):
        """
        Perform exports during the solution loop.
        
        This method handles all export operations at a given time step:
        - Custom query outputs (user-defined)
        - VTK file exports
        - CSV data exports
        
        Parameters
        ----------
        t : float Current simulation time
        """
        self.query_output(self.pb, t)
        self.export.export_results(t)
        self.export.csv.csv_export(t)
        
    def query_output(self, problem, t):
        """
        User-defined query outputs during simulation.
        
        This method can be overridden to implement custom output operations
        such as computing specific quantities, logging information, or
        performing specialized post-processing during the simulation.
        
        Parameters
        ----------
        problem : Problem The problem object containing all simulation data
        t : float Current simulation time
        """
        pass
    
    def final_output(self, problem):
        """
        Final output operations after simulation completion.
        
        This method can be overridden to implement final post-processing
        operations such as computing global quantities, generating reports,
        or performing cleanup operations.
        
        Parameters
        ----------
        problem : Problem The problem object containing all simulation data
        """
        pass