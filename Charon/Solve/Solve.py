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
Created on Tue Mar  8 15:51:14 2022

@author: bouteillerp
"""
from .displacement_solve import ExplicitDisplacementSolver
from .energy_solve import ExplicitEnergySolver, DiffusionSolver
from .plastic_solve import PlasticSolve
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
    def __init__(self, problem, dictionnaire, **kwargs):
        self.pb = problem
        self.t = 0        
        self.setup_export(dictionnaire)
        self.set_iterative_solver_parameters()
        self.set_time_step(**kwargs)
        self.update_Pth()
        self.set_solver()
        self.pb.set_time_dependant_BCs(self.load_steps)
        self.compteur_output = kwargs.get("compteur", 1)
        self._create_time_and_bcs_update()
        self._create_problem_solve()
        self._create_output()
        
    def _create_time_and_bcs_update(self):
        """Fusionne update_time et update_bcs en une seule fonction optimisée"""
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
            pb_update_bcs(j)
            for constant, values, v_constant, speeds in bcs_refs:
                constant.value = values[j]
                v_constant.value = speeds[j]
            for constant, values in loading_refs:
                constant.value = values[j]
        self.update_time_and_bcs = update_time_and_bcs
    
    def _create_output(self):
        """Crée une version optimisée de output() sans test conditionnel"""
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
        Crée une version optimisée de problem_solve en fonction de l'analyse
        et des options activées, éliminant tous les tests conditionnels.
        """
        # Liste des opérations à effectuer
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
        
        # Opérations communes à tous les types d'analyse
        if self.pb.multiphase_analysis and hasattr(self.pb.multiphase, 'evol') and self.pb.multiphase.evol:
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
        problem's export settings.
        
        Parameters
        ----------
        **kwargs : dict Additional parameters, potentially including 'Prefix'
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
        Initialise les solveurs successivement appelés.
        """
        if self.pb.analysis == "explicit_dynamic":
            self.explicit_displacement_solver = ExplicitDisplacementSolver(self.pb.u, self.pb.v, self.dt, \
                                                                            self.pb.m_form, self.pb.form, self.pb.bcs)
        if not self.pb.iso_T:
            if self.pb.adiabatic:
                self.energy_solver = ExplicitEnergySolver(self.dt, self.pb.T, self.pb.therm.C_tan, self.pb.pint_vol)
            else:
                self.energy_solver = DiffusionSolver(self.dt, self.pb.T, self.pb.T_, self.pb.dT, \
                                                     self.pb.pint_vol, self.pb.therm.C_tan, \
                                                     self.pb.bilinear_flux_form, \
                                                     self.pb.bcs_T, self.pb.kinematic, self.pb.dx)
            
        elif self.pb.analysis == "static":
            self.set_static_u_solver()
            self.Nitermax=2000
                
        if self.pb.is_hypoelastic:
                self.hypo_elastic_solver = HypoElasticSolve(self.pb.material.devia, self.dt)
                
        if self.pb.damage_analysis:
            damage_solver_mapper = {"PhaseField" : PhaseFieldSolve, 
                                    "StaticJohnson" : StaticJohnsonSolve,
                                    "DynamicJohnson" : DynamicJohnsonSolve,
                                    "InertialJohnson" : InertialJohnsonSolve
                                    }
            damage_class = damage_solver_mapper[self.pb.constitutive.damage_model]
            self.damage_solver = damage_class(self.pb.constitutive.damage, 
                                              self.pb.kinematic, self.pb.mesh.comm, 
                                              self.pb.dx, self.dt)
            
        if self.pb.plastic_analysis:
            self.plastic_solver = PlasticSolve(self.pb.constitutive.plastic, self.pb.u)
            
        if self.pb.multiphase_analysis:
            if any(self.pb.multiphase.multiphase_evolution):
                self.pb.multiphase.evol = True
                self.multiphase_solver = MultiphaseSolver(self.pb.multiphase, self.dt, self.pb.constitutive.p, \
                                                          self.pb.T, self.pb.material, self.pb.load)
            else:
                self.pb.multiphase.evol = False

                    
    def set_time_step(self, **kwargs):
        """
        Initialise un objet de la classe TimeStepping.

        Parameters
        ----------
        **kwargs : Tfin, Temps de fin de simulation.
                    Scheme, schema de discretisation.
        """
        self.time_stepping = TimeStepping(self.pb.analysis, self.pb.mesh, self.pb.material, **kwargs)
        self.load_steps = self.time_stepping.load_steps
        self.Tfin = self.time_stepping.Tfin
        self.dt = self.time_stepping.dt
        
    def update_Pth(self):       
        if self.pb.multiphase_analysis and self.pb.multiphase.explosive:
            self.pb.pint_vol += self.pb.multiphase.Delta_e_vol_chim / self.dt
         
    def set_static_u_solver(self):      
        param = default_Newton_displacement_solver_parameters()
        self.problem_u = petsc.NonlinearProblem(self.pb.form, self.pb.u, self.pb.bcs.bcs)
        self.solver = NewtonSolver(self.pb.mesh.comm, self.problem_u)
        self.solver.atol = param.get("absolute_tolerance")
        self.solver.rtol = param.get("relative_tolerance")
        self.solver.convergence_criterion = param.get("convergence_criterion")
        
    def set_iterative_solver_parameters(self):
        pass
            
    def update_pressure(self):
        if not self.pb.multiphase_analysis:
            self.pb.constitutive.p.x.array[:] = self.pb.material.eos.update_pressure()
        if self.pb.multiphase_analysis:
            for i, mat in enumerate(self.pb.material):
                if mat.eos_type == "Tabulated":
                    self.pb.constitutive.p_list[i].x.array[:] = self.pb.material.eos.update_pressure()        
            

    def staggered_solve(self):
        """
        Solveur échelonné pour la résolution des problèmes élasto-endommageable en statique
        """
        niter=0
        evol_dam = self.pb.damage_analysis
        evol_plas = self.pb.plastic_analysis
        if evol_dam:   
            self.damage_solver.inf_damage()
        while niter < self.Nitermax and (evol_dam or evol_plas):
            #Displacement solve
            self.solver.solve(self.pb.u)
            # print("La norme du résidu est égale à", assemble(self.pb.form).norm("l2"))
            if evol_dam:                          
                evol_dam = self.damage_solver.damage_evolution()
            if evol_plas:
                self.plastic_solver.solve()
                evol_plas = False
            niter+=1
            # print("   Iteration {}".format(niter))

    def solve(self):
        """
        Boucle temporelle, à chaque pas les différents solveurs 
        (déplacement, énergie...) sont successivement appelés        

        Parameters
        ----------
        **kwargs : Int, si un compteur donné par un int est spécifié
                        l'export aura lieu tout les compteur-pas de temps.
        """
        
        num_time_steps = self.time_stepping.num_time_steps
        # Utilisation de tqdm pour créer une barre de progression
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
        self.query_output(self.pb, t)
        self.export.export_results(t)
        self.export.csv.csv_export(t)
        
    def query_output(self, problem, t):
        pass
    
    def final_output(self, problem):
        pass