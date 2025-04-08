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
Created on Mon Sep 26 17:56:59 2022

@author: bouteillerp
"""
from ..utils.default_parameters import default_energy_solver_order
from .TimeIntegrator import first_order_rk1, first_order_rk2, first_order_rk4, ButcherIntegrator
from .hybrid_solver import create_linear_solver

from dolfinx.fem import Function, Expression
        
class ExplicitEnergySolver:
    def __init__(self, dt, T, C_tan, PVol):
        """
        Initialise la résolution en énergie explicite adiabatique

        Parameters
        ----------
        dt : float, pas de temps
        T : Function, champ de température actuelle
        C_tan : float ou Function, capacité thermique volumique tangente
        PVol : Function, puissance volumique source.
        """
        self.dt = dt
        self.T = T
        self.dot_T = PVol / C_tan
        self.dot_T_expr = Expression(self.dot_T, self.T.function_space.element.interpolation_points())
        self.dot_T_func = Function(self.T.function_space)
        def derivative_calculator(update_velocity=False):
            # Le paramètre update_velocity n'est pas utilisé ici, mais est requis par l'interface
            return self.dot_T_expr
        
        # Initialiser l'intégrateur avec les tableaux de Butcher
        self.integrator = ButcherIntegrator(derivative_calculator)
        
    def energy_solve(self):
        """
        Actualisation explicite du champ de température
        """
        order = default_energy_solver_order()
        # Mapper les ordres aux schémas disponibles
        scheme_map = {1: "RK1", 2: "RK2", 4: "RK4"}
        scheme = scheme_map.get(order, "RK1")  # Par défaut RK1 si l'ordre n'est pas trouvé
        
        # Utiliser l'intégrateur avec le tableau de Butcher approprié
        self.integrator.solve(scheme, self.T, self.dot_T_expr, self.dot_T_func, self.dt)
        
    # def energy_solve(self):
    #     """ 
    #     Actualisation explicite du champ de température, il est possible
    #     d'utiliser d'autres méthodes de Runge-Kutta en cas d'évolution brutale.
    #     """
    #     order = default_energy_solver_order()
    #     order_selector = {1 : first_order_rk1, 2 : first_order_rk2, 4 : first_order_rk4}
    #     order_selector.get(order)(self.T, self.dot_T_expr, self.dot_T_func ,self.dt)
        

        
class DiffusionSolver:
    def __init__(self, dt, T, T_, dT, PVol, C_tan, flux_form, T_bcs, kinematic, dx):
        """
        Initialise le solveur pour la diffusion thermique

        Parameters
        ----------
        dt : float, pas de temps
        T : Function, champ de température actuelle
        T_ : TestFunction, champ test de température
        dT : TrialFunction, trial function de la température
        PVol : Function, puissance volumique source
        C_tan : float ou Function, capacité thermique volumique tangente
        flux_form : Form, forme linéaire du flux thermique
        T_bcs : DirichletBC, condition aux limites de dirichlet pour T
        kinematic : Objet de la classe kinematic.
        dx : Mesure d'intégration
        """
        self.set_transient_thermal_form(dt, T, T_, dT, PVol, C_tan, flux_form, kinematic, dx)
        self.set_T_solver(T, T_bcs)
        
    def set_transient_thermal_form(self, dt, T, T_, dT, PVol, C_tan, flux_form, kin, dx):
        """
        Définition des formes bilinéaires et linéaires pour la résolution
        du problème variationnel de diffusion      

        Parameters
        ----------
        dt : Float, pas de temps.
        T : Function, température actuelle.
        T_ : TestFunction, Fonction test pour la température.
        dT : TrialFunction, Fonction test pour la température.
        PVol : Expression, puissance volumique.
        C_tan : Expression, capacité thermique massique tangente.
        flux_form : Form, forme bilinéaire du flux thermique.
        dx : Measure, mesure d'intégration.
        """
        self.bilinear_therm_form = kin.measure(C_tan * dT * T_ / dt, dx) + flux_form
        self.linear_therm_form = kin.measure((C_tan * T / dt + PVol) * T_,  dx)

    def set_T_solver(self, T, T_bcs):
        """
        Initialise le solveur pour la résolution du problème variationnel de diffusion
        """
        self.problem_T = create_linear_solver(self.bilinear_therm_form, self.linear_therm_form, T, 
                                              bcs=T_bcs, solver_type="hybrid")
        
    def energy_solve(self):
        self.problem_T.solve()
        
    # def print_statistics(self):
    #     self.problem_T.print_statistics(detailed=True)