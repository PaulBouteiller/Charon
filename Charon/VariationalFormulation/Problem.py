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
Problem Base Module
==================

This module provides the foundation for defining and solving mechanical problems
using the finite element method with FEniCSx.

The module implements the base Problem class, which serves as the framework
for all specific problem types (1D, 2D, 3D). It handles the setup of the
variational formulation, boundary conditions, material properties, and solver
parameters.

Key components:
- BoundaryConditions: Management of displacement and temperature constraints
- Loading: Application of external forces and pressure
- Problem: Base framework for mechanical problems
"""

from ..ConstitutiveLaw.ConstitutiveLaw import ConstitutiveLaw
from ..ConstitutiveLaw.Thermal import Thermal
from ..utils.kinematic import Kinematic
from ..utils.default_parameters import default_damping_parameters
from ..utils.MyExpression import MyConstant, Tabulated_BCs
# from ..utils.quadrature import Quadrature
from ..utils.interpolation import create_function_from_expression

from .multiphase import Multiphase


from mpi4py import MPI
from basix.ufl import element
from dolfinx.fem.petsc import set_bc
from petsc4py.PETSc import ScalarType
from numpy import array

from dolfinx.fem import (functionspace, locate_dofs_topological, dirichletbc, 
                         form, assemble_scalar, Constant, Function, Expression, function)
from dolfinx.mesh import meshtags

from ufl import (action, inner, FacetNormal, TestFunction, TrialFunction, dot, SpatialCoordinate)

class BoundaryConditions:
    """
    Class containing displacement boundary conditions.
    
    This class manages Dirichlet boundary conditions for displacement,
    velocity, and acceleration fields, supporting both constant and
    time-dependent values.
    
    Attributes
    ----------
    V : dolfinx.fem.FunctionSpace
        Function space for the displacement field
    facet_tag : dolfinx.mesh.MeshTags
        Tags identifying different regions of the boundary
    bcs : list
        List of Dirichlet boundary conditions
    v_bcs : list
        List of velocity boundary conditions
    a_bcs : list
        List of acceleration boundary conditions
    bcs_axi : list
        List of axisymmetry boundary conditions
    bcs_axi_homog : list
        List of homogeneous axisymmetry boundary conditions
    my_constant_list : list
        List of time-dependent boundary condition expressions
    """
    def __init__(self, V, facet_tag):
        """
        Initialize boundary conditions.
        
        Parameters
        ----------
        V : dolfinx.fem.FunctionSpace Function space for the displacement field
        facet_tag : dolfinx.mesh.MeshTags Tags identifying different regions of the boundary
        """
        self.V = V
        self.facet_tag = facet_tag
        self.bcs = []
        self.v_bcs = []
        self.a_bcs = []
        self.bcs_axi = []
        self.bcs_axi_homog = []
        self.my_constant_list = []
        
    def current_space(self, space, isub):
        """
        Get the current function space or subspace.
        
        Parameters
        ----------
        space : dolfinx.fem.FunctionSpace Base function space
        isub : int or None Subspace index, or None for the whole space
            
        Returns
        -------
        dolfinx.fem.FunctionSpace The selected function space or subspace
        """
        if isub is None:
            return space
        else:
            return space.sub(isub)

    def add_component(self, space, isub, bcs, region, value=ScalarType(0)):
        """
        Add a Dirichlet boundary condition to the list.
        
        Parameters
        ----------
        space : dolfinx.fem.FunctionSpace Function space for the constrained field
        isub : int or None Subspace index, or None for a scalar field
        bcs : list List to which the boundary condition will be added
        region : int Tag identifying the boundary region
        value : float, Constant, or Expression, optional Value to impose, by default 0
        """
        def bc_value(value):
            if isinstance(value, float) or isinstance(value, Constant):
                return value
            elif isinstance(value, MyConstant):
                return value.Expression.constant
         
        dof_loc = locate_dofs_topological(self.current_space(space, isub), self.facet_tag.dim, self.facet_tag.find(region))
        bcs.append(dirichletbc(bc_value(value), dof_loc, self.current_space(space, isub)))
        if isinstance(value, MyConstant):
            self.my_constant_list.append(value.Expression)
            
    def add_associated_speed_acceleration(self, space, isub, region, value=ScalarType(0)):
        """
        Add associated velocity and acceleration boundary conditions.
        
        Parameters
        ----------
        space, isub, region, value : see add_component parameters
        """
        def associated_speed(value):
            if isinstance(value, float) or isinstance(value, Constant):
                return value
            elif isinstance(value, MyConstant):
                return value.Expression.v_constant
            
        def associated_acceleration(value):
            if isinstance(value, float) or isinstance(value, Constant):
                return value
            elif isinstance(value, MyConstant):
                return value.Expression.a_constant
         
        dof_loc = locate_dofs_topological(self.current_space(space, isub), self.facet_tag.dim, self.facet_tag.find(region))
        self.v_bcs.append(dirichletbc(associated_speed(value), dof_loc, self.current_space(space, isub)))
        self.a_bcs.append(dirichletbc(associated_acceleration(value), dof_loc, self.current_space(space, isub)))
        
    def add_T(self, V_T, T_bcs, value, region):
        """
        Impose a Dirichlet boundary condition on the temperature field.
        
        Parameters
        ----------
        V_T : dolfinx.fem.FunctionSpace Function space for the temperature field
        T_bcs : list List of Dirichlet BCs for the temperature field
        value : ScalarType or Expression Value to impose
        region : int Tag identifying the boundary region
        """
        self.add_component(V_T, None, T_bcs, region, value)
        
    def remove_all_bcs(self):
        """
        Remove all boundary conditions.
        
        Clears all boundary condition lists, including displacement,
        velocity, acceleration, and time-dependent expressions.
        """
        print("Remove_bcs")
        self.bcs = []
        self.v_bcs = []
        self.a_bcs = []
        self.my_constant_list = []
                
class Loading:
    """
    Class for managing external loads.
    
    This class creates the external work form (Wext) representing the
    work done by external forces, such as boundary tractions or body forces.
    
    Attributes
    ----------
    kinematic        : Kinematic Object handling kinematics transformations
    my_constant_list : list List of time-dependent loading expressions
    function_list    : list  List of loading functions
    Wext             : ufl.form.Form Form representing the external work
    """
    def __init__(self, mesh, u_, dx, kinematic):
        """
        Initialize the external work form.
        
        Parameters
        ----------
        mesh : dolfinx.mesh.Mesh Computational mesh
        u_ : ufl.TestFunction Test function for the displacement field
        dx : ufl.Measure Integration measure
        kinematic : Kinematic Object handling kinematics transformations
        """
        self.kinematic = kinematic
        self.my_constant_list = []
        self.function_list = []
        self.Wext = kinematic.measure(Constant(mesh, ScalarType(0)) * u_, dx)
        self.n = FacetNormal(mesh)
        
    def add_loading(self, value, u_, dx):
        """
        Add external loads.
        
        If dx is a volume measure, this adds body forces;
        if dx is a surface measure, this adds Neumann boundary conditions.
        
        Parameters
        ----------
        value : ScalarType or Expression Value of the load
        u_ : ufl.TestFunction Test function for the displacement field
        dx : ufl.Measure Integration measure
        """
        if isinstance(value, MyConstant):
            if hasattr(value, "function"):
                self.Wext += self.kinematic.measure(inner(value.Expression.constant * value.function, u_), dx)
            else: 
                self.Wext += self.kinematic.measure(inner(value.Expression.constant, u_), dx)   
            self.my_constant_list.append(value.Expression)
        
        elif isinstance(value, Tabulated_BCs):
            pass
        else:
            assert(value!=0.)
            self.Wext += self.kinematic.measure(inner(value, u_), dx)
            
    def select(self, value):
        """
        Select the appropriate value for a boundary condition.
        
        Parameters
        ----------
        value : various types Input value or expression
            
        Returns
        -------
        function.Expression or None Selected representation of the value
        """
        if isinstance(value, function.Expression):
            return value
    
    def add_pressure(self, p, u_, ds):
        """
        Add pressure (normal surface force) on the exterior surface.
        
        Parameters
        ----------
        p : ScalarType or Expression Pressure value
        u_ : ufl.TestFunction Test function for the displacement field
        ds : ufl.Measure Surface integration measure
        """
        def value(value):
            if isinstance(value, MyConstant):
                return value.Expression.constant
            else:
                return value
        self.Wext += self.kinematic.measure(-value(p) * dot(self.n, u_), ds)
        
class Problem:
    """
    Base class for mechanical problems.
    
    This class is one of the main elements of the CharonX code. It defines
    the problem formulation by calling the selected mechanical models.
    """
    def __init__(self, material, simulation_dic):
        """
        Initialize the problem.
        
        Sets up the mesh, material properties, boundary conditions,
        function spaces, and variational forms for the mechanical problem.
        
        Parameters
        ----------
        material : Material or list
            Material properties, or list of materials for multiphase problems
        simulation_dic : dict
                - analysis: Type of analysis
                - damage: Damage model
                - plastic: Plasticity model
                - isotherm: Whether to use isothermal analysis
                - adiabatic: Whether to use adiabatic analysis
                - Thermal_material: Material for thermal properties
        """
        # Configure analysis type
        self._init_analysis_type(simulation_dic)
        
        # Initialize material
        self.material = material
        
        self._init_mpi()
        
        # Initialize mesh and MPI configuration
        self._transfer_data_from_mesh_manager(simulation_dic)
        if self.name in ["Axisymmetric", "CylindricalUD", "SphericalUD"]:
            self.r = SpatialCoordinate(self.mesh)[0]
        else: 
            self.r = None
        
        # Initialize kinematics and damping
        self.kinematic = Kinematic(self.name, self.r)
        self.damping = simulation_dic.get("damping", default_damping_parameters())
        
        # Configure function spaces and unknown functions
        self._init_spaces_and_functions()
        
        # Configure multiphase analysis
        self._init_multiphase(simulation_dic)
        
        # Initialize density fields
        self.rho_0_field_init, self.relative_rho_field_init_list = self.rho_0_field()
        
        # Configure polycrystal if needed
        self._init_polycristal()
        
        # Determine law types
        self._determine_law_types()
        
        # Initialize constitutive law
        self._init_constitutive_law()
        
        # Initialize temperature and auxiliary fields
        self._init_temperature_and_auxiliary()
        
        # Configure explosive if needed
        if self.multiphase_analysis and self.multiphase.explosive:
            self.set_explosive()
        
        # Configure thermal analysis if needed
        self._init_thermal_analysis()
        
        # Initialize loading
        self._init_loading(simulation_dic)
        
        # Configure variational forms
        self._init_variational_forms()
        
        # Configure boundary conditions
        self._init_boundary_conditions(simulation_dic)
    
    def _init_mpi(self):
        """
        Initialize MPI configuration for parallel computation.
        
        Sets up the MPI environment and determines whether the computation
        is running in parallel or serial mode.
        """
        if MPI.COMM_WORLD.Get_size() > 1:
            print("Parallel computation")
            self.mpi_bool = True
        else:
            print("Serial computation")
            self.mpi_bool = False
    
    def _init_analysis_type(self, dictionnaire):
        """
        Configure the type of analysis to perform.
        
        Sets up flags and parameters for the specific type of analysis,
        such as static, dynamic, with or without damage, etc.
        
        Parameters
        ----------
        dictionnaire : dict Configuration parameters:
                        - analysis: Type of analysis
                        - damage: Damage model
                        - plastic: Plasticity model
                        - isotherm: Whether to use isothermal analysis
                        - adiabatic: Whether to use adiabatic analysis
                        - Thermal_material: Material for thermal properties
        """
        self.analysis = dictionnaire.get("analysis", "explicit_dynamic")
        self.damage_dictionnary = dictionnaire.get("damage", {})
        self.plasticity_dictionnary = dictionnaire.get("plasticity", {})        
        if self.analysis == "Pure_diffusion":
            self.adiabatic = False
            self.iso_T = False
        else:
            self.iso_T = dictionnaire.get("isotherm", False)
            self.adiabatic = dictionnaire.get("adiabatic", True)

        if not self.adiabatic:
            self.mat_th = dictionnaire.get("Thermal_material", None)
            
        self.plastic_analysis = bool(self.plasticity_dictionnary)
        self.damage_analysis = bool(self.damage_dictionnary)
        
    def _transfer_data_from_mesh_manager(self, simulation_dic):
        mesh_manager = simulation_dic["mesh_manager"]
        self.mesh = mesh_manager.mesh
        self.quad = mesh_manager.quad
        self.h = mesh_manager.h
        self.dim = mesh_manager.dim
        self.fdim = mesh_manager.fdim
        self.dx = mesh_manager.dx
        self.dx_l = mesh_manager.dx_l
        self.ds = mesh_manager.ds
        self.u_deg = mesh_manager.u_deg
        self.facet_tag = mesh_manager.facet_tag
    
    def _init_spaces_and_functions(self):
        """
        Initialize function spaces and unknown functions.
        
        Sets up the finite element spaces for displacement, stress, etc.,
        and creates the corresponding function objects.
        """
        self.set_finite_element()
        self.set_function_space()
        self.set_functions()
    
    def _init_multiphase(self, dictionnaire):
        """
        Initialize multiphase analysis if needed.
        
        Sets up the multiphase object and configuration if the material
        is defined as a list of materials.

        """
        self.multiphase_analysis = isinstance(self.material, list)
        if self.multiphase_analysis:
            self.n_mat = len(self.material)
            self.multiphase = Multiphase(self.n_mat, self.quad)
        else:
            self.n_mat = 1
            self.multiphase = None
    
    def _init_polycristal(self):
        """
        Initialize polycrystal configuration if needed.
        
        Sets up polycrystal-specific properties for anisotropic materials.
        """
        if (self.multiphase_analysis and any(mat.dev_type == "Anisotropic" for mat in self.material)) or \
           (not self.multiphase_analysis and self.material.dev_type == "Anisotropic"):
            self.set_polycristal()
    
    def _determine_law_types(self):
        """
        Determine the types of constitutive laws used.
        
        Identifies whether the material uses tabulated EOS, hypoelastic
        formulation, or pure hydrostatic behavior.
        """
        def is_in_list(material, attribut, keyword):
            is_mult = isinstance(material, list)
            return (is_mult and any(getattr(mat, attribut) == keyword for mat in material)) or \
                  (not is_mult and getattr(material, attribut) == keyword)

        self.is_tabulated = is_in_list(self.material, "eos_type", "Tabulated")
        self.is_hypoelastic = is_in_list(self.material, "dev_type", "Hypoelastic")
        self.is_pure_hydro = is_in_list(self.material, "dev_type", None)
    
    def _init_constitutive_law(self):
        """
        Initialize the constitutive law.
        
        Creates a ConstitutiveLaw object with the appropriate configuration
        for the material and problem type.
        """
        self.constitutive = ConstitutiveLaw(
            self.u, self.material, self.plasticity_dictionnary,
            self.damage_dictionnary, self.multiphase,
            self.name, self.kinematic, self.quad,
            self.damping, self.is_hypoelastic,
            self.relative_rho_field_init_list, self.h
        )
    
    def _init_temperature_and_auxiliary(self):
        """
        Initialize temperature and auxiliary fields.
        
        Sets up the initial temperature and creates auxiliary fields derived
        from the primary variables.
        """
        self.set_initial_temperature()
        self.set_auxiliary_field()
    
    def _init_thermal_analysis(self):
        """
        Initialize thermal analysis if needed.
        
        Sets up thermal properties and heat transfer formulation for
        non-isothermal analysis.
        """
        if self.analysis != "static" and not self.iso_T:
            self.therm = Thermal(
                self.material, self.multiphase, self.kinematic, 
                self.T, self.T0, self.constitutive.p
            )
            self.set_T_dependant_massic_capacity()
            self.therm.set_tangent_thermal_capacity() 
            self.set_volumic_thermal_power()
    
    def _init_loading(self, simulation_dic):
        """
        Initialize loading conditions.
        
        Creates a Loading object and sets up external forces and boundary tractions.
        """
        self.load = Constant(self.mesh, ScalarType((1)))
        self.loading = self.loading_class()(self.mesh, self.u_, self.dx, self.kinematic)
        loading_config = simulation_dic.get("loading_conditions", [])
        for loading in loading_config:
            loading_type = loading["type"]
            tag = loading["tag"]
            value = loading["value"]
            if loading_type == "surfacique" and self.analysis == "static":
                getattr(self.loading, "add_"+ loading["component"])(value * self.load, self.u_, self.ds(tag))
            elif loading_type == "surfacique" and self.analysis == "explicit_dynamic":
                getattr(self.loading, "add_"+ loading["component"])(value, self.u_, self.ds(tag))
            else:
                raise ValueError("loading type must be either surfacique or volumique")
    
    def _init_variational_forms(self):
        """
        Initialize variational forms.
        
        Sets up the weak forms for the mechanical problem, including terms
        for stiffness, mass, and external work.
        """
        print("Starting setting up variational formulation")
        self.set_form()
        
        if not self.adiabatic:
            self.flux_bilinear_form()

        if self.damage_analysis:
            self.constitutive.set_damage_driving(self.u, self.J_transfo)
            
        if self.plastic_analysis:
            self.constitutive.set_plastic_driving()
    
    def _init_boundary_conditions(self, simulation_dic):
        """
        Initialize boundary conditions.
        
        Creates a BoundaryConditions object and sets up the specific
        boundary conditions for the problem.
        """
        self.bcs = self.boundary_conditions_class()(self.V, self.facet_tag, self.name)
        boundary_conditions_config = simulation_dic.get("boundary_conditions", [])
        for bc_config in boundary_conditions_config:
            tag = bc_config["tag"]
            value = bc_config.get("value", ScalarType(0))
            getattr(self.bcs, "add_" + bc_config["component"])(region = tag, value = value)
        
    def set_function_space(self):
        """
        Initialize function spaces.
        
        Creates the appropriate function spaces for temperature, displacement,
        and other fields based on the problem configuration.
        """
        if self.adiabatic:
            self.V_T = self.quad.quadrature_space(["Scalar"])
        else:
            FE_T_elem = element("Lagrange", self.mesh.basix_cell(), degree=self.u_deg)
            self.V_T = functionspace(self.mesh, FE_T_elem)
        self.V = functionspace(self.mesh, self.U_e)
        self.V_quad_UD = self.quad.quadrature_space(["Scalar"])
        self.V_Sig = functionspace(self.mesh, self.Sig_e)
        self.V_devia = functionspace(self.mesh, self.devia_e)
        
    def set_functions(self):
        """
        Initialize unknown fields for the thermo-mechanical problem.
        
        Creates Function objects for displacement, velocity, and temperature.
        """
        self.u_ = TestFunction(self.V)
        self.du = TrialFunction(self.V)
        self.u = Function(self.V, name="Displacement")
        self.v = Function(self.V, name="Velocities")
        self.T = Function(self.V_T, name="Temperature")
        
    def set_form(self):
        """
        Initialize variational forms for the solver.
        
        Sets up the residual form and, for dynamic problems, the mass form.
        """
        a_res = self.k(self.sig, self.conjugate_strain())
        L_form = self.loading.Wext
        self.form = a_res - L_form
        if self.analysis == "explicit_dynamic":
            self.m_form = self.m(self.du, self.u_)
        
    def k(self, sigma, eps):
        """
        Define the stiffness form.
        
        Parameters
        ----------
        sigma : ufl.tensors.ListTensor Current stress
        eps : ufl.tensors.ListTensor Strain conjugate to sigma
            
        Returns
        -------
        ufl.form.Form Stiffness form
        """
        return self.kinematic.measure(self.inner(sigma, eps), self.dx)
    
    def m(self, du, u_):
        """
        Define the mass bilinear form.
        
        Parameters
        ----------
        du : ufl.TrialFunction Trial function
        u_ : ufl.TestFunction Test function
            
        Returns
        -------
        ufl.form.Form Mass form
        """
        return self.kinematic.measure(self.rho_0_field_init * inner(du, u_), self.dx_l)
        
    def set_auxiliary_field(self):
        """
        Initialize auxiliary fields for the thermo-mechanical problem.
        
        Creates fields derived from the primary unknowns, such as the
        Jacobian of the transformation, density, and stress.
        """
        self.J_transfo = self.kinematic.J(self.u)
        self.rho = self.rho_0_field_init / self.J_transfo
        self.sig = self.current_stress(self.u, self.v, self.T, self.T0, self.J_transfo)        
        self.D = self.kinematic.Eulerian_gradient(self.v, self.u)
        
        self.sig_expr = Expression(self.sig, self.V_Sig.element.interpolation_points())
        self.sig_func = Function(self.V_Sig, name="Stress")
        if not self.is_pure_hydro:
            s_expr = self.extract_deviatoric(self.constitutive.s)
            # self.sig_VM = Expression(sqrt(3./2 * inner(s_expr, s_expr)), self.V_quad_UD.element.interpolation_points())
            # self.sig_VM_func = Function(self.V_quad_UD, name = "VonMises") 
            self.s_expr = Expression(s_expr, self.V_devia.element.interpolation_points())
            self.s_func = Function(self.V_devia, name = "Deviateur")  
        self.p_expr = Expression(self.constitutive.p, self.V_quad_UD.element.interpolation_points())
        self.p_func = Function(self.V_quad_UD, name="Pression")
        
    def rho_0_field(self):
        """
        Define the initial density field.
        
        For multimaterial models, computes the initial density as a weighted
        sum of phase densities, and returns relative density fields for each phase.
        
        Returns
        -------
        tuple (rho_0_field_init, relative_rho_field_init_list)
                - Initial density field
                - List of relative density fields for each phase
        """
        if isinstance(self.material, list):
            if all([self.material[0].rho_0 == self.material[i].rho_0 for i in range(self.n_mat)]):
                return self.material[0].rho_0, [1 for _ in range(self.n_mat)]
            else:
                rho_sum = sum(c * mat.rho_0 for c, mat in zip(self.multiphase.c, self.material))
                rho_0_field_init = create_function_from_expression(
                    self.V_quad_UD, 
                    Expression(rho_sum, self.V_quad_UD.element.interpolation_points())
                )
                relative_rho_field_init_list = [Function(self.V_quad_UD) for _ in range(self.n_mat)]
                for i in range(self.n_mat):
                    relative_rho_field = Expression(self.material[i].rho_0 / rho_sum, self.V_quad_UD.element.interpolation_points()) 
                    relative_rho_field_init_list[i].interpolate(relative_rho_field)

                return rho_0_field_init, relative_rho_field_init_list
        else:
            return self.material.rho_0, 1

    def current_stress(self, u, v, T, T0, J):
        """
        Define the current stress in the material.
        
        Computes the stress based on deformation and velocity, applying
        degradation factors in case of damage.
        
        Parameters
        ----------
        u  : dolfinx.fem.Function Displacement field
        v  : dolfinx.fem.Function Velocity field
        T  : dolfinx.fem.Function Current temperature field
        T0 : dolfinx.fem.Function Initial temperature field
        J  : ufl.algebra.Product Jacobian of the transformation
            
        Returns
        -------
        ufl.tensors.ListTensor
            Current stress tensor
        """
        sigma = self.undamaged_stress(u, v, T, T0, J)
        if self.damage_analysis:
            sigma *= self.constitutive.damage.g_d
        return sigma
    
    def set_initial_temperature(self):
        """
        Initialize temperature for the study.
        
        For materials following Mie-Gruneisen EOS, the T field may
        represent internal energy rather than temperature.
        """
        T0 = 293.15
        self.T0 = Function(self.V_T)
        self.T0.x.petsc_vec.set(T0)
        self.T.x.petsc_vec.set(T0)
                
    def set_volumic_thermal_power(self):
        """
        Define the volumetric power of internal forces.
        
        Computes the heat generation term from mechanical work.
        """
        self.pint_vol = self.inner(self.sig, self.D)
        
    def flux_bilinear_form(self):
        """
        Define the bilinear form for thermal flux.
        
        Sets up the weak form for heat conduction in non-adiabatic analysis.
        """
        self.dT = TrialFunction(self.V_T)
        self.T_ = TestFunction(self.V_T)
        j = self.therm.thermal_constitutive_law(self.mat_th, self.kinematic.grad_scal(self.dT))
        self.bilinear_flux_form = self.kinematic.measure(self.dot_grad_scal(j, self.kinematic.grad_scal(self.T_)), self.dx)
        
    def set_time_dependant_BCs(self, load_steps):
        """
        Define the list giving the temporal evolution of loading.
        
        Parameters
        ----------
        load_steps : list List of time steps
        """
        for constant in self.loading.my_constant_list:
            constant.set_time_dependant_array(load_steps)
        for constant in self.bcs.my_constant_list:
            constant.set_time_dependant_array(load_steps)
    
    def set_T_dependant_massic_capacity(self):
        """
        Set temperature-dependent specific heat capacity.
        
        To be overridden in derived classes.
        """
        pass
    
    def update_bcs(self, num_pas):
        """
        Update boundary conditions at a given time step.
        
        Parameters
        ----------
        num_pas : int Current time step number
        """
        pass
    
    def set_polycristal(self):
        """
        Set up polycrystal configuration.
        
        To be overridden in derived classes.
        """
        pass
    

    def user_defined_displacement(self, t):
        """
        Apply user-defined displacement at time t.
        
        Parameters
        ----------
        t : float Current time
        """
        set_bc(self.u.x.petsc_vec, self.bcs.bcs)
    
    def set_gen_F(self, boundary_flag, value):
        """
        Define the resultant force on a given surface.
        
        Computes the reaction force by testing the residual with a
        carefully chosen test function.
        
        Parameters
        ----------
        boundary_flag : int Flag of the boundary where the resultant is to be recovered
        value : ScalarType Value to impose for the test function
            
        Returns
        -------
        ufl.form.Form Linear form representing the action of the residual on the test function
            
        Notes
        -----
        This follows the approach described in:
        https://comet-fenics.readthedocs.io/en/latest/demo/tips_and_tricks/computing_reactions.html
        """
        v_reac = Function(self.V)
        dof_loc = locate_dofs_topological(self.V, self.facet_tag.dim, self.facet_tag.find(boundary_flag))
        set_bc(v_reac.x.petsc_vec, [dirichletbc(value, dof_loc, self.V)])
        return form(action(self.form, v_reac))
    
    def set_F(self, boundary_flag, coordinate):
        """
        Initialize the resultant force along a coordinate.
        
        Parameters
        ----------
        boundary_flag : int Flag of the boundary where the resultant is to be recovered
        coordinate : str Coordinate for which to recover the reaction ("x", "y", "z", "r")
            
        Returns
        -------
        ufl.form.Form  Linear form representing the reaction force
        """
        if self.dim == 1:
            return self.set_gen_F(boundary_flag, ScalarType(1.))
        elif self.dim == 2:
            if coordinate == "r" or coordinate =="x":
                return self.set_gen_F(boundary_flag, ScalarType((1., 0)))
            elif coordinate == "y" or coordinate =="z":
                return self.set_gen_F(boundary_flag, ScalarType((0, 1.)))
        elif self.dim == 3:
            if coordinate == "x":
                return self.set_gen_F(boundary_flag, ScalarType((1., 0, 0)))
            elif coordinate == "y" :
                return self.set_gen_F(boundary_flag, ScalarType((0, 1., 0)))
            elif coordinate == "z" :
                return self.set_gen_F(boundary_flag, ScalarType((0, 0, 1.)))
    
    def get_F(self, form):
        """
        Export the reaction associated with a Dirichlet boundary condition.
        
        Parameters
        ----------
        form : ufl.form.Form Linear form representing the reaction force
            
        Returns
        -------
        float Integral of the linear form over the boundary
        """
        return assemble_scalar(form)