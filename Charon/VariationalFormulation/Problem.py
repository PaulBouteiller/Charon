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
from ..utils.interpolation import create_function_from_expression

from .multiphase import Multiphase


from mpi4py import MPI
from basix.ufl import element
from dolfinx.fem.petsc import set_bc
from petsc4py.PETSc import ScalarType

from dolfinx.fem import (functionspace, locate_dofs_topological, dirichletbc, 
                         form, assemble_scalar, Constant, Function, Expression, function)

from ufl import (action, inner, FacetNormal, TestFunction, TrialFunction, dot, SpatialCoordinate)

class BoundaryConditions:
    """
    Manager for displacement boundary conditions in mechanical problems.
    
    This class handles Dirichlet boundary conditions for displacement,
    velocity, and acceleration fields, supporting both constant and
    time-dependent values.
    
    Attributes
    ----------
    V : dolfinx.fem.FunctionSpace
        Function space for the displacement field
    facet_tag : dolfinx.mesh.MeshTags
        Mesh tags identifying boundary regions
    bcs : list of dolfinx.fem.DirichletBC
        Displacement boundary conditions
    v_bcs : list of dolfinx.fem.DirichletBC
        Velocity boundary conditions  
    a_bcs : list of dolfinx.fem.DirichletBC
        Acceleration boundary conditions
    bcs_axi : list of dolfinx.fem.DirichletBC
        Axisymmetry boundary conditions
    bcs_axi_homog : list of dolfinx.fem.DirichletBC
        Homogeneous axisymmetry boundary conditions
    my_constant_list : list of MyExpression
        Time-dependent boundary condition expressions
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
        Add a Dirichlet boundary condition to the specified list.
        
        Parameters
        ----------
        space : dolfinx.fem.FunctionSpace
            Function space for the constrained field
        isub : int or None
            Subspace index for vector fields, None for scalar fields
        bcs : list
            List to which the boundary condition will be added
        region : int
            Tag identifying the boundary region
        value : float, Constant, MyConstant, optional
            Value to impose, by default 0
            
        Notes
        -----
        For time-dependent values, use MyConstant objects which will be
        automatically added to the time-dependent expressions list.
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
    Manager for external loads in mechanical problems.
    
    This class constructs the external work form (Wext) representing
    work done by external forces, including body forces, surface tractions,
    and pressure loads.
    
    Attributes
    ----------
    kinematic : Kinematic
        Object handling kinematic transformations and measures
    my_constant_list : list of MyExpression  
        Time-dependent loading expressions
    function_list : list
        Additional loading functions (reserved for future use)
    Wext : ufl.Form
        Variational form representing external work
    n : ufl.FacetNormal
        Outward unit normal vector on mesh boundaries
    u_ : ufl.TestFunction 
    Test function for the displacement field
    """
    def sub_component(self, u_, sub):
        if sub == None:
            return u_
        else:
            return u_[sub]
        
    def __init__(self, mesh, u_, dx, kinematic, sub = None):
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
        self.u_ = u_ 
        self.Wext = kinematic.measure(Constant(mesh, ScalarType(0)) * self.sub_component(self.u_, sub), dx)
        self.n = FacetNormal(mesh)
        
    def add_loading(self, value, dx, sub = None):
        """
        Add external loads to the variational form.
        
        Parameters
        ----------
        value : ScalarType, Expression, MyConstant, or Tabulated_BCs
            Load value or expression
        dx : ufl.Measure
            Integration measure (dx for body forces, ds for surface tractions)
            
        Notes
        -----
        - For volume measures (dx): adds body forces
        - For surface measures (ds): adds Neumann boundary conditions
        - Time-dependent loads should use MyConstant objects
        """
        u_component = self.sub_component(self.u_, sub)
        if isinstance(value, MyConstant):
            if hasattr(value, "function"):
                self.Wext += self.kinematic.measure(inner(value.Expression.constant * value.function, u_component), dx)
            else: 
                self.Wext += self.kinematic.measure(inner(value.Expression.constant, u_component), dx)   
            self.my_constant_list.append(value.Expression)
        
        elif isinstance(value, Tabulated_BCs):
            pass
        else:
            assert(value!=0.)
            self.Wext += self.kinematic.measure(inner(value, u_component), dx)
            
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
    
    def add_pressure(self, p, ds):
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
        self.Wext += self.kinematic.measure(-value(p) * dot(self.n, self.u_), ds)
        
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
        print("Starting setting up variational formulation")
        self.set_form()
        
        if not self.adiabatic:
            self.flux_bilinear_form()
        
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
        Configure the analysis type and related parameters.
        
        Sets up flags for static/dynamic analysis, damage modeling,
        plasticity, and thermal coupling based on the simulation dictionary.
        
        Parameters
        ----------
        dictionnaire : dict
            Configuration dictionary with keys:
            - 'analysis' : str, analysis type ('static', 'explicit_dynamic', etc.)
            - 'damage' : dict, damage model parameters
            - 'plasticity' : dict, plasticity model parameters  
            - 'isotherm' : bool, isothermal analysis flag
            - 'adiabatic' : bool, adiabatic analysis flag
            - 'Thermal_material' : ThermalMaterial, thermal properties
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
        Initialize multiphase analysis configuration.
        
        Creates a Multiphase object if the material is defined as a list,
        enabling simulation of materials with multiple phases.
        
        Parameters
        ----------
        dictionnaire : dict
            Simulation configuration dictionary (currently unused but kept
            for future multiphase-specific parameters)
        """
        self.multiphase_analysis = isinstance(self.material, list)
        if self.multiphase_analysis:
            self.n_mat = len(self.material)
            self.multiphase = Multiphase(self.n_mat, self.quad, dictionnaire['multiphase'])
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
        Analyze material properties to determine constitutive law types.
        
        Identifies special material behaviors that require specific handling:
        - Tabulated equations of state
        - Hypoelastic formulations  
        - Pure hydrostatic materials (no deviatoric stress)
        
        Sets the corresponding boolean flags for use throughout the solver.
        """
        def has_law_type(material, attribute, keyword):
            """Check if material(s) have a specific law type."""
            materials = material if isinstance(material, list) else [material]
            return any(getattr(mat, attribute) == keyword for mat in materials)
    
        self.is_tabulated = has_law_type(self.material, "eos_type", "Tabulated")
        self.is_hypoelastic = has_law_type(self.material, "dev_type", "Hypoelastic")  
        self.is_pure_hydro = has_law_type(self.material, "dev_type", None)
    
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
            self.damping,
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
            Define the volumetric power of internal forces.
            
            Computes the heat generation term from mechanical work.
        """
        if self.analysis != "static" and not self.iso_T:
            self.therm = Thermal(
                self.material, self.multiphase, self.kinematic, 
                self.T, self.T0, self.constitutive.p
            )
            self.set_T_dependant_massic_capacity()
            self.therm.set_tangent_thermal_capacity() 
            self.pint_vol = self.inner(self.sig, self.D)
    
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
            component = loading["component"]
            
            if loading_type == "surfacique":
                load_value = value * self.load if self.analysis == "static" else value
                getattr(self.loading, f"add_{component}")(load_value, self.ds(tag))
            elif loading_type == "volumique":
                getattr(self.loading, f"add_{component}")(value, self.dx)
            else:
                raise ValueError(f"Unknown loading type: {loading_type}") 
    
    def _init_boundary_conditions(self, simulation_dic):
        """
        Initialize boundary conditions.
        
        Creates a BoundaryConditions object and sets up the specific
        boundary conditions for the problem.
        """
        self.bcs = self.boundary_conditions_class()(self.V, self.facet_tag, self.name)
        
        boundary_conditions_config = simulation_dic.get("boundary_conditions", [])
        for bc_config in boundary_conditions_config:
            component = bc_config["component"]
            tag = bc_config["tag"]
            value = bc_config.get("value", ScalarType(0))
            
            if hasattr(self.bcs, f"add_{component}"):
                getattr(self.bcs, f"add_{component}")(region=tag, value=value)
            else:
                raise ValueError(f"Unknown boundary condition component: {component}")
        
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
        Initialize auxiliary fields derived from primary variables.
        
        Creates derived quantities needed for post-processing and solver
        operations, including:
        - Jacobian of deformation
        - Current density
        - Stress tensor and components
        - Velocity gradient (rate of deformation)
        
        Also sets up Expression and Function objects for field output.
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
        Compute the current Cauchy stress tensor.
        
        Calculates stress based on the current deformation state,
        applying damage degradation if damage analysis is enabled.
        
        Parameters
        ----------
        u : dolfinx.fem.Function
            Current displacement field
        v : dolfinx.fem.Function  
            Current velocity field
        T : dolfinx.fem.Function
            Current temperature field
        T0 : dolfinx.fem.Function
            Reference temperature field
        J : ufl.Expression
            Jacobian of the deformation gradient
            
        Returns
        -------
        ufl.Expression
            Cauchy stress tensor (form depends on problem dimension)
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