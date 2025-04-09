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
Created on Fri Mar 11 09:36:05 2022

@author: bouteillerp
"""
from ..ConstitutiveLaw.ConstitutiveLaw import ConstitutiveLaw
from ..ConstitutiveLaw.Thermal import Thermal
from ..utils.kinematic import Kinematic
from ..utils.default_parameters import default_damping_parameters, default_fem_parameters
from ..utils.MyExpression import MyConstant, Tabulated_BCs
from ..utils.quadrature import Quadrature
from ..utils.interpolation import create_function_from_expression

from .multiphase import Multiphase
from .mesh_manager import MeshManager

from mpi4py import MPI
from basix.ufl import element
from dolfinx.fem.petsc import set_bc
from petsc4py.PETSc import ScalarType
from numpy import array

from dolfinx.fem import (functionspace, locate_dofs_topological, dirichletbc, 
                         form, assemble_scalar, Constant, Function, Expression, function)
from dolfinx.mesh import meshtags

from ufl import (action, inner, FacetNormal, TestFunction, TrialFunction, dot)



class BoundaryConditions:
    """
    La classe BoundaryConditions contient les conditions aux limites en déplacement
    """
    def __init__(self, V, facet_tag):
        self.V = V
        self.facet_tag = facet_tag
        self.bcs = []
        self.v_bcs= []
        self.a_bcs= []
        self.bcs_axi = []
        self.bcs_axi_homog = []
        self.my_constant_list = []
        
    def current_space(self, space, isub):
        if isub is None:
            return space
        else:
            return space.sub(isub)

    def add_component(self, space, isub, bcs, region, value = ScalarType(0)):
        """
        Ajout une condition au limites de Dirichlet à la list bcs

        Parameters
        ----------
        space : functionspace, espace fonctionnel sur lequel vis la fonction sur 
                                laquelle on cherche à appliquer la CL.
        isub : Int ou None, indice du sous espace de space sur lequel on souhaite 
                            appliquer la CL.
        bcs : List, liste auxquelles va être rajouté la CL.
        region : Int, drapeau de la région sur lequel on souhaite appliquer la CL.
        value : Float, Constant ou expression, optional
                    Valeur de la CL. The default is ScalarType(0).
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
            
    def add_associated_speed_acceleration(self, space, isub, region, value = ScalarType(0)):     
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
        
        
    def add_T(self, V_T, T_bcs, value, region = 1):
        """
        Impose une CL de dirichlet au champ de température

        Parameters
        ----------
        V_T : functionspace, espace fonctionnel du champ de température.
        T_bcs : List, list des Cl de Dirichlet pour le champ de température.
        value : ScalarType ou Expression, valeur de la CL à appliquer.
        region : Int, drapeau de la région où appliquer les CLs
        """
        self.add_component(V_T, None, T_bcs, region, value)
        
    def remove_all_bcs(self):
        print("Remove_bcs")
        self.bcs = []
        self.v_bcs = []
        self.a_bcs = []
        self.my_constant_list = []
                
class Loading:
    """
    La classe loading créé l'objet self.Wext égale au travail des efforts exterieurs
    """
    def __init__(self, mesh, u_, dx, kinematic):
        """
        Initialise la forme linéaire Wext

        Parameters
        ----------
        mesh : Mesh, maillage du domaine.
        u_ : TestFunction, fonction test du déplacement.
        dx : Measure, mesure d'intégration.
        """
        self.kinematic = kinematic
        self.my_constant_list = []
        self.function_list = []
        self.Wext = kinematic.measure(Constant(mesh, ScalarType(0)) * u_, dx)
        
    def add_loading(self, value, u_, dx):
        """
        Ajoute des efforts extérieurs, si dx est la mesure volumique,
        il s'agit de force volumique, si dx est une mesure surfacique,
        il s'agit de conditions aux limites de Neumann

        Parameters
        ----------
        value : ScalarType ou Expression, valeur de la CL en effort.
        u_ : TestFunction, fonction test du déplacement.
        dx : Measure, mesure d'intégration.
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
        # if isinstance(value, MyExpression):
        #     return value.Expression.eval
        if isinstance(value, function.Expression):
            return value
        # elif isinstance(value, tabulated):
            #ToDo si on veut mettre une CL tabulée on peut le faire ici
            # en mettant à jour la fonction avec sa valeur.
            # return 
    
    def add_pressure(self, p, mesh, u_, ds):
        """
        Ajoute une pression (force surfacique normal aux frontières)
        sur la surface extérieur de mesure d'intégration ds.

        Parameters
        ----------
        p : ScalarType ou Expression, valeur de la pression.
        mesh : Mesh, maillage du domaine.
        u_ : TestFunction, fonction test du déplacement.
        dx : Measure, mesure surfacique d'intégration.
        """
        n = FacetNormal(mesh)
        def value(value):
            if isinstance(value, MyConstant):
                return value.Expression.constant
            else:
                return value
        self.Wext += self.kinematic.measure(-value(p) * dot(n, u_), ds)
        
class Problem:
    """
    La classe Problem est un des éléments principal du code Charon. C'est elle qui défini
    la formulation du problème en appelant les modèles mécaniques retenus.
    """
    def __init__(self, material, initial_mesh=None, **kwargs):
        # Initialisation du maillage et configuration MPI
        self._init_mesh(initial_mesh)
        self._init_mpi()
        
        # Initialisation des paramètres et du schéma d'intégration
        self._init_parameters(kwargs)
        self.quad = Quadrature(self.mesh, self.u_deg, self.schema)
        
        # Configuration du type d'analyse
        self._init_analysis_type(kwargs)
        
        # Initialisation du matériau
        self.material = material
        
        # Initialisation du gestionnaire de maillage
        self._init_mesh_manager()
        
        # Configuration des conditions aux limites et mesures d'intégration
        self._init_boundary_and_measures()
        
        # Configuration de l'anisotropie
        self.set_anisotropy()
        
        # Initialisation de la cinématique et de l'amortissement
        self.kinematic = Kinematic(self.name, self.r, self.n0)
        self.damping = self.set_damping()
        
        # Configuration des espaces fonctionnels et fonctions inconnues
        self._init_spaces_and_functions()
        
        # Configuration pour l'analyse multiphase
        self._init_multiphase(kwargs)
        
        # Initialisation des champs de masse volumique
        self.rho_0_field_init, self.relative_rho_field_init_list = self.rho_0_field()
        
        # Configuration pour polycristal si nécessaire
        self._init_polycristal()
        
        # Détermination des types de lois
        self._determine_law_types()
        
        # Initialisation de la loi constitutive
        self._init_constitutive_law()
        
        # Configuration pour analyse d'endommagement si nécessaire
        if self.damage_analysis:
            self.set_damage()
        
        # Configuration pour analyse plastique si nécessaire
        if self.plastic_analysis:
            self.set_plastic()
        
        # Initialisation de la température et des champs auxiliaires
        self._init_temperature_and_auxiliary()
        
        # Configuration pour explosif si nécessaire
        if self.multiphase_analysis and self.multiphase.explosive:
            self.set_explosive()
        
        # Configuration thermique si nécessaire
        self._init_thermal_analysis()
        
        # Initialisation du chargement
        self._init_loading()
        
        # Configuration des formulations variationnelles
        self._init_variational_forms()
        
        # Configuration des conditions aux limites
        self._init_boundary_conditions()
        
        # Initialisation de la vitesse initiale
        self.set_initial_speed()
    
    def _init_mesh(self, initial_mesh):
        """Initialise le maillage du problème."""
        if initial_mesh is None:
            self.mesh = self.define_mesh()
        else:
            self.mesh = initial_mesh
    
    def _init_mpi(self):
        """Initialise la configuration MPI."""
        if MPI.COMM_WORLD.Get_size() > 1:
            print("Parallel computation")
            self.mpi_bool = True
        else:
            print("Serial computation")
            self.mpi_bool = False
            
    def _init_parameters(self, kwargs):
        """Initialise les paramètres FEM du problème."""
        self.fem_parameters()
    
    def _init_analysis_type(self, kwargs):
        """Configure le type d'analyse à effectuer."""
        self.analysis = kwargs.get("analysis", "explicit_dynamic")
        self.damage_model = kwargs.get("damage", None)
        self.plastic_model = kwargs.get("plastic", None)
        self.iso_T = kwargs.get("isotherm", False)
        
        if self.analysis == "Pure_diffusion":
            self.adiabatic = False
            assert not self.iso_T
        else:
            self.adiabatic = kwargs.get("adiabatic", True)

        if not self.adiabatic:
            self.mat_th = kwargs.get("Thermal_material", None)
            
        self.plastic_analysis = self.plastic_model is not None
        self.damage_analysis = self.damage_model is not None
    
    def _init_mesh_manager(self):
        """Initialise le gestionnaire de maillage."""
        self.mesh_manager = MeshManager(self.mesh, self.name)
        self.h = self.mesh_manager.h
        self.dim = self.mesh_manager.dim
        self.fdim = self.mesh_manager.fdim
    
    def _init_boundary_and_measures(self):
        """Initialise les conditions aux limites et mesures d'intégration."""
        self.set_boundary()
        self.set_measures()
        self.r = self.mesh_manager.r
        self.facet_tag = self.mesh_manager.facet_tag
    
    def _init_spaces_and_functions(self):
        """Initialise les espaces fonctionnels et fonctions inconnues."""
        self.set_finite_element()
        self.set_function_space()
        self.set_functions()
    
    def _init_multiphase(self, kwargs):
        """Initialise l'analyse multiphase si nécessaire."""
        self.multiphase_analysis = isinstance(self.material, list)
        if self.multiphase_analysis:
            self.multiphase = Multiphase(len(self.material), self.quad)
            self.set_multiphase()
        else:
            self.multiphase = None
    
    def _init_polycristal(self):
        """Initialise la configuration polycristalline si nécessaire."""
        if (self.multiphase_analysis and any(mat.dev_type == "Anisotropic" for mat in self.material)) or \
           (not self.multiphase_analysis and self.material.dev_type == "Anisotropic"):
            self.set_polycristal()
    
    def _determine_law_types(self):
        """Détermine les types de lois utilisées."""
        def is_in_list(material, attribut, keyword):
            is_mult = isinstance(material, list)
            return (is_mult and any(getattr(mat, attribut) == keyword for mat in material)) or \
                  (not is_mult and getattr(material, attribut) == keyword)

        self.is_tabulated = is_in_list(self.material, "eos_type", "Tabulated")
        self.is_hypoelastic = is_in_list(self.material, "dev_type", "Hypoelastic")
        self.is_pure_hydro = is_in_list(self.material, "dev_type", "None")
    
    def _init_constitutive_law(self):
        """Initialise la loi constitutive."""
        self.constitutive = ConstitutiveLaw(
            self.u, self.material, self.plastic_model,
            self.damage_model, self.multiphase,
            self.name, self.kinematic, self.quad,
            self.damping, self.is_hypoelastic,
            self.relative_rho_field_init_list, self.h
        )
    
    def _init_temperature_and_auxiliary(self):
        """Initialise la température et les champs auxiliaires."""
        self.set_initial_temperature()
        self.set_auxiliary_field()
    
    def _init_thermal_analysis(self):
        """Initialise l'analyse thermique si nécessaire."""
        if self.analysis != "static" and not self.iso_T:
            self.therm = Thermal(
                self.material, self.multiphase, self.kinematic, 
                self.T, self.T0, self.constitutive.p
            )
            self.set_T_dependant_massic_capacity()
            self.therm.set_tangent_thermal_capacity() 
            self.set_volumic_thermal_power()
    
    def _init_loading(self):
        """Initialise le chargement."""
        self.load = Constant(self.mesh, ScalarType((1)))
        self.loading = self.loading_class()(self.mesh, self.u_, self.dx, self.kinematic)
        self.set_loading()
    
    def _init_variational_forms(self):
        """Initialise les formulations variationnelles."""
        print("Starting setting up variational formulation")
        self.set_form()
        
        if not self.adiabatic:
            self.flux_bilinear_form()

        if self.damage_analysis:
            self.constitutive.set_damage_driving(self.u, self.J_transfo)
            
        if self.plastic_analysis:
            self.constitutive.set_plastic_driving()
    
    def _init_boundary_conditions(self):
        """Initialise les conditions aux limites."""
        self.bcs = self.boundary_conditions_class()(self.V, self.facet_tag, self.name)
        self.set_boundary_condition()

    def set_output(self):
        return {}
    
    def query_output(self, t):
        return {}
    
    def final_output(self):
        pass
    
    def csv_output(self):
        return {}
    
    def prefix(self):
        return "problem"
    
    def set_measures(self):
        """Configure les mesures d'intégration pour le problème.
        
        Utilise le gestionnaire de maillage pour définir les mesures
        d'intégration avec le degré polynomial approprié.
        """
        self.mesh_manager.set_measures(self.quad)
        self.dx = self.mesh_manager.dx
        self.dx_l = self.mesh_manager.dx_l
        self.ds = self.mesh_manager.ds
        
    def set_function_space(self):  
        """
        Initialise les espaces fonctionnels
        """
        if self.adiabatic:
            self.V_T = self.quad.quadrature_space(["Scalar"])
        else:
            FE_T_elem = element("Lagrange", self.mesh.basix_cell(), degree = self.u_deg)
            self.V_T = functionspace(self.mesh, FE_T_elem)
        self.V = functionspace(self.mesh, self.U_e)
        self.V_quad_UD = self.quad.quadrature_space(["Scalar"])
        self.V_Sig = functionspace(self.mesh, self.Sig_e)
        self.V_devia = functionspace(self.mesh, self.devia_e)
        
    def set_functions(self):   
        """ 
        Initialise les champs inconnues du problème thermo-mécanique
        """
        self.u_ = TestFunction(self.V)
        self.du = TrialFunction(self.V)
        self.u = Function(self.V, name = "Displacement")
        self.v = Function(self.V, name = "Velocities")
        self.T = Function(self.V_T, name = "Temperature")
        
    def set_form(self):
        """
        Initialise les formes variationnelles prises en input de l'objet solve
        """
        a_res = self.k(self.sig, self.conjugate_strain())
        L_form = self.loading.Wext
        self.form = a_res - L_form
        if self.analysis == "explicit_dynamic":
            self.m_form = self.m(self.du, self.u_)
        
    def k(self, sigma, eps):
        """
        Définition de la forme bilinéaire "rigidité"

        Parameters
        ----------
        sigma : Function, Contrainte actuelle
        eps : Function, déformation conjuguée à sigma
        """
        return self.kinematic.measure(self.inner(sigma, eps), self.dx)
    
    def m(self, du, u_):
        """
        Définition de la forme bilinéaire de masse

        Parameters
        ----------
        du : TrialFunction, champ test.
        u_ : TestFunction, champ test.
        """
        return self.kinematic.measure(self.rho_0_field_init * inner(du, u_), self.dx_l)
        
    def set_auxiliary_field(self):
        """
        Initialise quelques champs auxiliaires qui permettent d'écrire de 
        manière plus concise le problème thermo-mécanique
        """        
        self.J_transfo = self.kinematic.J(self.u)
        self.rho = self.rho_0_field_init / self.J_transfo
        self.sig = self.current_stress(self.u, self.v, self.T, self.T0, self.J_transfo)        
        self.D = self.kinematic.Eulerian_gradient(self.v, self.u)
        # if not self.is_pure_hydro:
        #     s_expr = self.extract_deviatoric(self.constitutive.s)
            # self.sig_VM = Expression(sqrt(3./2 * inner(s_expr, s_expr)), self.V_quad_UD.element.interpolation_points())
            # self.sig_VM_func = Function(self.V_quad_UD, name = "VonMises") 
            # self.s_expr = Expression(s_expr, self.V_devia.element.interpolation_points())
            # self.s_func = Function(self.V_devia, name = "Deviateur")    
        
        self.sig_expr = Expression(self.sig, self.V_Sig.element.interpolation_points())
        self.sig_func = Function(self.V_Sig, name = "Stress")
        
        self.p_expr = Expression(self.constitutive.p, self.V_quad_UD.element.interpolation_points())
        self.p_func = Function(self.V_quad_UD, name = "Pression")

    
    def set_damping(self):
        """
        Initialise les paramètres de la pseudo-viscosité
        """
        return default_damping_parameters()
        
    def rho_0_field(self):
        """
        Définition du champ de masse volumique initial, si toutes les phases
        possèdent la même masse volumique, on renvoie cette valeur commune. 
        La méthode renvoie également des champs de contraintes relatives
        rho_0_phase/rho_0_init pour les modèles multimatériaux afin d'assurer
        la conservation de la masse dans les équations d'états. 
        """
        if isinstance(self.material, list):
            if all([self.material[0].rho_0 == self.material[i].rho_0 for i in range(len(self.material))]):
                return self.material[0].rho_0, [1 for j in range(len(self.material))]
            else:
                rho_sum = sum(c * mat.rho_0 for c, mat in zip(self.multiphase.c, self.material))
                rho_0_field_init = create_function_from_expression(
                    self.V_quad_UD, 
                    Expression(rho_sum, self.V_quad_UD.element.interpolation_points())
                )
                relative_rho_field_init_list = [Function(self.V_quad_UD) for i in range(len(self.material))]
                for i in range(len(self.material)):
                    relative_rho_field = Expression(self.material[i].rho_0 / rho_sum, self.V_quad_UD.element.interpolation_points()) 
                    relative_rho_field_init_list[i].interpolate(relative_rho_field)

                return rho_0_field_init, relative_rho_field_init_list
        else:
            return self.material.rho_0, 1

    def current_stress(self, u, v, T, T0, J):
        """
        Définie la contrainte actuelle dans le matériau en fonction
        de la déformation et de la vitesse (utilisée pour la pseudo-viscosité)
        la contrainte est pondérée par des fonctions de dégradation
        dans le cas endommageable

        Parameters
        ----------
        u : Function, champ de déplacement.
        v : Function, champ de vitesse.
        T : Function, champ de température.
        T0 : Function, champ de température initiale.
        J : Function, Jacobien de la transformation.

        Returns
        -------
        sigma : Contrainte actuelle.

        """
        sigma = self.undamaged_stress(u, v, T, T0, J)
        if self.damage_analysis:
            sigma *= self.constitutive.damage.g_d
        return sigma
    
    def set_initial_temperature(self):
        """
        Initialise la température de l'étude, si un des matériaux
        suit une loi d'état de Mie Gruneisen, le champ T dans ce cas
        désigne l'énergie interne !!!
        """
        T0 = 293.15
        self.T0 = Constant(self.mesh, ScalarType(T0))
        self.T.x.petsc_vec.set(T0)
                
    def set_volumic_thermal_power(self):
        """
        Définition de la puissance volumique des efforts intérieurs
        """
        self.pint_vol = self.inner(self.sig, self.D)
        
    def flux_bilinear_form(self):
        """
        Définition de la forme bilinéaire correspondant au flux thermique
        """
        self.dT = TrialFunction(self.V_T)
        self.T_ = TestFunction(self.V_T)
        j = self.therm.thermal_constitutive_law(self.mat_th, self.kinematic.grad_scal(self.dT))
        self.bilinear_flux_form = self.kinematic.measure(self.dot_grad_scal(j, self.kinematic.grad_scal(self.T_)), self.dx)
        
    def set_time_dependant_BCs(self, load_steps):
        """
        Définition de la liste donnant l'évolution temporelle du chargement.
        Parameters
        ----------
        load_steps : List, liste des pas de temps.
        """
        for constant in self.loading.my_constant_list:
            constant.set_time_dependant_array(load_steps)
        for constant in self.bcs.my_constant_list:
            constant.set_time_dependant_array(load_steps)
            
    def set_anisotropy(self):
        if isinstance(self.material, list):
            if any(mat.dev_type in ["NeoHook_Transverse", "Lu_Transverse"] for mat in self.material):
                self.set_anisotropic_direction()
            else:
                self.n0 = None
        else:
            if self.material.dev_type in ["NeoHook_Transverse", "Lu_Transverse"]:
                self.set_anisotropic_direction()
            else:
                self.n0 = None
        
    def set_boundary(self):
        print("Warning no boundary has been tagged inside CHARONX")
        self.mesh_manager.facet_tag = meshtags(self.mesh, self.fdim, array([]), array([]))
    
    def set_loading(self):
        pass
    
    def set_boundary_condition(self):
        pass

    def set_velocity_boundary_condition(self):
        pass
    
    def set_initial_speed(self):
        pass
    
    def user_defined_constitutive_law(self):
        pass
    
    def set_T_dependant_massic_capacity(self):
        pass
    
    def update_bcs(self, num_pas):
        pass
    
    def set_polycristal(self):
        pass
    
    def fem_parameters(self):
        fem_parameters = default_fem_parameters()
        self.u_deg = fem_parameters.get("u_degree")
        self.schema= fem_parameters.get("schema")
        
    def user_defined_displacement(self, t):
        set_bc(self.u.x.petsc_vec, self.bcs.bcs)
    
    def set_initial_conditions(self):
        pass
    
    def set_gen_F(self, boundary_flag, value):
        """
        Fonction générique définissant la résultante d'une force sur une surface donnée
        en étudiant l'action du champ de déplacement solution sur le résidu
        voir https://comet-fenics.readthedocs.io/en/latest/demo/tips_and_tricks/computing_reactions.html

        Parameters
        ----------
        boundary_flag : Int, drapeau de la frontière sur laquelle on souhaite récupérer
        la résultante.
        value : ScalarType.
        Returns
        -------
        Form, form linéaire, action du résidu sur un champ test bien 
                        choisi prenant une valeur unitaire sur la CL de 
                        Dirichlet où on souhaite récupérer la réaction.
        """
        v_reac = Function(self.V)
        dof_loc = locate_dofs_topological(self.V, self.facet_tag.dim, self.facet_tag.find(boundary_flag))
        set_bc(v_reac.x.petsc_vec, [dirichletbc(value, dof_loc, self.V)])
        return form(action(self.form, v_reac))
    
    def set_F(self, boundary_flag, coordinate):
        """
        Initialise la résultante F selon la coordonnée coordinate
        sur la frontière déterminée par le drapeau boundary_flag.

        Parameters
        ----------
        boundary_flag : Int, drapeau de la frontière sur laquelle on souhaite récupérer
        la résultante.
        coordinate : Str, coordonnée pour laquelle on souhaite récupérer la réaction
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
        Export la réaction associé à la condition aux limites de Dirichlet.

        Parameters
        ----------
        form : Form, forme linéaire, action du résidu sur un champ test bien 
                     choisi prenant une valeur unitaire sur la CL de 
                     Dirichlet où on souhaite récupérer la réaction.
        Returns
        -------
        Scalar, intégrale de la forme linéaire sur la frontière.
        """
        return assemble_scalar(form)