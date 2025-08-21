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
Unified Boundary Conditions Module
=================================

This module provides a unified framework for managing boundary conditions
across all problem dimensions (1D, 2D, 3D) and coordinate systems.

Classes:
--------
BoundaryConditions : Unified base class for all boundary conditions
    Handles displacement, velocity, and acceleration constraints
    Supports component mapping for different coordinate systems
    Manages time-dependent boundary conditions
"""

from dolfinx.fem import locate_dofs_topological, dirichletbc
from petsc4py.PETSc import ScalarType
from ..utils.time_dependent_expressions import MyConstant


class BoundaryConditions:
    """
    Unified manager for boundary conditions in mechanical problems.
    
    This class handles Dirichlet boundary conditions for displacement,
    velocity, and acceleration fields, supporting both constant and
    time-dependent values across all problem dimensions.
    
    Attributes
    ----------
    V : dolfinx.fem.FunctionSpace Function space for the displacement field
    facet_tag : dolfinx.mesh.MeshTags Mesh tags identifying boundary regions
    bcs : list of dolfinx.fem.DirichletBC Displacement boundary conditions
    v_bcs : list of dolfinx.fem.DirichletBC Velocity boundary conditions  
    a_bcs : list of dolfinx.fem.DirichletBC Acceleration boundary conditions
    bcs_axi : list of dolfinx.fem.DirichletBC Axisymmetry boundary conditions
    bcs_axi_homog : list of dolfinx.fem.DirichletBC Homogeneous axisymmetry boundary conditions
    my_constant_list : list of MyExpression Time-dependent boundary condition expressions
    dim : int, optional Problem dimension (1, 2, or 3)
    mapping : dict, optional Component name to index mapping
    """
    
    # Mapping des noms de composantes vers les indices
    COMPONENT_MAPPINGS = {
        1: {'U': None},
        2: {'Ux': 0, 'Uy': 1,# Plane strain      
            'Ur': 0, 'Uz': 1}, # Axisymmetric   
        3: {'Ux': 0, 'Uy': 1, 'Uz': 2}
    }
    
    def __init__(self, V, facet_tag, name=None, dim=None):
        """
        Initialize boundary conditions.
        
        Parameters
        ----------
        V : dolfinx.fem.FunctionSpace Function space for the displacement field
        facet_tag : dolfinx.mesh.MeshTags Tags identifying different regions of the boundary
        name : str, optional Problem name identifier
        dim : int, optional Problem dimension for component mapping
        """
        self.V = V
        self.facet_tag = facet_tag
        self.name = name
        
        # Initialize boundary condition lists
        self.bcs = []
        self.v_bcs = []
        self.a_bcs = []
        self.bcs_axi = []
        self.bcs_axi_homog = []
        self.my_constant_list = []
        self.T_bcs = []
        
        # Setup component mapping if dimension provided
        if dim is not None:
            self.dim = dim
            self.mapping = self.COMPONENT_MAPPINGS[dim]
        
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
        space : dolfinx.fem.FunctionSpace Function space for the constrained field
        isub : int or None Subspace index for vector fields, None for scalar fields
        bcs : list List to which the boundary condition will be added
        region : int Tag identifying the boundary region
        value : float, Constant, MyConstant, optional Value to impose, by default 0
            
        Notes
        -----
        For time-dependent values, use MyConstant objects which will be
        automatically added to the time-dependent expressions list.
        """
        def bc_value(value):
            if isinstance(value, float) or hasattr(value, 'value'):  # Constant-like
                return value
            elif isinstance(value, MyConstant):
                return value.Expression.constant
         
        dof_loc = locate_dofs_topological(self.current_space(space, isub), 
                                         self.facet_tag.dim, 
                                         self.facet_tag.find(region))
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
            if isinstance(value, float) or hasattr(value, 'value'):
                return value
            elif isinstance(value, MyConstant):
                return value.Expression.v_constant
            
        def associated_acceleration(value):
            if isinstance(value, float) or hasattr(value, 'value'):
                return value
            elif isinstance(value, MyConstant):
                return value.Expression.a_constant
         
        dof_loc = locate_dofs_topological(self.current_space(space, isub), 
                                         self.facet_tag.dim, 
                                         self.facet_tag.find(region))
        self.v_bcs.append(dirichletbc(associated_speed(value), dof_loc, self.current_space(space, isub)))
        self.a_bcs.append(dirichletbc(associated_acceleration(value), dof_loc, self.current_space(space, isub)))
        
    def add_component_by_name(self, component_name, region, value=ScalarType(0)):
        """
        Add a boundary condition using component name.
        
        Parameters
        ----------
        component_name : str Name of the component ('Ux', 'Uy', 'Uz', 'Ur', 'U', etc.)
        region : int Tag identifying the boundary region
        value : float, Constant, MyConstant, optional Value to impose, by default 0
            
        Raises
        ------
        ValueError If component name is not supported for current dimension
        AttributeError If dimension mapping is not initialized
        """
        if not hasattr(self, 'mapping'):
            raise AttributeError("Component mapping not initialized. Provide 'dim' in constructor.")
            
        if component_name not in self.mapping:
            raise ValueError(f"Component '{component_name}' not supported for dimension {self.dim}")
        
        component_idx = self.mapping[component_name]
        self.add_component(self.V, component_idx, self.bcs, region, value)
        self.add_associated_speed_acceleration(self.V, component_idx, region, value)
    
    def add_clamped(self, region):
        """
        Add a clamped boundary condition (fixed in all directions).
        
        Imposes zero displacement and zero velocity/acceleration on all
        components of the displacement field on the specified boundary region.
        
        Parameters
        ----------
        region : int Tag identifying the boundary region where the condition is applied
        """
        if hasattr(self, 'dim'):
            if self.dim == 1:
                self.add_component_by_name('U', region, ScalarType(0))
            else:
                for i in range(self.dim):
                    self.add_component(self.V, i, self.bcs, region, ScalarType(0))
                    self.add_associated_speed_acceleration(self.V, i, region, ScalarType(0))
        else:
            # Fallback: assume vector space
            try:
                # Try to get dimension from function space
                space_dim = self.V.num_sub_spaces
                for i in range(space_dim):
                    self.add_component(self.V, i, self.bcs, region, ScalarType(0))
                    self.add_associated_speed_acceleration(self.V, i, region, ScalarType(0))
            except:
                # Scalar space
                self.add_component(self.V, None, self.bcs, region, ScalarType(0))
                self.add_associated_speed_acceleration(self.V, None, region, ScalarType(0))
    
    def add_axi(self, region):
        """
        Add an axisymmetry boundary condition.
        
        This condition is necessary when a side of the domain lies on the
        symmetry axis to prevent field divergence.
        
        Parameters
        ----------
        region : int Tag identifying the boundary region
        """
        if hasattr(self, 'dim'):
            if self.dim == 1:
                component_idx = None
            else:
                component_idx = 0  # Radial component
        else:
            component_idx = 0  # Default to first component
            
        self.add_component(self.V, component_idx, self.bcs_axi, region, ScalarType(1))
        self.add_component(self.V, component_idx, self.bcs, region, ScalarType(0))
        self.add_associated_speed_acceleration(self.V, component_idx, region, ScalarType(0))
        
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