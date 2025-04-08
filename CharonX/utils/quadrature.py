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
Created on Thu Sep  7 15:01:40 2023

@author: bouteillerp, ce fichier contient les routines nécessaires à la condensation
de la matrice de masse dans le cas d'une résolution dynamique explicite.
Il permet de définir des points et des poids de Gauss personnalisés qui 
vont ensuite être utilisés pour intégrer la forme billinéaire de masse.
"""
from numpy import array, kron
from basix import make_quadrature, CellType, QuadratureType
from .default_parameters import default_fem_degree
from dolfinx.fem import functionspace
from basix.ufl import quadrature_element


class Quadrature():
    """
    A class for handling quadrature rules in finite element computations, 
    particularly for mass matrix condensation in explicit dynamic analysis.
    
    This class provides functionality to define custom Gauss points and weights
    that will be used to integrate bilinear mass forms.
    
    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        The computational mesh
    u_degree : int
        Degree of the displacement field interpolation
    schema : str
        Integration scheme type ('default', 'over', or 'reduit')
    
    Attributes
    ----------
    ref_el : str
        Name of the reference element (e.g., 'triangle', 'quadrilateral')
    dim : int
        Dimension of the mesh
    u_deg : int
        Degree of displacement field interpolation
    schema : str
        Integration scheme type
    lumped_metadata : dict
        Metadata for lumped mass integration
    metadata : dict
        Metadata for standard integration
    mesh : dolfinx.mesh.Mesh
        The computational mesh
    """
    def __init__(self, mesh, u_degree, schema):
        self.ref_el = mesh.ufl_cell().cellname()
        self.dim = mesh.topology.dim
        self.u_deg = u_degree
        self.schema = schema
        self.lumped_metadata = self.create_lumped_metadata()
        self.metadata = self.create_metadata()
        self.mesh = mesh
        
    def create_lumped_quadrature(self, ref_el, degree):
        """
        Defines Gauss points at mesh nodes and creates associated Gauss weights.
        
        This integration type, while less precise than standard Gauss integration,
        ensures that the mass matrix is diagonal.
        
        Parameters
        ----------
        ref_el : str
            Type of element used ('quadrilateral', 'triangle', 'hexahedron', 'tetrahedron')
        degree : int
            Interpolation degree of the displacement field
            
        Returns
        -------
        tuple
            x : numpy.ndarray
                Coordinates of Gauss points
            w : numpy.ndarray
                Weights of Gauss points
                
        Notes
        -----
        - For quadrilaterals, sum of weights must equal 1
        - For triangles, sum of weights must equal 0.5
        - Currently supports degrees 1 and 2 for most elements
        - Some 3D elements with degree 2 are not yet implemented
        """
        if ref_el == "quadrilateral" :
            #La somme des poids de Gauss doit être égale à 1 pour un rectangle afin
            # d'intégrer correctement une constante.
            if degree == 1:
                x, w = (array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
                        array([1/4.,]*4))
            elif degree == 2:
                x = array([[0.0, 0.0],
                            [0.5, 0.0],
                            [1.0, 0.0],
                            [0.0, 0.5],
                            [0.5, 0.5],
                            [1.0, 0.5],
                            [0.0, 1.0],
                            [0.5, 1.0],
                            [1.0, 1.0]])
                w_1D = array([1/6., 2/3., 1/6.])
                w = kron(w_1D, w_1D)
            elif degree > 2:
                x, w = make_quadrature(QuadratureType.gll, CellType.quadrilateral, degree+1)
                
        elif ref_el == "triangle" : 
            #La somme des poids de Gauss doit être égale à 0.5 pour un triangle afin
            # d'intégrer correctement une constante.
            if degree == 1:
                x = array([[0.0, 0.0],
                           [1.0, 0.0],
                           [0.0, 1.0]])
                w = array([1/6.,]*3)
            elif degree == 2:
                x = array([[0.0, 0.0],
                           [1.0, 0.0],
                           [0.0, 1.0],
                           [0.5, 0.0],
                           [0.5, 0.5],
                           [0.0, 0.5]])
                w = array([1/12.,]*6)
        elif ref_el == "hexahedron":
            if degree == 1:
                x = array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], 
                            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
                w = array([1/8.,]*8)
            elif degree == 2:
                raise ValueError("Not implemented yet")
        elif ref_el == "tetrahedron" :
            if degree == 1:
                x = array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],  [0.0, 0.0, 1.0]])
                w = array([1/4.,]*4)
            elif degree == 2:
                raise ValueError("Not implemented yet")
        return x, w
    
    def create_lumped_metadata(self):
        if self.ref_el == "interval":
            lumped_metadata = {"quadrature_rule": "GLL", "quadrature_degree" : 2 * (self.u_deg - 1)}
        else:
            points, weights = self.create_lumped_quadrature(self.ref_el, self.u_deg)
            lumped_metadata = {"quadrature_rule": "custom", "quadrature_degree" : self.u_deg, "quadrature_points" : points, "quadrature_weights" : weights}
        return lumped_metadata
    
    
    def default_quadrature_degree(self):
        """
        Determines the number of Gauss points for internal variables.
        
        Returns
        -------
        int
            Number of Gauss points based on element type and integration scheme
            
        Notes
        -----
        - For linear elements (u_deg=1):
          * 'default': normal integration
          * 'over': over-integration for quadrilateral/hexahedral elements
        - For quadratic elements (u_deg=2):
          * 'default': standard integration
          * 'reduit': under-integration to prevent shear-locking
        
        Raises
        ------
        ValueError
            If using degree 2 with interval elements (currently bugged)
        """
        if self.u_deg == 2 and self.ref_el == "interval":
            raise ValueError("Currently bugged")
        if self.u_deg == 1:
            if self.schema == "default":#normal integration for linear element
                return 1
            #over-integration for linear element to prevent hourglass
            elif self.schema == "over" and self.ref_el in ["quadrilateral", "hexahedron"]:
                return int(2**self.dim)
        elif self.u_deg == 2:
            if self.schema == "default":
                return int(2**self.dim)
            #under-integration for quadratic element to prevent shear-locking
            elif self.schema == "reduit":
                return 2
    
    def create_metadata(self):
        quad_deg = self.default_quadrature_degree()
        return {"quadrature_rule": "default", "quadrature_degree": quad_deg}
        # return {"quadrature_rule": "vertex", "quadrature_degree": quad_deg}
                
    def quad_element(self, shape):
        """
        Creates a Quadrature element of appropriate shape.
        
        Parameters
        ----------
        shape : list
            Defines the type and dimensions of the quadrature element:
            - ['Scalar']: for scalar quadrature element
            - ['Vector', size]: for vector quadrature element
            - ['Tensor', size1, size2]: for tensor quadrature element
            
        Returns
        -------
        basix.ufl.Element
            Quadrature element with specified properties
        """
        quad_deg = self.default_quadrature_degree()   
        if shape[0] == "Scalar":
            Qe = quadrature_element(self.ref_el, degree = quad_deg,
                                    value_shape=())
        elif shape[0] == "Vector":
            Qe = quadrature_element(self.ref_el, degree = quad_deg,
                                    value_shape=(shape[1],))
        elif shape[0] == "Tensor":
            Qe = quadrature_element(self.ref_el,  degree = quad_deg,
                                    value_shape=(shape[1], shape[2]))
        return Qe  
    
    def quadrature_space(self, shape):
        """
        Creates a function space using quadrature elements.
        
        Parameters
        ----------
        shape : list
            Shape specification for the quadrature element
            
        Returns
        -------
        dolfinx.fem.FunctionSpace
            Function space using the specified quadrature element
        """
        return functionspace(self.mesh, self.quad_element(shape))