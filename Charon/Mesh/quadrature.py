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
from numpy import array, kron, meshgrid, vstack
# from basix import make_quadrature, CellType, QuadratureType
from dolfinx.fem import functionspace
from basix.ufl import quadrature_element as quad_el
from ufl import FunctionSpace

class Quadrature():
    """
    A class for handling quadrature rules in finite element computations, 
    particularly for mass matrix condensation in explicit dynamic analysis.
    
    This class provides functionality to define custom Gauss points and weights
    that will be used to integrate bilinear mass forms.
    
    Attributes
    ----------
    ref_el : str Name of the reference element (e.g., 'triangle', 'quadrilateral')
    dim : int Dimension of the mesh
    u_deg : int Degree of displacement field interpolation
    schema : str Integration scheme type
    lumped_metadata : dict Metadata for lumped mass integration
    metadata : dict Metadata for standard integration
    mesh : dolfinx.mesh.Mesh The computational mesh
    """
    def __init__(self, mesh, mesh_type, u_degree, schema):
        """Parameters
        ----------
        mesh : dolfinx.mesh.Mesh The computational mesh
        u_degree : int Degree of the displacement field interpolation
        schema : str Integration scheme type ('default', 'over', or 'reduit')"""
        self.ref_el = mesh.ufl_cell().cellname()
        self.mesh_type = mesh_type
        if mesh_type == "dolfinx_mesh":
            self.dim = mesh.topology.dim
        elif mesh_type == "ufl_mesh":
            self.dim = mesh.geometric_dimension()
        self.u_deg = u_degree
        self.schema = schema
        self.lumped_metadata = self.create_lumped_metadata()
        self.metadata = self.create_metadata()
        self.mesh = mesh
        
    def create_lumped_quadrature(self, degree):
        """
        Defines Gauss points at mesh nodes and creates associated Gauss weights.
        
        This integration type, while less precise than standard Gauss integration,
        ensures that the mass matrix is diagonal.
        
        Parameters
        ----------
        ref_el : str Type of element used ('quadrilateral', 'triangle', 'hexahedron', 'tetrahedron')
        degree : int Interpolation degree of the displacement field
            
        Returns
        -------
        tuple x : numpy.ndarray Coordinates of Gauss points
              w : numpy.ndarray Weights of Gauss points
                
        Notes
        -----
        - For quadrilaterals, sum of weights must equal 1
        - For triangles, sum of weights must equal 0.5
        - Currently supports degrees 1 and 2 for most elements
        - Some 3D elements with degree 2 are not yet implemented
        """
        if degree>2: raise ValueError("degree must be lower than 2")
        if self.ref_el == "interval":
            if degree == 1:
                x, w = (array([[0.0], [1.0]]), array([1/2.,]*2))
            elif degree == 2:
                x = array([[0.0], [0.5], [1.0]])
                w = array([1/6, 4/6, 1/6])
        elif self.ref_el == "quadrilateral" :
            if degree == 1:
                x = array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
                w = array([1/4.,]*4)
            elif degree == 2:
                # P2 : Gauss-Lobatto-Legendre 3x3
                points_1D = array([0.0, 0.5, 1.0])
                weights_1D = array([1/6, 4/6, 1/6])  # coins, milieux, centre
                # Coordonnées 2D via meshgrid
                X, Y = meshgrid(points_1D, points_1D, indexing='ij')
                x = vstack([X.ravel(), Y.ravel()]).T
                # Poids 2D par produit tensoriel
                w = kron(weights_1D, weights_1D)
        elif self.ref_el == "triangle" : 
            #La somme des poids de Gauss doit être égale à 0.5 pour un triangle afin
            # d'intégrer correctement une constante.
            if degree == 1:
                x = array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
                w = array([1/6.,]*3)
            elif degree == 2:
                x = array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
                           [0.5, 0.0], [0.5, 0.5], [0.0, 0.5]])
                from sympy import symbols, integrate
                # symboliques pour calcul exact sur triangle référence x>=0,y>=0,x+y<=1
                x,y = symbols('x y')
                l1 = 1 - x - y
                l2 = x
                # P2 basis (barycentric)
                phi_v1 = l1*(2*l1 - 1)
                phi_m12 = 4*l1*l2

                def integral(expr):
                    return integrate(integrate(expr, (y, 0, 1 - x)), (x, 0, 1))
                # diagonales de la matrice consistante (M_ii = ∫ φ_i^2)
                Mii_v = float(integral(phi_v1**2))   # même valeur pour v2,v3
                Mii_m = float(integral(phi_m12**2))  # même pour les 3 milieux
                # somme et facteur d'échelle pour HRZ
                diag = array([Mii_v, Mii_v, Mii_v, Mii_m, Mii_m, Mii_m])
                M_total = 0.5
                factor = M_total / diag.sum()
                w = diag * factor
        elif self.ref_el == "hexahedron":
            if degree == 1:
                x = array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], 
                            [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
                w = array([1/8.,]*8)
            elif degree == 2:
                points_1D = array([0.0, 0.5, 1.0])
                weights_1D = array([1/6, 4/6, 1/6])
                
                # Coordonnées 3D
                X, Y, Z = meshgrid(points_1D, points_1D, points_1D, indexing='ij')
                x = vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
                
                # Poids 3D via produit tensoriel
                w = kron(weights_1D, kron(weights_1D, weights_1D))
                
        elif self.ref_el == "tetrahedron" :
            if degree == 1:
                x = array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],  [0.0, 0.0, 1.0]])
                w = array([1/24.,]*4)
            elif degree == 2:
                from sympy import symbols, integrate
                
                # symboliques pour tétraèdre référence x>=0, y>=0, z>=0, x+y+z <=1
                x, y, z = symbols('x y z')
                l1 = 1 - x - y - z
                l2 = x
                # P2 basis functions
                phi_v1 = l1*(2*l1 - 1)
                # 6 milieux d'arêtes
                phi_m12 = 4*l1*l2
                def integral(expr):
                    return float(integrate(integrate(integrate(expr, (z, 0, 1-x-y)), (y, 0, 1-x)), (x, 0, 1)))
                
                # diagonales de la matrice consistante
                Mii_v = integral(phi_v1**2)
                Mii_m = integral(phi_m12**2)  # n'importe quel milieu
                
                diag = array([Mii_v]*4 + [Mii_m]*6)
                V_total = 1/6
                factor = V_total / diag.sum()
                w = diag * factor
                
                # Points P2 tétraèdre
                x = array([[0,0,0], [1,0,0], [0,1,0], [0,0,1],
                           [0.5,0,0], [0.5,0.5,0], [0.5,0,0.5],
                           [0,0.5,0], [0,0.5,0.5], [0,0,0.5]])
        return x, w
    
    def create_lumped_metadata(self):
        points, weights = self.create_lumped_quadrature(self.u_deg)
        lumped_metadata = {"quadrature_rule": "custom", "quadrature_degree" : self.u_deg, 
                           "quadrature_points" : points, "quadrature_weights" : weights}
        return lumped_metadata
            
    def default_quadrature_degree(self):
        """
        Determines the quadrature degree to be used for stiffness matrix integration.
    
        Returns
        -------
        int Quadrature degree, maximum degree of polynomials integrated 
                                exactly by the quadrature rule).
    
        Notes
        -----
        - For the stiffness matrix, the integrand involves products of derivatives
          of shape functions:
            * Shape functions of degree p = u_deg
            * Their derivatives are of degree (p - 1)
            * Products of two derivatives are of degree (2p - 2)
          Therefore, an exact integration rule requires:
              quadrature_degree = 2 * u_deg - 2
    
        - Integration schemes:
            * 'default': exact integration with quadrature_degree = 2*u_deg - 2
            * 'reduit': reduced integration (quadrature_degree = 2*u_deg - 3),
                        often used to alleviate locking (e.g. shear locking)
            * 'over': over-integration (quadrature_degree = 2*u_deg),
                      sometimes used for stabilisation (e.g. to prevent hourglassing)
        """
        if self.schema == "default":
            return 2 * (self.u_deg - 1)
        elif self.schema == "reduit":
            return max(1, 2 * self.u_deg - 3)
        elif self.schema == "over":
            return 2 * self.u_deg 
            raise ValueError(f"Unknown integration schema {self.schema}")
    
    def create_metadata(self):
        quad_deg = self.default_quadrature_degree()
        return {"quadrature_rule": "default", "quadrature_degree": quad_deg}
                
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
        basix.ufl.Element Quadrature element with specified properties
        """
        q_deg = self.default_quadrature_degree()   
        if shape[0] == "Scalar":
            Qe = quad_el(self.ref_el, degree = q_deg, value_shape=())
        elif shape[0] == "Vector":
            Qe = quad_el(self.ref_el, degree = q_deg, value_shape=(shape[1],))
        elif shape[0] == "Tensor":
            Qe = quad_el(self.ref_el, degree = q_deg, value_shape=(shape[1], shape[2]))
        return Qe  
    
    def quadrature_space(self, shape):
        """
        Creates a function space using quadrature elements.
        
        Parameters
        ----------
        shape : list Shape specification for the quadrature element
            
        Returns
        -------
        dolfinx.fem.FunctionSpace Function space using the specified quadrature element
        """
        quad_el = self.quad_element(shape)
        if self.mesh_type == "dolfinx_mesh":
            return functionspace(self.mesh, quad_el)
        elif self.mesh_type == "ufl_mesh":
            return FunctionSpace(self.mesh, quad_el)