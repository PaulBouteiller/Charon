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
Mesh Manager Module
==================

This module provides a comprehensive set of tools for managing meshes and
integration domains in finite element simulations.

The MeshManager class encapsulates all mesh-related operations, including 
boundary marking, and the definition of integration measures. 
"""

from dolfinx.mesh import locate_entities_boundary, meshtags
from ufl import Measure
from numpy import hstack, argsort, finfo, full_like, array, zeros, where, unique
from dolfinx.fem import functionspace, Function
from .quadrature import Quadrature
from ..utils.default_parameters import default_fem_parameters

class MeshManager:
    """
    Manager for mesh operations and integration domains.
    
    This class encapsulates all operations related to the mesh,
    including the creation of facet submeshes, boundary marking,
    and definition of integration measures.
    
    Attributes
    ----------
    mesh : dolfinx.mesh.Mesh Main computational mesh
    name : str Problem type identifier (e.g., "Axisymmetric", "PlaneStrain")
    dim : int Topological dimension of the mesh
    fdim : int Dimension of facets (dim-1)
    h : dolfinx.fem.Function Function containing local mesh size at each element
    facet_tag : dolfinx.mesh.MeshTags, optional
        Tags identifying different regions of the boundary
    """
    def __init__(self, mesh, dictionnaire):
        """
        Initialize the mesh manager.
        
        Parameters
        ----------
        mesh : dolfinx.mesh.Mesh Main computational mesh
        name : str Problem type identifier (e.g., "Axisymmetric", "PlaneStrain")
        """
        # Initialize FEM parameters
        fem_parameters_dic = dictionnaire.get("fem_parameters", default_fem_parameters())
        self.u_deg = fem_parameters_dic["u_degree"]
        self.schema= fem_parameters_dic["schema"]
        self.mesh_type = self.set_mesh_type(mesh)
        # Initialize quadrature
        self.quad = Quadrature(mesh, self.mesh_type, self.u_deg, self.schema)
        self.mesh = mesh
        self.dim = self.quad.dim
        self.fdim = self.dim - 1
        if "tags" in dictionnaire and "coordinate" in dictionnaire and "positions" in dictionnaire:
            self.mark_boundary(dictionnaire["tags"], dictionnaire["coordinate"], dictionnaire["positions"])
        elif "facet_tag" in dictionnaire:
            self.facet_tag = dictionnaire["facet_tag"]
        else:
            print("Warning no boundary has been tagged inside CHARONX \
                  Boundary conditions cannot be used")
            if self.mesh_type == "dolfinx_mesh":
                self.facet_tag = meshtags(self.mesh, self.fdim, array([]), array([]))
            elif self.mesh_type == "ufl_mesh":
                self.facet_tag = None
        self.set_measures(self.quad)
        # self.cell_type = self.get_cell_type(self.mesh_type)
        self.cell_type = self.mesh.ufl_cell().cellname()
        
        self.cell_tags = dictionnaire.get("cell_tags", None)
        
    def set_mesh_type(self, mesh):
        import dolfinx
        import ufl
        if isinstance(mesh, dolfinx.mesh.Mesh):
            return "dolfinx_mesh"
        elif isinstance(mesh, ufl.Mesh):
            return "ufl_mesh"

    def mark_boundary(self, flag_list, coord_list, localisation_list, tol=finfo(float).eps):
        """
        Mark boundaries of the mesh.
        
        Identifies and tags facets on the boundary based on their coordinates,
        assigning numeric identifiers to different boundary regions.
        
        Parameters
        ----------
        flag_list : list of int Numeric identifiers for boundary regions
        coord_list : list of str Coordinate variables used to identify regions ('x', 'y', 'z', 'r')
        localisation_list : list of float oordinate values for boundary detection
        tol : float, optional Tolerance for coordinate matching, by default machine epsilon
            
        Notes
        -----
        This method creates mesh tags that can be used to identify boundary
        regions when imposing boundary conditions or computing boundary integrals.
        """
        # Get facets and their flags
        facets, full_flag = self._set_facet_flags(flag_list, coord_list, localisation_list, tol)
        
        # # Add custom facet flags if needed
        # facets, full_flag = self.set_custom_facet_flags(facets, full_flag)
        
        # Assemble and sort marked facets
        marked_facets = hstack(facets)
        marked_values = hstack(full_flag)
        sorted_facets = argsort(marked_facets)
        
        # Create mesh tags
        self.facet_tag = meshtags(self.mesh, self.fdim, 
                                  marked_facets[sorted_facets], 
                                  marked_values[sorted_facets])
        
    # def set_custom_facet_flags(self, facets, full_flag):
    #     """
    #     Add custom facet flags to the boundary marking.
        
    #     This method can be overridden in derived classes to add
    #     custom boundary markings beyond the standard coordinate-based approach.
        
    #     Parameters
    #     ----------
    #     facets : list of numpy.ndarray List of arrays containing facet indices
    #     full_flag : list of numpy.ndarray List of arrays containing corresponding flags
            
    #     Returns
    #     -------
    #     tuple of (list, list) Potentially modified (facets, full_flag) lists
    #     """
    #     return facets, full_flag
        
    def _set_facet_flags(self, flag_list, coord_list, localisation_list, tol):
        """
        Define facet flags based on coordinates and values.
        
        Internal method that identifies facets based on their spatial coordinates
        and assigns the corresponding flags.
        
        Parameters
        ----------
        flag_list : list of int Numeric identifiers for boundary regions
        coord_list : list of str Coordinate variables used to identify regions
        localisation_list : list of float Coordinate values for boundary detection
        tol : float Tolerance for coordinate matching
            
        Returns
        -------
        tuple of (list, list)
            Lists of facet indices and their corresponding flags
        """
        facets = []
        full_flag = []
        
        for flag, coord, loc in zip(flag_list, coord_list, localisation_list):
            # Filter function to identify facets
            def boundary_filter(x):
                return abs(x[self.index_coord(coord)] - loc) < tol
                
            # Locate matching entities
            found_facets = locate_entities_boundary(self.mesh, self.fdim, boundary_filter)
            facets.append(found_facets)
            full_flag.append(full_like(found_facets, flag))
            
        return facets, full_flag
        
    def index_coord(self, coord):
        """
        Return the index corresponding to the spatial variable.
        
        Maps coordinate names to their indices in the coordinate system.
        
        Parameters
        ----------
        coord : str Coordinate name: "x", "y", "z", or "r"
            
        Returns
        -------
        int Index of the coordinate in the coordinate system
            
        Raises
        ------
        ValueError If an invalid coordinate is provided
        """
        if coord in ("x", "r"):
            return 0
        elif coord == "y":
            return 1
        elif coord == "z":
            if self.dim == 2:  # Axisymmetric
                return 1
            else:
                return 2
        else:
            raise ValueError(f"Invalid coordinate: {coord}")
    
    def get_boundary_element_size(self, flag):
        """
        Calculate the areas of elements connected to facets with a given flag.
    
        Parameters
        ----------
        flag : int Flag identifying the facets of interest
    
        Returns
        -------
        tuple of (numpy.ndarray, numpy.ndarray)
            Arrays containing the areas and centroids of the elements
        """
        # Find facets marked with the flag
        marked_facets = where(self.facet_tag.values == flag)[0]
        facet_indices = self.facet_tag.indices[marked_facets]
    
        # Facet-to-cell connectivity
        facet_to_cell_map = self.mesh.topology.connectivity(1, 2)
    
        # Find cells connected to marked facets
        connected_cells = []
        for facet in facet_indices:
            if facet_to_cell_map.offsets[facet] != facet_to_cell_map.offsets[facet + 1]:
                connected_cells.extend(facet_to_cell_map.links(facet))
    
        connected_cells = unique(connected_cells)  # Remove duplicates
    
        # Calculate areas of connected cells
        areas = []
        centroids_x = []
        
        for cell in connected_cells:
            # Get vertices of the cell
            cell_vertices = self.mesh.geometry.x[self.mesh.topology.connectivity(2, 0).links(cell)]
            
            # Calculate area and centroid based on element type
            area, centroid_x = self._calculate_cell_properties(cell_vertices)
            
            areas.append(area)
            centroids_x.append(centroid_x)
    
        return array(areas), array(centroids_x)
    
    def _calculate_cell_properties(self, vertices):
        """
        Calculate the area and centroid of a cell.
        
        Parameters
        ----------
        vertices : numpy.ndarray Vertices of the cell
            
        Returns
        -------
        tuple of (float, float) Area and x-coordinate of the centroid
            
        Raises
        ------
        ValueError If the number of vertices is not supported
        """
        if len(vertices) == 3:  # Triangle
            x1, y1 = vertices[0][0], vertices[0][1]
            x2, y2 = vertices[1][0], vertices[1][1]
            x3, y3 = vertices[2][0], vertices[2][1]
            
            # Calculate area of the triangle
            area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
            centroid_x = (x1 + x2 + x3) / 3
            
        elif len(vertices) == 4:  # Quadrilateral
            x1, y1 = vertices[0][0], vertices[0][1]
            x2, y2 = vertices[1][0], vertices[1][1]
            x3, y3 = vertices[2][0], vertices[2][1]
            x4, y4 = vertices[3][0], vertices[3][1]
            
            # Calculate area of the quadrilateral (sum of two triangles)
            area = (0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) +
                    0.5 * abs(x1 * (y3 - y4) + x3 * (y4 - y1) + x4 * (y1 - y3)))
            centroid_x = (x1 + x2 + x3 + x4) / 4
            
        else:
            raise ValueError(f"Unsupported number of vertices: {len(vertices)}")
            
        return area, centroid_x
    
    def set_measures(self, quadrature):
        """
        Define integration measures for the problem.
        
        Creates the volume and surface integration measures with
        appropriate quadrature rules for accurate integration.
        
        Parameters
        ----------
        quadrature : Quadrature Quadrature scheme to use for integration
        """
        # Define integration measures
        self.dx = Measure("dx", domain = self.mesh, metadata = quadrature.metadata)
        self.dx_l = Measure("dx", domain = self.mesh, metadata = quadrature.lumped_metadata)
        self.ds = Measure('ds')(subdomain_data = self.facet_tag)
        
    def calculate_mesh_size(self, mesh, dim):
        """
        Calculate the local size of mesh elements.
        
        Creates a function containing the characteristic size of each
        element in the mesh, used for error estimation and stability parameters.
        
        Returns
        -------
        dolfinx.fem.Function Function containing the local mesh size at each element
        """
        # Create function space to store the size
        h_loc = Function(functionspace(mesh, ("DG", 0)), name="MeshSize")
        
        # Calculate size for each cell
        num_cells = mesh.topology.index_map(dim).size_local
        h_local = zeros(num_cells)
        
        for i in range(num_cells):
            h_local[i] = mesh.h(dim, array([i]))
        
        # Assign calculated values
        h_loc.x.array[:] = h_local
        
        return h_loc