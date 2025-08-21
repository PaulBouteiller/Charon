"""
MeshManager Template - Complete Configuration Options
====================================================
This template shows all available mesh configuration options and required parameters
for the Charon finite element framework MeshManager class.

Created on: [DATE]
@author: [YOUR_NAME]
"""

from Charon import MeshManager
from Charon.Mesh.gmsh_mesh import create_1D_meshn create_2D_rectangle
from dolfinx.mesh import create_interval, create_rectangle, create_box, CellType
from mpi4py.MPI import COMM_WORLD
import numpy as np

# =============================================================================
# MESH CREATION OPTIONS
# =============================================================================

# Choose ONE of the following mesh creation methods:

# -----------------------------------------------------------------------------
# 1. SIMPLE 1D MESHES (Intervals)
# -----------------------------------------------------------------------------
# Create 1D interval mesh
mesh_1d = create_interval(COMM_WORLD, 100, [np.array(0.0), np.array(1.0)])

# Alternative using Charon function
mesh_1d_charon = create_1D_mesh(0.0, 1.0, 100)

# -----------------------------------------------------------------------------
# 2. SIMPLE 2D MESHES (Rectangles)
# -----------------------------------------------------------------------------
# Basic rectangle mesh with triangular elements
mesh_2d_tri = create_rectangle(COMM_WORLD, [(0.0, 0.0), (1.0, 0.5)], [20, 10], CellType.triangle)

# Rectangle mesh with quadrilateral elements
mesh_2d_quad = create_rectangle(COMM_WORLD, [(0.0, 0.0), (1.0, 0.5)], [20, 10], CellType.quadrilateral)

# Alternative using Charon function
mesh_2d_charon = create_2D_rectangle(0.0, 0.0, 1.0, 0.5, 20, 10)

# -----------------------------------------------------------------------------
# 3. SIMPLE 3D MESHES (Boxes)
# -----------------------------------------------------------------------------
# Basic box mesh with tetrahedral elements
mesh_3d_tet = create_box(COMM_WORLD, [np.array([0, 0, 0]), np.array([1, 0.5, 0.3])], 
                         [20, 10, 6], CellType.tetrahedron)

# Box mesh with hexahedral elements
mesh_3d_hex = create_box(COMM_WORLD, [np.array([0, 0, 0]), np.array([1, 0.5, 0.3])], 
                         [20, 10, 6], CellType.hexahedron)

# -----------------------------------------------------------------------------
# 4. SPECIALIZED GMSH MESHES
# -----------------------------------------------------------------------------

# Axisymmetric sphere (hollow sphere mesh)
mesh_axi_sphere, _, facets_sphere = axi_sphere(
    Ri=5.0,        # Inner radius
    Re=10.0,       # Outer radius  
    N_theta=40,    # Circumferential elements
    N_radius=10,   # Radial elements
    tol_dyn=1e-5,  # Tolerance for axis
    quad=True,     # Use quadrilateral elements
    write=False    # Don't write mesh file
)

# Double shell axisymmetric mesh
mesh_double_shell, _, facets_double = axi_double_coquille(
    Ri=5.0,        # Inner radius
    Re=15.0,       # Outer radius
    R_mid=10.0,    # Mid radius
    N_theta=40,    # Circumferential elements
    Nr_int=5,      # Inner radial elements
    Nr_out=8,      # Outer radial elements
    tol_dyn=1e-5,
    quad=True
)

# Perforated plate
mesh_perforated, _, facets_perforated = generate_perforated_plate(
    W=10.0,        # Width
    H=5.0,         # Height
    R=1.0,         # Hole radius
    mesh_size=0.1  # Mesh size
)

# Quarter perforated plate (for symmetry)
mesh_quarter_perforated, _, facets_quarter = quarter_perforated_plate(
    width=5.0,     # Width
    heigth=5.0,    # Height
    radius=1.0,    # Hole radius
    hsize=0.1,     # Mesh size
    quad=True      # Use quadrilateral elements
)

# Double rectangle (bi-material)
mesh_double_rect, _, facets_double_rect = double_rectangle(
    xbl=0.0,       # Bottom left x
    ybl=0.0,       # Bottom left y
    h1=2.0,        # First region height
    h2=3.0,        # Second region height
    L=10.0,        # Length
    Nx=20,         # Elements along length
    Nh1=8,         # Elements in first region
    Nh2=12,        # Elements in second region
    quad=True
)

# Broken unit square (crack simulation)
mesh_broken_square, _, facets_broken = broken_unit_square(
    N_largeur=20,  # Elements along width
    Nh=20,         # Elements along height
    Largeur=1.0,   # Domain width
    tol=1e-3,      # Crack tolerance
    raff=2,        # Refinement ratio (0=no refinement)
    quad=True
)

# Hotspot plate (thermal concentration)
mesh_hotspot, _, facets_hotspot = hotspot_plate(
    W=10.0,        # Width
    H=5.0,         # Height
    R=1.0,         # Hotspot radius
    mesh_size=[0.05, 0.2]  # [fine, coarse] mesh sizes
)

# =============================================================================
# MESH MANAGER CONFIGURATION OPTIONS
# =============================================================================

# Choose ONE of the following MeshManager configuration methods:

# -----------------------------------------------------------------------------
# 1. SIMPLE BOUNDARY MARKING (Coordinate-based)
# -----------------------------------------------------------------------------

# Example 1: 1D interval with two boundaries
dictionnaire_1d_simple = {
    "tags": [1, 2],                    # Boundary flags
    "coordinate": ["x", "x"],          # Coordinate directions  
    "positions": [0.0, 1.0]           # Coordinate values
}
mesh_manager_1d = MeshManager(mesh_1d, dictionnaire_1d_simple)

# Example 2: 2D rectangle with four boundaries
dictionnaire_2d_simple = {
    "tags": [1, 2, 3, 4],                           # Boundary flags
    "coordinate": ["x", "x", "y", "y"],             # Coordinate directions
    "positions": [0.0, 1.0, 0.0, 0.5]              # Coordinate values
}
mesh_manager_2d = MeshManager(mesh_2d_quad, dictionnaire_2d_simple)

# Example 3: 3D box with six boundaries
dictionnaire_3d_simple = {
    "tags": [1, 2, 3, 4, 5, 6],                           # Boundary flags
    "coordinate": ["x", "x", "y", "y", "z", "z"],         # Coordinate directions
    "positions": [0.0, 1.0, 0.0, 0.5, 0.0, 0.3]          # Coordinate values
}
mesh_manager_3d = MeshManager(mesh_3d_hex, dictionnaire_3d_simple)

# Example 4: Axisymmetric mesh (cylindrical/spherical coordinates)
dictionnaire_axi = {
    "tags": [1, 2],                    # Boundary flags
    "coordinate": ["r", "r"],          # Radial coordinates
    "positions": [5.0, 10.0]          # Inner and outer radii
}
mesh_manager_axi = MeshManager(mesh_1d, dictionnaire_axi)

# -----------------------------------------------------------------------------
# 2. GMSH-GENERATED FACET TAGS
# -----------------------------------------------------------------------------

# When using GMSH meshes with pre-defined boundary tags
dictionnaire_gmsh_facets = {
    "facet_tag": facets_sphere  # Use facet tags from GMSH generation
}
mesh_manager_gmsh = MeshManager(mesh_axi_sphere, dictionnaire_gmsh_facets)

# -----------------------------------------------------------------------------
# 3. ADVANCED FEM PARAMETERS
# -----------------------------------------------------------------------------

# Custom FEM parameters for advanced control
dictionnaire_advanced_fem = {
    "tags": [1, 2, 3, 4],
    "coordinate": ["x", "x", "y", "y"],
    "positions": [0.0, 1.0, 0.0, 0.5],
    "fem_parameters": {
        "u_degree": 2,           # Displacement field polynomial degree (1 or 2)
        "schema": "reduit"       # Integration scheme: "default", "over", "reduit"
    }
}
mesh_manager_advanced = MeshManager(mesh_2d_quad, dictionnaire_advanced_fem)

# -----------------------------------------------------------------------------
# 4. NO BOUNDARY CONDITIONS (Warning case)
# -----------------------------------------------------------------------------

# Empty dictionary - will generate warning about no boundary conditions
dictionnaire_empty = {}
mesh_manager_no_bc = MeshManager(mesh_2d_quad, dictionnaire_empty)

# =============================================================================
# COMPLETE EXAMPLES BY PROBLEM TYPE
# =============================================================================

# -----------------------------------------------------------------------------
# EXAMPLE 1: 1D Cartesian (Bar/Rod)
# -----------------------------------------------------------------------------
mesh_1d_bar = create_1D_mesh(0.0, 50.0, 100)
dictionnaire_1d_bar = {
    "tags": [1, 2],
    "coordinate": ["x", "x"],
    "positions": [0.0, 50.0],
    "fem_parameters": {"u_degree": 1, "schema": "default"}
}
mesh_manager_1d_bar = MeshManager(mesh_1d_bar, dictionnaire_1d_bar)

# -----------------------------------------------------------------------------
# EXAMPLE 2: 1D Cylindrical (Hollow cylinder)
# -----------------------------------------------------------------------------
R_int, R_ext = 5.0, 10.0
mesh_1d_cyl = create_1D_mesh(R_int, R_ext, 50)
dictionnaire_1d_cyl = {
    "tags": [1, 2],
    "coordinate": ["r", "r"], 
    "positions": [R_int, R_ext]
}
mesh_manager_1d_cyl = MeshManager(mesh_1d_cyl, dictionnaire_1d_cyl)

# -----------------------------------------------------------------------------
# EXAMPLE 3: 1D Spherical (Hollow sphere)
# -----------------------------------------------------------------------------
mesh_1d_sph = create_1D_mesh(R_int, R_ext, 50)
dictionnaire_1d_sph = {
    "tags": [1, 2],
    "coordinate": ["r", "r"],
    "positions": [R_int, R_ext]
}
mesh_manager_1d_sph = MeshManager(mesh_1d_sph, dictionnaire_1d_sph)

# -----------------------------------------------------------------------------
# EXAMPLE 4: 2D Plane Strain
# -----------------------------------------------------------------------------
mesh_2d_plane = create_rectangle(COMM_WORLD, [(0.0, 0.0), (10.0, 5.0)], 
                                 [40, 20], CellType.quadrilateral)
dictionnaire_2d_plane = {
    "tags": [1, 2, 3, 4],
    "coordinate": ["x", "x", "y", "y"],
    "positions": [0.0, 10.0, 0.0, 5.0],
    "fem_parameters": {"u_degree": 2, "schema": "reduit"}  # Avoid shear locking
}
mesh_manager_2d_plane = MeshManager(mesh_2d_plane, dictionnaire_2d_plane)

# -----------------------------------------------------------------------------
# EXAMPLE 5: 2D Axisymmetric (using GMSH sphere)
# -----------------------------------------------------------------------------
mesh_axi, _, facets_axi = axi_sphere(5.0, 10.0, 40, 20, tol_dyn=1e-5)
dictionnaire_axi_gmsh = {
    "facet_tag": facets_axi,
    "fem_parameters": {"u_degree": 2, "schema": "default"}
}
mesh_manager_axi_gmsh = MeshManager(mesh_axi, dictionnaire_axi_gmsh)

# -----------------------------------------------------------------------------
# EXAMPLE 6: 3D Tridimensional
# -----------------------------------------------------------------------------
mesh_3d = create_box(COMM_WORLD, [np.array([0, 0, 0]), np.array([50, 4, 4])], 
                     [100, 8, 8], CellType.hexahedron)
dictionnaire_3d = {
    "tags": [1, 2, 3, 4, 5, 6],
    "coordinate": ["x", "x", "y", "y", "z", "z"],
    "positions": [0, 50, 0, 4, 0, 4],
    "fem_parameters": {"u_degree": 1, "schema": "over"}  # Prevent hourglass
}
mesh_manager_3d = MeshManager(mesh_3d, dictionnaire_3d)

# -----------------------------------------------------------------------------
# EXAMPLE 7: PERFORATED PLATE CONFIGURATION
# -----------------------------------------------------------------------------
mesh_perf, _, facets_perf = generate_perforated_plate(20.0, 10.0, 2.0, 0.2)
dictionnaire_perf = {
    "facet_tag": facets_perf,
    "fem_parameters": {"u_degree": 2, "schema": "default"}
}
mesh_manager_perf = MeshManager(mesh_perf, dictionnaire_perf)

# -----------------------------------------------------------------------------
# EXAMPLE 8: CRACK/FRACTURE CONFIGURATION
# -----------------------------------------------------------------------------
mesh_crack, _, facets_crack = broken_unit_square(40, 40, 1.0, tol=1e-3, raff=3)
dictionnaire_crack = {
    "facet_tag": facets_crack,
    "fem_parameters": {"u_degree": 1, "schema": "default"}
}
mesh_manager_crack = MeshManager(mesh_crack, dictionnaire_crack)

# =============================================================================
# PARAMETER REFERENCE GUIDE
# =============================================================================

"""
PARAMETER REFERENCE:

1. BASIC PARAMETERS:
   - tags: List of integer flags for boundary identification
   - coordinate: List of coordinate names ("x", "y", "z", "r")
   - positions: List of coordinate values for boundary location

2. FEM PARAMETERS:
   - u_degree: Polynomial degree for displacement field
     * 1: Linear elements (P1)
     * 2: Quadratic elements (P2)
   
   - schema: Integration scheme
     * "default": Standard integration
     * "over": Over-integration (prevents hourglass in linear elements)
     * "reduit": Under-integration (prevents shear-locking in quadratic elements)

3. MESH TYPES:
   - Interval: 1D problems (bars, rods)
   - Triangle/Quadrilateral: 2D problems (plane strain, axisymmetric)
   - Tetrahedron/Hexahedron: 3D problems

4. COORDINATE SYSTEMS:
   - Cartesian: x, y, z
   - Cylindrical: r, θ, z (r for radial in 1D/2D)
   - Spherical: r, θ, φ (r for radial in 1D)

5. BOUNDARY IDENTIFICATION:
   Method 1 (Coordinate-based):
   - Use tags, coordinate, positions for simple geometries
   
   Method 2 (GMSH facet tags):
   - Use facet_tag for complex geometries generated with GMSH

INTEGRATION SCHEME RECOMMENDATIONS:

Linear Elements (u_degree=1):
- Use "default" for most cases
- Use "over" for quadrilateral/hexahedral meshes to prevent hourglass modes

Quadratic Elements (u_degree=2):
- Use "default" for most cases  
- Use "reduit" for plane strain/3D problems to prevent shear-locking

TYPICAL COMBINATIONS BY PROBLEM TYPE:

1D Problems:
- Cartesian: Bar, beam problems
- Cylindrical: Hollow cylinder under pressure
- Spherical: Hollow sphere under pressure

2D Problems:
- Plane Strain: Thick structures, out-of-plane constraint
- Axisymmetric: Rotational symmetry (cylinders, spheres)

3D Problems:
- Tridimensional: General 3D structures
- Complex geometries: Use GMSH for mesh generation

BOUNDARY FLAG CONVENTIONS:
- Use consecutive integers starting from 1
- Common patterns:
  * 1D: [1, 2] for left/right boundaries
  * 2D: [1, 2, 3, 4] for left/right/bottom/top
  * 3D: [1, 2, 3, 4, 5, 6] for x-/x+/y-/y+/z-/z+
  * Axisymmetric: Custom flags for inner/outer/axis boundaries
"""