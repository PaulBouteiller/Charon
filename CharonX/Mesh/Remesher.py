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
Created on Wed Feb 28 16:18:27 2024

@author: bouteillerp
"""
import meshio
from mpi4py import MPI
import gmsh
from dolfinx.io import XDMFFile, gmshio
from dolfinx.io.gmshio import model_to_mesh
from dolfinx.mesh import locate_entities_boundary, meshtags
from dolfinx.fem import create_nonmatching_meshes_interpolation_data
from dolfinx.cpp.mesh import entities_to_geometry
from dolfinx.cpp.io import perm_gmsh
from numpy import int32, arange, isnan, asarray
from dolfinx.mesh import create_unit_square, CellType


def init_gmsh():
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 0)  # to disable meshing info
    gdim = 2
    rank = 0
    gmsh.model.add("Model")
    geom = gmsh.model.geo
    return gdim, rank, geom

def return_mesh(model, comm, rank, gdim, quad = True, degree = 1):
    gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    if quad:
        gmsh.option.setNumber("Mesh.Algorithm", 8)
        gmsh.option.setNumber("Mesh.RecombineAll", 1)
        gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    gmsh.model.mesh.generate(gdim)
    gmsh.model.mesh.setOrder(degree)
    gmsh.model.mesh.optimize("Netgen")

    domain, meshtags, facets = model_to_mesh(model, comm, rank, gdim=gdim)
    gmsh.finalize()
    with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "w") as infile:
        infile.write_mesh(domain)
    return domain, meshtags, facets

def dolfinx_mesh_to_msh_with_meshio(input_mesh):
    name = "old_mesh"
    ufl_celltype = input_mesh.ufl_cell().cellname()
    if ufl_celltype =="triangle":
        raise ValueError("Not done yet")
        celltype = "triangle"
        reshape_size = 3
    elif ufl_celltype =="quadrilateral":
        celltype = "quad"
        reshape_size = 4

    cells = input_mesh.topology.connectivity(input_mesh.topology.dim, 0).array.reshape((-1, reshape_size))
    gdim = input_mesh.topology.dim
    num_cells_owned_by_proc = input_mesh.topology.index_map(gdim).size_local
    cell_entitites = entities_to_geometry(input_mesh._cpp_object, gdim, arange(num_cells_owned_by_proc, dtype = int32), False)
    permutation = asarray(perm_gmsh(input_mesh.topology.cell_types[0], cell_entitites.shape[1]), dtype = int32)
    
    cell_entitites = cell_entitites[:, permutation]
    cell_type = meshio.CellBlock("quad", cell_entitites)
   
    points = input_mesh.geometry.x
    mio_mesh = meshio.Mesh(points, [cell_type])
    meshio.write(name, mio_mesh, file_format="gmsh22")
    return name

def improved_gmsh_mesh_to_dolfin_mesh(input_mesh_name):
    gdim, model_rank, geom = init_gmsh()
    gmsh.open(input_mesh_name)
    gmsh.model.mesh.generate(2)
    
    tags_entites = [entite[1] for entite in gmsh.model.getEntities(gdim)]  # Récupère uniquement les tags des entités
    tag_groupe_physique = gmsh.model.addPhysicalGroup(gdim, tags_entites)
    gmsh.model.setPhysicalName(gdim, tag_groupe_physique, "Surface")
    
    # gmsh.fltk.run()
    # gmsh.model.mesh.optimize("Netgen")
    # gmsh.model.mesh.optimize("Laplace2D")
    # tags_surface = [entite[1] for entite in gmsh.model.getEntities(gdim)]
    # tag_groupe_physique = gmsh.model.addPhysicalGroup(gdim, 1, 1)
    # gmsh.model.mesh.set_smoothing(2, tag_groupe_physique, 10)
    # gmsh.model.mesh.
    
    # gmsh.option.setNumber("Mesh.RecombinationAlgorithm", 2)
    # gmsh.option.setNumber("Mesh.Algorithm", 8)
    # gmsh.option.setNumber("Mesh.RecombineAll", 1)
    # gmsh.option.setNumber("Mesh.SubdivisionAlgorithm", 1)
    gmsh.model.mesh.optimize("Relocate2D") 
    gmsh.model.mesh.refine()


    gmsh.model.mesh.generate(2)
    gmsh.fltk.run()
    domain, meshtags, facets = model_to_mesh(gmsh.model, MPI.COMM_WORLD, model_rank, gdim)
    gmsh.finalize()
    return domain, meshtags, facets

def improve_mesh(domain):
    old_mesh_name = dolfinx_mesh_to_msh_with_meshio(domain)
    return improved_gmsh_mesh_to_dolfin_mesh(old_mesh_name)

# def update_mesh(mesh):
#     old_mesh_name = dolfinx_mesh_to_msh(mesh)
#     new_mesh_name = refine_gmsh_mesh(old_mesh_name)
#     return mesh_to_dolfinx_mesh(new_mesh_name)

# Créer un maillage carré unitaire avec DOLFINx
# mesh = create_unit_square(MPI.COMM_WORLD, 10, 10, CellType.quadrilateral)
# improve_mesh(mesh)

def non_matching_interpolation(old_function, new_function, tol = 1e-9):
    interpolation_data = create_nonmatching_meshes_interpolation_data(
            new_function.function_space.mesh._cpp_object,
            new_function.function_space.element,
            old_function.function_space.mesh._cpp_object, tol)
    new_function.interpolate(old_function, nmm_interpolation_data=interpolation_data)
    
def deform_mesh(mesh, displacement_array):
    deformation_array = displacement_array.x.array.reshape((-1, mesh.geometry.dim))
    mesh.geometry.x[:, :mesh.geometry.dim] += deformation_array