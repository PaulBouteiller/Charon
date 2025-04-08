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
Created on Tue Apr  9 09:30:38 2024

@author: bouteillerp
"""
from .Solve import Solve
from petsc4py import PETSc
from math import isfinite
from numpy import int32, arange, isnan, asarray,  mod
from dolfinx.fem import Function
# import sys
# sys.path.append('../')  # Ajoute le chemin du dossier Mesh au chemin de recherche
# print(sys.path)
# from Gmsh_mesh import * 





import importlib.util




# import meshio
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
    with XDMFFile(MPI.COMM_WORLD, "new_mesh.xdmf", "w") as infile:
        infile.write_mesh(domain)
    return domain, meshtags, facets

def dolfinx_mesh_to_msh_with_meshio(input_mesh):
    with XDMFFile(MPI.COMM_WORLD, "mesh.xdmf", "w") as infile:
        infile.write_mesh(input_mesh)
    
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
    # gmsh.model.mesh.generate(2)
    
    tags_entites = [entite[1] for entite in gmsh.model.getEntities(gdim)]  # Récupère uniquement les tags des entités
    tag_groupe_physique = gmsh.model.addPhysicalGroup(gdim, tags_entites)
    gmsh.model.setPhysicalName(gdim, tag_groupe_physique, "Surface")
    # gmsh.model.mesh.optimize("Relocate2D") 
    gmsh.model.mesh.refine()

    return return_mesh(gmsh.model, MPI.COMM_WORLD, model_rank, gdim)
    # gmsh.model.mesh.generate(2)
    # gmsh.fltk.run()
    # domain, meshtags, facets = model_to_mesh(gmsh.model, MPI.COMM_WORLD, model_rank, gdim)
    # gmsh.finalize()
    # return domain, meshtags, facets

def improve_mesh(domain):
    old_mesh_name = dolfinx_mesh_to_msh_with_meshio(domain)
    return improved_gmsh_mesh_to_dolfin_mesh(old_mesh_name)

def non_matching_interpolation(old_function, new_function, interpolation_data):
    new_function.interpolate(old_function, nmm_interpolation_data = interpolation_data)
    
# def deform_mesh(mesh, displacement_func):
#     deformation_array = displacement_func.x.array.reshape((-1, mesh.geometry.dim))
#     mesh.geometry.x[:, :mesh.geometry.dim] += deformation_array
    
def deform_mesh(mesh, displacement_array):
    deformation_array = displacement_array.reshape((-1, mesh.geometry.dim))
    mesh.geometry.x[:, :mesh.geometry.dim] += deformation_array

def nm_interpolation_data(old_function_space, new_function_space, tol = 1e-9):
    return create_nonmatching_meshes_interpolation_data(new_function_space.mesh._cpp_object,
                                                        new_function_space.element, 
                                                        old_function_space.mesh._cpp_object, tol)








def load_script(path_to_script):
    # Déterminer le nom du module à partir du chemin
    module_name = "mon_module_dynamique"
    
    # Charger le module à partir du chemin spécifié
    spec = importlib.util.spec_from_file_location(module_name, path_to_script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Vous pouvez maintenant accéder aux objets du module chargé
    # Par exemple, module.ma_fonction()
    return module







class RestartSolve(Solve):
    def __init__(self, problem, path_name, **kwargs):        
        Solve.__init__(self, problem, restart = True, **kwargs)
        self.kwargs = kwargs
        print("Les mots clés sont", self.kwargs)
        self.set_store_solution()
        self.path_name = path_name
        self.iterative_solve_with_restart(**kwargs)

        
    def iterative_solve_with_restart(self, **kwargs):
        """
        Boucle temporelle, à chaque pas les différents solveurs 
        (déplacement, énergie...) sont successivement appelés        

        Parameters
        ----------
        **kwargs : Int, si un compteur donné par un int est spécifié
                        l'export aura lieu tout les compteur-pas de temps.
        """
        compteur_output = kwargs.get("compteur", 1)
        checkpoint_count = kwargs.get("checkpoint", 100)
        num_time_steps = self.time_stepping.num_time_steps
        self.compteur_reset = 0
        if compteur_output !=1:
            self.is_compteur = True
            self.compteur=0 
        else:
            self.is_compteur = False            
        j=0
        msg = "Time step {{:{l}d}}/{{:{l}d}}: {{:6f}}".format(l = len(str(num_time_steps)))
        while self.t < self.Tfin:
            self.update_time(j)
            self.update_bcs(j)
            self.problem_solve()
            self.check_convergence()
            if mod(j, compteur_output) == 0:
                print(msg.format(j+1, num_time_steps, self.t))
            if mod(j, checkpoint_count) == 0:
                self.store_solution(self.t)
            j+=1
            self.output(compteur_output)
        self.pb.final_output()
        self.export.final_csv_export()
        
    def check_convergence(self):
        convergence = self.NaN_checking_PETSc(self.pb.u.vector)
        if convergence:
            self.reset_problem(self.pb.mesh)
            raise ValueError("Divergence detected, restart from previous checkpoint")
        # print("Convergence")
        
    def reset_problem(self, mesh):
        deform_mesh(mesh, self.u_checkpoint)
        old_function_space = self.pb.V.clone()
        old_function = Function(old_function_space)
        new_mesh, _ , _= improve_mesh(self.pb.mesh)
        script = load_script(self.path_name)
        problem = script.set_problem(new_mesh)

        new_function_space = problem.V.clone()
        self.nm_interpolation_V = nm_interpolation_data(old_function_space, new_function_space, tol = 1e-9)
        old_function.x.array[:] = self.u_checkpoint
        problem.u.interpolate(old_function, nmm_interpolation_data = self.nm_interpolation_V)
        old_function.x.array[:] = self.v_checkpoint
        problem.v.interpolate(old_function, nmm_interpolation_data = self.nm_interpolation_V)

        self.kwargs.update({"TInit" : self.T_checkpoint})
        self.kwargs.update({"Prefix" : problem.prefix()+str(self.compteur_reset)})
        print("Les mots clés sont", self.kwargs)
        Solve.__init__(self, problem, restart = True, **self.kwargs)
        self.compteur_reset+=1
        self.t = self.T_checkpoint
        self.iterative_solve_with_restart(**self.kwargs)
        
        
        
    
    def set_store_solution(self):
        self.u_checkpoint = self.pb.u.x.array.copy()
        self.v_checkpoint = self.pb.v.x.array.copy()
    
    def store_solution(self, t):
        self.T_checkpoint = t
        self.u_checkpoint = self.pb.u.x.array.copy()
        self.v_checkpoint = self.pb.v.x.array.copy()
        
    def NaN_checking_PETSc(self, PETScVec):
        norm = PETScVec.norm(PETSc.NormType.NORM_2)
        return not isfinite(norm)
    
    def Nan_checking_numpy(self, NumpyArray):
        return isnan(NumpyArray).any()