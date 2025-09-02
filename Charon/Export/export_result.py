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
Created on Fri Mar 11 09:28:55 2022

@author: bouteillerp
"""
from ..utils.parameters.default import default_post_processing_parameters
from .csv_export import OptimizedCSVExport

from dolfinx.io import XDMFFile, VTKFile
from os import remove, path
from mpi4py.MPI import COMM_WORLD
from dolfinx.fem import functionspace, Function, Expression
from ufl import as_vector



class ExportResults:
    def __init__(self, problem, output_file_name, dictionnaire, dictionnaire_csv):
        """
        Initialise l'export des résultats.

        Parameters
        ----------
        problem : Objet de la classe Problem, problème mécanique qui a été résolu.
        output_file_name : String, nom du dossier dans lequel sera stocké les résultats.
        dictionnaire : Dictionnaire, dictionnaire contenant le nom des variables
                        que l'on souhaite exporter au format XDMF (Paraview).
        dictionnaire_csv : Dictionnaire, dictionnaire contenant le nom des variables
                            que l'on souhaite exporter au format csv.
        """
        self.pb = problem
        self.is_plastic = self.pb.plastic_analysis
        self.is_damage = self.pb.damage_analysis
        self.quad = problem.quad
        self.dico = dictionnaire
        self.dico_csv = dictionnaire_csv
        self.param = default_post_processing_parameters()
        self.file_name = self.save_dir(output_file_name) + self.param["file_results"]
        self.model_meca = self.pb.name
        if self.param["writer"] == "xdmf":
            if path.isfile(self.file_name) and COMM_WORLD.rank == 0:
                remove(self.file_name)
                remove(self.file_name.replace(".xdmf", ".h5"))
                print("File has been found and deleted.")
            file_results = XDMFFile(self.pb.mesh.comm, self.file_name, "w")
            file_results.write_mesh(self.pb.mesh)
            self.file_results = XDMFFile(self.pb.mesh.comm, self.file_name, "a")
        elif self.param["writer"] == "VTK":
            if path.isfile(self.file_name) and COMM_WORLD.rank == 0 :
                remove(self.file_name)
                print("File has been found and deleted.")
            file_results = VTKFile(self.pb.mesh.comm, self.file_name, "w")
            file_results.write_mesh(self.pb.mesh)
            self.file_results = VTKFile(self.pb.mesh.comm, self.file_name, "a")
        
        self.setup_export_fields()
        self.csv = OptimizedCSVExport(self.save_dir(output_file_name), 
                                      problem, dictionnaire_csv, export = self)

    def save_dir(self, name):
        """
        Renvoie nom du dossier dans lequel sera stocké les résultats.
        Parameters
        ----------
        name : String, nom du dossier dans lequel sera stocké les résultats.
        """
        savedir = name + "-" + "results" + "/"
        return savedir
    
    def set_sig_element(self):
        if self.model_meca == "CartesianUD" :
            return self.quad.quad_element(["Scalar"])
        elif self.model_meca == "CylindricalUD":
            return self.quad.quad_element(["Vector", 2])
        elif self.model_meca == "SphericalUD":
            return self.quad.quad_element(["Vector", 3])
        elif self.model_meca == "PlaneStrain":
            return self.quad.quad_element(["Vector", 3])
        elif self.model_meca == "Axisymmetric":
            return self.quad.quad_element(["Vector", 4])
        else:
            return self.quad.quad_element(["Tensor", 3, 3])
        
    def set_s_element(self):
        if self.model_meca in["CartesianUD", "CylindricalUD", "SphericalUD"]:
            return self.quad.quad_element(["Vector", 3])
        elif self.model_meca in ["PlaneStrain", "Axisymmetric"]:
            return self.quad.quad_element(["Vector", 4])
        else:
            return self.quad.quad_element(["Tensor", 3, 3])
        
    def extract_deviatoric(self, deviatoric):
        if self.model_meca in ["CartesianUD", "CylindricalUD", "SphericalUD"]:
            return as_vector([deviatoric[0, 0], deviatoric[1, 1], deviatoric[2, 2]])
        elif self.model_meca == "PlaneStrain":
            return as_vector([deviatoric[0, 0], deviatoric[1, 1], deviatoric[2, 2], deviatoric[0, 1]])
        elif self.model_meca == "Axisymmetric":
            return self.pb.kinematic.tensor_3d_to_compact(deviatoric, symmetric = True)
        else:
            return deviatoric
            
    def setup_export_fields(self):
        """Configure tous les espaces et expressions nécessaires pour l'export"""
        def set_func_expr(field_name, expr, V):
            setattr(self, f"V_{field_name}", V)
            setattr(self, f"{field_name}", Function(V, name = field_name))
            setattr(self, f"{field_name}_expr", Expression(expr, V.element.interpolation_points()))

        # Contraintes
        if self.dico.get("sig") or self.dico_csv.get("sig"):
            sig_element = self.set_sig_element()
            set_func_expr("sig", self.pb.sig, functionspace(self.pb.mesh, sig_element))

        # Déviateur
        if self.dico.get("s") or self.dico_csv.get("s"):
            s_element = self.set_s_element()
            s_expr = self.extract_deviatoric(self.pb.constitutive.s)
            set_func_expr("s", s_expr, functionspace(self.pb.mesh, s_element))

        # Pression
        if self.dico.get("p") or self.dico_csv.get("p"):
            set_func_expr("p", self.pb.constitutive.p, self.pb.V_quad_UD)
            
        # Densité
        if self.dico.get("rho") or self.dico_csv.get("rho"):
            set_func_expr("rho", self.pb.rho, self.pb.V_quad_UD)
            
        if self.dico.get("J") or self.dico_csv.get("J"):
            set_func_expr("J", self.pb.J_transfo, self.pb.V_quad_UD)
        
        # Taux de déformation
        if self.dico.get("D"):
            D_element = self.set_sig_element()
            set_func_expr("D", self.pb.D, functionspace(self.pb.mesh, D_element))
        
        # Concentration
        if self.dico.get("c") is not None and hasattr(self.pb, 'multiphase'):
            V_c = self.pb.multiphase.V_c
            n_mat = len(self.pb.material)
            self.c_list = [Function(V_c, name=f"Concentration ( {i} )")for i in range(n_mat)]    
            self.c_expr_list= [Expression(self.pb.multiphase.c[i], V_c.element.interpolation_points())
                               for i in range(n_mat)]

    def export_results(self, t):
        """Exporte les résultats au format XDMF."""
        if self.dico == {}:
            return
        
        if self.dico.get("U"):
            self.file_results.write_function(self.pb.u, t)
            
        if self.dico.get("v"):
            self.file_results.write_function(self.pb.v, t)
            
        if self.dico.get("d") and self.is_damage:
            self.file_results.write_function(self.pb.constitutive.damage.d, t)
            
        if self.dico.get("eps_p") and self.is_plastic:
            self.file_results.write_function(self.pb.constitutive.plastic.eps_p, t)
            
        if self.dico.get("sig"):
            self.sig.interpolate(self.sig_expr)
            self.file_results.write_function(self.sig, t)
    
        if self.dico.get("p"):
            self.p.interpolate(self.p_expr)
            self.file_results.write_function(self.p, t)
    
        if self.dico.get("T"):
            self.file_results.write_function(self.pb.T, t)
            
        if self.dico.get("s"):
            self.s.interpolate(self.s_expr)
            self.file_results.write_function(self.s, t)