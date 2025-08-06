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
from ..utils.default_parameters import default_post_processing_parameters

from dolfinx.io import XDMFFile, VTKFile
from os import remove, path
from dolfinx.fem import Function, Expression
from mpi4py.MPI import COMM_WORLD
from dolfinx.fem import functionspace
from ufl import as_vector

from .csv_export2 import OptimizedCSVExport

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

            
        # Centraliser la création des espaces et expressions
        self.export_spaces = {}
        self.export_expressions = {}
        self.export_functions = {}
        
        self.setup_export_fields()
        self.csv = OptimizedCSVExport(
            self.save_dir(output_file_name), 
            problem, 
            dictionnaire_csv,
            export_context=self  # Passer la référence
        )

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
        
    def set_devia_element(self):
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
        # Contraintes
        if self.dico.get("Sig") or self.dico_csv.get("Sig"):
            sig_element = self.set_sig_element()
            self.export_spaces['V_Sig'] = functionspace(self.pb.mesh, sig_element)
            self.export_expressions['sig'] = Expression(
                self.pb.sig, 
                self.export_spaces['V_Sig'].element.interpolation_points()
            )
            self.export_functions['sig_func'] = Function(
                self.export_spaces['V_Sig'], 
                name="Stress"
            )
        
        # Déviateur
        if self.dico.get("deviateur") or self.dico_csv.get("deviateur"):
            devia_element = self.set_devia_element()
            self.export_spaces['V_devia'] = functionspace(self.pb.mesh, devia_element)
            s_expr = self.extract_deviatoric(self.pb.constitutive.s)
            self.export_expressions['s'] = Expression(
                s_expr, 
                self.export_spaces['V_devia'].element.interpolation_points()
            )
            self.export_functions['s_func'] = Function(
                self.export_spaces['V_devia'], 
                name="Deviateur"
            )
        
        # Pression
        if self.dico.get("Pressure") or self.dico_csv.get("Pressure"):
            self.export_expressions['p'] = Expression(
                self.pb.constitutive.p, 
                self.pb.V_quad_UD.element.interpolation_points()
            )
            self.export_functions['p_func'] = Function(
                self.pb.V_quad_UD, 
                name="Pression"
            )
        
    def set_expression(self):      
        def get_index(key, length):
            if key == "all" or key:
                return [i for i in range(length)]
            elif isinstance(key, list):
                return key
        if self.dico.get("Sig"):
            self.pb.sig_expr = Expression(self.pb.sig, self.pb.V_Sig.element.interpolation_points())
            self.pb.sig_func = Function(self.pb.V_Sig, name="Stress")

        if self.dico.get("deviateur"):  

            # s_expr = self.kinematic.tensor_3d_to_compact(self.constitutive.s)
            s_expr = self.extract_deviatoric(self.pb.constitutive.s)
            # self.sig_VM = Expression(sqrt(3./2 * inner(s_expr, s_expr)), self.V_quad_UD.element.interpolation_points())
            # self.sig_VM_func = Function(self.V_quad_UD, name = "VonMises") 
            self.s_expr = Expression(s_expr, self.pb.V_devia.element.interpolation_points())
            self.s_func = Function(self.pb.V_devia, name = "Deviateur")
            
        if self.dico.get("Pressure"):  
            self.p_expr = Expression(self.pb.constitutive.p, self.pb.V_quad_UD.element.interpolation_points())
            self.p_func = Function(self.pb.V_quad_UD, name="Pression")
            
        if self.dico.get("D"):
            self.D_func = Function(self.pb.V_Sig, name = "Taux déformation")
            self.D_expression = Expression(self.pb.D, self.pb.V_Sig.element.interpolation_points())
            
        if self.dico.get("c")!=None:
            self.c_index_list = get_index(self.dico.get("c"), len(self.pb.material))
            self.c_func_list = [Function(self.pb.multiphase.V_c, name = "Concentration ( %i)" % (i)) for i in self.c_index_list]
            self.c_expression_list = [Expression(self.pb.multiphase.c[i], self.pb.multiphase.V_c.element.interpolation_points()) for i in self.c_index_list]

    def export_results(self, t):
        """Exporte les résultats au format XDMF."""
        if self.dico == {}:
            return
        
        if self.dico.get("U"):
            self.file_results.write_function(self.pb.u, t)
            
        if self.dico.get("v"):
            self.file_results.write_function(self.pb.v, t)
            
        if self.dico.get("d") and hasattr(self.pb.constitutive, 'damage') and self.pb.constitutive.damage:
            self.file_results.write_function(self.pb.constitutive.damage.d, t)
            
        if self.dico.get("eps_p") and hasattr(self.pb.constitutive, 'plastic') and self.pb.constitutive.plastic:
            self.file_results.write_function(self.pb.constitutive.plastic.eps_p, t)
            
        if self.dico.get("Sig"):
            sig_func = self.export_functions.get('sig_func')
            sig_expr = self.export_expressions.get('sig')
            if sig_func and sig_expr:
                sig_func.interpolate(sig_expr)
                self.file_results.write_function(sig_func, t)
    
        if self.dico.get("Pressure"):
            p_func = self.export_functions.get('p_func')
            p_expr = self.export_expressions.get('p')
            if p_func and p_expr:
                p_func.interpolate(p_expr)
                self.file_results.write_function(p_func, t)
    
        if self.dico.get("T"):
            self.file_results.write_function(self.pb.T, t)
            
        if self.dico.get("deviateur"):
            s_func = self.export_functions.get('s_func')
            s_expr = self.export_expressions.get('s')
            if s_func and s_expr:
                s_func.interpolate(s_expr)
                self.file_results.write_function(s_func, t)