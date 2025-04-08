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

from .csv_export import OptimizedCSVExport

class ExportResults:
    def __init__(self, problem, name, dictionnaire, dictionnaire_csv):
        """
        Initialise l'export des résultats.

        Parameters
        ----------
        problem : Objet de la classe Problem, problème mécanique qui a été résolu.
        name : String, nom du dossier dans lequel sera stocké les résultats.
        dictionnaire : Dictionnaire, dictionnaire contenant le nom des variables
                        que l'on souhaite exporter au format XDMF (Paraview).
        dictionnaire_csv : Dictionnaire, dictionnaire contenant le nom des variables
                            que l'on souhaite exporter au format csv.
        """
        self.pb = problem
        self.name = name
        self.dico = dictionnaire
        self.dico_csv = dictionnaire_csv
        self.param = default_post_processing_parameters()
        self.file_name = self.save_dir(name) + self.param["file_results"]
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
            # XDMFFile(self.pb.mesh.comm, self.file_name, "w").write_mesh(self.pb.mesh)
        self.set_expression()
        self.csv = OptimizedCSVExport(self.save_dir(name), name, problem, problem.name, dictionnaire_csv)

    def save_dir(self, name):
        """
        Renvoie nom du dossier dans lequel sera stocké les résultats.
        Parameters
        ----------
        name : String, nom du dossier dans lequel sera stocké les résultats.
        """
        savedir = name + "-" + "results" + "/"
        return savedir
    
    def set_expression(self):
        
        def get_index(key, length):
            if key == "all" or key:
                return [i for i in range(length)]
            elif isinstance(key, list):
                return key
            
        if self.dico.get("D"):
            self.D_func = Function(self.pb.V_Sig, name = "Taux déformation")
            self.D_expression = Expression(self.pb.D, self.pb.V_Sig.element.interpolation_points())
            
        if self.dico.get("c")!=None:
            self.c_index_list = get_index(self.dico.get("c"), len(self.pb.material))
            self.c_func_list = [Function(self.pb.multiphase.V_c, name = "Concentration ( %i)" % (i)) for i in self.c_index_list]
            self.c_expression_list = [Expression(self.pb.multiphase.c[i], self.pb.multiphase.V_c.element.interpolation_points()) for i in self.c_index_list]

    def export_results(self, t):
        """
        Exporte les résultats au format XDMF.

        Parameters
        ----------
        t : Float, temps de la simulation auquel les résultats sont exportés.
        """
        if  self.dico == {}:
            return
        
        if self.dico.get("U"):
            self.file_results.write_function(self.pb.u, t)
            
        if self.dico.get("v"):
            self.file_results.write_function(self.pb.v, t)
            
        if self.dico.get("a"):
            self.file_results.write_function(self.pb.a, t)
            
        if self.dico.get("d") and self.pb.constitutive.damage_model != None:
            self.file_results.write_function(self.pb.constitutive.damage.d, t)
            
        if self.dico.get("eps_p") and self.pb.constitutive.plastic_model != None:
            self.file_results.write_function(self.pb.constitutive.plastic.eps_p, t)
            
        if self.dico.get("D"):
            self.D_func.interpolate(self.D_expression)
            self.file_results.write_function(self.D_func, t)
            
        if self.dico.get("Sig"):
            self.pb.sig_func.interpolate(self.pb.sig_expr)
            self.file_results.write_function(self.pb.sig_func, t)

        if self.dico.get("Pressure"):
            self.pb.p_func.interpolate(self.pb.p_expr)
            self.file_results.write_function(self.pb.p_func, t)

        if self.dico.get("T"):
            self.file_results.write_function(self.pb.T, t)
            
        if self.dico.get("deviateur"):
            if self.pb.material.dev_type == "Hypoelastic":
                self.file_results.write_function(self.pb.constitutive.deviator.s, t)
            else:
                self.pb.s_func.interpolate(self.pb.s_expr)
                self.file_results.write_function(self.pb.s_func, t)
                
        if self.dico.get("c"):
            for i in range(len(self.c_index_list)):
                self.c_func_list[i].interpolate(self.c_expression_list[i])
                self.file_results.write_function(self.c_func_list[i], t)