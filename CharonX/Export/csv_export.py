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
Created on Tue Sep 10 11:28:37 2024

@author: bouteillerp
"""
from ..utils.gather_mpi_functions import gather_coordinate, gather_function

from dolfinx.fem import Expression, Function, locate_dofs_topological
from mpi4py.MPI import COMM_WORLD
from pandas import DataFrame, concat
from csv import writer, reader, field_size_limit
from numpy import ndarray, array, char
from os import remove, rename

class OptimizedCSVExport:
    def __init__(self, save_dir, name, pb, model_meca, dico_csv):
        self.save_dir = save_dir
        self.name = name
        self.pb = pb
        self.model_meca = model_meca
        self.dico_csv = dico_csv
        self.file_handlers = {}
        self.csv_writers = {}
        self.coordinate_data = {}
        self.initialize_export_settings()
        self.initialize_csv_files()

    def csv_name(self, name):
        return self.save_dir+ f"{name}.csv"

    def initialize_export_settings(self):
        self.setup_displacement_export()
        self.setup_velocity_export()
        self.setup_damage_export()
        self.setup_temperature_export()
        self.setup_pressure_export()
        self.setup_density_export()
        self.setup_plastic_strain_export()
        self.setup_stress_export()
        self.setup_deviatoric_stress_export()
        self.setup_VonMises_export()
        self.setup_concentration_export()
        self.setup_free_surface_export()

    def setup_displacement_export(self):
        if "U" in self.dico_csv:
            self.csv_export_U = True
            self.U_dte = self.dofs_to_exp(self.pb.V, self.dico_csv.get("U"))#Suffixe dte:dofs_to_export
            self.coordinate_data["U"] = self.get_coordinate_data(self.pb.V, self.U_dte)
            if self.pb.dim == 1:
                self.u_name_list = ["u"]
            elif self.pb.dim == 2:
                self.U_cte = [self.comp_to_export(self.U_dte, i) for i in range(2)]
                self.u_name_list = ["u_{x}", "u_{y}"] if self.model_meca == "PlaneStrain" else ["u_{r}", "u_{z}"]
            elif self.pb.dim == 3:
                self.U_cte = [self.comp_to_export(self.U_dte, i) for i in range(3)]
                self.u_name_list = ["u_{x}", "u_{y}", "u_{z}"]
        else:
            self.csv_export_U = False
            
    def setup_velocity_export(self):
        if "v" in self.dico_csv:
            self.csv_export_v = True
            self.v_dte = self.dofs_to_exp(self.pb.V, self.dico_csv.get("v"))
            self.coordinate_data["v"] = self.get_coordinate_data(self.pb.V, self.v_dte)
            if self.pb.dim == 1:
                self.v_name_list = ["v"]
            elif self.pb.dim == 2:
                self.v_cte = [self.comp_to_export(self.v_dte, i) for i in range(2)]
                self.v_name_list = ["u_{x}", "u_{y}"] if self.model_meca == "PlaneStrain" else ["v_{r}", "v_{z}"]
            elif self.pb.dim == 3:
                self.v_cte = [self.comp_to_export(self.v_dte, i) for i in range(3)]
                self.v_name_list = ["v_{x}", "v_{y}", "v_{z}"]
        else:
            self.csv_export_v = False            

    def setup_damage_export(self):
        if "d" in self.dico_csv:
            V_d = self.pb.constitutive.damage.V_d
            self.csv_export_d = True
            self.d_dte = self.dofs_to_exp(V_d, self.dico_csv.get("d"))
            self.coordinate_data["d"] = self.get_coordinate_data(V_d, self.d_dte)
        else:
            self.csv_export_d = False

    def setup_temperature_export(self):
        if "T" in self.dico_csv:
            self.csv_export_T = True
            self.T_dte = self.dofs_to_exp(self.pb.V_T, self.dico_csv.get("T"))
            self.coordinate_data["T"] = self.get_coordinate_data(self.pb.V_T, self.T_dte)
        else:
            self.csv_export_T = False

    def setup_pressure_export(self):
        if "Pressure" in self.dico_csv:
            self.csv_export_P = True
            V_p = self.pb.V_quad_UD
            self.p_dte = self.dofs_to_exp(V_p, self.dico_csv.get("Pressure"))
            self.coordinate_data["Pressure"] = self.get_coordinate_data(V_p, self.p_dte)
        else:
            self.csv_export_P = False

    def setup_density_export(self):
        if "rho" in self.dico_csv:
            self.csv_export_rho = True
            V_rho = self.pb.V_quad_UD
            self.rho_dte = self.dofs_to_exp(V_rho, self.dico_csv.get("rho"))
            self.coordinate_data["rho"] = self.get_coordinate_data(V_rho, self.rho_dte)
            self.rho_expr = Expression(self.pb.rho, V_rho.element.interpolation_points())
            self.rho_func = Function(V_rho, name="rho")
        else:
            self.csv_export_rho = False

    def setup_plastic_strain_export(self):
        if "eps_p" in self.dico_csv:
            V_epsp = self.pb.constitutive.plastic.Vepsp
            self.csv_export_eps_p = True
            self.epsp_dte = self.dofs_to_exp(V_epsp, self.dico_csv.get("eps_p"))
            self.coordinate_data["eps_p"] = self.get_coordinate_data(V_epsp, self.epsp_dte)
            if self.pb.dim == 1:
                self.epsp_cte = [self.comp_to_export(self.epsp_dte, i) for i in range(3)]
                self.eps_p_name_list = ["epsp_{xx}", "epsp_{yy}", "epsp_{zz}"]
        else:
            self.csv_export_eps_p = False

    def setup_stress_export(self):
        if "Sig" in self.dico_csv:
            self.csv_export_Sig = True
            V_sig = self.pb.V_Sig
            self.sig_dte = self.dofs_to_exp(V_sig, self.dico_csv.get("Sig"))
            self.coordinate_data["Sig"] = self.get_coordinate_data(V_sig, self.sig_dte)
            self.setup_stress_components()
        else:
            self.csv_export_Sig = False

    def setup_stress_components(self):
        if self.model_meca == "CartesianUD":
            self.sig_cte = self.sig_dte
            self.sig_name_list = ["\sigma"]
        elif self.model_meca == "CylindricalUD":
            self.sig_cte = [self.comp_to_export(self.sig_dte, i) for i in range(2)]
            self.sig_name_list = [r"\sigma_{rr}", r"\sigma_{tt}"]
        elif self.model_meca == "SphericalUD":
            self.sig_cte = [self.comp_to_export(self.sig_dte, i) for i in range(3)]
            self.sig_name_list = [r"\sigma_{rr}", r"\sigma_{tt}", r"\sigma_{phiphi}"]
        elif self.model_meca == "PlaneStrain":
            self.sig_cte = [self.comp_to_export(self.sig_dte, i) for i in range(3)]
            self.sig_name_list = [r"\sigma_{xx}", r"\sigma_{yy}", r"\sigma_{xy}"]
        elif self.model_meca == "Axisymetric":
            self.sig_cte = [self.comp_to_export(self.sig_dte, i) for i in range(4)]
            self.sig_name_list = [r"\sigma_{rr}", r"\sigma_{tt}", r"\sigma_{zz}", r"\sigma_{rz}"]
        elif self.model_meca == "Tridimensionnal":
            self.sig_cte = [self.comp_to_export(self.sig_dte, i) for i in range(9)]
            self.sig_name_list = [r"\sigma_{xx}", r"\sigma_{xy}", r"\sigma_{xz}", 
                                  r"\sigma_{yx}", r"\sigma_{yy}",r"\sigma_{yz}", 
                                  r"\sigma_{zx}", r"\sigma_{zy}",r"\sigma_{zz}"]

    def setup_deviatoric_stress_export(self):
        if "deviateur" in self.dico_csv:
            self.csv_export_devia = True
            V_s = self.pb.V_devia
            self.s_dte = self.dofs_to_exp(V_s, self.dico_csv.get("deviateur"))
            self.coordinate_data["deviateur"] = self.get_coordinate_data(V_s, self.s_dte)
            self.setup_deviatoric_stress_components()
        else:
            self.csv_export_devia = False
            
    def setup_VonMises_export(self):
        if "VonMises" in self.dico_csv:
            self.csv_export_VM = True
            V_VM = self.pb.V_quad_UD
            self.VM_dte = self.dofs_to_exp(V_VM, self.dico_csv.get("VonMises"))
            self.coordinate_data["VonMises"] = self.get_coordinate_data(V_VM, self.VM_dte)
        else:
            self.csv_export_VM = False

    def setup_deviatoric_stress_components(self):
        if self.pb.dim == 1:
            self.s_cte = [self.comp_to_export(self.s_dte, i) for i in range(3)]
        elif self.pb.dim == 2:
            self.s_cte = [self.comp_to_export(self.s_dte, i) for i in range(4)]
        elif self.pb.dim == 3:
            self.s_cte = [self.comp_to_export(self.s_dte, i) for i in range(9)]
        
        if self.model_meca == "CartesianUD":
            self.s_name_list = ["s_{xx}", "s_{yy}", "s_{zz}"]
        elif self.model_meca == "CylindricalUD":
            self.s_name_list = ["s_{rr}", "s_{tt}", "s_{zz}"]
        elif self.model_meca == "SphericalUD":
            self.s_name_list = ["s_{rr}", "s_{tt}", "s_{phiphi}"]
        elif self.model_meca == "PlaneStrain":
            self.s_name_list = ["s_{xx}", "s_{yy}", "s_{zz}", "s_{xy}"]
        elif self.model_meca == "Axisymetric":
            self.s_name_list = ["s_{rr}", "s_{tt}", "s_{zz}", "s_{rz}"]
        elif self.model_meca == "Tridimensionnal":
            self.s_name_list = ["s_{xx}", "s_{xy}", "s_{xz}", 
                                "s_{yx}", "s_{yy}","s_{yz}", 
                                "s_{zx}", "s_{zy}","s_{zz}"]       

    def setup_concentration_export(self):
        if "c" in self.dico_csv:
            self.csv_export_c = True
            n_mat = len(self.pb.material)
            V_c = self.pb.multiphase.V_c
            self.c_dte = self.dofs_to_exp(V_c, self.dico_csv.get("c"))
            coordinate_data = self.get_coordinate_data(V_c, self.c_dte)
            self.coordinate_data.update(({f"Concentration{i}": coordinate_data  for i in range(n_mat)}))
            self.c_name_list = [f"Concentration{i}" for i in range(n_mat)]
        else:
            self.csv_export_c = False

    def setup_free_surface_export(self):
        if "FreeSurf_1D" in self.dico_csv:
            self.csv_FreeSurf_1D = True
            self.free_surf_dof = self.dofs_to_exp(self.pb.V, self.dico_csv.get("FreeSurf_1D"))
            self.time = []
            self.free_surf_v = []
        else:
            self.csv_FreeSurf_1D = False

    def initialize_csv_files(self):
        if COMM_WORLD.Get_rank() == 0:
            for field_name, export_info in self.dico_csv.items():
                if field_name == "c":
                    for i in range(len(self.pb.material)):
                        self.create_csv_file(f"Concentration{i}")
                else:
                    self.create_csv_file(field_name)

    def create_csv_file(self, field_name):
        file_path = self.csv_name(field_name)
        self.file_handlers[field_name] = open(file_path, 'w', newline='')
        self.csv_writers[field_name] = writer(self.file_handlers[field_name])
        headers = ["Time", field_name]
        self.csv_writers[field_name].writerow(headers)

    def dofs_to_exp(self, V, keyword):
        if isinstance(keyword, bool) and keyword is True:
            return "all"
        elif isinstance(keyword, list) and keyword[0] == "Boundary":
            return locate_dofs_topological(V, self.pb.facet_tag.dim, self.pb.facet_tag.find(keyword[1]))
        elif isinstance(keyword, ndarray):
            return keyword

    def comp_to_export(self, keyword, component):
        if isinstance(keyword, str):
            return keyword
        elif isinstance(keyword, ndarray):
            vec_dof_to_exp = keyword.copy()
            vec_dof_to_exp *= self.pb.dim
            vec_dof_to_exp += component
            return vec_dof_to_exp

    def csv_export(self, t):
        if not self.dico_csv:
            return
        if self.csv_export_U:
            if self.model_meca in ["CartesianUD", "CylindricalUD", "SphericalUD"]:
                self.export_field(t, "U", self.pb.u, self.U_dte)
            else:
                self.export_field(t, "U", self.pb.u, self.U_cte, subfield_name = self.u_name_list)
        if self.csv_export_v:
            if self.model_meca in ["CartesianUD", "CylindricalUD", "SphericalUD"]:
                self.export_field(t, "v", self.pb.v, self.v_dte)
            else:
                self.export_field(t, "v", self.pb.v, self.v_cte, subfield_name = self.v_name_list)
        if self.csv_export_d:
            self.export_field(t, "d", self.pb.constitutive.damage.d, self.d_dte)
        if self.csv_export_T:
            self.export_field(t, "T", self.pb.T, self.T_dte)
        if self.csv_export_P:
            self.pb.p_func.interpolate(self.pb.p_expr)
            self.export_field(t, "Pressure", self.pb.p_func, self.p_dte)
        if self.csv_export_rho:
            self.rho_func.interpolate(self.rho_expr)
            self.export_field(t, "rho", self.rho_func, self.rho_dte)
        if self.csv_export_eps_p:
            self.export_field(t, "eps_p", self.pb.constitutive.plastic.eps_p, self.epsp_cte, subfield_name = self.eps_p_name_list)
        if self.csv_export_Sig:
            self.pb.sig_func.interpolate(self.pb.sig_expr)
            self.export_field(t, "Sig", self.pb.sig_func, self.sig_cte, subfield_name = self.sig_name_list)
        if self.csv_export_devia:
            self.pb.s_func.interpolate(self.pb.s_expr)
            self.export_field(t, "deviateur", self.pb.s_func, self.s_cte, subfield_name = self.s_name_list)
        if self.csv_export_VM:
            self.pb.sig_VM_func.interpolate(self.pb.sig_VM)
            self.export_field(t, "VonMises", self.pb.sig_VM_func, self.VM_dte)
        if self.csv_export_c:
            for i, c_field in enumerate(self.pb.multiphase.c):
                self.export_field(t, f"Concentration{i}", c_field, self.c_dte)
        if self.csv_FreeSurf_1D:
            self.export_free_surface(t)
            
    def export_field(self, t, field_name, field, dofs_to_export, subfield_name = None):
        if isinstance(subfield_name, list):
            n_sub = len(subfield_name)
            for i in range(n_sub):
                data = self.gather_field_data(field, dofs_to_export[i], size = n_sub, comp = i)
                self.write_field_data(field_name, t, data)
        else:
            data = self.gather_field_data(field, dofs_to_export)
            self.write_field_data(field_name, t, data)        

    def get_coordinate_data(self, V, key):
        def mpi_gather(V):
            if self.pb.mpi_bool:
                return gather_coordinate(V)
            else:
                return V.tabulate_dof_coordinates()
        dof_coords = mpi_gather(V)
        def specific(array, coord, dof_to_exp):
            if isinstance(dof_to_exp, str):
                return array[:, coord]
            elif isinstance(dof_to_exp, ndarray):
                return array[dof_to_exp, coord]
            elif isinstance(dof_to_exp, ndarray):
                return array[dof_to_exp, coord]
        self.model_meca
        if COMM_WORLD.rank == 0:
            if self.model_meca == "CartesianUD":
                data = {"x": specific(dof_coords, 0, key)}
            elif self.model_meca in ["CylindricalUD", "SphericalUD"]:
                data = {"r": specific(dof_coords, 0, key)}
            elif self.model_meca == "PlaneStrain":
                data = {"x": specific(dof_coords, 0, key), "y": specific(dof_coords, 1, key)}
            elif self.model_meca =="Axisymetric":
                data = {"r": specific(dof_coords, 0, key), "z": specific(dof_coords, 1, key)}
            elif self.model_meca =="Tridimensionnal":
                data = {"x": specific(dof_coords, 0, key), "y": specific(dof_coords, 1, key), "z" : specific(dof_coords, 2, key)}
            return data

    def gather_field_data(self, field, dofs_to_export, size = None, comp = None):
        if self.pb.mpi_bool:
            field_data = gather_function(field)
        else:
            field_data = field.x.petsc_vec.array

        if isinstance(dofs_to_export, str) and dofs_to_export == "all" and size is None:
            return field_data
        elif isinstance(dofs_to_export, str) and dofs_to_export == "all" and size is not None:
            return field_data[comp::size]
        elif isinstance(dofs_to_export, ndarray):
            return field_data[dofs_to_export]
        else:
            return field_data
            
    def write_field_data(self, field_name, t, data):
        if COMM_WORLD.Get_rank() == 0:
            # Convertir en tableau NumPy si ce n'est pas déjà le cas
            if not isinstance(data, ndarray):
                data = array(data)
            
            # Aplatir le tableau si c'est un tableau multidimensionnel
            data = data.flatten()
            
            # Convertir les données en chaîne de caractères avec une précision fixe
            formatted_data = [f"{t:.6e}"]  # Temps avec 6 décimales
            formatted_data.extend(char.mod('%.6e', data))  # Données en notation scientifique
            
            # Écrire les données en tant que chaîne unique
            self.csv_writers[field_name].writerow([','.join(formatted_data)])
            self.file_handlers[field_name].flush()

    def export_free_surface(self, t):
        if COMM_WORLD.Get_rank() == 0:
            self.time.append(t)
            self.free_surf_v.append(self.pb.v.x.array[self.free_surf_dof][0])
            row = [t, self.free_surf_v[-1]]
            self.csv_writers["FreeSurf_1D"].writerow(row)
            self.file_handlers["FreeSurf_1D"].flush()

    def close_files(self):
        if COMM_WORLD.Get_rank() == 0:
            for handler in self.file_handlers.values():
                handler.close()
            self.post_process_all_files()

    def post_process_all_files(self):
        for field_name in self.dico_csv.keys():
            if field_name == "c":
                for i in range(len(self.pb.material)):
                    conc_name = f"Concentration{i}"
                    self.post_process_csv(conc_name)
            elif field_name in ["d", "T", "Pressure", "rho", "VonMises"]:
                self.post_process_csv(field_name)
            elif field_name == "U":
                self.post_process_csv(field_name, subfield_name = self.u_name_list)
            elif field_name == "v":
                self.post_process_csv(field_name, subfield_name = self.v_name_list)
            elif field_name == "Sig":
                self.post_process_csv(field_name, subfield_name = self.sig_name_list)
            elif field_name == "eps_p":
                self.post_process_csv(field_name, subfield_name = self.eps_p_name_list)
            elif field_name == "deviateur":
                self.post_process_csv(field_name, subfield_name = self.s_name_list)
            elif field_name == "FreeSurf_1D":
                pass
            else:
                raise ValueError("Wrong field name to post process")


    def post_process_csv(self, field_name, subfield_name=None):
        input_file = self.csv_name(field_name)
        temp_output_file = self.csv_name(f"{field_name}_processed")

        def parse_row(row):
            return array([float(val) for val in row.split(',')])
        field_size_limit(int(1e9))  # Augmenter à une valeur très élevée, par exemple 1 milliard
        with open(input_file, 'r') as f:
            csv_reader = reader(f)
            headers = next(csv_reader)# Lire la première ligne (en-têtes)
            data = [parse_row(row[0]) for row in csv_reader]# Lire le reste des données

        data = array(data)
        times = data[:, 0]
        values = data[:, 1:]

        coord = DataFrame(self.coordinate_data[field_name])
        if subfield_name is None:
            times_pd = [f"t={t}" for t in times]
        else:
            times_pd = [f"{subfield_name[compteur%len(subfield_name)]} t={t}" for compteur, t in enumerate(times)]
        datas = DataFrame({name: lst for name, lst in zip(times_pd, values)})
        result = concat([coord, datas], axis=1)

        result.to_csv(temp_output_file, index=False, float_format='%.6e')

        remove(input_file)
        rename(temp_output_file, input_file)