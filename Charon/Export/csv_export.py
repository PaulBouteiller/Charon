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
from ..utils.mpi.gather import gather_coordinate, gather_function

from dolfinx.fem import locate_dofs_topological
from mpi4py.MPI import COMM_WORLD
from pandas import DataFrame, concat
from csv import writer, reader, field_size_limit
from numpy import ndarray, array, char
from os import remove, rename


from dolfinx.fem import Function, dirichletbc, form, assemble_scalar
from petsc4py.PETSc import ScalarType
from dolfinx.fem.petsc import set_bc
from ufl import action

class OptimizedCSVExport:
    FIELD_COMPONENTS = {
        "CartesianUD": {
            "U": ["u"], "v": ["v"],
            "sig": [r"\sigma"],
            "s": ["s_{xx}", "s_{yy}", "s_{zz}"]
        },
        "CylindricalUD": {
            "U": ["u"], "v": ["v"],
            "sig": [r"\sigma_{rr}", r"\sigma_{tt}"],
            "s": ["s_{rr}", "s_{tt}", "s_{zz}"]
        },
        "SphericalUD": {
            "U": ["u"], "v": ["v"],
            "sig": [r"\sigma_{rr}", r"\sigma_{tt}", r"\sigma_{phiphi}"],
            "s": ["s_{rr}", "s_{tt}", "s_{phiphi}"]
        },
        "PlaneStrain": {
            "U": ["u_{x}", "u_{y}"], "v": ["v_{x}", "v_{y}"],
            "sig": [r"\sigma_{xx}", r"\sigma_{yy}", r"\sigma_{xy}"],
            "s": ["s_{xx}", "s_{yy}", "s_{zz}", "s_{xy}"]
        },
        "Axisymmetric": {
            "U": ["u_{r}", "u_{z}"], "v": ["v_{r}", "v_{z}"],
            "sig": [r"\sigma_{rr}", r"\sigma_{tt}", r"\sigma_{zz}", r"\sigma_{rz}"],
            "s": ["s_{rr}", "s_{tt}", "s_{zz}", "s_{rz}"]
        },
        "Tridimensional": {
            "U": ["u_{x}", "u_{y}", "u_{z}"], 
            "v": ["v_{x}", "v_{y}", "v_{z}"],
            "sig": [r"\sigma_{xx}", r"\sigma_{xy}", r"\sigma_{xz}", 
                   r"\sigma_{yx}", r"\sigma_{yy}", r"\sigma_{yz}", 
                   r"\sigma_{zx}", r"\sigma_{zy}", r"\sigma_{zz}"],
            "s": ["s_{xx}", "s_{xy}", "s_{xz}", 
                         "s_{yx}", "s_{yy}", "s_{yz}", 
                         "s_{zx}", "s_{zy}", "s_{zz}"]
        }
    }
    
    def __init__(self, save_dir, pb, dico_csv, export = None):
        self.save_dir = save_dir
        self.export = export
        self.pb = pb
        self.dico_csv = dico_csv
        self.file_handlers = {}
        self.csv_writers = {}
        self.coordinate_data = {}
        self.initialize_export_settings()
        self.initialize_csv_files()
        self.export_times = []

    def csv_name(self, name):
        return self.save_dir+ f"{name}.csv"

    def initialize_export_settings(self):
        self.setup_displacement_export()
        self.setup_velocity_export()
        self.setup_damage_export()
        self.setup_dilatation_export()
        self.setup_temperature_export()
        self.setup_pressure_export()
        self.setup_density_export()
        self.setup_plastic_strain_export()
        self.setup_stress_export()
        self.setup_deviatoric_stress_export()
        self.setup_concentration_export()
        self.setup_free_surface_export()
        self.setup_reaction_force_export()
        
    def setup_simple_field(self, field_name, space):
        if field_name in self.dico_csv:
            setattr(self, f"csv_export_{field_name}", True)
            dofs = self.dofs_to_exp(space, self.dico_csv.get(field_name))
            setattr(self, f"{field_name}_dte", dofs)
            self.coordinate_data[field_name] = self.get_coordinate_data(space, dofs)
        else:
            setattr(self, f"csv_export_{field_name}", False)
            
    def setup_displacement_export(self):
        if "U" in self.dico_csv:
            self.csv_export_U = True
            self.U_dte = self.dofs_to_exp(self.pb.V, self.dico_csv.get("U"))
            self.coordinate_data["U"] = self.get_coordinate_data(self.pb.V, self.U_dte)
            
            components = self.FIELD_COMPONENTS[self.pb.name]["U"]
            self.u_name_list = components
            if len(components) > 1:
                self.U_cte = [self.comp_to_export(self.U_dte, i) for i in range(len(components))]
        else:
            self.csv_export_U = False
            
    def setup_velocity_export(self):
        if "v" in self.dico_csv:
            self.csv_export_v = True
            self.v_dte = self.dofs_to_exp(self.pb.V, self.dico_csv.get("v"))
            self.coordinate_data["v"] = self.get_coordinate_data(self.pb.V, self.v_dte)
            components = self.FIELD_COMPONENTS[self.pb.name]["v"]
            self.v_name_list = components
            if len(components) > 1:
                self.v_cte = [self.comp_to_export(self.v_dte, i) for i in range(len(components))]
        else:
            self.csv_export_v = False       

    def setup_temperature_export(self):
        self.setup_simple_field("T", self.pb.V_T)

    def setup_pressure_export(self):
        self.setup_simple_field("p", self.pb.V_quad_UD)
        
    def setup_damage_export(self):
        if self.export.is_damage:
            self.setup_simple_field("d", self.pb.constitutive.damage.V_d)
        else:
            self.csv_export_d = False

    def setup_density_export(self):
        self.setup_simple_field("rho", self.pb.V_quad_UD)
        
    def setup_dilatation_export(self):
        self.setup_simple_field("J", self.pb.V_quad_UD)

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
        if "sig" in self.dico_csv:
            self.csv_export_sig = True
            self.sig_dte = self.dofs_to_exp(self.export.V_sig, self.dico_csv.get("sig"))
            self.coordinate_data["sig"] = self.get_coordinate_data(self.export.V_sig, self.sig_dte)
            components = self.FIELD_COMPONENTS[self.pb.name]["sig"]
            self.sig_name_list = components
            if len(components) == 1:
                self.sig_cte = self.sig_dte
            else:
                self.sig_cte = [self.comp_to_export(self.sig_dte, i) for i in range(len(components))]
        else:
            self.csv_export_sig = False

    def setup_deviatoric_stress_export(self):
        if "s" in self.dico_csv:
            self.csv_export_devia = True
            self.s_dte = self.dofs_to_exp(self.export.V_s, self.dico_csv.get("s"))
            self.coordinate_data["s"] = self.get_coordinate_data(self.export.V_s, self.s_dte)
            components = self.FIELD_COMPONENTS[self.pb.name]["s"]
            self.s_name_list = components
            self.s_cte = [self.comp_to_export(self.s_dte, i) for i in range(len(components))]  
        else:
            self.csv_export_devia = False

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
            
    def setup_reaction_force_export(self):
        def set_gen_F(boundary_flag, value):
            """
            Define the resultant force on a given surface.
            
            Computes the reaction force by testing the residual with a
            carefully chosen test function.
            
            Parameters
            ----------
            boundary_flag : int Flag of the boundary where the resultant is to be recovered
            value : ScalarType Value to impose for the test function
                
            Returns
            -------
            ufl.form.Form Linear form representing the action of the residual on the test function
                
            Notes
            -----
            This follows the approach described in:
            https://comet-fenics.readthedocs.io/en/latest/demo/tips_and_tricks/computing_reactions.html
            """
            v_reac = Function(self.pb.V)
            dof_loc = locate_dofs_topological(self.pb.V, self.pb.facet_tag.dim, self.pb.facet_tag.find(boundary_flag))
            set_bc(v_reac.x.petsc_vec, [dirichletbc(value, dof_loc, self.pb.V)])
            return form(action(self.pb.form, v_reac))
        
        def set_F(boundary_flag, coordinate):
            """
            Initialize the resultant force along a coordinate.
            
            Parameters
            ----------
            boundary_flag : int Flag of the boundary where the resultant is to be recovered
            coordinate : str Coordinate for which to recover the reaction ("x", "y", "z", "r")
                
            Returns
            -------
            ufl.form.Form  Linear form representing the reaction force
            """
            if self.pb.dim == 1:
                return set_gen_F(boundary_flag, ScalarType(1.))
            elif self.pb.dim == 2:
                if coordinate == "r" or coordinate =="x":
                    return set_gen_F(boundary_flag, ScalarType((1., 0)))
                elif coordinate == "y" or coordinate =="z":
                    return set_gen_F(boundary_flag, ScalarType((0, 1.)))
            elif self.pb.dim == 3:
                if coordinate == "x":
                    return set_gen_F(boundary_flag, ScalarType((1., 0, 0)))
                elif coordinate == "y" :
                    return set_gen_F(boundary_flag, ScalarType((0, 1., 0)))
                elif coordinate == "z" :
                    return set_gen_F(boundary_flag, ScalarType((0, 0, 1.)))
        if "reaction_force" in self.dico_csv:
            flag = self.dico_csv["reaction_force"]["flag"]
            component = self.dico_csv["reaction_force"]["component"]
            self.reaction_form = set_F(flag, component)
            self.csv_reaction_force = True
            self.reaction_force = []
        else:
            self.csv_reaction_force = False            

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
        self.export_times.append(t)
        if not self.dico_csv:
            return
        if self.csv_export_U:
            if self.pb.name in ["CartesianUD", "CylindricalUD", "SphericalUD"]:
                self.export_field(t, "U", self.pb.u, self.U_dte)
            else:
                self.export_field(t, "U", self.pb.u, self.U_cte, subfield_name = self.u_name_list)
        if self.csv_export_v:
            if self.pb.name in ["CartesianUD", "CylindricalUD", "SphericalUD"]:
                self.export_field(t, "v", self.pb.v, self.v_dte)
            else:
                self.export_field(t, "v", self.pb.v, self.v_cte, subfield_name = self.v_name_list)
        if self.csv_export_d:
            self.export_field(t, "d", self.pb.constitutive.damage.d, self.d_dte)
        if self.csv_export_T:
            self.export_field(t, "T", self.pb.T, self.T_dte)
        if self.csv_export_p:
            self.export.p.interpolate(self.export.p_expr)
            self.export_field(t, "p", self.export.p, self.p_dte)
        if self.csv_export_rho:
            self.export.rho.interpolate(self.export.rho_expr)
            self.export_field(t, "rho", self.export.rho, self.rho_dte)
        if self.csv_export_J:
            self.export.J.interpolate(self.export.J_expr)
            self.export_field(t, "J", self.export.J, self.J_dte)
        if self.csv_export_eps_p:
            self.export_field(t, "eps_p", self.pb.constitutive.plastic.eps_p, self.epsp_cte, subfield_name = self.eps_p_name_list)
        if self.csv_export_sig:
            self.export.sig.interpolate(self.export.sig_expr)
            self.export_field(t, "sig", self.export.sig, self.sig_cte, subfield_name = self.sig_name_list)
        if self.csv_export_devia:
            self.export.s.interpolate(self.export.s_expr)
            self.export_field(t, "s", self.export.s, self.s_cte, subfield_name = self.s_name_list)
        if self.csv_export_c:
            for i, c_field in enumerate(self.pb.multiphase.c):
                self.export_field(t, f"Concentration{i}", c_field, self.c_dte)
        if self.csv_FreeSurf_1D:
            if COMM_WORLD.Get_rank() == 0:
                self.time.append(t)
                self.free_surf_v.append(self.pb.v.x.array[self.free_surf_dof][0])
                row = [t, self.free_surf_v[-1]]
                self.csv_writers["FreeSurf_1D"].writerow(row)
                self.file_handlers["FreeSurf_1D"].flush()
        if self.csv_reaction_force:
            self.reaction_force.append(assemble_scalar(self.reaction_form))
            
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
        dof_coords = gather_coordinate(V) if self.pb.mpi_bool else V.tabulate_dof_coordinates()
        def specific(array, coord, dof_to_exp):
            if isinstance(dof_to_exp, str):
                return array[:, coord]
            elif isinstance(dof_to_exp, ndarray):
                return array[dof_to_exp, coord]
            elif isinstance(dof_to_exp, ndarray):
                return array[dof_to_exp, coord]
        self.pb.name
        if COMM_WORLD.rank == 0:
            if self.pb.name == "CartesianUD":
                data = {"x": specific(dof_coords, 0, key)}
            elif self.pb.name in ["CylindricalUD", "SphericalUD"]:
                data = {"r": specific(dof_coords, 0, key)}
            elif self.pb.name == "PlaneStrain":
                data = {"x": specific(dof_coords, 0, key), "y": specific(dof_coords, 1, key)}
            elif self.pb.name =="Axisymmetric":
                data = {"r": specific(dof_coords, 0, key), "z": specific(dof_coords, 1, key)}
            elif self.pb.name =="Tridimensional":
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
            
    def write_list(self, header, value_list, file_name):
        with open(file_name, 'w', newline='') as f:
            csv_writer = writer(f)
            csv_writer.writerow([header])
            for t in value_list:
                csv_writer.writerow([f"{t:.6e}"])
        for handler in self.file_handlers.values():
            handler.close()
        

    def close_files(self):
        if COMM_WORLD.Get_rank() == 0:
            self.write_list("Time", self.export_times, self.csv_name("export_times"))
            if self.csv_reaction_force:
                self.write_list("Reaction force", self.reaction_force, self.csv_name("reaction_force"))                
            self.post_process_all_files()

    def post_process_all_files(self):
        for field_name in self.dico_csv.keys():
            if field_name == "c":
                for i in range(len(self.pb.material)):
                    conc_name = f"Concentration{i}"
                    self.post_process_csv(conc_name)
            elif field_name in ["d", "T", "p", "rho",  "J"]:
                self.post_process_csv(field_name)
            elif field_name == "U":
                self.post_process_csv(field_name, subfield_name = self.u_name_list)
            elif field_name == "v":
                self.post_process_csv(field_name, subfield_name = self.v_name_list)
            elif field_name == "sig":
                self.post_process_csv(field_name, subfield_name = self.sig_name_list)
            elif field_name == "eps_p":
                self.post_process_csv(field_name, subfield_name = self.eps_p_name_list)
            elif field_name == "s":
                self.post_process_csv(field_name, subfield_name = self.s_name_list)
            elif field_name == "FreeSurf_1D":
                pass
            elif field_name == "reaction_force":
                a=1
                pass
            else:
                raise ValueError(f"{field_name} can not be post process")


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