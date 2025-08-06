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
from numpy import ndarray, array
from os import remove, rename

class OptimizedCSVExport:
    def __init__(self, save_dir, pb, dico_csv, export_context = None):
        self.save_dir = save_dir
        self.pb = pb
        self.export_context = export_context
        self.dico_csv = dico_csv
        self.file_handlers = {}
        self.csv_writers = {}
        self.initialize_export_settings()
        self.initialize_csv_files()

    def csv_name(self, name):
        return self.save_dir + f"{name}.csv"

    def initialize_export_settings(self):
        """Configuration centralisée de tous les exports"""
        # Configuration centralisée : [V_space, expr, func_name, components]
        FIELD_CONFIGS = {
            "U": [self.pb.V, None, None, self._get_displacement_components()],
            "v": [self.pb.V, None, None, self._get_velocity_components()], 
            "d": [getattr(self.pb.constitutive.damage, 'V_d', None) if hasattr(self.pb.constitutive, 'damage') and self.pb.constitutive.damage else None, None, None, None],
            "T": [self.pb.V_T, None, None, None],
            "J": [self.pb.V_quad_UD, self.pb.J_transfo, "Dilatation", None],
            "Pressure": [self.pb.V_quad_UD, None, None, None],
            "rho": [self.pb.V_quad_UD, self.pb.rho, "rho", None],
            "eps_p": [getattr(self.pb.constitutive.plastic, 'Vepsp', None) if hasattr(self.pb.constitutive, 'plastic') and self.pb.constitutive.plastic else None, None, None, ["epsp_{xx}", "epsp_{yy}", "epsp_{zz}"]],
            "Sig": [self.get_export_space('V_Sig'), None, None, self._get_stress_components()],
            "deviateur": [self.get_export_space('V_devia'), None, None, self._get_devia_components()],
            "VonMises": [self.pb.V_quad_UD, None, None, None],
        }
        
        for field_name, (V, expr, func_name, components) in FIELD_CONFIGS.items():
            if field_name in self.dico_csv:
                self._setup_field(field_name, V, expr, func_name, components)
        
        # Cas spéciaux
        self._setup_concentration_export()
        self._setup_free_surface_export()

    def _setup_field(self, field_name, V, expr, func_name, components):
        """Méthode générique de configuration"""
        if V is None:  # Skip si l'espace de fonction n'existe pas
            setattr(self, f"csv_export_{field_name}", False)
            return
            
        setattr(self, f"csv_export_{field_name}", True)
        dte = self.dofs_to_exp(V, self.dico_csv[field_name])
        setattr(self, f"{field_name}_dte", dte)
        
        if expr:
            setattr(self, f"{field_name}_expr", Expression(expr, V.element.interpolation_points()))
            setattr(self, f"{field_name}_func", Function(V, name=func_name))
        
        if components:
            setattr(self, f"{field_name}_cte", [self.comp_to_export(dte, i) for i in range(len(components))])
            setattr(self, f"{field_name}_name_list", components)

    def _get_displacement_components(self):
        """Retourne les noms des composantes de déplacement"""
        if self.pb.dim == 1:
            return ["u"]
        elif self.pb.dim == 2:
            return ["u_{x}", "u_{y}"] if self.pb.name == "PlaneStrain" else ["u_{r}", "u_{z}"]
        else:  # dim == 3
            return ["u_{x}", "u_{y}", "u_{z}"]

    def _get_velocity_components(self):
        """Retourne les noms des composantes de vitesse"""
        if self.pb.dim == 1:
            return ["v"]
        elif self.pb.dim == 2:
            return ["v_{x}", "v_{y}"] if self.pb.name == "PlaneStrain" else ["v_{r}", "v_{z}"]
        else:  # dim == 3
            return ["v_{x}", "v_{y}", "v_{z}"]

    def get_export_space(self, name):
        if self.export_context:
            return self.export_context.export_spaces.get(name)
        return None
    
    def get_export_expression(self, name):
        if self.export_context:
            return self.export_context.export_expressions.get(name)
        return None
    
    def get_export_function(self, name):
        if self.export_context:
            return self.export_context.export_functions.get(name)
        return None

    def _get_stress_components(self):
        """Retourne les noms des composantes de contrainte selon le modèle"""
        stress_configs = {
            "CartesianUD": [r"\sigma"],
            "CylindricalUD": [r"\sigma_{rr}", r"\sigma_{tt}"],
            "SphericalUD": [r"\sigma_{rr}", r"\sigma_{tt}", r"\sigma_{phiphi}"],
            "PlaneStrain": [r"\sigma_{xx}", r"\sigma_{yy}", r"\sigma_{xy}"],
            "Axisymmetric": [r"\sigma_{rr}", r"\sigma_{tt}", r"\sigma_{zz}", r"\sigma_{rz}"],
            "Tridimensional": [r"\sigma_{xx}", r"\sigma_{xy}", r"\sigma_{xz}", 
                              r"\sigma_{yx}", r"\sigma_{yy}", r"\sigma_{yz}", 
                              r"\sigma_{zx}", r"\sigma_{zy}", r"\sigma_{zz}"]
        }
        return stress_configs.get(self.pb.name, [])

    def _get_devia_components(self):
        """Retourne les noms des composantes déviatoriques selon le modèle"""
        devia_configs = {
            "CartesianUD": ["s_{xx}", "s_{yy}", "s_{zz}"],
            "CylindricalUD": ["s_{rr}", "s_{tt}", "s_{zz}"],
            "SphericalUD": ["s_{rr}", "s_{tt}", "s_{phiphi}"],
            "PlaneStrain": ["s_{xx}", "s_{yy}", "s_{zz}", "s_{xy}"],
            "Axisymmetric": ["s_{rr}", "s_{tt}", "s_{zz}", "s_{rz}"],
            "Tridimensional": ["s_{xx}", "s_{xy}", "s_{xz}", 
                              "s_{yx}", "s_{yy}", "s_{yz}", 
                              "s_{zx}", "s_{zy}", "s_{zz}"]
        }
        return devia_configs.get(self.pb.name, [])

    def _setup_concentration_export(self):
        """Configuration spéciale pour les concentrations"""
        if "c" in self.dico_csv:
            self.csv_export_c = True
            n_mat = len(self.pb.material)
            V_c = self.pb.multiphase.V_c
            self.c_dte = self.dofs_to_exp(V_c, self.dico_csv["c"])
            self.c_name_list = [f"Concentration{i}" for i in range(n_mat)]
        else:
            self.csv_export_c = False

    def _setup_free_surface_export(self):
        """Configuration spéciale pour la surface libre"""
        if "FreeSurf_1D" in self.dico_csv:
            self.csv_FreeSurf_1D = True
            self.free_surf_dof = self.dofs_to_exp(self.pb.V, self.dico_csv["FreeSurf_1D"])
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
        """Export principal utilisant un dictionnaire de dispatch"""
        if not self.dico_csv:
            return
    
        export_methods = {
            "U": lambda t: self._export_displacement(t),
            "v": lambda t: self._export_velocity(t),
            "d": lambda t: self._export_damage(t),
            "T": lambda t: self.export_field(t, "T", self.pb.T, self.T_dte),
            "Pressure": lambda t: self._export_with_interpolation(t, "Pressure", 'p_func', 'p', self.Pressure_dte),
            "rho": lambda t: self._export_with_interpolation(t, "rho", 'rho_func', 'rho', self.rho_dte),
            "J": lambda t: self._export_with_interpolation(t, "J", 'J_func', 'J', self.J_dte),
            "eps_p": lambda t: self._export_plastic_strain(t),
            "Sig": lambda t: self._export_with_interpolation(t, "Sig", 'sig_func', 'sig', self.Sig_cte, self.Sig_name_list),
            "deviateur": lambda t: self._export_with_interpolation(t, "deviateur", 's_func', 's', self.deviateur_cte, self.deviateur_name_list),
            "VonMises": lambda t: self._export_with_interpolation(t, "VonMises", 'sig_VM_func', 'sig_VM', self.VonMises_dte),
            "c": lambda t: self._export_concentration(t),
            "FreeSurf_1D": lambda t: self.export_free_surface(t)
        }

        for field_name in self.dico_csv:
            if hasattr(self, f"csv_export_{field_name}") and getattr(self, f"csv_export_{field_name}"):
                export_methods.get(field_name, lambda t: None)(t)

    def _export_displacement(self, t):
        """Export des déplacements"""
        if self.pb.name in ["CartesianUD", "CylindricalUD", "SphericalUD"]:
            self.export_field(t, "U", self.pb.u, self.U_dte)
        else:
            self.export_field(t, "U", self.pb.u, self.U_cte, self.u_name_list)

    def _export_velocity(self, t):
        """Export des vitesses"""
        if self.pb.name in ["CartesianUD", "CylindricalUD", "SphericalUD"]:
            self.export_field(t, "v", self.pb.v, self.v_dte)
        else:
            self.export_field(t, "v", self.pb.v, self.v_cte, self.v_name_list)

    def _export_damage(self, t):
        """Export du dommage (avec vérification)"""
        if hasattr(self.pb.constitutive, 'damage') and self.pb.constitutive.damage and hasattr(self, 'd_dte'):
            self.export_field(t, "d", self.pb.constitutive.damage.d, self.d_dte)

    def _export_plastic_strain(self, t):
        """Export des déformations plastiques (avec vérification)"""
        if hasattr(self.pb.constitutive, 'plastic') and self.pb.constitutive.plastic and hasattr(self, 'epsp_cte'):
            self.export_field(t, "eps_p", self.pb.constitutive.plastic.eps_p, self.epsp_cte, self.eps_p_name_list)

    def _export_with_interpolation(self, t, field_name, func_key, expr_key, dofs_to_export, subfield_names=None):
        """Export d'un champ nécessitant une interpolation"""
        func = self.get_export_function(func_key)
        expr = self.get_export_expression(expr_key)
        
        if func and expr:
            func.interpolate(expr)
            self.export_field(t, field_name, func, dofs_to_export, subfield_names)

    def _export_stress(self, t):
        """Export des contraintes"""
        self._export_with_interpolation(
            t, "Sig", 'sig_func', 'sig', 
            self.Sig_cte, self.Sig_name_list
        )
    
    def _export_deviator(self, t):
        """Export du déviateur"""
        self._export_with_interpolation(
            t, "deviateur", 's_func', 's', 
            self.deviateur_cte, self.deviateur_name_list
        )

    def _export_concentration(self, t):
        """Export des concentrations"""
        for i, c_field in enumerate(self.pb.multiphase.c):
            self.export_field(t, f"Concentration{i}", c_field, self.c_dte)
            
    def export_field(self, t, field_name, field, dofs_to_export, subfield_name=None):
        if isinstance(subfield_name, list):
            n_sub = len(subfield_name)
            for i in range(n_sub):
                data = self.gather_field_data(field, dofs_to_export[i], size=n_sub, comp=i)
                self.write_field_data(field_name, t, data)
        else:
            data = self.gather_field_data(field, dofs_to_export)
            self.write_field_data(field_name, t, data)        

    def gather_field_data(self, field, dofs_to_export, size=None, comp=None):
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
            data = array(data).flatten()
            formatted_data = [f"{t:.6e}"] + [f"{val:.6e}" for val in data]
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
        """Post-traitement avec dispatch"""
        post_process_config = {
            "c": lambda: [self.post_process_csv(f"Concentration{i}") for i in range(len(self.pb.material))],
            "U": lambda: self.post_process_csv("U", getattr(self, 'u_name_list', None)),
            "v": lambda: self.post_process_csv("v", getattr(self, 'v_name_list', None)),
            "Sig": lambda: self.post_process_csv("Sig", getattr(self, 'sig_name_list', None)),
            "eps_p": lambda: self.post_process_csv("eps_p", getattr(self, 'eps_p_name_list', None)),
            "deviateur": lambda: self.post_process_csv("deviateur", getattr(self, 's_name_list', None)),
            "FreeSurf_1D": lambda: None  # Pas de post-traitement
        }
        
        # Champs simples (sans sous-champs)
        simple_fields = ["d", "T", "Pressure", "rho", "VonMises", "J"]
        
        for field_name in self.dico_csv.keys():
            if field_name in post_process_config:
                post_process_config[field_name]()
            elif field_name in simple_fields:
                self.post_process_csv(field_name)
            else:
                raise ValueError(f"{field_name} can not be post process")

    def post_process_csv(self, field_name, subfield_name=None):
        input_file = self.csv_name(field_name)
        temp_output_file = self.csv_name(f"{field_name}_processed")

        def parse_row(row):
            return array([float(val) for val in row.split(',')])
        
        field_size_limit(int(1e9))
        with open(input_file, 'r') as f:
            csv_reader = reader(f)
            headers = next(csv_reader)
            data = [parse_row(row[0]) for row in csv_reader]

        data = array(data)
        times = data[:, 0]
        values = data[:, 1:]

        # Coordonnées simplifiées
        coord_data = self.get_coordinate_data(field_name)
        coord = DataFrame(coord_data)
        
        if subfield_name is None:
            times_pd = [f"t={t}" for t in times]
        else:
            times_pd = [f"{subfield_name[compteur%len(subfield_name)]} t={t}" for compteur, t in enumerate(times)]
        
        datas = DataFrame({name: lst for name, lst in zip(times_pd, values)})
        result = concat([coord, datas], axis=1)

        result.to_csv(temp_output_file, index=False, float_format='%.6e')

        remove(input_file)
        rename(temp_output_file, input_file)

    def get_coordinate_data(self, field_name):
        """Version simplifiée pour récupérer les coordonnées"""
        V_mapping = {
            "U": self.pb.V, 
            "v": self.pb.V, 
            "T": self.pb.V_T, 
            "d": getattr(self.pb.constitutive.damage, 'V_d', None) if hasattr(self.pb.constitutive, 'damage') and self.pb.constitutive.damage else None,
            "J": self.pb.V_quad_UD, 
            "Pressure": self.pb.V_quad_UD, 
            "rho": self.pb.V_quad_UD, 
            "VonMises": self.pb.V_quad_UD, 
            "eps_p": getattr(self.pb.constitutive.plastic, 'Vepsp', None) if hasattr(self.pb.constitutive, 'plastic') and self.pb.constitutive.plastic else None,
            "Sig": self.get_export_space('V_Sig'),  # CORRIGER ICI
            "deviateur": self.get_export_space('V_devia')  # CORRIGER ICI
        }
        
        # Gestion des concentrations
        if field_name.startswith("Concentration"):
            V = self.pb.multiphase.V_c
            key = self.c_dte
        else:
            V = V_mapping.get(field_name)
            if V is None:  # Si l'espace n'existe pas, retourner un dict vide
                return {}
            key = getattr(self, f"{field_name}_dte", "all")
        
        if COMM_WORLD.rank != 0:
            return {}
            
        dof_coords = gather_coordinate(V) if self.pb.mpi_bool else V.tabulate_dof_coordinates()
        
        def extract_coord(array, coord_idx, dof_key):
            if isinstance(dof_key, str):
                return array[:, coord_idx]
            elif isinstance(dof_key, ndarray):
                return array[dof_key, coord_idx]
        
        coord_names = {
            "CartesianUD": ["x"], "CylindricalUD": ["r"], "SphericalUD": ["r"],
            "PlaneStrain": ["x", "y"], "Axisymmetric": ["r", "z"], 
            "Tridimensional": ["x", "y", "z"]
        }.get(self.pb.name, ["x"])
        
        return {name: extract_coord(dof_coords, i, key) for i, name in enumerate(coord_names)}