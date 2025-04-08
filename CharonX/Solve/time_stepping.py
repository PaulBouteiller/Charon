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
Created on Tue Sep 13 10:52:01 2022

@author: bouteillerp
"""
from numpy import linspace, concatenate, array
from ..utils.default_parameters import default_dynamic_parameters
from mpi4py import MPI

class TimeStepping:
    def __init__(self, analysis, mesh, mat, **kwargs):
        """
        Initialise l'objet time stepping
        """
        self.analysis = analysis
        self.set_time_stepping(mesh, mat, **kwargs)
        
    def set_time_stepping(self, mesh, mat, **kwargs):
        """
        Initialise les pas de chargement.

        Parameters
        ----------
        scheme : String, schema pour la discrétisation temporelle.
        TFin : Float, temps final de la simulation.
        dt_min : Float, pas de temps minimal.
        """
        
        self.scheme = kwargs.get("time_step_scheme", "default")
        if self.analysis == "static":
            default_Tfin = 1
        else:
            default_Tfin = None
        self.Tfin = kwargs.get("TFin", default_Tfin)
        TInit = kwargs.get("TInit", 0)
        if self.Tfin == None:
            raise ValueError("A final time has to be given")

        if self.analysis == "explicit_dynamic":
            self.initial_CFL(mesh, mat)
            dt = kwargs.get("dt", self.dt_CFL)
            assert(dt <= self.dt_CFL)
        else:
            dt = kwargs.get("dt", None)
        if self.analysis in "static":
            self.set_virtual_time_step(**kwargs)
        else:
            if self.scheme == "default" or self.scheme == "fixed":
                self.regular_time_stepping(dt, self.Tfin, TInit)
            elif self.scheme == "adaptative":
                self.set_adaptative_time_stepping()
            else:
                raise ValueError("Unknown time stepping scheme")

    def initial_CFL(self, mesh, mat):
        """
        Défini le pas de temps minimal respectant la condition CFL

        Parameters
        ----------
        mesh : Mesh, maillage du domaine.
        mat : Objet de la classe material, matériau à l'étude.
        """
        celerity = self.celerity(mat)
        tdim = mesh.topology.dim
        num_cells = mesh.topology.index_map(tdim).size_local
        self.h = min(mesh.h(tdim, array(range(num_cells))))
        print("La célérité des ondes est", celerity)
        CFL_estimate = self.h / celerity
        print("Estim_t_crit", CFL_estimate)
        self.dt_CFL = default_dynamic_parameters()["CFL_ratio"] * CFL_estimate
        print("dt is : {0:.2e} on Proc {1:d}".format(self.dt_CFL, MPI.COMM_WORLD.rank))
        self.dt_CFL = MPI.COMM_WORLD.allreduce(self.dt_CFL, op = MPI.MIN)
        print("Shared critical time step:", self.dt_CFL)
        
    def celerity(self, material):
        """
        Retourne la célérité maximale des ondes élastiques
        Parameters
        ----------
        material : Objet ou liste d'objets de la classe material.
        Returns
        -------
        Float, célérité maximale des ondes

        """
        if isinstance(material, list):
            celerity = [mat.celerity for mat in material]
            return max(celerity)
        else:
            return material.celerity
        
    def regular_time_stepping(self, dt, TFin, TInit):
        """
        Initialise l'incrémentation régulière par défaut

        Parameters
        ----------
        dt : Float, pas de temps.
        TFin : Float, temps final de la simulation.
        """
        self.num_time_steps = int((TFin - TInit) / dt)
        self.load_steps = linspace(TInit, TFin, self.num_time_steps + 1)
        self.dt = TFin/(self.num_time_steps)
        
    def set_virtual_time_step(self, **kwargs):
        """ 
        Définie les pas de chargements pour une simulation en statique
        """
        if self.scheme == "default":
            self.num_time_steps = kwargs["npas"]
            self.load_steps = linspace(0, self.Tfin, self.num_time_steps)
            
        elif self.scheme == "piecewise_linear":
            self.piecewise_time_tepping(kwargs["discretisation_list"])
        self.dt = None
    
    def set_adaptative_time_stepping(self):
        """Calcul le nouveau pas de temps CFL"""
        raise ValueError("Non fonctionnel pour l'instant")
        self.dt =  self.dt_CFL
        self.actual_mesh_size = self.J_transfo * self.h
        self.dt_field = self.actual_mesh_size / self.material.GP_celerity(self.e_m_int)
        self.load_steps = [0]
        
        # #REGARDER LES PERFORMANCES
        # DG0 = dolfinx.fem.FUnctionSpace(mesh, ("DG", 0))
        # v = ufl.TestFunction(DG0)
        # cell_area_form = dolfinx.fem.form(v*ufl.dx)
        # cell_area = dolfinx.fem.assemble_vector(cell_area_form)
        # print(cell_area.array)
        
        # #Pour calculer les dimensions d'un triangle
        # detJ = abs(ufl.JacobianDeterminant(mesh) * ufl.classes.ReferenceCellVolume(mesh))
        # expr = dolfinx.fem.Expression(detJ,DG0.element.interpolation_points())
        # vol = dolfinx.fem.Function(DG0)
        # vol.interpolate(expr)
        # print(vol.x.array)
        
    def adaptative_time_stepping(self, dt_old):
        """
        Permet de définir un pas de temps adaptatif,
        non fonctionnel pour l'instant
        """
        dt_CFL = default_dynamic_parameters()["CFL_ratio"] * self.h / self.update_wave_velocity()
        new_dt = min(1.1 * dt_old, self.dt_CFL)
        return new_dt
    
    def piecewise_time_tepping(self, discretisation):
        """
        Définition d'une discrétisation temporelle affine par morceaux.

        Parameters
        ----------
        discretisation : List, liste contenant deux sous listes:
                                - la première contenant n éléments, elle définie les 
                                instants initiaux et finaux de chacun des intervalles.
                                - la second contenant n-1 entiers qui déterminent
                                le nombre de pas de temps dans chacun des intervalles
                                ainsi défini.
        """
        linspace_list = []
        for i in range(len(discretisation[1])):
            linspace_list.append(linspace(discretisation[0][i], discretisation[0][i+1], discretisation[1][i]))
        self.num_time_steps = sum(discretisation[1])
        self.load_steps = concatenate(linspace_list, axis = None)