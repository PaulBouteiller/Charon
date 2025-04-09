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
Created on Fri Mar 11 09:48:43 2022

@author: bouteillerp
"""
from .Problem import BoundaryConditions, Loading, Problem
from ufl import cofac, inner, div, sym, grad, dot
from petsc4py.PETSc import ScalarType
from basix.ufl import element

class TridimensionalBoundaryConditions(BoundaryConditions):
    def __init__(self, V, facets, name):
        BoundaryConditions.__init__(self, V, facets)

    def add_Ux(self, region=1, value = ScalarType(0), method="topological"):
        self.add_component(self.V, 0, self.bcs, region, value)
        self.add_associated_speed_acceleration(self.V, 0, region, value)

    def add_Uy(self, region=1, value = ScalarType(0), method="topological"):
        self.add_component(self.V, 1, self.bcs, region, value)
        self.add_associated_speed_acceleration(self.V, 1, region, value)

    def add_Uz(self, region=1, value = ScalarType(0), method="topological"):
        self.add_component(self.V, 2, self.bcs, region, value)
        self.add_associated_speed_acceleration(self.V, 2, region, value)
        
class TridimensionalLoading(Loading):
    def __init__(self, mesh, u, V, dx):
        Loading.__init__(self, mesh, u[0], V, dx)

    def add_Fx(self, value, u, dx, **kwargs):
        self.add_loading(value, u[0], dx, **kwargs)
        
    def add_Fy(self, value, u, dx, **kwargs):
        self.add_loading(value, u[1], dx, **kwargs)
        
    def add_Fz(self, value, u, dx, **kwargs):
        self.add_loading(value, u[2], dx, **kwargs)

class Tridimensional(Problem):
    @property
    def name(self):
        return "Tridimensional"

    def set_finite_element(self):
        cell = self.mesh.basix_cell()
        self.U_e = element("Lagrange", cell, degree = self.u_deg, shape = (3,))  
        self.Sig_e = self.quad.quad_element(["Tensor", 3, 3])
        self.devia_e = self.Sig_e
        
    def boundary_conditions_class(self):
        return TridimensionalBoundaryConditions
    
    def loading_class(self):
        return TridimensionalLoading
    
    def div(self, v):
        return div(v)

    def dot(self, a, b):
        return dot(a, b)
    
    def dot_grad_scal(self, tensor1, tensor2):
        self.dot(tensor1, tensor2)
        
    def inner(self, a, b):
        return inner(a, b)
    
    def undamaged_stress(self, u, v, T, T0, J):
        return self.constitutive.stress_3D(u, v, T, T0, J)
    
    def conjugate_strain(self):
        return dot(sym(grad(self.u_)), cofac(self.kinematic.F_3D(self.u)))
    
    def extract_deviatoric(self, deviatoric):
        return deviatoric