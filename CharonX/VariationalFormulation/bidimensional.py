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
Bidimensionnal.py contient les routines pour la définition d'un problème mécanique
2D (plan ou axisymétrique).
"""

from .Problem import BoundaryConditions, Loading, Problem
from ufl import cofac, as_vector, dot, grad, as_tensor, Identity
from petsc4py.PETSc import ScalarType
from basix.ufl import element

class BidimensionnalBoundaryConditions(BoundaryConditions):
    def __init__(self, V, facet_tag, name):
        self.name = name
        BoundaryConditions.__init__(self, V, facet_tag)
        
    def add_clamped(self, region):
        self.add_component(self.V, 0, self.bcs, region, ScalarType(0))
        self.add_associated_speed_acceleration(self.V, 0, region, ScalarType(0))
        self.add_component(self.V, 1, self.bcs, region, ScalarType(0))
        self.add_associated_speed_acceleration(self.V, 1, region, ScalarType(0))
        
class AxiBoundaryConditions(BidimensionnalBoundaryConditions):
    def add_Ur(self, region = 1, value = ScalarType(0)):
        """
        Ajoute une CL de Dirichlet u_{r} = value sur la frontière "region".
        """
        self.add_component(self.V, 0, self.bcs, region, value)
        self.add_associated_speed_acceleration(self.V, 0, region, value)

    def add_Uz(self, region = 1, value = ScalarType(0)):
        """
        Ajoute une CL de Dirichlet u_{z} = value sur la frontière "region".
        """       
        self.add_component(self.V, 1, self.bcs, region, value)
        self.add_associated_speed_acceleration(self.V, 1, region, value)

    def add_axi(self, region, value = None):
        """
        Ajoute une condition au limite d'axisymétrie, si un côté du domaine
        se trouve sur l'axe. Nécessaire afin d'éviter une divergence des champs.
        """
        self.add_component(self.V, 0, self.bcs_axi, region, ScalarType(1))
        self.add_component(self.V, 0, self.bcs, region, ScalarType(0))
        self.add_associated_speed_acceleration(self.V, 0, region, ScalarType(0))
        if value is not None:
            self.add_component(self.V, 1, self.bcs_axi, region, ScalarType(value))
            
class PlaneStrainBoundaryConditions(BidimensionnalBoundaryConditions):
    def add_Ux(self, region = 1, value = ScalarType(0)):
        """
        Ajoute une CL de Dirichlet u_{x} = value sur la frontière "region".
        """
        self.add_component(self.V, 0, self.bcs, region, value)
        self.add_associated_speed_acceleration(self.V, 0, region, value)

    def add_Uy(self, region = 1, value = ScalarType(0)):
        """
        Ajoute une CL de Dirichlet u_{y} = value sur la frontière "region".
        """
        self.add_component(self.V, 1, self.bcs, region, value)
        self.add_associated_speed_acceleration(self.V, 1, region, value)

class BidimensionnalLoading(Loading):
    def __init__(self, mesh, u_, dx, kinematic):
        Loading.__init__(self, mesh, u_[0], dx, kinematic)

class PlaneStrainLoading(BidimensionnalLoading):
    def add_Fx(self, value, u_, dx):
        """
        Ajoute une condition au limite en effort suivant ex.

        Parameters
        ----------
        value : ScalarType ou Expression, valeur de la CL à appliquer.
        u : TestFunction, champ de déplacement test.
        dx : Measure, mesure d'intégration qui donne la frontière sur laquelle
                    la CL sera appliquée.
        """
        self.add_loading(value, u_[0], dx)

    def add_Fy(self, value, u_, dx):
        self.add_loading(value, u_[1], dx)

class AxiLoading(BidimensionnalLoading):
    def add_Fr(self, value, u_, dx):
        """
        Ajoute une condition au limite en effort suivant \vec{e}_r.
        """
        self.add_loading(value, u_[0], dx)

    def add_Fz(self, value, u_, dx):
        self.add_loading(value, u_[1], dx)

class Bidimensionnal(Problem):
    def set_finite_element(self):
        """
        Définition des éléments finis utilisés pour le champ de déplacement
        et pour la contrainte.
        """
        cell = self.mesh.basix_cell()
        self.U_e = element("Lagrange", cell, degree = self.u_deg, shape = (2,))  
        self.Sig_e = self.quad.quad_element(["Vector", self.sig_dim_quadrature()])
        self.devia_e = self.quad.quad_element(["Vector", 4])

    def boundary_conditions_class(self):
        """
        Renvoie le nom de la classe des conditions aux limites en déplacement.
        """
        return self.bidim_boundary_conditions_class()

    def loading_class(self):
        """
        Renvoie le nom de la classe des conditions aux limites en efforts.
        """
        return self.bidim_loading_class()
    
    def undamaged_stress(self, u, v, T, T0, J):
        """
        Définie la contrainte actuelle dans le matériau représentée par un 
        quadri-vecteur ou un tri-vecteur.
        Parameters
        ----------
        u : Function, champ de déplacement.
        v : Function, champ de vitesse.
        T : Function, champ de température.
        T0 : Function, champ de température initiale.
        J : Function, Jacobien de la transformation.
        Returns
        -------
        sigma : Contrainte actuelle.
        """
        sig3D = self.constitutive.stress_3D(u, v, T, T0, J)
        return self.kinematic.tridim_to_reduit(sig3D, sym = True)
    
    def dot_grad_scal(self, tensor1, tensor2):
        return self.dot(tensor1, tensor2)
        
    def dot(self, tensor1, tensor2):
        """
        Produit contracté entre deux tenseurs.
        Parameters
        ----------
        tensor1 : Tenseurs
        tensor2 : Tenseurs
        """
        return dot(tensor1, tensor2)
    
class Plane_strain(Bidimensionnal):
    @property
    def name(self):
        return "PlaneStrain"
    
    def bidim_boundary_conditions_class(self):
        """
        Renvoie le nom de la classe des conditions aux limites en déplacement.
        """
        return PlaneStrainBoundaryConditions
    
    def bidim_loading_class(self):
        """
        Renvoie le nom de la classe des conditions aux limites en déplacement.
        """
        return PlaneStrainLoading
    
    def extract_deviatoric(self, s):
        return as_vector([s[0, 0], s[1, 1], s[2, 2], s[0, 1]])
    
    def sig_dim_quadrature(self):
        return 3

    def conjugate_strain(self):
        """
        Renvoie la déformation virtuelle conjuguée à la contrainte de Cauchy.
        """
        conj = dot(grad(self.u_),  cofac(Identity(2) + grad(self.u)))
        return self.kinematic.bidim_to_reduit(conj)
    
    def inner(self, vector_1, vector_2):
        """ 
        Renvoie le produit doublement contracté entre deux 
        vecteurs représentant des tenseurs 2x2
        Parameters
        ----------
        vector_1 : Quadri-vecteur représentant un tenseur "en X"
        vector_2 : Quadri-vecteur symétrisé représentant un tenseur "en X" 
        """
        shape_1 = vector_1.ufl_shape[0]
        shape_2 = vector_2.ufl_shape[0]
        if shape_1== 3 and shape_2==4:
            return vector_1[0] * vector_2[0] + vector_1[1] * vector_2[1] + \
                    + vector_1[2] * (vector_2[2] + vector_2[3]) 

class Axisymetric(Bidimensionnal):
    """
    Défini un problème axisymétrique, par convention l'axe de symétrie est l'axe Oy
    """
    @property
    def name(self):
        return "Axisymetric"
    
    def bidim_boundary_conditions_class(self):
        """
        Renvoie le nom de la classe des conditions aux limites en déplacement.
        """
        return AxiBoundaryConditions
    
    def bidim_loading_class(self):
        """
        Renvoie le nom de la classe des conditions aux limites en déplacement.
        """
        return AxiLoading
    
    def sig_dim_quadrature(self):
        return 4
    
    def extract_deviatoric(self, deviatoric):
        return self.kinematic.tridim_to_reduit(deviatoric, sym = True)
    
    def inner(self, vector_1, vector_2):
        """ 
        Renvoie le produit doublement contracté entre deux 
        vecteurs représentant des tenseurs "en x"", le tenseur x_tens1 doit être symétrique
        Parameters
        ----------
        xtens_1_vector : Quadri-vecteur représentant un tenseur "en X"
        xtens_2_vector : Quadri-vecteur symétrisé représentant un tenseur "en X" 
        """
        shape_1 = vector_1.ufl_shape[0]
        shape_2 = vector_2.ufl_shape[0]
        if shape_1== 4 and shape_2==5:
            return vector_1[0] * vector_2[0] + vector_1[1] * vector_2[1] + \
                    vector_1[2] * vector_2[2] + vector_1[3] * (vector_2[3] + vector_2[4]) 

    def cofac3D(self, x_tens):
        """
        Renvoie $\COM A^{\top}=\det(A)A^{-1}$ d'un tenseur en x
        Parameters
        ----------
        x_tens : Tenseur 3x3 avec une structure "en x"
        """
        TComA = as_tensor([[x_tens[2, 2] * x_tens[1, 1], 0, -x_tens[0, 2] * x_tens[1, 1]],
                           [0, x_tens[0, 0] * x_tens[2, 2] - x_tens[0, 2] * x_tens[2, 0], 0],
                           [-x_tens[2, 0] * x_tens[1, 1], 0, x_tens[0, 0] * x_tens[1, 1]]])
        return TComA

    def conjugate_strain(self):
        """
        Retourne la partie symétrique de la déformation conjuguée au tenseur 
        des contraintes de Cauchy
        """
        cofF = self.cofac3D(self.kinematic.F_3D((self.u)))
        return self.kinematic.tridim_to_reduit(dot(self.kinematic.grad_3D(self.u_), cofF))