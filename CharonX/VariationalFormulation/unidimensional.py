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
Le fichier Unidimensional contient l'ensemble des routines 
nécessaires à la définition d'un problème unidimensionnel.
Cela recouvre les modèles cartésien 1D, cylindrique et sphérique.
"""
from .Problem import BoundaryConditions, Loading, Problem
from ufl import as_vector, dot
from petsc4py.PETSc import ScalarType
from basix.ufl import element

class UnidimensionalBoundaryConditions(BoundaryConditions):
    def __init__(self, V, facets, name):
        BoundaryConditions.__init__(self, V, facets)

    def add_U(self, region = 1, value = ScalarType(0)):
        """
        Impose une CL de dirichlet à l'unique composante du champ de déplacement

        Parameters
        ----------
        region : Int, optional. Drapeau correspondant à la région
        où appliquer la CL de Dirichlet. The default is 1.
        value : ScalarType ou Expression, optional
        Valeur de la CL de Dirichlet. The default is ScalarType(0).
        """
        self.add_component(self.V, None, self.bcs, region, value)
        self.add_associated_speed_acceleration(self.V, None, region, value)
        
    def add_axi(self, region):
        """
        Ajoute une condition d'axisymétrie (utile pour les modèles cylindriques
        et sphérique).

        Parameters
        ----------
        region : Int, optional. Drapeau correspondant à la région
        où appliquer la CL de Dirichlet. The default is 1.
        """
        self.add_component(self.V, None, self.bcs_axi, region, ScalarType(1))
        self.add_component(self.V, None, self.bcs_axi_homog, region, ScalarType(0))
        self.add_component(self.V, None, self.bcs, region, ScalarType(0))
    
class UnidimensionalPeriodicBoundary:  
    pass

class UnidimensionalLoading(Loading):
    def __init__(self, mesh, u_, dx, kinematic):
        Loading.__init__(self, mesh, u_, dx, kinematic)
        
    def add_F(self, value, u_, dx, **kwargs):
        """
        Ajoute une condition au limite en effort.

        Parameters
        ----------
        value : ScalarType ou Expression, valeur de la CL à appliquer.
        u : TestFunction, champ de déplacement test.
        dx : Measure, mesure d'intégration qui donne la frontière sur laquelle
                    la CL sera appliquée.
        """
        self.add_loading(value, u_, dx)

class Unidimensional(Problem):
    def set_finite_element(self):
        """
        Initalise le type d'élements utilisé pour les champs inconnus
        """
        cell = self.mesh.basix_cell()
        self.U_e = element("Lagrange", cell, degree = self.u_deg)   
        if self.name == "CartesianUD" :
            self.Sig_e = self.quad.quad_element(["Scalar"])
        elif self.name == "CylindricalUD":
            self.Sig_e = self.quad.quad_element(["Vector", 2])
        elif self.name == "SphericalUD":
            self.Sig_e = self.quad.quad_element(["Vector", 3])
        self.devia_e = self.quad.quad_element(["Vector", 3])
        
    def boundary_conditions_class(self):
        return UnidimensionalBoundaryConditions
    
    def loading_class(self):
        return UnidimensionalLoading
    
    def dot_grad_scal(self, grad_scal_1, grad_scal_2):
        return grad_scal_1 * grad_scal_2

    def dot(self, reduit_1, reduit_2):
        """
        Produit contracté entre la représentation réduite de deux tenseurs.

        Parameters
        ----------
        reduit_1 : Vecteur de taille 1 ou 2, représentation réduite d'un tenseur diagonal.
        reduit_2 : Vecteur de taille 1 ou 2, représentation réduite d'un tenseur diagonal.
        """
        if self.name =="CartesianUD":
            return reduit_1 * reduit_2
        elif self.name =="CylindricalUD":
            return as_vector([reduit_1[i] * reduit_2[i] for i in range(2)])
        elif self.name =="SphericalUD":
            return as_vector([reduit_1[i] * reduit_2[i] for i in range(3)])
        
    def inner(self, reduit_1, reduit_2):
        """
        Produit doublement contracté entre la représentation réduite de deux
        tenseurs.

        Parameters
        ----------
        reduit_1 : Vecteur de taille 1 ou 2, représentation réduite d'un tenseur diagonal.
        reduit_2 : Vecteur de taille 1 ou 2, représentation réduite d'un tenseur diagonal.
        """
        if self.name =="CartesianUD":
            return reduit_1 * reduit_2
        elif self.name in ["CylindricalUD", "SphericalUD"]:
            return dot(reduit_1, reduit_2)
        
    def extract_deviatoric(self, deviatoric):
        return as_vector([deviatoric[0, 0], deviatoric[1, 1], deviatoric[2, 2]])
    
    def undamaged_stress(self, u, v, T, T0, J):
        """
        Définie la contrainte actuelle dans le matériau représentée par:
        - un scalaire \sigma
        - un 2-vecteur (\sigma_{rr}, \sigma_{\theta\theta}) pour les modèles 
        cylindrique ou sphérique.
        Parameters
        ----------
        u : Function, champ de déplacement.
        v : Function, champ de vitesse.
        T : Function, champ de température.
        T0 : Function, champ de température initiale.
        J : Function, Jacobien de la transformation.
        Returns
        -------
        sigma : Contrainte actuelle sous forme réduite.

        """
        return self.kinematic.tridim_to_reduit(self.constitutive.stress_3D(u, v, T, T0, J))
        
    def conjugate_strain(self):
        """
        Renvoie la déformation virtuelle sous forme réduite conjuguée à 
        la contrainte de Cauchy.
        """
        return self.dot(self.kinematic.grad_reduit(self.u_), self.cofac_reduit(self.u))
                
    def cofac_reduit(self, u):
        """
        Renvoie le cofacteur réduit du champ de déplacement
        Parameters
        ----------
        u : Function, champ de déplacement.
        """
        if self.name =="CartesianUD":
            return 1
        elif self.name =="CylindricalUD":               
            return as_vector([1 + self.u / self.r, 1 + u.dx(0)])
        elif self.name =="SphericalUD":
            return as_vector([(1 + self.u / self.r)**2, 
                              (1 + u.dx(0)) * (1 + self.u / self.r),
                              (1 + u.dx(0)) * (1 + self.u / self.r)])
    
class CartesianUD(Unidimensional):
    @property
    def name(self):
        return "CartesianUD"

class CylindricalUD(Unidimensional):
    @property
    def name(self):
        return "CylindricalUD"

class SphericalUD(Unidimensional):
    @property
    def name(self):
        return "SphericalUD"