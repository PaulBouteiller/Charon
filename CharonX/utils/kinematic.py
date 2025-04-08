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
Created on Mon Sep  5 13:59:18 2022

@author: bouteillerp
"""

from ufl import (grad, as_tensor, div, tr, Identity, dot, as_vector, det, inv,
                 conditional, ge)
from math import sqrt

class Kinematic:
    """
    Les méthodes implémentées ici  exploite les représentations réduites des 
    différents modèles afin d'accélérer les temps de calcul.
    """
    def __init__(self, name, r, anisotropic_dir):
        """
        Initialisation de l'objet kinematic. 

        Parameters
        ----------
        name : String, nom du modèle mécanique retenu, doit appartenir à:
                        [CartesianUD, CylindricalUD, SphericalUD, 
                         PlaneStrain, Axisymetric, Tridimensionnal].
        r : Coordonnées radiales dans les cas axi, cylindrique et sphérique.
        """
        self.name = name
        self.r = r
        self.n0 = anisotropic_dir

    def grad_scal(self, f):
        """
        Renvoie la représentation appropriée du gradient d'un champ scalaire f

        Parameters
        ----------
        f : Function, champ scalaire.
        """
        if self.name in ["CartesianUD", "CylindricalUD", "SphericalUD"]:
            return f.dx(0)
        else: 
            return grad(f)            

    def v_grad3D(self, f):
        """
        Renvoie le gradient 3D d'un champ scalaire f, ie un vecteur
        Parameters
        ----------
        f : Function, champ scalaire.
        """
        grad_f = grad(f)
        if self.name in ["CartesianUD", "CylindricalUD", "SphericalUD"]:
            return as_vector([grad_f[0], 0, 0])
        elif self.name == "PlaneStrain": 
            return as_vector([grad_f[0], grad_f[1], 0])
        elif self.name == "Axisymetric": 
            return as_vector([grad_f[0], 0, grad_f[1]])
        elif self.name == "Tridimensionnal": 
            return grad_f
        
    def grad_reduit(self, u, sym = False):
        """
        Renvoie le gradient reduit d'un champ de vectoriel u
        ----------
        u : Function, champ vectoriel (par exemple champ de deplacement).
        """
        if self.name == "CartesianUD":
            return u.dx(0)
        elif self.name == "CylindricalUD":
            return as_vector([u.dx(0), u / self.r])
        elif self.name == "SphericalUD":
            return as_vector([u.dx(0), u / self.r, u / self.r])
        elif self.name == "PlaneStrain":
            grad_u = grad(u)
            if sym:
                return as_vector([grad_u[0, 0], grad_u[1, 1], grad_u[0, 1]])
            else:
                return as_vector([grad_u[0, 0], grad_u[1, 1], grad_u[0, 1], grad_u[1, 0]])
        elif self.name == "Axisymetric":
            grad_u = grad(u)
            # condition = ge(self.r, 1e-3)
            if sym:
                true_grad = as_vector([grad_u[0, 0], u[0] / self.r, grad_u[1, 1], grad_u[0, 1]])
                # hop_grad = as_vector([grad_u[0, 0], grad_u[0, 0], grad_u[1, 1], grad_u[0, 1]])
            else:
                true_grad = as_vector([grad_u[0, 0], u[0] / self.r, grad_u[1, 1], 
                                  grad_u[0, 1], grad_u[1, 0]])         
                # hop_grad = as_vector([grad_u[0, 0],  grad_u[0, 0], grad_u[1, 1], 
                #                   grad_u[0, 1], grad_u[1, 0]])
            # return conditional(condition, true_grad, hop_grad)
            return true_grad
            # return hop_grad
        elif self.name == "Tridimensionnal":
            return grad(u)
    
    def reduit_to_3D(self, red, sym = False):
        """
        Renvoie la forme tridimensionnelle associée à une forme réduite  

        Parameters
        ----------
        red : Champ sous forme réduite.
        
        """
        if self.name == "CartesianUD":
            return as_tensor([[red, 0, 0], [0, 0, 0], [0, 0, 0]])
        elif self.name == "CylindricalUD":
            return as_tensor([[red[0], 0, 0], [0, red[1], 0], [0, 0, 0]])
        elif self.name == "SphericalUD":
            return as_tensor([[red[0], 0, 0], [0, red[1], 0], [0, 0, red[1]]])
        elif self.name == "PlaneStrain":
            if sym:
                return as_tensor([[red[0], red[2], 0], [red[2], red[1], 0], [0, 0, 0]])               
            else:
                return as_tensor([[red[0], red[2], 0], [red[3], red[1], 0], [0, 0, 0]])
        elif self.name == "Axisymetric":
            #Corrige le tenseur pour éviter la divergence en 0 
            # en utilisant la règle de l'hopital
            condition = ge(self.r, 1e-2)
            if sym:
                true_tens = as_tensor([[red[0], 0, red[3]], [0, red[1], 0], [red[3], 0, red[2]]])
                hop_tens = as_tensor([[red[0], 0, red[3]], [0, red[0], 0], [red[3], 0, red[2]]]) 
            else:
                true_tens = as_tensor([[red[0], 0, red[3]], [0, red[1], 0], [red[4], 0, red[2]]])
                hop_tens = as_tensor([[red[0], 0, red[3]], [0, red[0], 0], [red[4], 0, red[2]]]) 
            return conditional(condition, true_tens, hop_tens)
            # return true_tens
        elif self.name == "Tridimensionnal":
            if sym:
                return as_tensor([[red[0], red[3], red[4]], 
                                  [red[3], red[1], red[5]], 
                                  [red[4], red[5], red[2]]]) 
            else:
                return red
        
    def reduit_to_2D(self, red):
        """
        Renvoie la forme bidimensionnelle associée à une forme réduite
        """
        return as_tensor([[red[0], red[2]], [red[3], red[1]]])
    
    def bidim_to_reduit(self, tens2D):
        """
        Renvoie la forme réduite associée à un tenseur 2D
        """
        return as_vector([tens2D[0, 0], tens2D[1, 1], tens2D[0, 1], tens2D[1, 0]])

    def tridim_to_reduit(self, tens3D, sym = False):
        """
        Renvoie la forme réduite associée au tenseur tridimensionnel, les termes
        extra-diagonaux sont pondérés par un racine de 2, il s'agit de la notation
        de Mandel.
        Parameters
        ----------
        tens3D : Tenseur tridimensionnel.
        sym : Booléen, optional décrit si le tenseur d'origine était symétrique.
        Returns
        -------
        La forme réduite du tenseur 3D
        - Pour le modèle axisymétrique, il s'agit un quadri-vecteur 
        d'un tenseur en x
        """
        if self.name == "CartesianUD":
            return tens3D[0, 0]
        elif self.name =="CylindricalUD":
            return as_vector([tens3D[0, 0], tens3D[1, 1]])
        elif self.name =="SphericalUD":
            return as_vector([tens3D[0, 0], tens3D[1, 1], tens3D[2, 2]])
        elif self.name == "PlaneStrain":
            if sym:
                return as_vector([tens3D[0, 0], tens3D[1, 1], tens3D[0, 1]])
            else:
                return as_vector([tens3D[0, 0], tens3D[1, 1], tens3D[0, 1], tens3D[1, 0]])
        elif self.name == "Axisymetric":
            if sym:
                return as_vector([tens3D[0, 0], tens3D[1, 1], tens3D[2, 2], tens3D[0, 2]])
            else:
                return as_vector([tens3D[0, 0], tens3D[1, 1], tens3D[2, 2], 
                                  tens3D[0, 2], tens3D[2, 0]])
        elif self.name == "Tridimensionnal":
            if sym:
                return as_vector([tens3D[0, 0], tens3D[1, 1], tens3D[2, 2], 
                                  tens3D[0, 1], tens3D[0, 2], tens3D[1, 2]])
            else:
                return tens3D
            
    def tridim_to_mandel(self, tens3D):
        """
        Renvoie la forme réduite associée au tenseur tridimensionnel, les termes
        extra-diagonaux sont pondérés par un racine de 2, il s'agit de la notation
        de Mandel.
        Parameters
        ----------
        tens3D : Tenseur tridimensionnel symétrique.
        Returns
        -------
        La forme réduite du tenseur 3D
        - Pour le modèle axisymétrique, il s'agit un quadri-vecteur 
        d'un tenseur en x
        """
        sq2 = sqrt(2)
        if self.name in ["CartesianUD", "CylindricalUD", "SphericalUD"]:
            return as_vector([tens3D[0, 0], tens3D[1, 1], tens3D[2, 2]])
        elif self.name == "PlaneStrain":
            return as_vector([tens3D[0, 0], tens3D[1, 1], tens3D[2, 2], sq2 * tens3D[0, 1]])
        elif self.name == "Axisymetric":
            return as_vector([tens3D[0, 0], tens3D[1, 1], tens3D[2, 2], sq2 * tens3D[0, 2]])
        elif self.name == "Tridimensionnal":
            return as_vector([tens3D[0, 0], tens3D[1, 1], tens3D[2, 2], 
                              sq2 * tens3D[0, 1], sq2 * tens3D[0, 2], sq2 * tens3D[1, 2]])
        
    def mandel_to_tridim(self, red):
        """
        Renvoie la forme tridimensionnelle associée à une forme réduitede mandel

        Parameters
        ----------
        red : Champ sous forme réduite.
        
        """
        sq2 = sqrt(2)
        if self.name in ["CartesianUD", "CylindricalUD", "SphericalUD"]:
            return as_tensor([[red[0], 0, 0], [0, red[1], 0], [0, 0, red[2]]])
        if self.name == "PlaneStrain":
            return as_tensor([[red[0], red[3/sq2], 0], 
                              [red[3]/sq2, red[1], 0], 
                              [0, 0, red[2]]])
        elif self.name == "Axisymetric":
             return as_tensor([[red[0], 0, red[3]/sq2], 
                               [0, red[1], 0], 
                               [red[3]/sq2, 0, red[2]]])
        elif self.name =="Tridimensionnal":
            return as_tensor([[red[0], red[3], red[4]], 
                              [red[3], red[1], red[5]], 
                              [red[4], red[5], red[2]]]) 
        elif self.name =="Tridimensionnal":
            # return as_tensor([[red[0], red[3], red[4]], 
            #                   [red[3], red[1], red[5]], 
            #                   [red[4], red[5], red[2]]]) 
        
            return as_tensor([[red[0], red[3]/sq2, red[4]/sq2], 
                              [red[3]/sq2, red[1], red[5]/sq2], 
                              [red[4]/sq2, red[5]/sq2, red[2]]]) 


        
    def grad_3D(self, u, sym = False):
        """
        Renvoie le gradient tridimensionnel (une matrice) d'un champ de vectoriel
        ----------
        u : Function, champ vectoriel (par exemple champ de deplacement).
        """
        return self.reduit_to_3D(self.grad_reduit(u, sym = sym), sym = sym)
        
    def F_reduit(self, u):
        """
        Renvoie le gradient de la transformation. Ce gradient est représenté
        par un vecteur dans un cas 1D (diagonale du tenseur 3x3) et par un 
        tenseur 3x3 sinon.
        Parameters
        ----------
        u : Function, champ de déplacement.
        """
        if self.name == "CartesianUD":
            return as_vector([1 + u.dx(0), 1, 1])
        elif self.name == "CylindricalUD":
            return as_vector([1 + u.dx(0), 1 + u/self.r, 1])
        elif self.name == "SphericalUD":
            return as_vector([1 + u.dx(0), 1 + u/self.r, 1 + u/self.r])
        elif self.name in ["PlaneStrain", "Axisymetric", "Tridimensionnal"]:
            return Identity(3) + self.grad_3D(u)
        
    def div(self, v):
        """
        Renvoie la divergence du champ vectoriel v
        """
        if self.name in ["CartesianUD", "CylindricalUD", "SphericalUD"]:
             return v.dx(0)
        elif self.name in ["PlaneStrain", "Axisymetric", "Tridimensionnal"]:
             return div(v)
         
    def BI(self, u):
        """
        Renvoie la trace du tenseur de déformation de Cauchy-green gauche.
        Parameters
        ----------
        u : Function, champ de déplacement.
        """
        B = self.B(u)
        if self.name in ["CartesianUD", "CylindricalUD", "SphericalUD"]:
            return sum(B[i] for i in range(3))
        elif self.name in ["PlaneStrain", "Axisymetric", "Tridimensionnal"]:
            return tr(B)
          
    def BBarI(self, u):
        """
        Renvoie la trace du tenseur de déformation de Cauchy-green gauche isovolume.
        Parameters
        ----------
        u : Function, champ de déplacement.
        """
        return self.BI(u) / self.J(u)**(2./3)
        
    def invF_reduit(self, u):
        """
        Renvoie l'inverse du gradient de la transformation. Il est représenté
        par un vecteur dans un cas 1D (diagonale du tenseur 3x3) et par un 
        tenseur 3x3 sinon.
        Parameters
        ----------
        u : Function, champ de déplacement.
        """
        if self.name == "CartesianUD":
            return as_vector([1 / (1 + u.dx(0)), 1, 1])
        elif self.name == "CylindricalUD":
            return as_vector([1 / (1 + u.dx(0)), 1/(1 + u/self.r), 1])
        elif self.name == "SphericalUD":
            return as_vector([1 / (1 + u.dx(0)), 1 /(1 + u/self.r), 1 / (1 + u/self.r)])
        elif self.name == "PlaneStrain":
            inv_F2 = inv(Identity(2) + grad(u))
            return as_tensor([[inv_F2[0,0], inv_F2[0,1], 0], [inv_F2[1,0], inv_F2[1,1], 0], [0, 0, 1]])
        elif self.name == "Axisymetric":
            grad_u = grad(u)
            prefacteur = (1 + grad_u[0,0]) * (1 + grad_u[1,1]) -  grad_u[0,1] * (1 + grad_u[1,0])
            condition = ge(self.r, 1e-3)
            true_inv = as_tensor([[(1 + grad_u[1,1]) / prefacteur, 0, -grad_u[0,1] / prefacteur],
                              [0, 1 / (1 + u[0]/self.r), 0],
                              [-grad_u[1,0] / prefacteur, 0, (1 + grad_u[0,0]) / prefacteur]])
            hop_inv = as_tensor([[(1 + grad_u[1,1]) / prefacteur, 0, -grad_u[0,1] / prefacteur],
                              [0, 1 / (1 + grad_u[0,0]), 0],
                              [-grad_u[1,0] / prefacteur, 0, (1 + grad_u[0,0]) / prefacteur]])
            return conditional(condition, true_inv, hop_inv)
            
        elif self.name == "Tridimensionnal":
            return inv(Identity(3) + grad(u))
        
    def Eulerian_gradient(self, v, u):
        """
        Renvoie le gradient Eulerien de u c'est à dire \partial u/\partial x
        ou \partial x désigne la dérivée par rapport aux coordonnées actuelles.
        """
        invF_reduit = self.invF_reduit(u)
        grad_red = self.grad_reduit(v)
        if self.name == "CartesianUD":
            return grad_red * invF_reduit[0]
        elif self.name == "CylindricalUD":
            return as_vector([grad_red[i] * invF_reduit[i] for i in range(2)])
        elif self.name == "SphericalUD":
            return as_vector([grad_red[i] * invF_reduit[i] for i in range(3)])
        elif self.name == "PlaneStrain":
            return self.bidim_to_reduit(dot(self.reduit_to_2D(grad_red), inv(Identity(2) + grad(u))))
        elif self.name == "Axisymetric":
            return self.tridim_to_reduit(dot(self.reduit_to_3D(grad_red), invF_reduit))
        elif self.name == "Tridimensionnal":
            return dot(grad_red, invF_reduit)
    
    def B(self, u):
        """
        Renvoie le tenseur de Cauchy-Green gauche. Cette déformation est représentée
        par un vecteur dans un cas 1D (diagonale du tenseur 3x3) et par un 
        tenseur 3x3 sinon.
        Parameters
        ----------
        u : Function, champ de déplacement.
        """
        F = self.F_reduit(u)
        if self.name in ["CartesianUD", "CylindricalUD", "SphericalUD"]:
            return as_vector([F[0]**2, F[1]**2, F[2]**2])
        elif self.name in ["PlaneStrain", "Axisymetric", "Tridimensionnal"]:
            return dot(F, F.T)
        
    def J(self, u):
        """
        Renvoie le jacobien de la transformation. Ce Jacobien permet d'évaluer la
        dilatation locale.
        Parameters
        ----------
        u : Function, champ de déplacement.

        """
        if self.name =="CartesianUD":
            return 1 + u.dx(0)
        elif self.name =="CylindricalUD":
            return (1 + u.dx(0)) * (1 + u / self.r)
        elif self.name =="SphericalUD":
            return (1 + u.dx(0)) * (1 + u / self.r)**2
        elif self.name == "PlaneStrain":
            return det(Identity(2) + grad(u))
        elif self.name == "Axisymetric":
            F = self.F_reduit(u)
            return F[1, 1] * (F[0, 0] * F[2, 2] - F[0, 2] * F[2, 0])
        elif self.name == "Tridimensionnal":
            return det(Identity(3) + grad(u))
        
    def F_3D(self, u):
        """
        Renvoie le gradient de la transformation tridimensionnel.
        Parameters
        ----------
        u : Function, champ de déplacement.
        """
        return Identity(3) + self.grad_3D(u)
    
    def B_3D(self, u):
        """
        Renvoie le tenseur de Cauchy-Green gauche tridimensionnel.
        Parameters
        ----------
        u : Function, champ de déplacement.
        """
        F = self.F_3D(u)
        return dot(F, F.T)
    
    def C_3D(self, u):
        """
        Renvoie le tenseur de Cauchy-Green droit tridimensionnel.
        Parameters
        ----------
        u : Function, champ de déplacement.
        """
        F = self.F_3D(u)
        return dot(F.T, F)
    
    def measure(self, a, dx):
        """
        La mesure d'intégration est égale ici à:
        -dx dans le cas Cartésien 1D, 2D plan ou 3D.
        -rdr dans le cas cylindrique 1D ou axisymétrique.
        -r^{2}dr dans le cas sphérique 1D.
        
        Parameters
        ----------
        a : Champ à intégrer.
        dx : Measure, mesure d'intégration par défaut de FEniCS.
        """
        if self.name in ["CartesianUD", "PlaneStrain", "Tridimensionnal"]:
            return a * dx
        if self.name in ["CylindricalUD", "Axisymetric"]:
            return a * self.r * dx
        if self.name == "SphericalUD":
            return a * self.r**2 *dx 
        
    def relative_gradient_3D(self, u, u_old):
        """
        Renvoie le gradient de la transformation relative, entre deux
        configurations définies par leurs champs de déplacements respectifs.

        Parameters
        ----------
        u : Fonction, champ de déplacement actuel.
        u_old : Fonction, ancien champ de déplacement.

        """
        F_new = self.F_3D(u)
        inv_F_old = inv(self.F_3D(u_old))
        return dot(F_new, inv_F_old)
    
    def push_forward(self, tensor, u):
        """
        Renvoie le Push-forward d'un tenseur d'ordre 2 deux fois covariants.
        Parameters
        ----------
        tensor : Tenseur d'ordre 2.
        u : Function, champ de déplacement.
        """
        F = self.F_3D(u)
        return dot(dot(F, tensor), F.T)   
    
    def pull_back(self, tensor, u):
        """
        Renvoie le pull-back d'un tenseur d'ordre 2 deux fois covariants.
        Parameters
        ----------
        tensor : Tenseur d'ordre 2.
        u : Function, champ de déplacement.
        """
        F = self.F_3D(u)
        inv_F = inv(F)
        return dot(dot(inv_F, tensor), inv_F.T)   
    
    
    def actual_anisotropic_direction(self, u):
        """
        Renvoie le tenseur uniaxial actuel d'anisotropie
        Parameters
        ----------
        u : Fonction, champ de déplacement actuel.
        """
        n_t = dot(self.F_3D(u), self.n0)
        norm_nt = dot(n_t, n_t)**(1./2)
        return n_t/norm_nt
    
    def tridim_to_Voigt(self, tens):
        """
        Renvoie la représentation de Voigt d'un tenseur 3D, 
        la convention concernant la rangement des indices est
        "11, 22, 33, 12, 13, 23" 
        """
        return as_vector([tens[0,0], tens[1,1], tens[2,2],
                          2 * tens[1,2], 2 * tens[0,2], 2 * tens[0,1]])
    
    def Voigt_to_tridim(self, Voigt):
        """
        Renvoie la forme tridimensionnelle d'un vecteur mis en notation de Voigt,
        attention ne fonctionnera pas pour une déformation à cause du facteur 2
        """
        return as_tensor([[Voigt[0], Voigt[5], Voigt[4]],
                          [Voigt[5], Voigt[1], Voigt[3]],
                          [Voigt[4], Voigt[3], Voigt[2]]])