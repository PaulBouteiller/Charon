#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 10:51:13 2025

@author: bouteillerp
"""
"""
Module de cinématique pour les modèles mécaniques.

Ce module contient la classe Kinematic qui encapsule les opérations
cinématiques pour différentes dimensions et géométries.
"""

from ufl import (grad, as_tensor, div, tr, Identity, dot, as_vector, det, inv,
                 conditional, ge)
from math import sqrt

class Kinematic:
    """
    Encapsule les opérations cinématiques pour différentes dimensions et géométries.
    
    Cette classe fournit des méthodes pour calculer des gradients, tenseurs, 
    transformations et autres grandeurs cinématiques en adaptant automatiquement
    les calculs à la dimension et géométrie du problème.
    
    Attributes:
        name (str): Nom du modèle mécanique ('CartesianUD', 'PlaneStrain', etc.)
        r (Function): Coordonnées radiales dans les cas axisymétriques, cylindriques et sphériques
        n0 (Vector): Direction d'anisotropie initiale
    """
    def __init__(self, name, r, anisotropic_dir):
        """
        Initialise l'objet Kinematic.

        Parameters:
            name (str): Nom du modèle mécanique, doit appartenir à:
                       [CartesianUD, CylindricalUD, SphericalUD, 
                        PlaneStrain, Axisymetric, Tridimensionnal]
            r (Function): Coordonnées radiales dans les cas axisymétriques, cylindriques et sphériques
            anisotropic_dir (Vector): Direction d'anisotropie initiale
        """
        self.name = name
        self.r = r
        self.n0 = anisotropic_dir
        
        # Configurations pour les différents types de modèles
        self._model_config = {"dim1": ["CartesianUD", "CylindricalUD", "SphericalUD"],
                              "dim2": ["PlaneStrain", "Axisymetric"],
                              "dim3": ["Tridimensionnal"]}
        
    # =========================================================================
    # Méthodes de gradient
    # =========================================================================
    
    def grad_scal(self, f):
        """
        Renvoie la représentation appropriée du gradient d'un champ scalaire.

        Parameters:
            f (Function): Champ scalaire

        Returns:
            Expression: Gradient adapté à la dimension et géométrie
        """
        if self._is_1d():
            return f.dx(0)
        else: 
            return grad(f)
    
    def v_grad3D(self, f):
        """
        Renvoie le gradient 3D d'un champ scalaire sous forme vectorielle.

        Parameters:
            f (Function): Champ scalaire

        Returns:
            Vector: Gradient 3D adapté à la dimension et géométrie
        """
        grad_f = self.grad_scal(f)
        
        if self._is_1d():
            return as_vector([grad_f, 0, 0])
        elif self.name == "PlaneStrain": 
            return as_vector([grad_f[0], grad_f[1], 0])
        elif self.name == "Axisymetric": 
            return as_vector([grad_f[0], 0, grad_f[1]])
        else:  # Tridimensional
            return grad_f
    
    def grad_reduit(self, u, sym=False):
        """
        Renvoie le gradient réduit d'un champ vectoriel.
        
        La représentation est adaptée à la dimension et géométrie.

        Parameters:
            u (Function): Champ vectoriel
            sym (bool, optional): Si True, utilise une représentation symétrique. Défaut: False

        Returns:
            Expression: Gradient réduit
        """
        if self.name == "CartesianUD":
            return u.dx(0)
        elif self.name == "CylindricalUD":
            return as_vector([u.dx(0), u / self.r])
        elif self.name == "SphericalUD":
            return as_vector([u.dx(0), u / self.r, u / self.r])
        elif self.name == "PlaneStrain":
            grad_u = grad(u)
            return self._get_2d_reduced_grad(grad_u, sym)
        elif self.name == "Axisymetric":
            grad_u = grad(u)
            return self._get_axi_reduced_grad(grad_u, u, sym)
        else:  # Tridimensional
            return grad(u)
    
    def _get_2d_reduced_grad(self, grad_u, sym):
        """Méthode privée pour obtenir le gradient réduit 2D"""
        if sym:
            return as_vector([grad_u[0, 0], grad_u[1, 1], grad_u[0, 1]])
        else:
            return as_vector([grad_u[0, 0], grad_u[1, 1], grad_u[0, 1], grad_u[1, 0]])
    
    def _get_axi_reduced_grad(self, grad_u, u, sym):
        """Méthode privée pour obtenir le gradient réduit axisymétrique"""
        if sym:
            return as_vector([grad_u[0, 0], u[0] / self.r, grad_u[1, 1], grad_u[0, 1]])
        else:
            return as_vector([grad_u[0, 0], u[0] / self.r, grad_u[1, 1], 
                            grad_u[0, 1], grad_u[1, 0]])
    
    def grad_3D(self, u, sym=False):
        """
        Renvoie le gradient tridimensionnel d'un champ vectoriel.

        Parameters:
            u (Function): Champ vectoriel
            sym (bool, optional): Si True, utilise une représentation symétrique. Défaut: False

        Returns:
            Tensor: Gradient 3D sous forme tensorielle
        """
        return self.reduit_to_3D(self.grad_reduit(u, sym=sym), sym=sym)
    
    def Eulerian_gradient(self, v, u):
        """
        Renvoie le gradient Eulérien de v, c'est-à-dire ∂v/∂x où ∂x désigne 
        la dérivée par rapport aux coordonnées actuelles.

        Parameters:
            v (Function): Champ vectoriel à dériver
            u (Function): Champ de déplacement définissant la transformation

        Returns:
            Expression: Gradient eulérien
        """
        invF_reduit = self.invF_reduit(u)
        grad_red = self.grad_reduit(v)
        
        if self._is_1d():
            if self.name == "CartesianUD":
                return grad_red * invF_reduit[0]
            else:  # CylindricalUD or SphericalUD
                return as_vector([grad_red[i] * invF_reduit[i] for i in range(len(grad_red))])
        elif self.name == "PlaneStrain":
            return self.bidim_to_reduit(dot(self.reduit_to_2D(grad_red), inv(Identity(2) + grad(u))))
        elif self.name == "Axisymetric":
            return self.tridim_to_reduit(dot(self.reduit_to_3D(grad_red), invF_reduit))
        else:  # Tridimensional
            return dot(grad_red, invF_reduit)
    
    # =========================================================================
    # Méthodes de conversion de format
    # =========================================================================
    
    def reduit_to_3D(self, red, sym=False):
        """
        Convertit un tenseur en forme réduite vers sa forme tridimensionnelle complète.

        Parameters:
            red (Expression): Champ sous forme réduite
            sym (bool, optional): Si True, utilise une représentation symétrique. Défaut: False

        Returns:
            Tensor: Tenseur 3D correspondant
        """
        # Modèles 1D
        if self.name == "CartesianUD":
            return as_tensor([[red, 0, 0], [0, 0, 0], [0, 0, 0]])
        elif self.name == "CylindricalUD":
            return as_tensor([[red[0], 0, 0], [0, red[1], 0], [0, 0, 0]])
        elif self.name == "SphericalUD":
            return as_tensor([[red[0], 0, 0], [0, red[1], 0], [0, 0, red[1]]])
        
        # Modèles 2D
        elif self.name == "PlaneStrain":
            return self._plane_strain_to_3D(red, sym)
        elif self.name == "Axisymetric":
            return self._axisymetric_to_3D(red, sym)
        
        # Modèle 3D
        else:  # Tridimensional
            if sym:
                return as_tensor([[red[0], red[3], red[4]], 
                                  [red[3], red[1], red[5]], 
                                  [red[4], red[5], red[2]]])
            else:
                return red
    
    def _plane_strain_to_3D(self, red, sym):
        """Méthode privée pour convertir de PlaneStrain à 3D"""
        if sym:
            return as_tensor([[red[0], red[2], 0], [red[2], red[1], 0], [0, 0, 0]])
        else:
            return as_tensor([[red[0], red[2], 0], [red[3], red[1], 0], [0, 0, 0]])
    
    def _axisymetric_to_3D(self, red, sym):
        """Méthode privée pour convertir d'Axisymétrique à 3D"""
        condition = ge(self.r, 1e-2)
        
        if sym:
            true_tens = as_tensor([[red[0], 0, red[3]], [0, red[1], 0], [red[3], 0, red[2]]])
            hop_tens = as_tensor([[red[0], 0, red[3]], [0, red[0], 0], [red[3], 0, red[2]]]) 
        else:
            true_tens = as_tensor([[red[0], 0, red[3]], [0, red[1], 0], [red[4], 0, red[2]]])
            hop_tens = as_tensor([[red[0], 0, red[3]], [0, red[0], 0], [red[4], 0, red[2]]]) 
            
        return conditional(condition, true_tens, hop_tens)
    
    def reduit_to_2D(self, red):
        """
        Convertit un tenseur en forme réduite vers sa forme bidimensionnelle.

        Parameters:
            red (Expression): Champ sous forme réduite

        Returns:
            Tensor: Tenseur 2D correspondant
        """
        return as_tensor([[red[0], red[2]], [red[3], red[1]]])
    
    def bidim_to_reduit(self, tens2D):
        """
        Convertit un tenseur 2D vers sa forme réduite.

        Parameters:
            tens2D (Tensor): Tenseur 2D

        Returns:
            Vector: Forme réduite correspondante
        """
        return as_vector([tens2D[0, 0], tens2D[1, 1], tens2D[0, 1], tens2D[1, 0]])
    
    def tridim_to_reduit(self, tens3D, sym=False):
        """
        Convertit un tenseur 3D vers sa forme réduite.

        Parameters:
            tens3D (Tensor): Tenseur 3D
            sym (bool, optional): Si True, utilise une représentation symétrique. Défaut: False

        Returns:
            Expression: Forme réduite correspondante
        """
        # Modèles 1D
        if self.name == "CartesianUD":
            return tens3D[0, 0]
        elif self.name == "CylindricalUD":
            return as_vector([tens3D[0, 0], tens3D[1, 1]])
        elif self.name == "SphericalUD":
            return as_vector([tens3D[0, 0], tens3D[1, 1], tens3D[2, 2]])
        
        # Modèles 2D
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
        
        # Modèle 3D
        elif self.name == "Tridimensionnal":
            if sym:
                return as_vector([tens3D[0, 0], tens3D[1, 1], tens3D[2, 2], 
                                tens3D[0, 1], tens3D[0, 2], tens3D[1, 2]])
            else:
                return tens3D
    
    def tridim_to_mandel(self, tens3D):
        """
        Convertit un tenseur 3D vers sa représentation en notation de Mandel.
        
        Les termes extra-diagonaux sont pondérés par un facteur √2.

        Parameters:
            tens3D (Tensor): Tenseur 3D symétrique

        Returns:
            Vector: Représentation de Mandel
        """
        sq2 = sqrt(2)
        
        # Modèles 1D
        if self._is_1d():
            return as_vector([tens3D[0, 0], tens3D[1, 1], tens3D[2, 2]])
        
        # Modèles 2D
        elif self.name == "PlaneStrain":
            return as_vector([tens3D[0, 0], tens3D[1, 1], tens3D[2, 2], sq2 * tens3D[0, 1]])
        elif self.name == "Axisymetric":
            return as_vector([tens3D[0, 0], tens3D[1, 1], tens3D[2, 2], sq2 * tens3D[0, 2]])
        
        # Modèle 3D
        elif self.name == "Tridimensionnal":
            return as_vector([tens3D[0, 0], tens3D[1, 1], tens3D[2, 2], 
                            sq2 * tens3D[0, 1], sq2 * tens3D[0, 2], sq2 * tens3D[1, 2]])
    
    def mandel_to_tridim(self, red):
        """
        Convertit une représentation de Mandel vers un tenseur 3D.

        Parameters:
            red (Vector): Représentation de Mandel

        Returns:
            Tensor: Tenseur 3D correspondant
        """
        sq2 = sqrt(2)
        
        # Modèles 1D
        if self._is_1d():
            return as_tensor([[red[0], 0, 0], [0, red[1], 0], [0, 0, red[2]]])
        
        # Modèles 2D
        elif self.name == "PlaneStrain":
            return as_tensor([[red[0], red[3]/sq2, 0], 
                             [red[3]/sq2, red[1], 0], 
                             [0, 0, red[2]]])
        elif self.name == "Axisymetric":
            return as_tensor([[red[0], 0, red[3]/sq2], 
                             [0, red[1], 0], 
                             [red[3]/sq2, 0, red[2]]])
        
        # Modèle 3D
        elif self.name == "Tridimensionnal":
            return as_tensor([[red[0], red[3]/sq2, red[4]/sq2], 
                             [red[3]/sq2, red[1], red[5]/sq2], 
                             [red[4]/sq2, red[5]/sq2, red[2]]])
    
    def tridim_to_Voigt(self, tens):
        """
        Convertit un tenseur 3D vers sa représentation de Voigt.
        
        L'ordre des composantes est: [11, 22, 33, 12, 13, 23]

        Parameters:
            tens (Tensor): Tenseur 3D

        Returns:
            Vector: Représentation de Voigt
        """
        return as_vector([tens[0,0], tens[1,1], tens[2,2],
                        2 * tens[1,2], 2 * tens[0,2], 2 * tens[0,1]])
    
    def Voigt_to_tridim(self, Voigt):
        """
        Convertit une représentation de Voigt vers un tenseur 3D.
        
        Note: Ne fonctionne pas pour les déformations à cause du facteur 2.

        Parameters:
            Voigt (Vector): Représentation de Voigt

        Returns:
            Tensor: Tenseur 3D correspondant
        """
        return as_tensor([[Voigt[0], Voigt[5], Voigt[4]],
                         [Voigt[5], Voigt[1], Voigt[3]],
                         [Voigt[4], Voigt[3], Voigt[2]]])
    
    # =========================================================================
    # Méthodes de transformation
    # =========================================================================
    
    def F_reduit(self, u):
        """
        Renvoie le gradient de la transformation sous forme réduite.

        Parameters:
            u (Function): Champ de déplacement

        Returns:
            Expression: Gradient de transformation réduit
        """
        # Modèles 1D
        if self.name == "CartesianUD":
            return as_vector([1 + u.dx(0), 1, 1])
        elif self.name == "CylindricalUD":
            return as_vector([1 + u.dx(0), 1 + u/self.r, 1])
        elif self.name == "SphericalUD":
            return as_vector([1 + u.dx(0), 1 + u/self.r, 1 + u/self.r])
        
        # Modèles 2D et 3D
        else:
            return Identity(3) + self.grad_3D(u)
    
    def F_3D(self, u):
        """
        Renvoie le gradient de la transformation complet (3D).

        Parameters:
            u (Function): Champ de déplacement

        Returns:
            Tensor: Gradient de transformation 3D
        """
        return Identity(3) + self.grad_3D(u)
    
    def J(self, u):
        """
        Renvoie le jacobien de la transformation (mesure de la dilatation locale).

        Parameters:
            u (Function): Champ de déplacement

        Returns:
            Expression: Jacobien de la transformation
        """
        # Modèles 1D
        if self.name == "CartesianUD":
            return 1 + u.dx(0)
        elif self.name == "CylindricalUD":
            return (1 + u.dx(0)) * (1 + u / self.r)
        elif self.name == "SphericalUD":
            return (1 + u.dx(0)) * (1 + u / self.r)**2
        
        # Modèles 2D
        elif self.name == "PlaneStrain":
            return det(Identity(2) + grad(u))
        elif self.name == "Axisymetric":
            F = self.F_reduit(u)
            return F[1, 1] * (F[0, 0] * F[2, 2] - F[0, 2] * F[2, 0])
        
        # Modèle 3D
        else:
            return det(Identity(3) + grad(u))
    
    def invF_reduit(self, u):
        """
        Renvoie l'inverse du gradient de la transformation sous forme réduite.

        Parameters:
            u (Function): Champ de déplacement

        Returns:
            Expression: Inverse du gradient de transformation
        """
        # Modèles 1D
        if self.name == "CartesianUD":
            return as_vector([1 / (1 + u.dx(0)), 1, 1])
        elif self.name == "CylindricalUD":
            return as_vector([1 / (1 + u.dx(0)), 1/(1 + u/self.r), 1])
        elif self.name == "SphericalUD":
            return as_vector([1 / (1 + u.dx(0)), 1 /(1 + u/self.r), 1 / (1 + u/self.r)])
        
        # Modèles 2D
        elif self.name == "PlaneStrain":
            inv_F2 = inv(Identity(2) + grad(u))
            return as_tensor([[inv_F2[0,0], inv_F2[0,1], 0], [inv_F2[1,0], inv_F2[1,1], 0], [0, 0, 1]])
        elif self.name == "Axisymetric":
            return self._get_axisymetric_invF(u)
        
        # Modèle 3D
        else:
            return inv(Identity(3) + grad(u))
    
    def _get_axisymetric_invF(self, u):
        """Méthode privée pour calculer l'inverse du gradient en axisymétrique"""
        grad_u = grad(u)
        prefacteur = (1 + grad_u[0,0]) * (1 + grad_u[1,1]) - grad_u[0,1] * (1 + grad_u[1,0])
        condition = ge(self.r, 1e-3)
        
        true_inv = as_tensor([[(1 + grad_u[1,1]) / prefacteur, 0, -grad_u[0,1] / prefacteur],
                            [0, 1 / (1 + u[0]/self.r), 0],
                            [-grad_u[1,0] / prefacteur, 0, (1 + grad_u[0,0]) / prefacteur]])
        
        hop_inv = as_tensor([[(1 + grad_u[1,1]) / prefacteur, 0, -grad_u[0,1] / prefacteur],
                           [0, 1 / (1 + grad_u[0,0]), 0],
                           [-grad_u[1,0] / prefacteur, 0, (1 + grad_u[0,0]) / prefacteur]])
        
        return conditional(condition, true_inv, hop_inv)
    
    def relative_gradient_3D(self, u, u_old):
        """
        Renvoie le gradient de la transformation relative entre deux configurations.

        Parameters:
            u (Function): Champ de déplacement actuel
            u_old (Function): Ancien champ de déplacement

        Returns:
            Tensor: Gradient de transformation relative
        """
        F_new = self.F_3D(u)
        inv_F_old = inv(self.F_3D(u_old))
        return dot(F_new, inv_F_old)
    
    # =========================================================================
    # Méthodes de déformation
    # =========================================================================
    
    def B(self, u):
        """
        Renvoie le tenseur de Cauchy-Green gauche.

        Parameters:
            u (Function): Champ de déplacement

        Returns:
            Expression: Tenseur de Cauchy-Green gauche (forme adaptée à la dimension)
        """
        F = self.F_reduit(u)
        if self._is_1d():
            return as_vector([F[0]**2, F[1]**2, F[2]**2])
        else:
            return dot(F, F.T)
    
    def B_3D(self, u):
        """
        Renvoie le tenseur de Cauchy-Green gauche complet (3D).

        Parameters:
            u (Function): Champ de déplacement

        Returns:
            Tensor: Tenseur de Cauchy-Green gauche 3D
        """
        F = self.F_3D(u)
        return dot(F, F.T)
    
    def C_3D(self, u):
        """
        Renvoie le tenseur de Cauchy-Green droit complet (3D).

        Parameters:
            u (Function): Champ de déplacement

        Returns:
            Tensor: Tenseur de Cauchy-Green droit 3D
        """
        F = self.F_3D(u)
        return dot(F.T, F)
    
    def BI(self, u):
        """
        Renvoie la trace du tenseur de Cauchy-Green gauche.

        Parameters:
            u (Function): Champ de déplacement

        Returns:
            Expression: Trace du tenseur de Cauchy-Green gauche
        """
        B = self.B(u)
        if self._is_1d():
            return sum(B[i] for i in range(3))
        else:
            return tr(B)
    
    def BBarI(self, u):
        """
        Renvoie la trace du tenseur de Cauchy-Green gauche isovolume.

        Parameters:
            u (Function): Champ de déplacement

        Returns:
            Expression: Trace du tenseur isovolume
        """
        return self.BI(u) / self.J(u)**(2./3)
    
    # =========================================================================
    # Méthodes d'intégration et opérations diverses
    # =========================================================================
    
    def measure(self, a, dx):
        """
        Renvoie la mesure d'intégration adaptée à la géométrie.
        
        La mesure est:
        - dx dans le cas Cartésien 1D, 2D plan ou 3D
        - r*dr dans le cas cylindrique 1D ou axisymétrique
        - r²*dr dans le cas sphérique 1D

        Parameters:
            a (Expression): Champ à intégrer
            dx (Measure): Mesure d'intégration par défaut

        Returns:
            Expression: Mesure d'intégration adaptée
        """
        if self.name in ["CartesianUD", "PlaneStrain", "Tridimensionnal"]:
            return a * dx
        elif self.name in ["CylindricalUD", "Axisymetric"]:
            return a * self.r * dx
        elif self.name == "SphericalUD":
            return a * self.r**2 * dx
    
    def div(self, v):
        """
        Renvoie la divergence du champ vectoriel v.

        Parameters:
            v (Function): Champ vectoriel

        Returns:
            Expression: Divergence du champ
        """
        if self._is_1d():
            return v.dx(0)
        else:
            return div(v)
    
    def push_forward(self, tensor, u):
        """
        Renvoie le push-forward d'un tenseur d'ordre 2 deux fois covariant.

        Parameters:
            tensor (Tensor): Tenseur d'ordre 2
            u (Function): Champ de déplacement

        Returns:
            Tensor: Tenseur après push-forward
        """
        F = self.F_3D(u)
        return dot(dot(F, tensor), F.T)
    
    def pull_back(self, tensor, u):
        """
        Renvoie le pull-back d'un tenseur d'ordre 2 deux fois covariant.

        Parameters:
            tensor (Tensor): Tenseur d'ordre 2
            u (Function): Champ de déplacement

        Returns:
            Tensor: Tenseur après pull-back
        """
        F = self.F_3D(u)
        inv_F = inv(F)
        return dot(dot(inv_F, tensor), inv_F.T)
    
    # =========================================================================
    # Méthodes privées d'aide
    # =========================================================================
    
    def _is_1d(self):
        """Vérifie si le modèle est unidimensionnel"""
        return self.name in self._model_config["dim1"]