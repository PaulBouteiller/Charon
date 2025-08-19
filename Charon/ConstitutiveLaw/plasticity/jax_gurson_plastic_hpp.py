# Copyright 2025 CEA
"""
Modèle GTN simplifié basé sur Beardsmore et al. (2006)

@author: bouteillerp
"""

from .base_plastic import Plastic
from dolfinx.fem import Function

class GTNSimplePlasticity(Plastic):
    """GTN model simplifié selon Beardsmore et al. (2006)
    
    Système de 4 équations couplées non-linéaires:
    - f1: condition de consistance 
    - f2: fonction de charge GTN
    - f3: évolution déformation plastique équivalente microscopique
    - f4: évolution fraction volumique de vides
    """
    
    def _set_function(self, element, quadrature):
        """Initialize functions for GTN simple plasticity."""
        # Déformation plastique équivalente macroscopique
        self.V_eps_p_eq = quadrature.quadrature_space(["Scalar"])
        self.eps_p_eq = Function(self.V_eps_p_eq, name="Plastic_strain_eq")
        
        # Déformation plastique hydrostatique
        self.V_eps_p_m = quadrature.quadrature_space(["Scalar"])
        self.eps_p_m = Function(self.V_eps_p_m, name="Plastic_strain_hydro")
        
        # Déformation plastique équivalente microscopique (matériau dense)
        self.V_eps_p_eq_M = quadrature.quadrature_space(["Scalar"])
        self.eps_p_eq_M = Function(self.V_eps_p_eq_M, name="Microscopic_plastic_strain")
        
        # Fraction volumique de vides
        self.V_f = quadrature.quadrature_space(["Scalar"])
        self.f = Function(self.V_f, name="Void_fraction")
        
        # Paramètres GTN selon Beardsmore et al. (2006)
        self.q1 = 1.5
        self.q2 = 1.0
        
        # Paramètres de coalescence (équation 2 de l'article)
        self.fc = 0.15  # Fraction critique de coalescence
        self.ff = 0.25  # Fraction à rupture
        
        # Paramètres de nucléation (équation 20)
        self.fN = 0.04   # Fraction volumique due aux particules
        self.eps_N = 0.3 # Déformation moyenne de nucléation  
        self.sN = 0.1    # Écart-type
        
        # Initialisation
        self.f.x.array[:] = 0.001  # Porosité initiale
               
    def compute_deviatoric_stress(self, u, v, J, T, T0, material, deviator):
        """Calcul de la contrainte déviatorique pour GTN simple"""
        # Contrainte élastique trial
        if material.dev_type == "Hypoelastic":
            deviatoric = deviator.set_hypoelastic_deviator(u, v, J, material)
        else:
            deviatoric = deviator.set_elastic_dev(u, v, J, T, T0, material)
            
        # Correction plastique sera appliquée dans le solveur
        return deviatoric