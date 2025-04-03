"""Deviatoric stress models for material behaviors."""
#Abstract class
from .base_deviator import BaseDeviator
#Common isotropic deviator
from .none_deviator import NoneDeviator
from .isotropic_hpp import IsotropicHPPDeviator
from .neo_hook import NeoHookDeviator
from .mooney_rivlin import MooneyRivlinDeviator
#Fluid deviator
from .newtonian_fluid import NewtonianFluidDeviator
#Anisotropic deviator
from .transverse_isotropic import NeoHookTransverseDeviator, LuTransverseDeviator
from .anisotropic import AnisotropicDeviator

from ufl import (tr, sym, dev, Identity, dot, inner, skew)
from dolfinx.fem import Function, Expression

__all__ = [
    'BaseDeviator',
    'Deviator',
    'NoneDeviator',
    'NewtonianFluidDeviator',
    'IsotropicHPPDeviator', 
    'NeoHookDeviator',
    'MooneyRivlinDeviator',
    'NeoHookTransverseDeviator',
    'LuTransverseDeviator',
    'AnisotropicDeviator'
]

class Deviator:
    """Main interface for deviatoric stress calculations.
    
    This class handles calculations for deviatoric stress components
    and dispatches to appropriate specialized models.
    """
    
    def __init__(self, kinematic, model, quadrature, is_hypo):
        """Initialize the deviator interface.
        
        Parameters
        ----------
        kinematic : Kinematic Kinematic handler for tensor operations
        model : str Model name (e.g., "CartesianUD", "PlaneStrain")
        quadrature : QuadratureHandler Handler for quadrature spaces
        is_hypo : bool Whether to use hypoelastic formulation
        """
        self.kin = kinematic
        self.model = model
        self.is_hypo = is_hypo
        
        if is_hypo:
            self.set_hypoelastic_deviator_function_space(model, quadrature)
    
    def set_hypoelastic_deviator_function_space(self, quadrature):
        """Set up function spaces for hypoelastic formulation."""
        if self.model == "CartesianUD":
            self.V_s = quadrature.quadrature_space(["Scalar"])
        elif self.model in ["CylindricalUD", "SphericalUD"]:
            self.V_s = quadrature.quadrature_space(["Vector", 2])
        elif self.model == "PlaneStrain":
            self.V_s = quadrature.quadrature_space(["Vector", 3])
        elif self.model == "Axisymetric":
            self.V_s = quadrature.quadrature_space(["Vector", 4])
        elif self.model == "Tridimensionnal":
            self.V_s = quadrature.quadrature_space(["Tensor", 3, 3])
    
    def set_elastic_dev(self, u, v, J, T, T0, material):
        """Dispatch to the appropriate deviator model based on material type."""
        return material.devia.calculate_stress(u, v, J, T, T0, self.kin)
    
    def set_hypoelastic_deviator(self, u, v, J, mu):
        """Calculate the deviatoric stress for hypoelastic formulation."""
        self.s = Function(self.V_s, name="Deviator")
        s_3D = self.kin.reduit_to_3D(self.s, sym=True)
        L = self.kin.reduit_to_3D(self.kin.Eulerian_gradient(v, u))
        D = sym(L)
        
        B = self.kin.B_3D(u)

        # Jaumann rate formulation
        s_Jaumann_3D = mu/J**(5./3) * (dot(B, D) + dot(D, B) 
                                      - 2./3 * inner(B,D) * Identity(3)
                                      - 5./3 * tr(D) * dev(B))
        s_Jaumann = self.kin.tridim_to_reduit(s_Jaumann_3D, sym=True)
        
        if self.model in ["CartesianUD", "CylindricalUD", "SphericalUD"]:
            self.dot_s = Expression(s_Jaumann, self.V_s.element.interpolation_points())
        else:
            Omega = skew(L)
            Jaumann_corr = self.kin.tridim_to_reduit(
                dot(Omega, s_3D) - dot(s_3D, Omega), sym=True
            )
            self.dot_s = Expression(
                s_Jaumann + Jaumann_corr, 
                self.V_s.element.interpolation_points()
            )
        return s_3D