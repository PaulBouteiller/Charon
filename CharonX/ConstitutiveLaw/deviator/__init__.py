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
#Hypoelastic deviator
from .hypoelastic import HypoelasticDeviator
__all__ = ['BaseDeviator', 'Deviator', 'NoneDeviator', 'NewtonianFluidDeviator',
           'IsotropicHPPDeviator', 'NeoHookDeviator', 'MooneyRivlinDeviator',
           'NeoHookTransverseDeviator', 'LuTransverseDeviator',
           'AnisotropicDeviator', 'HypoelasticDeviator']

class Deviator:
    """Main interface for deviatoric stress calculations.
    
    This class acts as a facade to the underlying specialized deviatoric models,
    delegating calculations to the appropriate implementation based on the
    material type and simulation settings.
    
    Attributes
    ----------
    kin : Kinematic Kinematic handler for tensor operations
    model : str Model name (e.g., "CartesianUD", "PlaneStrain")
    is_hypo : bool Whether to use hypoelastic formulation
    hypo_deviator : HypoelasticDeviator, optional
            Instance of hypoelastic deviator if is_hypo is True
    """
    
    def __init__(self, kinematic, model, quadrature, material):
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
        
        def is_in_list(material, attribut, keyword):
            is_mult = isinstance(material, list)
            return (is_mult and any(getattr(mat, attribut) == keyword for mat in material)) or \
                (not is_mult and getattr(material, attribut) == keyword)

        self.is_hypo = is_in_list(material, "dev_type", "Hypoelastic")
        
        
        # self.is_hypo = material.dev_type == "Hypoelastic"
        self.quadrature = quadrature
        if self.is_hypo:
            material.devia.set_hypoelastic(kinematic, model, quadrature)
    
    def set_elastic_dev(self, u, v, J, T, T0, material):
        """Delegate to the appropriate deviator model based on material type.
        
        Parameters
        ----------
        u, v, J, T, T0 : Function See stress_3D method in ConstitutiveLaw.py for details.
        material : Material Material properties with deviator model
            
        Returns
        -------
        Function Deviatoric stress tensor
        """
        return material.devia.calculate_stress(u, v, J, T, T0, self.kin)
    
    def set_hypoelastic_deviator(self, u, v, J, material):
        """Calculate the deviatoric stress for hypoelastic formulation.
        
        This method handles both the initialization of the hypoelastic deviator
        (if needed) and the calculation of the current stress rate. The actual
        time integration and stress update is handled by HypoElasticSolver.
        
        Parameters
        ----------
        u : Function Displacement field
        v : Function Velocity field
        J : Function Jacobian of the transformation
        mu : float Shear modulus
            
        Returns
        -------
        Function 3D deviatoric stress tensor
        """
        return material.devia.calculate_stress_rate(u, v, J, material)