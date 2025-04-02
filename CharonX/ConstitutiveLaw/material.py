"""
Created on Wed Apr  2 11:14:17 2025

@author: bouteillerp
"""

# Dans material.py

from .eos import (
    IsotropicHPP_EOS, U_EOS, Vinet_EOS, JWL_EOS, MACAW_EOS,
    MG_EOS, xMG_EOS, PMG_EOS, GP_EOS, NewtonianFluid_EOS, Tabulated_EOS
)

class Material:
    """Base class for all material models.
    
    This class defines the core properties and behavior of materials in simulations.
    
    Attributes
    ----------
    rho_0 : float or Expression Initial mass density (kg/m³)
    C_mass : float or Function Mass thermal capacity (J/(K·kg))
    celerity : float Wave propagation speed in the material
    """
    
    def __init__(self, rho_0, C_mass, eos_type, dev_type, eos_params, deviator_params, **kwargs):
        """Initialize a material with basic properties and constitutive models.
        
        Parameters
        ----------
        rho_0 : float or Expression Initial mass density (kg/m³)
        C_mass : float or Function Mass thermal capacity (J/(K·kg))
        eos_type : str Type of equation of state (e.g., "IsotropicHPP", "U5", "MG")
        dev_type : str or None Type of deviatoric behavior (e.g., "IsotropicHPP", "NeoHook", None)
        eos_params : dict Parameters for the equation of state model
        deviator_params : dict Parameters for the deviatoric behavior model
        **kwargs : dict
            Additional parameters (e.g., activation energy for chemical reactions)
        """
        # Store basic properties
        self.rho_0 = rho_0
        self.C_mass = C_mass
        self.eos_type = eos_type
        self.dev_type = dev_type
        
        # Initialize constitutive models
        self.eos = self._create_eos_model(eos_type, eos_params)
        self.devia = self._create_deviator_model(dev_type, deviator_params)
        
        # Calculate wave speed
        self.celerity = self.eos.celerity(rho_0)
        
        # Store optional parameters
        self.e_activation = kwargs.get("e_activation", None)
        self.kin_pref = kwargs.get("kin_pref", None)
        
        # Log material properties
        self._log_material_properties()
    
    def _create_eos_model(self, eos_type, params):
        """Create the appropriate equation of state model.
        
        Parameters
        ----------
        eos_type : str
            Type of equation of state
        params : dict
            Parameters for the equation of state model
            
        Returns
        -------
        EOS
            Equation of state object
            
        Raises
        ------
        ValueError
            If the equation of state type is unknown
        """
        eos_classes = {
            "IsotropicHPP": IsotropicHPP_EOS,
            "U1": U_EOS,
            "U2": U_EOS,
            "U3": U_EOS,
            "U4": U_EOS,
            "U5": U_EOS,
            "U7": U_EOS,
            "U8": U_EOS,
            "Vinet": Vinet_EOS,
            "JWL": JWL_EOS,
            "MACAW": MACAW_EOS,
            "MG": MG_EOS,
            "xMG": xMG_EOS,
            "PMG": PMG_EOS,
            "GP": GP_EOS,
            "NewtonianFluid": NewtonianFluid_EOS,
            "Tabulated": Tabulated_EOS
        }
        
        if eos_type not in eos_classes:
            raise ValueError(f"Unknown equation of state: {eos_type}")
        
        return eos_classes[eos_type](params)
    
    def _create_deviator_model(self, dev_type, params):
        """Create the appropriate deviatoric behavior model.
        
        Parameters
        ----------
        dev_type : str or None
            Type of deviatoric behavior
        params : dict
            Parameters for the deviatoric behavior model
            
        Returns
        -------
        DeviatoricModel
            Deviatoric behavior object or None
            
        Raises
        ------
        ValueError
            If the deviatoric type is unknown
        """
        if dev_type is None:
            return None_deviatoric(params)
            
        deviatoric_classes = {
            "IsotropicHPP": IsotropicHPP_deviatoric,
            "NeoHook": IsotropicHPP_deviatoric,
            "Hypoelastic": IsotropicHPP_deviatoric,
            "MooneyRivlin": MooneyRivlin_deviatoric,
            "NeoHook_Transverse": NeoHook_Transverse_deviatoric,
            "Lu_Transverse": Lu_Transverse_deviatoric,
            "Anisotropic": Anisotropic_deviatoric
        }
        
        if dev_type not in deviatoric_classes:
            raise ValueError(f"Unknown deviatoric behavior: {dev_type}")
        
        return deviatoric_classes[dev_type](params)
    
    def _log_material_properties(self):
        """Log important material properties for reference."""
        print(f"Thermal capacity: {self.C_mass}")
        print(f"Initial density: {self.rho_0}")
        print(f"Wave speed: {self.celerity}")