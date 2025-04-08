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
Created on Wed Apr  2 11:14:17 2025

@author: bouteillerp
"""
from .eos import (IsotropicHPPEOS, UEOS, VinetEOS, JWLEOS, MACAWEOS,
                  MGEOS, xMGEOS, PMGEOS, GPEOS, NewtonianFluidEOS, TabulatedEOS)

from .deviator import (NoneDeviator, IsotropicHPPDeviator, MooneyRivlinDeviator,
                       NeoHookTransverseDeviator, LuTransverseDeviator, 
                       AnisotropicDeviator, NeoHookDeviator, HypoelasticDeviator)


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
        **kwargs : dict Additional parameters (e.g., activation energy for chemical reactions)
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
        print(f"Thermal capacity: {self.C_mass}")
        print(f"Initial density: {self.rho_0}")
        print(f"Wave speed: {self.celerity}")
    
    def _create_eos_model(self, eos_type, params):
        """Create the appropriate equation of state model.
        
        Parameters
        ----------
        eos_type : str Type of equation of state
        params : dict Parameters for the equation of state model
            
        Returns
        -------
        EOS Equation of state object
        """
        eos_classes = {"IsotropicHPP": IsotropicHPPEOS, "U1": UEOS, "U5": UEOS, "U8": UEOS,
                       "Vinet": VinetEOS, "JWL": JWLEOS, "MACAW": MACAWEOS,
                       "MG": MGEOS, "xMG": xMGEOS, "PMG": PMGEOS, "GP": GPEOS,
                       "NewtonianFluid": NewtonianFluidEOS, "Tabulated": TabulatedEOS}
        
        if eos_type not in eos_classes:
            raise ValueError(f"Unknown equation of state: {eos_type}")
        
        return eos_classes[eos_type](params)
    
    def _create_deviator_model(self, dev_type, params):
        """Create the appropriate deviatoric behavior model.
        
        Parameters
        ----------
        dev_type : str or None Type of deviatoric behavior
        params : dict Parameters for the deviatoric behavior model
            
        Returns
        -------
        DeviatoricModel Deviatoric behavior object or None
        """
        if dev_type is None:
            return NoneDeviator(params)
            
        deviatoric_classes = {"IsotropicHPP": IsotropicHPPDeviator,
                              "NeoHook": NeoHookDeviator,
                              "Hypoelastic": HypoelasticDeviator,
                              "MooneyRivlin": MooneyRivlinDeviator,
                              "NeoHook_Transverse": NeoHookTransverseDeviator,
                              "Lu_Transverse": LuTransverseDeviator,
                              "Anisotropic": AnisotropicDeviator}
        
        if dev_type not in deviatoric_classes:
            raise ValueError(f"Unknown deviatoric behavior: {dev_type}")
        
        return deviatoric_classes[dev_type](params)