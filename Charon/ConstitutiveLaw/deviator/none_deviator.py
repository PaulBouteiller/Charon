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
Created on Wed Apr  2 11:34:08 2025

@author: bouteillerp

Null Deviatoric Stress Model
===========================

This module implements a null deviatoric stress model, which always returns
zero for the deviatoric part of the stress tensor. This model is appropriate
for pure hydrostatic materials, such as perfect fluids, where the stress
consists only of pressure.
Classes:
--------
NoneDeviator : Null deviatoric model
    Returns zero deviatoric stress for all inputs
    Requires no parameters
    Provides consistent interface with other deviatoric models
"""

from ufl import Identity
from .base_deviator import BaseDeviator

class NoneDeviator(BaseDeviator):
    """Null deviatoric model for pure hydrostatic simulations.
    
    This model returns zero deviatoric stress, suitable for
    pure fluid simulations where only pressure contributes to stress.
    """
    
    def required_parameters(self):
        """Return the list of required parameters.
        
        Returns
        -------
        list Empty list (no parameters required)
        """
        return []
    
    def __init__(self, params):
        """Initialize the null deviatoric model.
        
        Parameters
        ----------
        params : dict Unused dictionary (no parameters needed)
        """
        super().__init__(params)
    
    def calculate_stress(self, u, v, J, T, T0, kinematic):
        """Calculate the deviatoric stress tensor (always zero).
        
        Parameters
        ----------
        u, v, J, T, T0 : Function See stress_3D method in ConstitutiveLaw.py for details.
            
        Returns
        -------
        Function Zero tensor (3x3)
        """
        return 0 * Identity(3)