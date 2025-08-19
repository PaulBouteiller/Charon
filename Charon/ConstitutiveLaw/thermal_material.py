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
Created on Tue Dec 20 17:36:38 2022

@author: bouteillerp
Thermal Material Properties Module
==================================

This module defines thermal material properties for heat transfer simulations.
It provides classes for different types of thermal conductivity models,
including linear and nonlinear isotropic formulations.

Classes:
--------
LinearThermal : Linear isotropic thermal material
    Constant thermal conductivity independent of temperature and pressure
    
NonLinearThermal : Nonlinear isotropic thermal material
    Temperature and pressure-dependent thermal conductivity
    Suitable for materials with significant thermal property variations
"""

class LinearThermal:
    """Linear isotropic thermal material model.
    
    This class represents materials with constant thermal conductivity,
    independent of temperature and pressure.
    
    Attributes
    ----------
    lmbda : float Thermal conductivity coefficient (W/(m·K))
    type : str Type identifier ("LinearIsotropic")
    """
    def __init__(self, lmbda):
        """Initialize linear thermal material.
        
        Parameters
        ----------
        lmbda : float
            Thermal conductivity coefficient (W/(m·K))
        """
        self.lmbda = lmbda
        print("Le coefficient de diffusion est", self.lmbda)
        self.type = "LinearIsotropic"
        
class NonLinearThermal:
    """Nonlinear isotropic thermal material model.
    
    This class represents materials with temperature and pressure-dependent
    thermal conductivity, following the relation:
    λ(T,P) = λ₀ + a₁/T + a₂·max(P,0)
    
    Attributes
    ----------
    lmbda : float
        Base thermal conductivity coefficient (W/(m·K))
    a1 : float
        Temperature dependence coefficient (W·K/m)
    a2 : float
        Pressure dependence coefficient (W/(m·K·Pa))
    type : str
        Type identifier ("NonLinearIsotropic")
    """
    def __init__(self, lmbda, a1, a2):
        """Initialize nonlinear thermal material.
        
        Parameters
        ----------
        lmbda : float
            Base thermal conductivity coefficient (W/(m·K))
        a1 : float
            Temperature dependence coefficient (W·K/m)
        a2 : float
            Pressure dependence coefficient (W/(m·K·Pa))
        """
        self.lmbda = lmbda
        self.a1 = a1
        self.a2 = a2
        print("Le coefficient de diffusion est", self.lmbda)
        print("Le coefficient de dépendance en température est", self.a1)
        print("Le coefficient de dépendance en pression est", self.a2)
        self.type = "NonLinearIsotropic"