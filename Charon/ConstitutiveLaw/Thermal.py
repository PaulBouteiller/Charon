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
Created on Wed Nov 23 16:33:35 2022

@author: bouteillerp
Thermal Behavior Module for Mechanical Simulations
=================================================

This module implements thermal behavior models for solids and fluids in mechanical
simulations. It provides functionality for calculating thermal energy, heat flux,
and temperature-dependent material properties.

The Thermal class manages the overall thermal behavior of the material system,
handling both single-phase and multiphase materials. It calculates thermal
capacities, conductivities, and constitutive relations for heat transfer.

Key features include:
- Calculation of tangent volumetric thermal capacity
- Support for multiphase materials with weighted averaging
- Implementation of various thermal constitutive laws
- Linear and nonlinear temperature-dependent conductivity models
- Coupling with mechanical fields (pressure, density)
"""
from ..utils.generic_functions import ppart

class Thermal:
    """Manager for thermal behavior in solids and fluids.
    
    This class handles the thermal aspects of material behavior, including
    heat capacity, thermal conductivity, and temperature-dependent properties.
    
    Attributes
    ----------
    material : Material or list
        Material properties (single material or list for multiphase)
    multiphase : Multiphase or None
        Object managing multiphase properties
    T0 : Function
        Initial temperature field
    T : Function
        Current temperature field
    P : Function
        Pressure field
    C_tan : Expression
        Tangent thermal capacity
    """
    
    def __init__(self, mat, multiphase, kinematic, T0, T, P):
        """Initialize thermal behavior manager.

        Parameters
        ----------
        mat : Material or list
            Material properties (single material or list for multiphase)
        multiphase : Multiphase or None
            Object managing multiphase properties
        kinematic : Kinematic
            Kinematic handler for tensor operations
        T0 : Function
            Initial temperature field
        T : Function
            Current temperature field
        P : Function
            Pressure field
        """
        self.mat = mat
        self.multiphase = multiphase
        self.T0 = T0
        self.T = T
        self.P = P
        
    def set_tangent_thermal_capacity(self):
        """Calculate the total tangent volumetric thermal capacity.
        
        Computes the volumetric thermal capacity (variation in internal energy
        with respect to temperature), handling both single-phase and
        multiphase materials.
        """
        if isinstance(self.mat, list):
            n_mat = len(self.mat)
            if all([self.mat[0].C_mass == self.mat[i].C_mass for i in range(n_mat)]):
                self.C_tan = self.partial_C_vol_tan(self.mat[0])
            else:
                self.C_tan = sum(c * self.partial_C_vol_tan(mat) for c, mat in zip(self.multiphase.c, self.mat))
        else:
            self.C_tan = self.partial_C_vol_tan(self.mat)
    
    def partial_C_vol_tan(self, mat):
        """Calculate partial volumetric thermal capacity for a material.
        
        Parameters
        ----------
        mat : Material
            Material properties
            
        Returns
        -------
        Expression
            Volumetric thermal capacity
        """
        CTan = mat.rho_0 * mat.C_mass
        return CTan   
    
    def thermal_constitutive_law(self, therm_mat, grad_dT):
        """Define the total thermal constitutive law.
        
        Calculates the heat flux based on the temperature gradient,
        handling both single-phase and multiphase materials.
        
        Parameters
        ----------
        therm_mat : ThermalMaterial or list
            Thermal material properties
        grad_dT : Expression
            Gradient of temperature variation
            
        Returns
        -------
        Expression
            Heat flux
        """
        if isinstance(therm_mat, list):
            return sum(g * self.partial_thermal_constitutive_law(mat) for g, mat in zip(self.multiphase.g, therm_mat))
        else:
            return self.partial_thermal_constitutive_law(therm_mat, grad_dT)
    
    def partial_thermal_constitutive_law(self, therm_mat, grad_dT):
        """Calculate the partial thermal constitutive law for a material.
        
        Parameters
        ----------
        therm_mat : ThermalMaterial
            Thermal material properties
        grad_dT : Expression
            Gradient of temperature variation
            
        Returns
        -------
        Expression
            Heat flux
            
        Raises
        ------
        ValueError
            If an unknown thermal model is specified
        """
        if therm_mat.type == "LinearIsotropic":
            return self.LinearFourrier(therm_mat.lmbda, grad_dT)
        elif therm_mat.type == "NonLinearIsotropic":
            return self.NonLinearFourrier(therm_mat.lmbda, therm_mat.a1, therm_mat.a2, grad_dT)
        
    def TATB_massique_capacity_correction(self, a, b, T, C_mass):
        """Apply correction to mass thermal capacity for TATB materials.

        Parameters
        ----------
        a, b : float
            Correction parameters
        T : Function
            Temperature field
        C_mass : float or Function
            Mass thermal capacity to correct
            
        Notes
        -----
        Requires temperature to be greater than 10K.
        """
        assert min(T.x.array)>10
        C_mass *= 1 / pow(1 + pow(T, a), b)
        
    def LinearFourrier(self, lmbda, grad_dT):
        """Implement linear isotropic Fourier's law.
        
        Parameters
        ----------
        lmbda : float
            Thermal conductivity coefficient
        grad_dT : Expression
            Gradient of temperature variation
            
        Returns
        -------
        Expression
            Heat flux
        """
        return lmbda * grad_dT
    
    def NonLinearFourrier(self, lmbda, a1, a2, grad_dT):
        """Implement nonlinear isotropic Fourier's law.
        
        Includes temperature and pressure dependence of thermal conductivity.
        
        Parameters
        ----------
        lmbda : float
            Base thermal conductivity coefficient
        a1 : float
            Temperature dependence coefficient
        a2 : float
            Pressure dependence coefficient
        grad_dT : Expression
            Gradient of temperature variation
            
        Returns
        -------
        Expression
            Heat flux
        """
        return (lmbda + a1/self.T + a2 * ppart(self.P)) * grad_dT