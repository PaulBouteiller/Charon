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
Phase Field Damage Model for Brittle Fracture
==============================================

This module implements phase field approaches to brittle fracture,
including AT1, AT2, and Wu formulations. These models represent cracks
as diffuse damage bands rather than discrete discontinuities.

Classes:
--------
PhaseFieldDamage : Phase field damage model for brittle fracture
    Implements AT1, AT2, and Wu formulations
    Provides energy-based approach with regularization
    Handles degradation functions for material stiffness
"""

from ufl import TestFunction, TrialFunction, dot, grad
from dolfinx.fem import Function, functionspace, Constant, Expression
from numpy import pi

from .base_damage import BaseDamage


class PhaseFieldDamage(BaseDamage):
    """Phase field damage model for brittle fracture.

    Implements phase field approaches (AT1, AT2, Wu) for brittle fracture,
    representing cracks as diffuse damage bands.
    
    Attributes
    ----------
    V_d : FunctionSpace Function space for damage field
    max_d : Function Maximum damage field (upper bound)
    d : Function Current damage field
    inf_d : Function Minimum damage field
    d_prev : Function Previous damage field (for prediction)
    d_ : TestFunction Test function for damage
    dd : TrialFunction Trial function for damage
    Gc : float Critical energy release rate
    l0 : float Regularization length
    E : float, optional Young's modulus (for Wu model)
    PF_model : str Phase field model type (AT1, AT2, Wu)
    g_d : Expression Degradation function
    energy : Expression, optional Elastic energy with damage
    fracture_energy : Expression, optional Energy dissipated by fracture
    sigma_c : float, optional Critical stress (Wu model)
    wu_softening_type : str, optional Softening type for Wu model
    """
    
    def __init__(self, mesh, quadrature, dictionnaire, u=None, J=None, pressure=None, material=None, kinematic=None):
        """Initialize the phase field damage model.

        Parameters
        ----------
        mesh : Mesh Computational mesh
        quadrature : QuadratureHandler Handler for quadrature integration
        dictionnaire : dict Parameters for phase field model
        """
        super().__init__(mesh, quadrature, dictionnaire, u, J, pressure, material, kinematic)

        
        # Initialize function spaces
        self.V_d = functionspace(mesh, ('CG', self.dam_parameters["degree"]))
        interp_points = self.V_d.element.interpolation_points()
        
        # Initialize damage functions
        self.max_d = Function(self.V_d, name="maximum damage")
        self.max_d.interpolate(Expression(Constant(mesh, 1 - self.residual_stiffness), interp_points))
        
        self.d = Function(self.V_d, name="Damage")
        self.d.interpolate(Expression(Constant(mesh, self.default_damage), interp_points))
        
        self.inf_d = Function(self.V_d, name="Inf Damage")
        self.inf_d.interpolate(Expression(Constant(mesh, self.default_damage), interp_points))
        
        self.d_prev = Function(self.V_d, name="Damage_predictor")
        self.d_prev.interpolate(Expression(Constant(mesh, self.default_damage), interp_points))
        
        # Initialize test and trial functions
        self.d_ = TestFunction(self.V_d)
        self.dd = TrialFunction(self.V_d)

    def set_damage(self, mesh, dictionnaire, PF_model="AT2"):
        """Initialize the phase field damage model parameters.

        Parameters
        ----------
        mesh : Mesh Computational mesh
        dictionnaire : dict
            Additional parameters:
            - Gc (float): Critical energy release rate
            - l0 (float): Regularization length
            - E (float, optional): Young's modulus (for Wu model)
            - sigma_c (float, optional): Critical stress (for Wu model)
            - wu_softening (str, optional): Softening type for Wu model
        PF_model : str, optional
            Phase field model type: "AT1", "AT2", or "wu", default is "AT2"
        """
        self.Gc = dictionnaire["Gc"]
        self.l0 = dictionnaire["l0"]
        self.E = dictionnaire.get("E", None)
        self.PF_model = PF_model
        
        if self.PF_model == "wu":
            self.sigma_c = dictionnaire["sigma_c"]
            self.wu_softening_type = dictionnaire["wu_softening"]
            
        self._set_degradation_function()
        
    def _set_degradation_function(self):
        """Initialize the stiffness degradation functions.
        
        Sets up the degradation function that weights material stiffness
        based on the damage field.
        """
        if self.PF_model in ["AT1", "AT2"]:
            self.g_d = (1 - self.d)**2
            if self.E is not None:
                sig_c_AT1 = (3 * self.E * self.Gc / (8 * self.l0))**(1./2)
                print("La contrainte critique du model AT1 vaut ici", sig_c_AT1)
        elif self.PF_model == "wu":
            self.g_d = self._wu_degradation_function(self.d, self.sigma_c, self.Gc, self.l0, self.E)
            
    def _wu_degradation_function(self, d, sigma_c, Gc, l_0, E):
        """Compute Wu's degradation function for cohesive phase field.
        
        Parameters
        ----------
        d : Function Damage field
        sigma_c : float or Expression Critical stress
        Gc : float or Expression Critical energy release rate
        l_0 : float or Expression Regularization length
        E : float or Expression Young's modulus
            
        Returns
        -------
        Expression Wu degradation function
            
        Raises
        ------
        ValueError If regularization length is too large for the given parameters
        """
        a_1 = 4 * E * Gc / (pi * l_0 * sigma_c**2)
        if isinstance(sigma_c, float):
            if a_1 <= 3./2.:
                raise ValueError("A smaller regularization length l_0 has to be chosen")
        p, a_2 = self._wu_softening()
        return (1 - d)**p / ((1 - d)**p + a_1 * d + a_1 * a_2 * d**2)
    
    def _wu_softening(self):
        """Get coefficients for Wu's softening laws.
        
        Returns
        -------
        tuple (p, a_2) coefficients for the chosen softening type
        """
        if self.wu_softening_type == "exp":
            p = 2.5
            a_2 = 2**(5./3) - 3
        elif self.wu_softening_type == "lin":
            p = 2
            a_2 = -0.5
        elif self.wu_softening_type == "bilin":
            p = 4
            a_2 = 2**(7./3) - 4.5
        return p, a_2
    
    def _initialize_driving_force(self, u, J, pressure, material, kinematic):
        """Initialize driving force for phase field models."""
        if material is not None and kinematic is not None:
            eHelm = self._helmholtz_energy(u, J, material, kinematic)
            self.set_NL_energy(eHelm)
            
    def _helmholtz_energy(self, u, J, mat, kinematic):
        """Return the Helmholtz free energy.
        
        This method delegates the calculation to the appropriate 
        EOS and deviator models.
        
        Parameters
        ----------
        u : Function Displacement field
        J : Expression Jacobian of the transformation
        mat : Material Material to study
            
        Returns
        -------
        Helmholtz free energy
        """
        # Get volumetric energy from EOS
        try:
            psi_vol = mat.eos.volumetric_helmholtz_energy(u, J, self.kinematic, mat.eos_type)
        except:
            raise ValueError("Phase field analysis has not been implemented for this eos")
        # Get isochoric energy from deviator
        try:
            psi_iso_vol = mat.devia.isochoric_helmholtz_energy(u, self.kinematic)
        except:
            raise ValueError("Phase field analysis has not been implemented for this deviatoric law")
        return psi_vol + psi_iso_vol

    def set_NL_energy(self, psi):
        """Set elastic and fracture energies for phase field models.
        
        Parameters
        ----------
        psi : Expression Undamaged elastic energy
        """
        self.energy = self.g_d * psi 
        self.fracture_energy = self._set_phase_field_fracture_energy(self.Gc, self.l0, self.d)
                
    def _set_phase_field_fracture_energy(self, Gc, l0, d):
        """Set the fracture energy density for phase field.
        
        Parameters
        ----------
        Gc : float or Expression Critical energy release rate
        l0 : float or Expression Regularization length
        d : Function Damage field
            
        Returns
        -------
        Expression Fracture energy density
        """
        if self.PF_model == "AT1":
            cw = 8. / 3
            w_d = d
        elif self.PF_model == "AT2":
            cw = 2.
            w_d = d**2
        elif self.PF_model == "wu":
            cw = pi
            xi = 2
            w_d = xi * d + (1 - xi) * d**2
        
        # Le gradient est bien cohérent quelque soit le modèle.
        return Gc / cw * (w_d / l0 + l0 * dot(grad(d), grad(d)))