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
Created on Wed Apr  2 11:37:27 2025

@author: bouteillerp
Anisotropic Hyperelastic Deviatoric Stress Model
================================================

This module implements a general anisotropic hyperelastic framework for the deviatoric
part of the stress tensor. It provides a flexible approach to modeling materials with
complex directional properties, supporting various symmetry classes including orthotropic,
transversely isotropic, and fully anisotropic behavior.

The implementation features:
- Support for direct specification of the stiffness tensor
- Built-in material parameter conversion for common symmetry classes
- Transformation capabilities to align material axes
- Optional nonlinear stiffness modulation
- Integrated calibration procedures for fitting model parameters to experimental data

Classes:
--------
AnisotropicDeviator : General anisotropic hyperelastic model
    Supports multiple initialization methods
    Implements the stress calculation for anisotropic hyperelasticity
    Provides tensor transformation utilities
    Includes calibration utilities for fitting to experimental data
"""
from ...utils.maths.fitting import fit_and_plot_shifted_polynomial_fixed
from ...utils.maths.tensor_rotations import rotation_matrix_direct, rotate_stifness, euler_to_rotation
from ...utils.time_dependent_expressions import interpolation_lin

from .base_deviator import BaseDeviator

from ufl import as_tensor, as_matrix, dev, inv, inner, dot, Identity
from numpy import array, diag, ndarray, insert, concatenate, asarray, full_like, identity
from numpy.linalg import norm

from dolfinx.fem import functionspace, Function
from dolfinx import default_scalar_type

def bulk_anisotropy_tensor(Rigi, module):
    voigt_M0 = [sum(Rigi[i,column] for i in range(3)) for column in range(6)]
    if module == "numpy":
        return array([[voigt_M0[0], voigt_M0[3], voigt_M0[4]],
                      [voigt_M0[3], voigt_M0[1], voigt_M0[5]],
                      [voigt_M0[4], voigt_M0[5], voigt_M0[2]]])
    else:
        return as_tensor([[voigt_M0[0], voigt_M0[3], voigt_M0[4]],
                          [voigt_M0[3], voigt_M0[1], voigt_M0[5]],
                          [voigt_M0[4], voigt_M0[5], voigt_M0[2]]])

def polynomial_expand(x, point, coeffs):
    return coeffs[0] + sum(coeff * (x - point)**(i+1) for i, coeff in enumerate(coeffs[1:]))

def polynomial_derivative(x, point, coeffs):
    return coeffs[1] * (x - point) + sum(coeff * (i+2) * (x - point)**(i+1) for i, coeff in enumerate(coeffs[2:]))

class AnisotropicDeviator(BaseDeviator):
    """General anisotropic hyperelastic deviatoric stress model.
    
    This model implements a general anisotropic formulation suitable for
    materials with complex directional properties. It includes integrated
    calibration procedures for fitting model parameters to experimental data.
    
    Attributes
    ----------
    C : array  Stiffness tensor in Voigt notation
    M0 : array Bulk anisotropy tensor derived from the stiffness tensor
    g_func_coeffs : array or None  Coefficients for coupling stress modulation functions
    f_func_coeffs : list or None  Coefficients for optional stiffness modulation functions
    """
    
    def required_parameters(self):
        """Return the list of required parameters."""
        return ["C"]
    
    def __init__(self, params):
        """Initialize the anisotropic deviatoric model.
        
        Parameters
        ----------
        params : dict
            Must contain:
            - "C": 6x6 stiffness tensor in Voigt notation
            
            Optional:
            - "f_func": Coefficients for stiffness modulation functions
            - "g_func": Coefficients for coupling stress modulation functions
            - "calibration_data": Dictionary for experimental calibration
        """
        self.params = params  # Store for required_parameters method
        super().__init__(params)
        
        # Only direct stiffness tensor accepted
        if "C" not in params:
            raise ValueError("AnisotropicDeviator requires a stiffness tensor 'C'. "
                           "Use stiffness_builders utility to construct C from material parameters.")
        
        self.RigiLin = params["C"]
        
        calibration_data = params.get("calibration_data", None)
        if calibration_data is not None:
            self._process_calibration_data(calibration_data)
        else:
            self.f_func_coeffs = params.get("f_func", None)
            self.g_func_coeffs = params.get("g_func", None)
        # Store coupling stress modulation parameters if provided
        if self.g_func_coeffs is not None:
            self.g_func_coeffs = insert(self.g_func_coeffs, 0, 1)
            
        # Store stiffness modulation parameters if provided
        if self.f_func_coeffs is not None:
            size = len(self.f_func_coeffs)
            for i in range(size):
                for j in range(size):
                    if isinstance(self.f_func_coeffs[i][j], ndarray):
                        self.f_func_coeffs[i][j] = insert(self.f_func_coeffs[i][j], 0, 1)
                        
    def _process_calibration_data(self, calibration_data):
        """Process calibration data to fit g and f functions.
        
        Parameters
        ----------
        calibration_data : dict
            Dictionary containing calibration data and settings
        """
        spherical_data = calibration_data.get("spherical_data", None)
        shear_data = calibration_data.get("shear_data", None)
        plot = calibration_data.get("plot", False)
        save = calibration_data.get("save", False)
        
        # First calibrate the g_func (volumetric coupling functions)
        if spherical_data is not None:
            self.g_data, self.g_func_coeffs = self._orthotropic_unified_gij_fit(spherical_data, plot, save)
            
        # Then calibrate the f_func (shear modulation functions) if shear data available
        if shear_data is not None and spherical_data is not None:
            self.f_func_coeffs = self.calibrate_fij(shear_data, spherical_data, plot, save)
        else:
            self.f_func_coeffs = None
        
    def set_orientation(self, mesh_manager, polycristal_dic):
        mesh = mesh_manager.mesh
        Q = functionspace(mesh, ("DG", 0))
        if "angle" in polycristal_dic and "axis" in polycristal_dic:
            angle_func = Function(Q)
            Q_axis = functionspace(mesh, ("DG", 0, (3, )))    
            axis_func = Function(Q_axis)
            for index, angle, axis in zip(polycristal_dic["tags"], polycristal_dic["angle"], polycristal_dic["axis"]):
                cells = mesh_manager.cell_tags.find(index)
                angle_func.x.array[cells] = full_like(cells, angle, dtype = default_scalar_type)
                axis_func.x.array[3 * cells] = full_like(cells, axis[0], dtype=default_scalar_type)
                axis_func.x.array[3 * cells + 1] = full_like(cells, axis[1], dtype=default_scalar_type)
                axis_func.x.array[3 * cells + 2] = full_like(cells, axis[2], dtype=default_scalar_type)
            self.R = rotation_matrix_direct(angle_func, axis_func)
        elif "euler_angle" in polycristal_dic:
            phi1_func, Phi_func, phi2_func = Function(Q), Function(Q), Function(Q)
            for index, euler_angle in zip(polycristal_dic["tags"], polycristal_dic["euler_angle"]):
                cells = mesh_manager.cell_tags.find(index)
                phi1_func.x.array[cells] = full_like(cells, euler_angle[0], dtype = default_scalar_type)
                Phi_func.x.array[cells] = full_like(cells, euler_angle[1], dtype=default_scalar_type)
                phi2_func.x.array[cells] = full_like(cells, euler_angle[2], dtype=default_scalar_type)
            self.R = euler_to_rotation(phi1_func, Phi_func, phi2_func)            
            
        
    def _orthotropic_unified_gij_fit(self, data, plot, save, tol_bulk_isotropy = 1):
        """Calibrate unified volumetric coupling functions from experimental data.
           
           This method fits a single polynomial function g(J) to describe the nonlinear
           coupling between volumetric deformation and deviatoric stress in anisotropic
           materials. The approach assumes g_xx = g_yy = g_zz for simplicity, reducing
           the number of parameters to fit.
           Parameters
           ----------
           data : dict
               Experimental data containing:
               - "J" : array_like
                   Jacobian values (volumetric deformation ratio). Should span
                   the expected deformation range in the simulation.
               - "s_xx", "s_yy", "s_zz" : array_like
                   Deviatoric stress components for each J value (Pa).
                   These should be from controlled volumetric tests.
               - "degree" : int
                   Polynomial degree for fitting (typically 2-4).
               - Additional plotting/saving options.
               
           plot : bool Whether to display fitting plots for quality assessment.
           save : bool Whether to save plots to files.
               
           Returns
           -------
           tuple
               - g_data : list of arrays
                   Original g_ij data points [g_xx, g_yy, g_zz] for validation.
               - poly_coeffs : ndarray or None
                   Polynomial coefficients for unified g(J) function.
                   None if material is isotropic (|dev(M₀)| < 1).
                   
           Notes
           -----
           Algorithm:
           1. Check if material is volumetrically isotropic by examining dev(M₀)
           2. If isotropic, return unity functions (no coupling needed)
           3. If anisotropic, extract g_ij from experimental data using:
              g_ij(J) = σ_dev,ij / (prefactor * M₀_dev,ij)
           4. Fit unified polynomial to combined g_xx, g_yy, g_zz data
           5. Validate fit quality and return coefficients
           """
        def prefactor(J):
            return (J - 1) / (3 * J) 
        
        def inv_prefactor(J):
            3 * J / (J - 1)

        # Vérifier si le matériau est isotrope
        J_list = data.get("J")
        M0_diag = diag(bulk_anisotropy_tensor(self.RigiLin, module = "numpy"))
        trace_M0 = sum(M0_diag)
        devM0 = M0_diag - 1./ 3 * trace_M0
        if norm(devM0) < 1:
            print("Le matériau est isotrope en volume")
            return 3 * [1 for _ in range(len(J_list))], None

        s_data = [data.get("s_xx"), data.get("s_yy"), data.get("s_zz")]
        
        # Calcul des coefficients gij pour chaque point expérimental
        # Note: la formule devrait être 3 * J / (J-1) pour éviter une division par zéro peut être plutôt inv_prefactor ?
        # pref_list = [prefactor(J) for J in J_list]
        # g_list = [[1 / (pref * devM0[i]) * s for pref, s in zip(pref_list, s_data[i])]
        #              for i in range(3)]
        inv_pref_list = [inv_prefactor(J) for J in J_list]
        g_list = [[inv_pref * s / devM0[i] for inv_pref, s in zip(inv_pref_list, s_data[i])]
                     for i in range(3)]
        # Extraction des composantes individuelles
        g_xx, g_yy, g_zz = asarray(g_list[0]), asarray(g_list[1]), asarray(g_list[2])
        # Fusionner les données pour le fitting
        J_combined = concatenate([J_list, J_list, J_list])
        g_combined = concatenate([g_xx, g_yy, g_zz])
        
        degree = data.get("degree")
        # Ajustement d'un polynôme unique pour toutes les composantes
        poly_fit_unified, r2_unified = fit_and_plot_shifted_polynomial_fixed(
            J_combined, g_combined, degree, plot, {"save" : save, "name" : "spherical"}, ylabel=r'$g$',
            title_prefix='Ajustement unifié de g_ij - ')  
        # Retourner les données originales et le polynôme unifié
        return [g_xx, g_yy, g_zz], poly_fit_unified

    def calibrate_fij(self, shear_data, spherical_data, plot, save):
        """Calibrate shear modulation functions from experimental shear test data.
        
        This method determines the functions f_ij(J) that modify the anisotropic
        shear response under volumetric deformation. These functions capture the
        coupling between shear stiffness and volume change, which is important
        for accurate anisotropic material modeling.
        ----------
        shear_data : dict
            Experimental shear test data containing:
            - "xy", "xz", "yz" : dict
                Data for each shear component with keys:
                - "J" : array_like, Jacobian values during shear test
                - "s" : array_like, Shear stress values (Pa)  
                - "gamma" : float, Applied shear strain magnitude
            - "degree" : int, Polynomial degree for fitting (typically 2-4)
            
        spherical_data : dict
            Previously fitted spherical data containing g_ij functions.
            Used to account for volumetric coupling when extracting f_ij.
            
        plot : bool Whether to display fitting plots for each shear component.
        save : bool Whether to save plots to files.
            
        Returns
        -------
        list of list
            6×6 matrix where f_func[i][j] contains polynomial coefficients
            for the f_ij modulation function. Only diagonal shear terms
            (indices 3,4,5 corresponding to yz, xz, xy) are fitted.
            Other entries are None.
            
        Notes
        -----
        Algorithm for each shear component:
        1. Extract experimental shear stress σ_ij and corresponding J values
        2. Interpolate g_ij functions at experimental J points (if available)
        3. Solve for f_ij using: f_ij = [σ_ij*J^(4/3)/γ - (J-1)/3*g*M_kl] / μ_ij
        4. Fit polynomial to f_ij vs J data
        5. Store coefficients in appropriate matrix location
        """
        # Dictionary to store results
        fij_dict = {}
        matrix_f_func = [[None for _ in range(6)] for _ in range(6)]
        
        # Define component mapping
        component_map = {"xy": {"name": r"$f_{66}$", "C_idx": 5, "m_idx": 1, "data_key": "xy"},
                         "xz": {"name": r"$f_{55}$", "C_idx": 4, "m_idx": 2, "data_key": "xz"},
                         "yz": {"name": r"$f_{44}$", "C_idx": 3, "m_idx": 2, "data_key": "yz"}}
        degree = shear_data.get("degree")
        # Process each shear component
        for direction, mapping in component_map.items():
            # Check if data exists for this component
            data = shear_data.get(mapping["data_key"])
            # Prepare and fit data
            J, fij_values = self._prepare_fij_data(data, spherical_data.get("J"), mapping["C_idx"], mapping["m_idx"])
            poly_fit, r2 = fit_and_plot_shifted_polynomial_fixed(
                J, fij_values, degree, plot, {"save" : save, "name" : direction},
                xlabel='J', ylabel = mapping["name"],
                title_prefix=f'Ajustement de {mapping["name"]} - ')
            # Store result
            fij_dict[mapping["name"]] = poly_fit
            matrix_f_func[mapping["C_idx"]][mapping["C_idx"]] = poly_fit
                
        return matrix_f_func
    
    def _prepare_fij_data(self, data, J_spherical, C_idx, m_idx):
        """Extract f_ij modulation function from experimental shear data.
        
        This helper method processes raw experimental shear test data to extract
        the shear modulation function f_ij(J) by accounting for volumetric coupling
        effects and normalizing by the reference shear modulus.
        
        Parameters
        ----------
        data : dict
            Experimental data for one shear component:
            - "J" : array_like, Jacobian values during test
            - "s" : array_like, Measured shear stress values (Pa)
            - "gamma" : float, Applied shear strain magnitude
        J_spherical : array_like J values from spherical test data (for g function interpolation).
        C_idx : int Index in stiffness matrix C corresponding to this shear component.
        m_idx : int Index in anisotropy tensor M₀ for volumetric coupling.
            
        Returns
        -------
        tuple - J_array : ndarray, Jacobian values from experiment
              - fij_values : ndarray, Extracted f_ij function values
            
        Notes
        -----
        The extraction uses the relationship:
        f_ij = [σ_ij * J^(4/3) / γ - (J-1)/3 * g * M₀_kl] / μ_ij
        
        Where:
        - σ_ij: Experimental shear stress
        - J^(4/3): Geometric correction for finite strain  
        - γ: Applied shear strain
        - g: Volumetric coupling function (from spherical data)
        - M₀_kl: Anisotropy tensor component
        - μ_ij: Reference shear modulus
        
        If no volumetric coupling data (g functions) is available,
        g is assumed to be unity (no coupling).
        """
        def compute_f(sij, J, Mkl, g, mu, gamma):
            """
            Calcule la fonction fij à partir de la contrainte de cisaillement.
            """
            return (-sij * J**(4/3) / gamma - (J-1) / 3 * g * Mkl) / mu
        
        # Extract data
        J_array = data.get("J")  # Jacobian
        sij_array = data.get("s")  # Shear stress component
        gamma = data.get("gamma")        
        # Get rigidity component
        mu = self.RigiLin[C_idx, C_idx]
        
        # Calculate appropriate g value for each point
        if self.g_func_coeffs is not None:
            g_values = interpolation_lin(J_spherical, self.g_data[m_idx], J_array)
        else:
            g_values = [1.0] * len(J_array)
        
        # Calculate fij values
        fij_values = [compute_f(sij, J, diag(self.M0)[m_idx], g, mu, gamma) 
                      for sij, J, g in zip(sij_array, J_array, g_values)]
        return J_array, fij_values
    
    def calculate_stress(self, u, v, J, T, T0, kinematic):
        """Calculate the deviatoric stress tensor for anisotropic hyperelasticity.
        
        Parameters
        ----------
        u, v, J, T, T0 : Function See stress_3D method in ConstitutiveLaw.py for details.
        kinematic : Kinematic Kinematic handler object
            
        Returns
        -------
        Function Deviatoric stress tensor
        """        
        def compute_pibar_contribution(M0, J, u):
            g_func = self.g_func_coeffs
            if g_func is None:
                pibar = 1./3 * (J-1) * M0
            elif isinstance(g_func, ndarray):
                print("Single fit") #Une seule fonction de pondération g
                pibar = 1./3 * (J-1) * polynomial_expand(J, 1, g_func)  * M0
            elif isinstance(g_func, list):
                gM0 = [[polynomial_expand(J, 1, g_func[i][j]) * M0[i, j] for i in range(3)] for j in range(3)]
                pibar = 1./3 * (J - 1) * as_tensor(gM0)
            return J**(-5./3) * dev(kinematic.push_forward(pibar, u))
        
        def compute_DEbar_contribution(M0, J, u, GLD_bar, inv_C):
            def symetrized_tensor_product(S1, S2):
                """
                Computes the symmetrized tensor product of two symmetric tensors in Voigt notation.
                
                Parameters
                ----------
                S1 : array-like First 3x3 symmetric tensor
                S2 : array-like Second 3x3 symmetric tensor
                
                Returns
                -------
                ufl.Matrix 6x6 matrix representing the symmetrized tensor product (S1⊗S2 + S2⊗S1) in Voigt notation
                """
                def sym_mat_to_vec(mat):
                    """
                    Converts a symmetric 3x3 matrix to a vector in Voigt notation order: [11, 22, 33, 23, 13, 12]
                    """
                    return [mat[0, 0], mat[1, 1], mat[2, 2], mat[1, 2], mat[0, 2], mat[0, 1]]
                list1 = sym_mat_to_vec(S1)
                list2 = sym_mat_to_vec(S2)
                mat_tot = [[list1[i] * list2[j] for j in range(6)] for i in range(6)]
                ufl_mat = as_matrix(mat_tot)
                return ufl_mat + ufl_mat.T
            
            g_func = self.g_func_coeffs
            GLDBar_V = kinematic.tensor_3d_to_voigt(GLD_bar)
            if g_func is None:
                D = 1./3 * symetrized_tensor_product(M0, inv_C)#Il doit y avoir un facteur 1/2 en trop dans ma théorie ????? ou alors un problème avec sym_mat_to_vec ????
            elif isinstance(g_func, ndarray):
                print("Single fit")
                D = 1./3 * ((J-1) * polynomial_derivative(J, 1, g_func) + 
                            polynomial_expand(J, 1, g_func)) * symetrized_tensor_product(M0, inv_C)
            elif isinstance(g_func, list):
                pibar_derivative = 1./3 * as_tensor([[M0[i, j] * (polynomial_expand(J, 1, g_func[i][j]) 
                                                                  + (J - 1) * polynomial_derivative(J, 1, g_func[i][j]))
                                                      for i in range(3)] for j in range(3)])
                D = 1./3 * symetrized_tensor_product(pibar_derivative(M0, g_func, J), inv_C)
            DE = kinematic.voigt_to_tensor_3d(dot(D, GLDBar_V))
            return kinematic.push_forward(DE, u)
        
        def compute_CBarEbar_contribution(J, u, GLD_bar, RigiLin):
            f_func = self.f_func_coeffs
            GLDBar_V = kinematic.tensor_3d_to_voigt(GLD_bar)
            size = 6
            if f_func is not None:
                RigiLinBar = [[None for _ in range(size)] for _ in range(size)]
                for i in range(size):
                    for j in range(size):
                        if f_func[i][j] is not None:
                            RigiLinBar[i][j] = polynomial_expand(J, 1, f_func[i][j]) * RigiLin[i, j]
                        else:
                            RigiLinBar[i][j] = RigiLin[i, j]
                if hasattr(self, "R"):
                    RotatedRigiLinBar = rotate_stifness(RigiLinBar, self.R)
                    CBarEbar = kinematic.voigt_to_tensor_3d(dot(RotatedRigiLinBar, GLDBar_V))
                else:
                    CBarEbar = kinematic.voigt_to_tensor_3d(dot(as_matrix(RigiLinBar), GLDBar_V))
            else:
                if hasattr(self, "R"):
                    RotatedRigiLinBar = rotate_stifness(RigiLin, self.R)
                    CBarEbar = kinematic.voigt_to_tensor_3d(dot(as_matrix(RotatedRigiLinBar), GLDBar_V))
                else:
                    CBarEbar = kinematic.voigt_to_tensor_3d(dot(as_matrix(RigiLin), GLDBar_V))
            return J**(-5./3) * dev(kinematic.push_forward(CBarEbar, u)) 
        
        def compute_EEbar_contribution(J, u, GLD_bar, inv_C, RigiLin):
            f_func = self.f_func_coeffs
            size = 6
            GLDBar_V = kinematic.tensor_3d_to_voigt(GLD_bar)
            
            DerivRigiLinBar = [[0 for _ in range(size)] for _ in range(size)]
            for i in range(size):
                for j in range(size):
                    if f_func[i][j] is not None:
                        DerivRigiLinBar[i][j] = polynomial_derivative(J, 1, f_func[i][j]) * RigiLin[i, j]
            if hasattr(self, "R"):
                RotatedDerivRigiLinBar = rotate_stifness(DerivRigiLinBar, self.R)
                EE = kinematic.voigt_to_tensor_3d(1./2 * inner(inv_C, GLD_bar) * dot(RotatedDerivRigiLinBar, GLDBar_V))
            else:
                EE = kinematic.voigt_to_tensor_3d(1./2 * inner(inv_C, GLD_bar) * dot(DerivRigiLinBar, GLDBar_V))
            return dev(kinematic.push_forward(EE, u))
        
        # Build different strain measure
        C = kinematic.right_cauchy_green_3d(u)
        C_bar = J**(-2./3) * C
        inv_C = inv(C)
        GLD_bar = 1./2 * (C_bar - Identity(3))
        # Third term 
        CBarEbar_contribution = compute_CBarEbar_contribution(J, u, GLD_bar, self.RigiLin)
        if self.f_func_coeffs is not None:
            EEbar_contribution = compute_EEbar_contribution(J, u, GLD_bar, inv_C, self.RigiLin)
        
        M0 = bulk_anisotropy_tensor(self.RigiLin, module = "numpy")
        devM0 = M0 - 1./3 * sum(M0[i,i] for i in range(3)) * identity(3)
        is_bulk_isotropic = norm(devM0)<1
        if not is_bulk_isotropic:
            if hasattr(self, "R"):
                rotated_C = rotate_stifness(self.RigiLin, self.R)
                M0 = bulk_anisotropy_tensor(rotated_C, module = "ufl")
            else:
                M0 = bulk_anisotropy_tensor(self.RigiLin, module = "ufl")
            pibar_contribution = compute_pibar_contribution(M0, J, u)
            DEbar_contribution = compute_DEbar_contribution(M0, J, u, GLD_bar, inv_C)
            if self.f_func_coeffs is not None:
                return pibar_contribution + DEbar_contribution + CBarEbar_contribution + EEbar_contribution
            else:
                return pibar_contribution + DEbar_contribution + CBarEbar_contribution
        else:
            if self.f_func_coeffs is not None:
                return CBarEbar_contribution + EEbar_contribution
            else:
                return CBarEbar_contribution