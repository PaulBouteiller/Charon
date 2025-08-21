"""
Material Template - Complete Configuration Options
=================================================
This template shows all available material model options and required parameters
for the Charon finite element framework. All the values are fictive and must be adapted by the user.
"""

from Charon import Material
import numpy as np

# =============================================================================
# BASIC MATERIAL PROPERTIES (Required for all materials)
# =============================================================================

# Physical properties
rho_0 = 7.8e-3      # Initial density [kg/m³] 
C_mass = 500        # Mass thermal capacity [J/(kg·K)]

# =============================================================================
# EQUATION OF STATE (EOS) OPTIONS
# =============================================================================

# Choose ONE of the following EOS types:

# -----------------------------------------------------------------------------
# 1. ISOTROPIC LINEAR ELASTIC (Small strain)
# -----------------------------------------------------------------------------
eos_type = "IsotropicHPP"
dico_eos_hpp = {
    "E": 210e9,         # Young's modulus [Pa]
    "nu": 0.3,          # Poisson's ratio [-]
    "alpha": 12e-6      # Thermal expansion coefficient [1/K]
}

# -----------------------------------------------------------------------------
# 2. HYPERELASTIC U-MODELS (Finite strain)
# -----------------------------------------------------------------------------
# Available variants: "U1", "U5", "U8"
eos_type = "U1"  # or "U5" or "U8"
dico_eos_u = {
    "kappa": 166.7e9,   # Bulk modulus [Pa]
    "alpha": 12e-6      # Thermal expansion coefficient [1/K]
}

# -----------------------------------------------------------------------------
# 3. VINET EOS (High pressure solids)
# -----------------------------------------------------------------------------
eos_type = "Vinet"
dico_eos_vinet = {
    "iso_T_K0": 166.7e9,    # Isothermal bulk modulus [Pa]
    "T_dep_K0": -1e6,       # Temperature dependence of K0 [Pa/K]
    "iso_T_K1": 4.0,        # Pressure derivative of K0 [-]
    "T_dep_K1": -1e-3       # Temperature dependence of K1 [1/K]
}

# -----------------------------------------------------------------------------
# 4. JWL EOS (Explosives)
# -----------------------------------------------------------------------------
eos_type = "JWL"
dico_eos_jwl = {
    "A": 373.77e9,      # First energy coefficient [Pa]
    "R1": 4.15,         # First rate coefficient [-]
    "B": 3.747e9,       # Second energy coefficient [Pa]
    "R2": 0.9,          # Second rate coefficient [-]
    "w": 0.35           # Energy fraction coefficient [-]
}

# -----------------------------------------------------------------------------
# 5. MACAW EOS (Extreme conditions)
# -----------------------------------------------------------------------------
eos_type = "MACAW"
dico_eos_macaw = {
    "A": 1e11,          # Cold curve parameter A [Pa]
    "B": 2.0,           # Cold curve parameter B [-]
    "C": 1.5,           # Cold curve parameter C [-]
    "eta": 0.5,         # Thermal parameter eta [-]
    "vinf": 1e-3,       # Thermal parameter vinf [m³/kg]
    "rho0": 7800,       # Reference density [kg/m³]
    "theta0": 300,      # Reference temperature [K]
    "a0": 0.5,          # Thermal parameter a0 [-]
    "m": 1.0,           # Thermal parameter m [-]
    "n": 2.0,           # Thermal parameter n [-]
    "Gammainf": 0.5,    # Asymptotic Grüneisen parameter [-]
    "Gamma0": 2.0,      # Reference Grüneisen parameter [-]
    "cvinf": 500        # Asymptotic heat capacity [J/(kg·K)]
}

# -----------------------------------------------------------------------------
# 6. MIE-GRÜNEISEN FAMILY
# -----------------------------------------------------------------------------
# Standard Mie-Grüneisen
eos_type = "MG"
dico_eos_mg = {
    "C": 5000,          # Linear coefficient [Pa]
    "D": 1e9,           # Quadratic coefficient [Pa]
    "S": 1e12,          # Cubic coefficient [Pa]
    "gamma0": 2.0       # Grüneisen coefficient [-]
}

# Extended Mie-Grüneisen
eos_type = "xMG"
dico_eos_xmg = {
    "c0": 5000,         # Sound speed at zero pressure [m/s]
    "gamma0": 2.0,      # Grüneisen coefficient [-]
    "s1": 1.5,          # Empirical parameter s1 [-]
    "s2": 0.0,          # Empirical parameter s2 [-]
    "s3": 0.0,          # Empirical parameter s3 [-]
    "b": 0.5            # Volumetric parameter [-]
}

# Polynomial Mie-Grüneisen
eos_type = "PMG"
dico_eos_pmg = {
    "Pa": 1e5,          # Atmospheric pressure [Pa]
    "Gamma0": 2.0,      # Grüneisen coefficient [-]
    "D": 1e9,           # Polynomial coefficient D [Pa]
    "S": 1e12,          # Polynomial coefficient S [Pa]
    "H": 1e15,          # Polynomial coefficient H [Pa]
    "c0": 5000          # Sound speed [m/s]
}

# -----------------------------------------------------------------------------
# 7. IDEAL GAS EOS
# -----------------------------------------------------------------------------
eos_type = "GP"
dico_eos_gas = {
    "gamma": 1.4,       # Polytropic coefficient [-]
    "e_max": 1e6        # Estimated maximum specific internal energy [J/kg]
}

# -----------------------------------------------------------------------------
# 8. TABULATED EOS (Requires JAX)
# -----------------------------------------------------------------------------
eos_type = "Tabulated"
# Option 1: Using DataFrame
import pandas as pd
# Create example DataFrame: rows=temperature, columns=J values, values=pressure
T_values = [300, 400, 500, 600]
J_values = [0.8, 0.9, 1.0, 1.1, 1.2]
P_data = np.random.rand(len(T_values), len(J_values)) * 1e6  # Example pressure data
df_tabulated = pd.DataFrame(P_data, index=T_values, columns=J_values)

dico_eos_tabulated_df = {
    "c0": 5000,         # Wave speed [m/s]
    "Dataframe": df_tabulated
}

# Option 2: Using arrays directly
dico_eos_tabulated_arrays = {
    "c0": 5000,         # Wave speed [m/s]
    "T": np.array(T_values),    # Temperature array [K]
    "J": np.array(J_values),    # Jacobian array [-]
    "P": P_data                 # Pressure array [Pa]
}

# =============================================================================
# DEVIATORIC BEHAVIOR OPTIONS
# =============================================================================

# Choose ONE of the following deviatoric types:

# -----------------------------------------------------------------------------
# 1. NO DEVIATORIC BEHAVIOR (Pure hydrostatic)
# -----------------------------------------------------------------------------
dev_type = None
dico_devia_none = {}

# -----------------------------------------------------------------------------
# 2. ISOTROPIC LINEAR ELASTIC (Small strain)
# -----------------------------------------------------------------------------
dev_type = "IsotropicHPP"
# Option A: Direct shear modulus specification
dico_devia_hpp_mu = {
    "mu": 80.77e9       # Shear modulus [Pa]
}
# Option B: Young's modulus and Poisson's ratio
dico_devia_hpp_e_nu = {
    "E": 210e9,         # Young's modulus [Pa]
    "nu": 0.3           # Poisson's ratio [-]
}

# -----------------------------------------------------------------------------
# 3. NEO-HOOKEAN HYPERELASTIC
# -----------------------------------------------------------------------------
dev_type = "NeoHook"
dico_devia_neo = {
    "mu": 80.77e9       # Shear modulus [Pa]
}

# -----------------------------------------------------------------------------
# 4. MOONEY-RIVLIN HYPERELASTIC
# -----------------------------------------------------------------------------
dev_type = "MooneyRivlin"
dico_devia_mooney = {
    "mu": 80.77e9,      # Primary shear modulus [Pa]
    "mu_quad": 20e9     # Secondary shear modulus [Pa]
}

# -----------------------------------------------------------------------------
# 5. HYPOELASTIC (Rate-dependent)
# -----------------------------------------------------------------------------
dev_type = "Hypoelastic"
dico_devia_hypo = {
    "mu": 80.77e9       # Shear modulus [Pa]
}

# -----------------------------------------------------------------------------
# 6. ANISOTROPIC HYPERELASTIC
# -----------------------------------------------------------------------------
dev_type = "Anisotropic"

# Option A: Direct stiffness tensor (6x6 Voigt notation)
C_matrix = np.array([
    [250e9, 120e9, 120e9,     0,     0,     0],
    [120e9, 250e9, 120e9,     0,     0,     0],
    [120e9, 120e9, 250e9,     0,     0,     0],
    [    0,     0,     0, 80e9,     0,     0],
    [    0,     0,     0,     0, 80e9,     0],
    [    0,     0,     0,     0,     0, 80e9]
])
dico_devia_aniso_direct = {
    "C": C_matrix
}

# Option B: Orthotropic parameters
dico_devia_aniso_ortho = {
    "ET": 15e9,         # Young's modulus (transverse) [Pa]
    "EL": 130e9,        # Young's modulus (longitudinal) [Pa]
    "EN": 15e9,         # Young's modulus (normal) [Pa]
    "nuLT": 0.3,        # Poisson's ratio (longitudinal-transverse) [-]
    "nuLN": 0.3,        # Poisson's ratio (longitudinal-normal) [-]
    "nuTN": 0.4,        # Poisson's ratio (transverse-normal) [-]
    "muLT": 5e9,        # Shear modulus (longitudinal-transverse) [Pa]
    "muLN": 5e9,        # Shear modulus (longitudinal-normal) [Pa]
    "muTN": 3e9         # Shear modulus (transverse-normal) [Pa]
}

# Option C: Transversely isotropic parameters
dico_devia_aniso_transverse = {
    "ET": 15e9,         # Young's modulus (transverse) [Pa]
    "EL": 130e9,        # Young's modulus (longitudinal) [Pa]
    "nuT": 0.4,         # Poisson's ratio (transverse) [-]
    "nuL": 0.3,         # Poisson's ratio (longitudinal) [-]
    "muL": 5e9          # Shear modulus (longitudinal) [Pa]
}

# Optional: Rotation and advanced parameters for anisotropic materials
dico_devia_aniso_advanced = {
    **dico_devia_aniso_ortho,  # Base orthotropic parameters
    "rotation": 0.785,         # Rotation angle [rad] (45 degrees)
    "f_func": None,            # Stiffness modulation functions (advanced)
    "g_func": np.array([0.1, 0.05]),  # Coupling stress modulation coefficients
    "calibration_data": {      # Experimental calibration data
        "spherical_data": {
            "J": np.array([0.9, 1.0, 1.1]),
            "s_xx": np.array([1e6, 0, -1e6]),
            "s_yy": np.array([1e6, 0, -1e6]),
            "s_zz": np.array([1e6, 0, -1e6]),
            "degree": 2
        },
        "shear_data": {
            "xy": {"J": np.array([0.9, 1.0, 1.1]), "s": np.array([5e5, 0, -5e5]), "gamma": 0.1},
            "degree": 2
        },
        "plot": False,
        "save": False
    }
}

# =============================================================================
# MATERIAL CREATION EXAMPLES
# =============================================================================

# Example 1: Simple linear elastic steel
steel_elastic = Material(
    rho_0=7800,
    C_mass=500,
    eos_type="IsotropicHPP",
    dev_type="IsotropicHPP",
    eos_params=dico_eos_hpp,
    deviator_params=dico_devia_hpp_mu
)

# Example 2: Elastic-plastic steel with J2 plasticity
steel_plastic = Material(
    rho_0=7800,
    C_mass=500,
    eos_type="IsotropicHPP",
    dev_type="IsotropicHPP",
    eos_params=dico_eos_hpp,
    deviator_params=dico_devia_hpp_mu
)

# Example 3: Hyperelastic rubber with Neo-Hookean model
rubber = Material(
    rho_0=1200,
    C_mass=1800,
    eos_type="U5",
    dev_type="NeoHook",
    eos_params=dico_eos_u,
    deviator_params=dico_devia_neo
)

# Example 4: Explosive material with JWL EOS
explosive = Material(
    rho_0=1600,
    C_mass=1000,
    eos_type="JWL",
    dev_type=None,
    eos_params=dico_eos_jwl,
    deviator_params=dico_devia_none
)

# Example 5: Anisotropic composite material
composite = Material(
    rho_0=1500,
    C_mass=1200,
    eos_type="IsotropicHPP",
    dev_type="Anisotropic",
    eos_params=dico_eos_hpp,
    deviator_params=dico_devia_aniso_ortho
)

# Example 6: Gas material
gas = Material(
    rho_0=1.2,
    C_mass=1000,
    eos_type="GP",
    dev_type=None,
    eos_params=dico_eos_gas,
    deviator_params=dico_devia_none
)

# =============================================================================
# NOTES AND GUIDELINES
# =============================================================================

"""
PARAMETER SELECTION GUIDELINES:

1. EOS Selection:
   - IsotropicHPP: Small strain linear elasticity
   - U1/U5/U8: Finite strain hyperelasticity
   - Vinet: High pressure solids
   - JWL: Explosives and detonation products
   - MACAW: Materials under extreme conditions
   - MG/xMG/PMG: Shock-loaded materials
   - GP: Ideal gases
   - Tabulated: Experimental data-driven

2. Deviatoric Selection:
   - None: Pure hydrostatic behavior (fluids)
   - IsotropicHPP: Linear elastic isotropic solids
   - NeoHook/MooneyRivlin: Hyperelastic materials (rubbers)
   - Hypoelastic: Rate-dependent behavior
   - Anisotropic: Composite materials, crystals

3. Plasticity Selection:
   - HPP_Plasticity: Small strain classical plasticity
   - Finite_Plasticity: Large deformation plasticity
   - J2_JAX: Enhanced performance J2 model
   - JAX_Gurson: Porous materials with void evolution
   - HPP_Gurson: Simplified porous plasticity

4. Damage Selection:
   - PhaseField: Brittle fracture with crack propagation
   - Johnson models: Ductile damage with porosity evolution

UNITS CONSISTENCY:
- Length: [m]
- Mass: [kg]
- Time: [s]
- Force: [N]
- Pressure/Stress: [Pa]
- Temperature: [K]
- Energy: [J]

TYPICAL MATERIAL COMBINATIONS:
- Metals: IsotropicHPP + IsotropicHPP + J2 plasticity
- Rubbers: U5 + NeoHook
- Composites: IsotropicHPP + Anisotropic
- Explosives: JWL + None
- Gases: GP + None
- Concrete: IsotropicHPP + IsotropicHPP + PhaseField damage
"""