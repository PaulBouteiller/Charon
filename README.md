# Charon: A Solid Mechanics FEM Package

Charon is a finite element solver for solid mechanics problems. Developed as an extension to DOLFINx (part of the FEniCSx ecosystem), Charon provides capabilities for simulating complex material behaviors including elasticity, plasticity, damage, and thermal effects. The code is designed to facilitate research of new constitutive laws, with portions of the code leveraging JAX for automatic differentiation.

## Physical models and numerical methods

Charon solves solid mechanics problems using the finite element method. The current version provides the following features:

### Kinematics and time integration
- Static and explicit dynamic simulations
- Support for large deformations and finite strains
- Hypoelastic and hyperelastic formulations

### Material models
- **Equations of state**:
  - Linear elastic (IsotropicHPP)
  - Hyperelastic (Neo-Hookean, Mooney-Rivlin)
  - Various U-model EOS (U1, U5, U8)
  - Mie-Grüneisen, Vinet, JWL, MACAW
  - Gas model
- **Deviatoric behaviors**:
  - Isotropic linear elasticity
  - Neo-Hookean and Mooney-Rivlin hyperelasticity
  - Transversely isotropic models
  - Fully anisotropic models
- **Advanced material behaviors**:
  - Plasticity (small strain and finite strain)
  - Damage models (Phase Field, Johnson-Cook)
  - Multiphase materials with evolving concentrations
  - Thermal coupling

### Computational features
- Stabilization methods for shock problems (artificial viscosity)
- JAX-based return mapping algorithms for complex constitutive behaviors

## Example applications

Charon is designed for simulations involving complex material behaviors such as:
- Damage and fracture mechanics
- Multiphase material behaviors
- Thermomechanical coupling effects

## Installation

Charon depends on DOLFINx and other components of the FEniCSx ecosystem. See https://github.com/FEniCS/dolfinx for installation of FEniCSx. To install:

```bash
# Install optional dependencies for advanced features
python -m pip install jax Diffrax
# Clone and install CharonX
git clone https://github.com/username/CharonX.git
cd CharonX
pip install -e .
