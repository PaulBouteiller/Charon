# CharonX: A Differentiable Solid Mechanics FEM Package

CharonX is a finite element solver for solid mechanics problems. Developed as an extension to DOLFINx (part of the FEniCSx ecosystem), CharonX provides robust capabilities for simulating complex material behaviors including elasticity, plasticity, damage, and thermal effects. The code is designed to facilitate research of new constitutive law, with portions of the code leveraging JAX for automatic differentiation.

## Physical models and numerical methods

CharonX solves solid mechanics problems using the finite element method. The current version provides the following features:

### Kinematics and time integration
- Static and dynamic simulations
- Explicit time stepping schemes (LeapFrog, Yoshida)
- Support for large deformations and finite strains
- Hypoelastic and hyperelastic formulations

### Material models
- **Equations of state**:
  - Linear elastic (IsotropicHPP)
  - Hyperelastic (Neo-Hookean, Mooney-Rivlin)
  - Various U-model EOS (U1, U5, U8)
  - Mie-Grüneisen, Vinet, JWL, MACAW
  - Gas and fluid models

- **Deviatoric behaviors**:
  - Isotropic linear elasticity
  - Neo-Hookean and Mooney-Rivlin hyperelasticity
  - Transversely isotropic models
  - Fully anisotropic models
  - Newtonian fluid deviatoric stress

- **Advanced material behaviors**:
  - Plasticity (small strain and finite strain)
  - Damage models (Phase Field, Johnson-Cook)
  - Multiphase materials with evolving concentrations
  - Thermal coupling

### Computational features
- Quadrature-based function spaces for better volumetric locking behavior
- Stabilization methods for shock problems (artificial viscosity)
- JAX-based return mapping algorithms for complex constitutive behaviors
- Multi-core parallelization via MPI
- Export capabilities to XDMF, VTK, and CSV formats

## Example applications

CharonX is designed for simulations involving complex material behaviors such as:
- Damage and fracture mechanics
- Multiphase material behaviors
- Explosive deformation
- Thermomechanical coupling effects

## Installation

CharonX depends on DOLFINx and other components of the FEniCSx ecosystem. To install:

```bash
# Install FEniCSx prerequisites
python -m pip install numpy scipy matplotlib

# Install FEniCSx
python -m pip install fenics-dolfinx

# Install optional dependencies for advanced features
python -m pip install jax jaxlib

# Clone and install CharonX
git clone https://github.com/username/CharonX.git
cd CharonX
pip install -e .
