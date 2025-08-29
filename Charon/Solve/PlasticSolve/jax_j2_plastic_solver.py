"""
JAX-based J2 Plasticity Solver
==============================

High-performance J2 plasticity solver using JAX for automatic differentiation
and vectorized operations.

@author: bouteillerp
"""
from dolfinx.fem import Function, Expression
from ...utils.petsc_operations import petsc_assign

try:
    from jax.numpy import clip, sqrt, reshape, array, eye
    from jax.lax import cond
    from jax import vmap, jit, jacfwd
    import jax.numpy as jnp
    from .jax_newton_solver import JAXNewton
except:
    print(f"Warning: Optional module jax not found. Some functionality may be limited.")
    # raise ImportError("JAX is required for this solver")

def to_vect(tensor_3x3, length):
    """Convert 3x3 tensor to vector format.

    Parameters
    ----------
    tensor_3x3 : array 3x3 tensor
    length : int Target vector length (3, 4, 6, or 9)
        
    Returns
    -------
    array Vectorized tensor
    """
    if length == 3:
        return array([tensor_3x3[0,0], tensor_3x3[1,1], tensor_3x3[2,2]])
    elif length == 4:
        return array([
            tensor_3x3[0,0], tensor_3x3[1,1], tensor_3x3[2,2], tensor_3x3[0,1]
        ])
    elif length == 6:
        return array([
            tensor_3x3[0,0], tensor_3x3[1,1], tensor_3x3[2,2],
            tensor_3x3[0,1], tensor_3x3[0,2], tensor_3x3[1,2]
        ])
    else:
        return tensor_3x3.flatten()

def to_mat(vector):
    """Convert vector to 3x3 tensor format.
    
    Parameters
    ----------
    vector : array Vector representation of tensor
        
    Returns
    -------
    array 3x3 tensor
    """
    if len(vector) == 3:
        return jnp.diag(vector)
    elif len(vector) == 4:
        return array([
            [vector[0], vector[3], 0],
            [vector[3], vector[1], 0],
            [0, 0, vector[2]]
        ])
    elif len(vector) == 6:
        return array([
            [vector[0], vector[3], vector[4]],
            [vector[3], vector[1], vector[5]],
            [vector[4], vector[5], vector[2]]
        ])
    elif len(vector) == 9:
        return vector.reshape(3, 3)
    else:
        raise ValueError(f"Format de vecteur non supporté: {len(vector)}")

# Tensor operations
dev = jit(lambda tensor: tensor - jnp.trace(tensor) / 3.0 * eye(3))
tr = jit(lambda tensor: jnp.trace(tensor))  
det = jit(lambda tensor: jnp.linalg.det(tensor))

def dev_norm_3d(tensor_vec):
    """Deviatoric norm for 3-component vector."""
    tr_tensor = jnp.sum(tensor_vec)
    dev_diag = tensor_vec - tr_tensor / 3.0
    return jnp.sqrt(2/3 * jnp.sum(dev_diag**2))

def dev_norm_4d(tensor_vec):
    """Deviatoric norm for 4-component vector."""
    tr_tensor = jnp.sum(tensor_vec[:3])
    dev_diag = tensor_vec[:3] - tr_tensor / 3.0
    return jnp.sqrt(2/3 * (jnp.sum(dev_diag**2) + 2 * tensor_vec[3]**2))

def dev_norm_6d(tensor_vec):
    """Deviatoric norm for 6-component vector."""
    tr_tensor = jnp.sum(tensor_vec[:3])
    dev_diag = tensor_vec[:3] - tr_tensor / 3.0
    return jnp.sqrt(2/3 * (jnp.sum(dev_diag**2) + 2 * jnp.sum(tensor_vec[3:]**2)))

class JAXJ2PlasticSolver:
    """JAX-based J2 plasticity solver with automatic differentiation.
    
    Implements J2 plasticity using JAX for high performance and
    automatic differentiation of the return mapping algorithm.
    
    Attributes
    ----------
    plastic : JAXJ2Plasticity J2 plasticity model instance
    mu : float Shear modulus
    yield_stress : callable Yield stress function
    len : int Length of plastic variable array
    clipped_equiv_stress : callable Clipped equivalent stress function
    normal : callable Normal to yield surface function
    newton_solver : JAXNewton Newton solver for return mapping
    batched_constitutive_update : callable Vectorized constitutive update function
    n_gauss : int Number of Gauss points
    Be_Bar_trial_func : Function Trial elastic left Cauchy-Green tensor function
    Be_Bar_trial_expr : Expression Expression for trial tensor
    """
    def __init__(self, problem, plastic, u):
        """Initialize the JAX J2 plasticity solver.

        Parameters
        ----------
        problem : Problem Problem instance (unused but kept for interface consistency)
        plastic : JAXJ2Plasticity J2 plasticity model instance
        u : Function Current displacement field
        """
        self.plastic = plastic
        self._setup_solver()
        self.batched_constitutive_update = jit(vmap(self.constitutive_update, in_axes=(0, 0)))
        self.n_gauss = len(self.plastic.p.x.array)
        self.Be_Bar_trial_func = Function(self.plastic.V_Be_bar)
        expr = self.plastic.kin.tensor_3d_to_mandel_compact(self.plastic.Be_bar_trial(u, self.plastic.u_old))
        self.Be_Bar_trial_expr = Expression(expr, self.plastic.V_Be_bar.element.interpolation_points())
        dev_norm_funcs = {3: dev_norm_3d, 4: dev_norm_4d, 6: dev_norm_6d}
        self.dev_norm = dev_norm_funcs[self.plastic.len_plas]
            
    def _setup_solver(self):
        """Configure solver with pre-compiled functions."""
        self.mu = self.plastic.mu
        self.yield_stress = self.plastic.yield_stress
        self.len = self.plastic.len_plas
        self.clipped_equiv_stress = jit(lambda s: clip(jnp.linalg.norm(s), a_min=1e-8))
        self.normal = jit(jacfwd(self.clipped_equiv_stress))
        self.newton_solver = JAXNewton(rtol=1e-8, atol=1e-8, niter_max=100)
    
    def _compute_yield_criterion(self, be_bar_trial, p_old):
        """Compute yield criterion for J2 plasticity."""
        sig_eq_trial = self.mu * self.dev_norm(be_bar_trial)
        return sig_eq_trial - sqrt(2 / 3) * self.yield_stress(p_old)
    
    def constitutive_update(self, be_bar_trial, p_old):
        """Constitutive update with JAX Newton solver.
        
        Parameters
        ----------
        be_bar_trial : array Trial elastic left Cauchy-Green tensor
        p_old : float Previous cumulative plastic strain
            
        Returns
        -------
        tuple (updated_be_bar, plastic_strain_increment)
        """
        yield_criterion = self._compute_yield_criterion(be_bar_trial, p_old)
        
        def r_p(dx):
            """Residual for plasticity condition."""
            be_bar, dp = dx[:-1], dx[-1]
            be_bar_3x3 = to_mat(be_bar)
            s = self.mu * dev(be_bar_3x3)
            
            r_elastic = lambda dp: dp
            r_plastic = lambda dp: (
                self.clipped_equiv_stress(s) - sqrt(2 / 3) * self.yield_stress(p_old + dp)
            ) / self.mu
            
            return cond(yield_criterion < 0.0, r_elastic, r_plastic, dp)
        
        def r_be(dx):
            """Residual for elastic tensor evolution."""
            be_bar, dp = dx[:-1], dx[-1]
            be_bar_3x3 = to_mat(be_bar)
            be_bar_trial_3x3 = to_mat(be_bar_trial)
            s = self.mu * dev(be_bar_3x3)
            
            r_elastic = lambda be_bar, dp: to_vect(
                be_bar_3x3 - be_bar_trial_3x3, self.len
            )
            r_plastic = lambda be_bar, dp: to_vect(
                dev(be_bar_3x3 - be_bar_trial_3x3)
                + 2 * sqrt(3 / 2) * dp * tr(be_bar_3x3) / 3 * self.normal(s)
                + eye(3) * (det(be_bar_3x3) - 1),
                self.len,
            )
            
            return cond(yield_criterion < 0.0, r_elastic, r_plastic, be_bar_3x3, dp)
        
        # Résolution
        self.newton_solver.set_residual((r_be, r_p))
        x0 = jnp.concatenate([be_bar_trial, jnp.array([0.0])])
        x, data = self.newton_solver.solve(x0)
        
        return x[:-1], x[-1]

    def solve(self):
        """Solve J2 plasticity problem.
        
        Updates elastic left Cauchy-Green tensor and cumulative plastic strain
        using vectorized JAX operations.
        """
        self.Be_Bar_trial_func.interpolate(self.Be_Bar_trial_expr)
        Bbar_trial = reshape(self.Be_Bar_trial_func.x.array, (self.n_gauss, self.len))
        be_bar, dp = self.batched_constitutive_update(Bbar_trial, self.plastic.p.x.array)
        self.plastic.Be_bar.x.array[:] = be_bar.ravel()
        self.plastic.p.x.array[:] += dp
        petsc_assign(self.plastic.u_old, self.plastic.u)