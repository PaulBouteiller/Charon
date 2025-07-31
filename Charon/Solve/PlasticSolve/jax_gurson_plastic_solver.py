"""
JAX-based Gurson Plasticity Solver
==================================

High-performance Gurson-Tvergaard-Needleman plasticity solver using JAX
for porous ductile materials with void evolution.

@author: bouteillerp
"""

try:
    from jax.numpy import clip, sqrt, linalg, reshape, array, concatenate, exp
    from jax.numpy import zeros as jnp_zeros
    from jax.lax import cond
    from jax import vmap, jit
    from optimistix import LevenbergMarquardt, root_find
except ImportError:
    raise ImportError("JAX and Optimistix are required for this solver")

from math import pi

class JAXGursonPlasticSolver:
    """JAX-based Gurson-Tvergaard-Needleman plasticity solver.
    
    Implements GTN model for porous ductile materials accounting for
    void growth, nucleation, and coalescence using JAX for performance.
    
    Attributes
    ----------
    plastic : JAXGursonPlasticity
        Gurson plasticity model instance
    u : Function
        Current displacement field
    mu : float
        Shear modulus
    yield_stress : callable
        Yield stress function
    equivalent_stress : callable
        Equivalent stress function
    clipped_equiv_stress : callable
        Clipped equivalent stress function
    normal : callable
        Normal to yield surface function
    residual_function : callable
        Residual function for return mapping
    solver : LevenbergMarquardt
        Levenberg-Marquardt solver
    batched_constitutive_update : callable
        Vectorized constitutive update function
    n_gauss : int
        Number of Gauss points
    """
    
    def __init__(self, plastic, u):
        """Initialize the JAX Gurson plasticity solver.
        
        Parameters
        ----------
        plastic : JAXGursonPlasticity
            Gurson plasticity model instance
        u : Function
            Current displacement field
        """
        self.plastic = plastic
        self.u = u
        self._setup_solver()
        self.batched_constitutive_update = jit(vmap(self.constitutive_update, in_axes=(0, 0, 0)))
        self.n_gauss = len(self.plastic.p.x.array)
            
    def _setup_solver(self):
        """Configure Levenberg-Marquardt solver for Gurson model."""
        self.mu = self.plastic.mu
        self.yield_stress = self.plastic.yield_stress
        self.equivalent_stress = lambda x: linalg.norm(x)
        self.clipped_equiv_stress = lambda s: clip(self.equivalent_stress(s), a_min=1e-6)
        self.normal = lambda s: s / self.clipped_equiv_stress(s)
        
        def residual_function(x, args):
            """Combined residual function for return mapping."""
            be_bar_trial, p_old, f_old = args
            be_bar, dp, df = x[:-2], x[-2], x[-1]
            
            r_be = self._r_be_plastic(be_bar, dp, be_bar_trial)
            r_p = self._r_p_plastic_gurson(p_old, dp, be_bar, f_old + df)
            r_f = self._r_f_plastic(f_old, df, dp, be_bar, p_old)
            
            return concatenate([r_be, array([r_p, r_f])])
            
        self.residual_function = residual_function
        self.solver = LevenbergMarquardt(rtol=1e-8, atol=1e-8)

    def _gurson_yield_function(self, sig_eq, sig_h, f_star):
        """Gurson yield function.
        
        Parameters
        ----------
        sig_eq : float
            Equivalent stress
        sig_h : float
            Hydrostatic stress
        f_star : float
            Effective porosity
            
        Returns
        -------
        float
            Yield function value
        """
        q1, q2, q3 = self.plastic.q1, self.plastic.q2, self.plastic.q3
        sig_y = self.yield_stress(0)
        
        term1 = (sig_eq / sig_y) ** 2
        term2 = 2 * q1 * f_star * linalg.cosh(q2 * sig_h / (2 * sig_y))
        term3 = (q1 * f_star) ** 2
        
        return term1 + term2 - 1 - term3

    def _compute_f_star(self, f):
        """Compute effective porosity according to GTN model.
        
        Parameters
        ----------
        f : float
            Current porosity
            
        Returns
        -------
        float
            Effective porosity accounting for coalescence
        """
        fc, ff, q1 = self.plastic.fc, self.plastic.ff, self.plastic.q1
        fu = 1 / q1
        
        return cond(
            f <= fc,
            lambda f: f,
            lambda f: fc + (fu - fc) * (f - fc) / (ff - fc),
            f
        )

    def _r_p_plastic_gurson(self, p_old, dp, be_bar, f):
        """Residual for Gurson plasticity criterion.
        
        Parameters
        ----------
        p_old : float
            Previous cumulative plastic strain
        dp : float
            Plastic strain increment
        be_bar : array
            Elastic left Cauchy-Green tensor
        f : float
            Current porosity
            
        Returns
        -------
        float
            Plasticity residual
        """
        s = self.mu * reduced_3D_dev(be_bar)
        sig_eq = self.clipped_equiv_stress(s)
        sig_h = reduced_3D_tr(be_bar) / 3
        f_star = self._compute_f_star(f)
        
        return self._gurson_yield_function(sig_eq, sig_h, f_star) / self.mu
    
    def _r_be_plastic(self, be_bar, dp, be_bar_trial):
        """Residual for elastic tensor evolution (Gurson).
        
        Parameters
        ----------
        be_bar : array
            Current elastic left Cauchy-Green tensor
        dp : float
            Plastic strain increment
        be_bar_trial : array
            Trial elastic left Cauchy-Green tensor
            
        Returns
        -------
        array
            Tensor evolution residual
        """
        det_be_bar = reduced_3D_det(be_bar)
        dev_be_bar = reduced_3D_dev(be_bar)
        s = self.mu * dev_be_bar
            
        return dev_be_bar - reduced_3D_dev(be_bar_trial) \
                + 2 * sqrt(3 / 2) * dp * reduced_3D_tr(be_bar) / 3 * self.normal(s) \
                + (det_be_bar - 1) * reduced_unit_array(self.plastic.len_plas)
                
    def _r_f_plastic(self, f_old, df, dp, be_bar, p_old):
        """Residual for porosity evolution.
        
        Parameters
        ----------
        f_old : float
            Previous porosity
        df : float
            Porosity increment
        dp : float
            Plastic strain increment
        be_bar : array
            Elastic left Cauchy-Green tensor
        p_old : float
            Previous cumulative plastic strain
            
        Returns
        -------
        float
            Porosity evolution residual
        """
        # Void growth
        tr_be_bar = reduced_3D_tr(be_bar)
        f_growth = (1 - f_old) * dp * tr_be_bar / 3
        
        # Nucleation (simplified)
        eN, sN, fN = 0.3, 0.1, 0.04
        f_nucleation = fN / (sN * sqrt(2 * pi)) * \
                      exp(-0.5 * ((p_old - eN) / sN) ** 2) * dp
        
        return df - f_growth - f_nucleation
        
    def _compute_trial_state(self, be_bar_trial, p_old, f_old):
        """Compute trial state for Gurson model.
        
        Parameters
        ----------
        be_bar_trial : array
            Trial elastic left Cauchy-Green tensor
        p_old : float
            Previous cumulative plastic strain
        f_old : float
            Previous porosity
            
        Returns
        -------
        tuple
            Trial state variables
        """
        dev_be_bar_trial = reduced_3D_dev(be_bar_trial)
        s_trial = self.mu * dev_be_bar_trial
        sig_eq_trial = self.clipped_equiv_stress(s_trial)
        sig_h_trial = reduced_3D_tr(be_bar_trial) / 3
        f_star = self._compute_f_star(f_old)
        
        yield_criterion = self._gurson_yield_function(sig_eq_trial, sig_h_trial, f_star)
        return dev_be_bar_trial, s_trial, sig_eq_trial, yield_criterion
    
    def constitutive_update(self, be_bar_trial, p_old, f_old):
        """Constitutive update with Gurson model.
        
        Parameters
        ----------
        be_bar_trial : array
            Trial elastic left Cauchy-Green tensor
        p_old : float
            Previous cumulative plastic strain
        f_old : float
            Previous porosity
            
        Returns
        -------
        array
            Updated elastic left Cauchy-Green tensor
        """
        dev_be_bar_trial, s_trial, sig_eq_trial, yield_criterion = \
            self._compute_trial_state(be_bar_trial, p_old, f_old)
        
        return cond(
            yield_criterion < 0.0,
            lambda x: x,  # Elastic case
            lambda x: self._return_mapping_gurson(x, p_old, f_old),  # Plastic case
            be_bar_trial
        )
        
    def _return_mapping_gurson(self, be_bar_trial, p_old, f_old):
        """Return mapping algorithm for Gurson yield surface.
        
        Parameters
        ----------
        be_bar_trial : array
            Trial elastic left Cauchy-Green tensor
        p_old : float
            Previous cumulative plastic strain
        f_old : float
            Previous porosity
            
        Returns
        -------
        array
            Updated elastic left Cauchy-Green tensor
        """
        x0 = jnp_zeros((self.plastic.len_plas + 2,))  # +2 for dp and df
        x0 = x0.at[:-2].set(be_bar_trial)
        y0 = jnp_zeros(self.plastic.len_plas + 2)
        
        sol = root_find(
            lambda x, args: self.residual_function(x, (be_bar_trial, p_old, f_old)), 
            self.solver, 
            y0, 
            x0
        )
        be_bar = sol.value[:-2]
        return be_bar

    def solve(self):
        """Solve Gurson plasticity problem with JAX.
        
        Updates elastic tensor and porosity using vectorized operations.
        """
        self.plastic.Be_Bar_trial_func.interpolate(self.plastic.Be_Bar_trial)
        Bbar_trial = reshape(self.plastic.Be_Bar_trial_func.x.array, (self.n_gauss, self.plastic.len_plas))
        p_array = self.plastic.p.x.array
        f_array = self.plastic.f.x.array
        
        be_bar = self.batched_constitutive_update(Bbar_trial, p_array, f_array)
        self.plastic.Be_Bar_old.x.array[:] = be_bar.ravel()