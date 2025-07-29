# Copyright 2025 CEA
"""
Solver pour la plasticité J2 avec JAX

@author: bouteillerp
"""

try:
    from jax.numpy import clip, sqrt, linalg, reshape, array, concatenate
    from jax.numpy import zeros as jnp_zeros
    from jax.lax import cond
    from jax import vmap, jit
    from optimistix import LevenbergMarquardt, root_find
except Exception:
    raise ImportError("JAX et Optimistix sont requis pour ce solveur")

def reduced_3D_tr(x):
    return sum(x[:3])

def reduced_3D_dev(x, trace = None):
    tr = reduced_3D_tr(x)
    return x - 1./3 * tr * reduced_unit_array(len(x))

def reduced_3D_det(x):
    if len(x)==3:
        return x[0] * x[1] * x[2]
    elif len(x)==6:
        return x[0]*x[1]*x[2] + x[1]*x[2]*x[3]/sqrt(2) \
            - x[4]**2*x[2]/2 - x[5]**2*x[0]/2 - x[3]**2*x[2]/2 
            
def reduced_unit_array(length):
    if length == 3:
        return array([1, 1, 1])
    elif length == 6:
        return array([1, 1, 1, 0, 0, 0])


class JAXJ2PlasticSolver:
    """Solveur pour la plasticité J2 avec JAX et Optimistix"""
    
    def __init__(self, plastic, u):
        """
        Initialise le solveur J2 JAX
        
        Parameters
        ----------
        plastic : JAXJ2Plasticity
            Instance de la classe JAXJ2Plasticity
        u : Function
            Champ de déplacement
        """ 
        self.plastic = plastic
        # self.u = u
        self._setup_solver()
        self.batched_constitutive_update = jit(vmap(self.constitutive_update, in_axes=(0, 0)))
        self.n_gauss = len(self.plastic.p.x.array)
            
    def _setup_solver(self):
        """Configure le solveur Levenberg-Marquardt"""
        self.mu = self.plastic.mu
        self.yield_stress = self.plastic.yield_stress
        self.equivalent_stress = lambda x: linalg.norm(x)
        self.clipped_equiv_stress = lambda s: clip(self.equivalent_stress(s), a_min=1e-6)
        self.normal = lambda s: s / self.clipped_equiv_stress(s)
        
        def residual_function(x, args):
            be_bar_trial, p_old = args
            be_bar, dp = x[:-1], x[-1]
            dev_be_bar_trial = reduced_3D_dev(be_bar_trial)
            s_trial = self.mu * dev_be_bar_trial
            sig_eq_trial = self.clipped_equiv_stress(s_trial)
            
            r_be = self._r_be_plastic(be_bar, dp, be_bar_trial, dev_be_bar_trial)
            r_p = self._r_p_plastic(p_old, dp, be_bar, sig_eq_trial)
            
            return concatenate([r_be, array([r_p])])
            
        self.residual_function = residual_function
        self.solver = LevenbergMarquardt(rtol=1e-8, atol=1e-8)

    def _r_p_plastic(self, p_old, dp, be_bar, sig_eq_trial):
        """Résidu du critère de plasticité"""
        s = self.mu * reduced_3D_dev(be_bar)
        return (self.clipped_equiv_stress(s) - sqrt(2 / 3) * self.yield_stress(p_old + dp)) / self.mu
    
    def _r_be_plastic(self, be_bar, dp, be_bar_trial, dev_be_bar_trial):
        """Résidu de l'équation d'évolution du tenseur élastique"""
        det_be_bar = reduced_3D_det(be_bar)
        dev_be_bar = reduced_3D_dev(be_bar)
        s = self.mu * dev_be_bar
            
        return dev_be_bar - dev_be_bar_trial \
                + 2 * sqrt(3 / 2) * dp * reduced_3D_tr(be_bar) / 3 * self.normal(s) \
                + (det_be_bar - 1) * reduced_unit_array(self.plastic.len_plas)
        
    def _compute_trial_state(self, be_bar_trial, p_old):
        """Calcul de l'état d'essai"""
        dev_be_bar_trial = reduced_3D_dev(be_bar_trial)
        s_trial = self.mu * dev_be_bar_trial
        sig_eq_trial = self.clipped_equiv_stress(s_trial)
        yield_criterion = sig_eq_trial - sqrt(2 / 3) * self.yield_stress(p_old)
        return dev_be_bar_trial, s_trial, sig_eq_trial, yield_criterion
    
    def constitutive_update(self, be_bar_trial, p_old):
        """Mise à jour constitutive avec test élastique/plastique"""
        dev_be_bar_trial, s_trial, sig_eq_trial, yield_criterion = \
        self._compute_trial_state(be_bar_trial, p_old)
        
        return cond(
            yield_criterion < 0.0,
            lambda x: x,  # Cas élastique
            lambda x: self._return_mapping(x, p_old, dev_be_bar_trial, sig_eq_trial),  # Cas plastique
            be_bar_trial
        )
        
    def _return_mapping(self, be_bar_trial, p_old, dev_be_bar_trial, sig_eq_trial):
        """Algorithme de retour sur la surface de charge"""
        x0 = jnp_zeros((self.plastic.len_plas + 1,))
        x0 = x0.at[:-1].set(be_bar_trial)
        y0 = jnp_zeros(self.plastic.len_plas + 1)
        
        sol = root_find(
            lambda x, args: self.residual_function(x, (be_bar_trial, p_old)), 
            self.solver, 
            y0, 
            x0
        )
        be_bar, dp = sol.value[:-1], sol.value[-1]
        return be_bar

    def solve(self):
        """
        Résout le problème de plasticité J2 avec JAX
        
        Met à jour le tenseur élastique de Cauchy-Green gauche
        """
        self.plastic.Be_Bar_trial_func.interpolate(self.plastic.Be_Bar_trial)
        Bbar_trial = reshape(self.plastic.Be_Bar_trial_func.x.array, (self.n_gauss, self.plastic.len_plas))
        be_bar = self.batched_constitutive_update(Bbar_trial, self.plastic.p.x.array)
        self.plastic.Be_Bar_old.x.array[:] = be_bar.ravel()