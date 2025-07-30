# Copyright 2025 CEA
"""
Solver pour la plasticité J2 avec JAX - Version nettoyée et optimisée

@author: bouteillerp
"""
from dolfinx.fem import Function, Expression
from ...utils.petsc_operations import petsc_assign

try:
    from jax.numpy import clip, sqrt, reshape, array, eye
    from jax.lax import cond
    from jax import vmap, jit, jacfwd
    import jax.numpy as jnp
except Exception:
    raise ImportError("JAX est requis pour ce solveur")

from .jax_newton_solver import JAXNewton

def to_vect(tensor_3x3, length):
    """Convertit un tenseur 3x3 en vecteur selon la longueur demandée"""
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
    """Convertit un vecteur en tenseur 3x3 selon sa longueur"""
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

def dev(tensor):
    """Partie déviatorique d'un tenseur 3x3"""
    return tensor - jnp.trace(tensor) / 3.0 * eye(3)

def tr(tensor):
    """Trace d'un tenseur"""
    return jnp.trace(tensor)

def det(tensor):
    """Déterminant d'un tenseur"""
    return jnp.linalg.det(tensor)

class JAXJ2PlasticSolver:
    """Solveur pour la plasticité J2 avec JAX - Version optimisée"""
    
    def __init__(self, problem, plastic, u):
        self.plastic = plastic
        self._setup_solver()
        self.batched_constitutive_update = jit(vmap(self.constitutive_update, in_axes=(0, 0)))
        self.n_gauss = len(self.plastic.p.x.array)
        self.Be_Bar_trial_func = Function(self.plastic.V_Be_bar)
        expr = self.plastic.kin.tridim_to_mandel(self.plastic.Be_bar_trial(u, self.plastic.u_old))
        self.Be_Bar_trial_expr = Expression(expr, self.plastic.V_Be_bar.element.interpolation_points())
            
    def _setup_solver(self):
        """Configure le solveur avec fonctions pré-compilées"""
        self.mu = self.plastic.mu
        self.yield_stress = self.plastic.yield_stress
        self.len = self.plastic.len_plas
        self.clipped_equiv_stress = jit(lambda s: clip(jnp.linalg.norm(s), a_min=1e-8))
        self.normal = jit(jacfwd(self.clipped_equiv_stress))
        self.newton_solver = JAXNewton(rtol=1e-8, atol=1e-8, niter_max=100)

    def _compute_yield_criterion(self, be_bar_trial, p_old):
        """Calcul du critère de plasticité"""
        be_bar_trial_3x3 = to_mat(be_bar_trial)
        s_trial = self.mu * dev(be_bar_trial_3x3)
        sig_eq_trial = self.clipped_equiv_stress(s_trial)
        return sig_eq_trial - sqrt(2 / 3) * self.yield_stress(p_old)
    
    def constitutive_update(self, be_bar_trial, p_old):
        """Mise à jour constitutive avec JAXNewton"""
        yield_criterion = self._compute_yield_criterion(be_bar_trial, p_old)
        
        def r_p(dx):
            """Résidu de la condition de plasticité"""
            be_bar, dp = dx[:-1], dx[-1]
            be_bar_3x3 = to_mat(be_bar)
            s = self.mu * dev(be_bar_3x3)
            
            r_elastic = lambda dp: dp
            r_plastic = lambda dp: (
                self.clipped_equiv_stress(s) - sqrt(2 / 3) * self.yield_stress(p_old + dp)
            ) / self.mu
            
            return cond(yield_criterion < 0.0, r_elastic, r_plastic, dp)
        
        def r_be(dx):
            """Résidu de l'évolution du tenseur élastique"""
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
        """Résout le problème de plasticité J2"""
        self.Be_Bar_trial_func.interpolate(self.Be_Bar_trial_expr)
        Bbar_trial = reshape(self.Be_Bar_trial_func.x.array, (self.n_gauss, self.len))
        be_bar, dp = self.batched_constitutive_update(Bbar_trial, self.plastic.p.x.array)
        self.plastic.Be_bar.x.array[:] = be_bar.ravel()
        self.plastic.p.x.array[:] += dp
        petsc_assign(self.plastic.u_old, self.plastic.u)