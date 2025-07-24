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
Created on Mon Sep 26 17:58:18 2022

@author: bouteillerp
"""
from dolfinx.fem import Function

from ..utils.petsc_operations import petsc_add, petsc_assign

try:
    from jaxopt import LevenbergMarquardt
except Exception:
    print("jaxopt has not been loaded therefore complexe return mapping cannot be used")

try:
    from jax.numpy import clip, sqrt, zeros, linalg, reshape, array, concatenate
    from jax.lax import cond
    from jax import vmap, jit
except Exception:
    print("JAX has not been loaded therefore complexe return mapping cannot be used")

def reduced_3D_tr(x):
    return sum(x[:3])

def reduced_3D_dev(x, trace = None):
    tr = reduced_3D_tr(x)
    return x - 1./3 * tr * reduced_unit_array(len(x))

def reduced_3D_det(x):
    if len(x)==3:
        return x[0]*x[1]*x[2]
    elif len(x)==6:
        return x[0]*x[1]*x[2] + x[1]*x[2]*x[3]/sqrt(2) \
            - x[4]**2*x[2]/2 - x[5]**2*x[0]/2 - x[3]**2*x[2]/2 
            
def reduced_unit_array(length):
    if length == 3:
        return array([1, 1, 1])
    elif length == 6:
        return array([1, 1, 1, 0, 0, 0])

class PlasticSolve:
    def __init__(self, plastic, u):
        """
        Initialise le solveur pour la plasticité.

        Parameters
        ----------
        plastic : Objet de la classe plastic.
        u : Function, champ de déplacement.
        """
        self.plastic = plastic
        self.u = u
        if self.plastic.plastic_model == "J2_JAX":
            self.set_Jax()
            self.n_gauss = len(self.plastic.p.x.array)
            
 
    def set_Jax(self):
        self.mu = self.plastic.mu
        self.yield_stress = self.plastic.yield_stress
        self.equivalent_stress = lambda x: linalg.norm(x)
        self.clipped_equiv_stress = lambda s: clip(self.equivalent_stress(s), a_min=1e-8)
        # self.normal = jacfwd(self.clipped_equiv_stress)
        
        self.normal = lambda s : s / (self.equivalent_stress(s) + 1e-6)
        
        def residual_function(x, be_bar_trial, p_old):
            be_bar, dp = x[:-1], x[-1]
            dev_be_bar_trial = reduced_3D_dev(be_bar_trial)
            s_trial = self.mu * dev_be_bar_trial
            sig_eq_trial = self.clipped_equiv_stress(s_trial)
            
            # Combiner les résidus
            r_be = self.r_be_plastic(be_bar, dp, be_bar_trial, dev_be_bar_trial)
            r_p = self.r_p_plastic(p_old, dp, be_bar, sig_eq_trial)
            
            return concatenate([r_be, array([r_p])])
        
        self.solver = LevenbergMarquardt(residual_fun=residual_function, tol=1e-8, maxiter=100)
        self.batched_constitutive_update = jit(vmap(self.constitutive_update, in_axes=(0, 0)))



    def r_p_plastic(self, p_old, dp, be_bar, sig_eq_trial, *args):
        s = self.mu * reduced_3D_dev(be_bar)
        return (self.clipped_equiv_stress(s) - sqrt(2 / 3) * self.yield_stress(p_old + dp)) / self.mu
    
    def r_be_plastic(self, be_bar, dp, be_bar_trial, dev_be_bar_trial, *args):
        det_be_bar = reduced_3D_det(be_bar)
        dev_be_bar = reduced_3D_dev(be_bar)
        s = self.mu * dev_be_bar
            
        return dev_be_bar - dev_be_bar_trial \
                + 2 * sqrt(3 / 2) * dp * reduced_3D_tr(be_bar) / 3 * self.normal(s) \
                + (det_be_bar - 1) * reduced_unit_array(self.plastic.len_plas)
    
        
    def compute_trial_state(self, be_bar_trial, p_old):
        """Regroupe tous les calculs d'essai"""
        dev_be_bar_trial = reduced_3D_dev(be_bar_trial)
        s_trial = self.mu * dev_be_bar_trial
        sig_eq_trial = self.clipped_equiv_stress(s_trial)
        yield_criterion = sig_eq_trial - sqrt(2 / 3) * self.yield_stress(p_old)
        return dev_be_bar_trial, s_trial, sig_eq_trial, yield_criterion
    
    def constitutive_update(self, be_bar_trial, p_old):
        dev_be_bar_trial, s_trial, sig_eq_trial, yield_criterion = \
        self.compute_trial_state(be_bar_trial, p_old)
        
        return cond(
            yield_criterion < 0.0,
            lambda x: x,  # Cas élastique : retourne l'argument directement
            lambda x: self.return_mapping(x, p_old, dev_be_bar_trial, sig_eq_trial),  # Cas plastique
            be_bar_trial
                        )
    def return_mapping(self, be_bar_trial, p_old, dev_be_bar_trial, sig_eq_trial):
        x0 = zeros((self.plastic.len_plas + 1,))
        x0 = x0.at[:-1].set(be_bar_trial)
        
        sol = self.solver.run(x0, be_bar_trial, p_old)
        be_bar, dp = sol.params[:-1], sol.params[-1]
        p_old += dp
        return be_bar

        
    def solve(self):
        """
        Projection et actualisation de la déformation plastique
        """
        if self.plastic.plastic_model == "HPP_Plasticity":
            if self.plastic.hardening == "Isotropic":
                self.plastic.Delta_p.interpolate(self.plastic.Delta_p_expression)
                petsc_add(self.plastic.p.x.petsc_vec, self.plastic.Delta_p.x.petsc_vec)
            self.plastic.delta_eps_p.interpolate(self.plastic.Delta_eps_p_expression)
            petsc_add(self.plastic.eps_p.x.petsc_vec, self.plastic.delta_eps_p.x.petsc_vec)
                
                
                
            
        elif self.plastic.plastic_model == "Finite_Plasticity":
            self.plastic.barI_e.interpolate(self.plastic.barI_e_expr)
            self.plastic.dev_Be.interpolate(self.plastic.dev_Be_expr)
            petsc_assign(self.plastic.u_old, self.u)
        elif self.plastic.plastic_model == "J2_JAX":
            self.plastic.Be_Bar_trial_func.interpolate(self.plastic.Be_Bar_trial)
            Bbar_trial = reshape(self.plastic.Be_Bar_trial_func.x.array, (self.n_gauss, self.plastic.len_plas))
            be_bar = self.batched_constitutive_update(Bbar_trial, self.plastic.p.x.array)
            self.plastic.Be_Bar_old.x.array[:] = be_bar.ravel()