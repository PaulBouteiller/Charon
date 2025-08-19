"""
GTN Simple Solver avec Newton JAX
=================================

Implémentation du solveur GTN selon Beardsmore et al. (2006)
utilisant le solveur Newton JAX existant

@author: bouteillerp
"""

try:
    import jax.numpy as jnp
    from jax import jit, vmap
    from .jax_newton_solver import JAXNewton
except ImportError:
    raise ImportError("JAX is required for this solver")

from ...utils.petsc_operations import petsc_assign

class GTNSimpleJAXSolver:
    """Solveur GTN simplifié avec Newton JAX selon Beardsmore et al. (2006)
    
    Résout le système de 4 équations non-linéaires (équations 23a-23d):
    f1: condition de consistance plastique
    f2: fonction de charge GTN = 0  
    f3: évolution déformation plastique microscopique
    f4: évolution fraction volumique de vides
    """
    
    def __init__(self, problem, plastic, u):
        """Initialize GTN simple solver with JAX Newton"""
        self.plastic = plastic
        self.u = u
        self.problem = problem
        
        # Paramètres matériau
        self.mu = plastic.mu
        # Calcul module de compressibilité K
        try:
            nu = problem.constitutive.nu
            self.K = 2 * self.mu * (1 + nu) / (3 * (1 - 2 * nu))
        except:
            # Valeur par défaut si nu non disponible
            self.K = 3 * self.mu  # Correspond à nu ≈ 0.3
        
        # Paramètres GTN
        self.q1 = plastic.q1
        self.q2 = plastic.q2
        self.fc = plastic.fc
        self.ff = plastic.ff
        self.fN = plastic.fN
        self.eps_N = plastic.eps_N
        self.sN = plastic.sN
        
        # Contrainte d'écoulement (supposée constante pour simplicité)
        self.sig_0 = self.plastic.sig_yield
    
        # Configuration solveur Newton
        self.newton_solver = JAXNewton(rtol=1e-8, atol=1e-8, niter_max=50)
        self._setup_residuals()
        
        # Vectorisation pour points de Gauss
        self.n_gauss = len(plastic.f.x.array)
        self.batched_solve = jit(vmap(self.solve_local_problem, in_axes=(0, 0, 0, 0)))
        
    def compute_f_star(self, f):
        """Porosité effective - équation (2)"""
        fu = 1.0 / self.q1
        return jnp.where(
            f <= self.fc,
            f,
            jnp.where(
                f < self.ff,
                self.fc + (fu - self.fc) / (self.ff - self.fc) * (f - self.fc),
                fu
            )
        )
        
    def gurson_yield_function(self, sig_eq, sig_m, f):
        """Fonction de charge GTN - équation (1)"""
        f_star = self.compute_f_star(f)
        
        term1 = (sig_eq / self.sig_0)**2
        
        # Protection overflow
        arg = 3.0 * self.q2 * sig_m / (2.0 * self.sig_0)
        cosh_term = jnp.where(
            jnp.abs(arg) > 10.0,
            jnp.exp(jnp.abs(arg)),
            jnp.cosh(arg)
        )
        
        term2 = 2.0 * self.q1 * f_star * cosh_term
        term3 = 1.0 + (self.q1 * f_star)**2
        
        return term1 + term2 - term3

    def _setup_residuals(self):
        """Configuration des résidus selon équations (23a-23d) Beardsmore et al."""
            
        @jit
        def nucleation_rate(eps_p_eq_M):
            """Taux de nucléation - équation (20)"""
            return (self.fN / (self.sN * jnp.sqrt(2.0 * jnp.pi))) * \
                   jnp.exp(-0.5 * ((eps_p_eq_M - self.eps_N) / self.sN)**2)
        
        def residuals(x):
            """Système de 4 équations - équations (23a-23d)"""
            Delta_eps_p_eq, Delta_eps_p_m, Delta_eps_p_eq_M, Delta_f = x
            
            # Variables d'état actuelles (passées en argument lors de l'appel)
            sig_eq_trial, sig_m_trial, f_old, eps_p_eq_M_old = self.current_state
            
            # Contraintes actuelles après correction plastique
            sig_eq = sig_eq_trial - 3.0 * self.mu * Delta_eps_p_eq  # équation (15)
            sig_m = sig_m_trial - 3.0 * self.K * Delta_eps_p_m      # équation (13b)
            
            # Dérivées de g pour condition de consistance
            f_star = self.compute_f_star(f_old + Delta_f)
            arg = 3.0 * self.q2 * sig_m / (2.0 * self.sig_0)
            
            dg_dsig_eq = 2.0 * sig_eq / (self.sig_0**2)
            dg_dsig_m = 2.0 * self.q1 * f_star * jnp.sinh(arg) * 3.0 * self.q2 / (2.0 * self.sig_0)
            
            # f1: Condition de consistance - équation (23a)
            f1 = dg_dsig_m * Delta_eps_p_eq - 3.0 * dg_dsig_eq * Delta_eps_p_m
            
            # f2: Fonction de charge GTN - équation (23b) 
            f2 = self.gurson_yield_function(sig_eq, sig_m, f_old + Delta_f)
            
            # f3: Évolution déformation plastique microscopique - équation (23c)
            work_term = sig_eq * Delta_eps_p_eq + 3.0 * sig_m * Delta_eps_p_m
            f3 = Delta_eps_p_eq_M - work_term / (self.sig_0 * (1.0 - f_old - Delta_f))
            
            # f4: Évolution fraction volumique - équation (23d)
            growth_term = 3.0 * (1.0 - f_old) * Delta_eps_p_m  # Croissance
            nucleation_term = nucleation_rate(eps_p_eq_M_old + Delta_eps_p_eq_M) * Delta_eps_p_eq_M
            f4 = Delta_f - growth_term - nucleation_term
            
            return jnp.array([f1, f2, f3, f4])
        
        self.residuals = residuals
        self.newton_solver.set_residual(residuals)
        
    def solve_local_problem(self, sig_eq_trial, sig_m_trial, f_old, eps_p_eq_M_old):
       """Résolution locale du problème GTN pour un point de Gauss"""
       from jax.lax import cond
       
       # Test élastique
       f_star_trial = self.compute_f_star(f_old)
       
       # Fonction de charge trial
       yield_trial = self.gurson_yield_function(sig_eq_trial, sig_m_trial, f_star_trial)
       
       def plastic_case(_):
           # Cas plastique: résolution avec Newton
           self.current_state = (sig_eq_trial, sig_m_trial, f_old, eps_p_eq_M_old)
           
           # Initialisation: estimations basées sur J2 classique
           Delta_eps_p_eq_init = yield_trial / (6.0 * self.mu)
           Delta_eps_p_m_init = 0.1 * Delta_eps_p_eq_init
           
           x0 = jnp.array([
               Delta_eps_p_eq_init,    # Δε_eq^p
               Delta_eps_p_m_init,     # Δε_m^p  
               Delta_eps_p_eq_init,    # Δε_eq_M^p
               0.01 * Delta_eps_p_eq_init  # Δf
           ])
           
           solution, data = self.newton_solver.solve(x0)
           return solution
       
       return cond(
           yield_trial <= 0.0,
           lambda _: jnp.array([0.0, 0.0, 0.0, 0.0]),  # Élastique
           plastic_case,  # Plastique
           None
       )
        
    def solve(self):
        """Résolution du problème GTN sur tous les points de Gauss"""
        
        # Calcul des contraintes d'essai élastiques
        sig_eq_trials = self.get_trial_equivalent_stress()
        sig_m_trials = self.get_trial_hydrostatic_stress()
        
        # États actuels
        f_old_array = self.plastic.f.x.array.copy()
        eps_p_eq_M_old_array = self.plastic.eps_p_eq_M.x.array.copy()
        
        # Résolution vectorisée
        solutions = self.batched_solve(
            sig_eq_trials, sig_m_trials, f_old_array, eps_p_eq_M_old_array
        )
        
        # Mise à jour des variables internes
        Delta_eps_p_eq = solutions[:, 0]
        Delta_eps_p_m = solutions[:, 1] 
        Delta_eps_p_eq_M = solutions[:, 2]
        Delta_f = solutions[:, 3]
        
        # Accumulation des variables d'état
        self.plastic.eps_p_eq.x.array[:] += Delta_eps_p_eq
        self.plastic.eps_p_m.x.array[:] += Delta_eps_p_m
        self.plastic.eps_p_eq_M.x.array[:] += Delta_eps_p_eq_M
        self.plastic.f.x.array[:] += Delta_f
        
        # Mise à jour déplacement précédent
        petsc_assign(self.plastic.u_old, self.u)
        
    def get_trial_equivalent_stress(self):
        """Contrainte équivalente d'essai élastique
        
        À adapter selon votre structure pour extraire les contraintes
        depuis le problème constitutif
        """
        # Version simplifiée - à remplacer par votre implémentation
        try:
            # Essayer d'extraire depuis le problème constitutif
            constitutive = self.problem.constitutive
            # Supposons que vous avez un moyen d'obtenir le tenseur de contrainte trial
            stress_trial = constitutive.get_trial_stress()  # À implémenter
            sig_eq = constitutive.compute_von_mises(stress_trial)  # À implémenter
            return jnp.array(sig_eq)
        except:
            # Valeur par défaut pour test
            return jnp.ones(self.n_gauss) * self.plastic.sig_yield * 1.1
        
    def get_trial_hydrostatic_stress(self):
        """Contrainte hydrostatique d'essai élastique
        
        À adapter selon votre structure pour extraire les contraintes
        depuis le problème constitutif
        """
        # Version simplifiée - à remplacer par votre implémentation
        try:
            # Essayer d'extraire depuis le problème constitutif
            constitutive = self.problem.constitutive
            stress_trial = constitutive.get_trial_stress()  # À implémenter
            sig_h = constitutive.compute_hydrostatic(stress_trial)  # À implémenter
            return jnp.array(sig_h)
        except:
            # Valeur par défaut pour test
            return jnp.ones(self.n_gauss) * self.plastic.sig_yield * 0.3
            
    def compute_stress_correction(self):
        """Calcule la correction de contrainte après résolution plastique
        
        À utiliser pour mettre à jour les contraintes dans le problème principal
        """
        # Corrections plastiques calculées
        Delta_eps_p_eq = self.plastic.eps_p_eq.x.array
        Delta_eps_p_m = self.plastic.eps_p_m.x.array
        
        # Corrections de contrainte
        Delta_sig_eq = -3.0 * self.mu * Delta_eps_p_eq
        Delta_sig_m = -3.0 * self.K * Delta_eps_p_m
        
        return Delta_sig_eq, Delta_sig_m