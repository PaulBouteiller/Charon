"""
Module implémentant un solveur de Newton en JAX avec différentiation automatique implicite.

Ce module fournit une implémentation efficace et différentiable d'un solveur de Newton
pour résoudre des systèmes d'équations non-linéaires. Il utilise JAX pour la 
différentiation automatique et est particulièrement adapté pour être intégré dans
des chaînes de calcul différentiables.
"""
try:
    from typing import NamedTuple
    from jax.numpy import logical_and, concatenate, isscalar, linalg, atleast_1d
    from jax.lax import custom_root, while_loop
    from jax import jacfwd
except Exception:
    print("JAX has not been loaded therefore complexe return mapping cannot be used")

class SolverParameters(NamedTuple):
    """Classe définissant les paramètres du solveur de Newton.
    
    Attributes
    ----------
    rtol : float
        Tolérance relative pour le critère de convergence
    atol : float
        Tolérance absolue pour le critère de convergence
    niter_max : int
        Nombre maximum d'itérations autorisées
    """

    rtol: float
    atol: float
    niter_max: int
    

def _solve_linear_system(x, J, b):
    """Résout le système linéaire Jx = b.
    
    Cette fonction gère à la fois les cas scalaires et vectoriels.
    
    Parameters
    ----------
    x : array_like Point courant (utilisé pour déterminer la dimension du problème)
    J : array_like Matrice jacobienne
    b : array_like Second membre
        
    Returns
    -------
    array_like
        Solution du système linéaire
    """
    if isscalar(x):
        return b / J
    else:
        dx = linalg.solve(J, b)
    return dx

def newton_solve(x0, res_func, dr_dx, params):
    def compute_residual(x):
        return res_func(x)

    def run_newton_step(state):
        norm_res, niter, x, res = state
        dx = _solve_linear_system(x, dr_dx(x), -res)
        x += dx
        res_new = compute_residual(x)
        norm_res_new = linalg.norm(x)
        
        return norm_res_new, niter + 1, x, res_new
    
    def convergence_check(state, norm_res0):
        norm_res, niter, _, _ = state
        cond_a_tol = norm_res > params.atol
        cond_r_tol = norm_res > params.rtol * norm_res0
        cond_iter = niter < params.niter_max
        return logical_and(logical_and(cond_a_tol, cond_r_tol), cond_iter)
    
    # État initial
    res = compute_residual(x0)
    norm_res0 = linalg.norm(res)
    init_state = (norm_res0, 0, x0, res)
    
    # Boucle principale
    final_state = while_loop(lambda s: convergence_check(s, norm_res0),
                                          run_newton_step, init_state)
    
    norm_res, niter, x_sol, res_sol = final_state
    return x_sol, (niter, norm_res0, norm_res, res_sol)

class JAXNewton:
    """Une classe implémentant un solveur de Newton en JAX.
    
    Cette classe fournit une interface simple pour résoudre des systèmes
    non-linéaires avec l'algorithme de Newton. Elle utilise la différentiation
    automatique de JAX pour calculer les jacobiennes et permet une différentiation
    implicite personnalisée.
    
    Attributes
    ----------
    params : SolverParameters
        Paramètres du solveur
    r : callable
        Fonction résidu
    dr_dx : callable
        Jacobienne du résidu
    """

    def __init__(self, rtol=1e-8, atol=1e-8, niter_max=2000):
        """Newton solver

        Parameters
        ----------
        rtol : float, optional
            Relative tolerance for the Newton method, by default 1e-8
        atol : float, optional
            Absolute tolerance for the Newton method, by default 1e-8
        niter_max : int, optional
            Maximum number of allowed iterations, by default 200
        """
        self.params = SolverParameters(rtol, atol, niter_max)

    def set_residual(self, r):
        """Set the residual  vector r(x)
        Parameters
        ----------
        r : callable, list, tuple
            Residual to solve for. r(x) is a function of R^n to R^n. Alternatively, r can be a list/tuple
            of functions with the same signature. The resulting system corresponds to a concatenation of all
            individual residuals.
        """
        # residual
        if isinstance(r, list) or isinstance(r, tuple):
            self.r = lambda x: concatenate([atleast_1d(ri(x)) for ri in r])
        else:
            self.r = r
            
    def set_Jacobian(self, dr_dx = None):
        """
        dr_dx : callable, optional
            The jacobian of the residual. dr_dx(x) is a function of R^n to R^{n}xR^n. If None (default),
            JAX computes the residual using forward-mode automatic differentiation.
        """
        #Jacobian
        if dr_dx is None:
            self.dr_dx = jacfwd(self.r)
        else:
            self.dr_dx = dr_dx

    def solve(self, x):
        """Résout le système non-linéaire r(x) = 0.
        
        Cette méthode utilise custom_root de JAX pour permettre une
        différentiation implicite personnalisée du solveur.
        
        Parameters
        ----------
        x : array_like
            Point initial
            
        Returns
        -------
        tuple
            (x_sol, data) où x_sol est la solution et data contient les
            informations sur la convergence
        """
        # return newton_solve(x, self.r, self.dr_dx, self.params)
        solve = lambda f, x: newton_solve(x, f, jacfwd(f), self.params)
        tangent_solve = lambda g, y: _solve_linear_system(x, jacfwd(g)(y), y)
        return custom_root(self.r, x, solve, tangent_solve, has_aux=True)