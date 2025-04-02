import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def sigma_ext(t, amplitude):
    """
    Créneau d'amplitude 'amplitude' 
    """
    return amplitude if t >= 0.0 else 0.0

def solve_g(R_out, cp, lambd, mu, amplitude, 
            tmax, num_points=2000):
    """
    Résout l'ODE pour g(xi) :
      (lambda + 2mu)/R_out * g'(xi) + (lambda - 2mu)/R_out^2 * g(xi) = sigma_ext(xi - R_out/cp)
    On pose:
      c = (lambda + 2mu)
      d = (lambda - 2mu)
    => c/R_out * g'(xi) + d/R_out^2 * g(xi) = sigma_ext(...)
    => g'(xi) = [R_out/c] * [sigma_ext(...) - d/R_out^2 * g(xi)]
    
    On intègre de xi=0 à xi=tmax + R_out/cp (approx).
    
    Paramètres:
    -----------
    R_out : rayon externe
    cp    : vitesse onde P = sqrt((lambda+2mu)/rho)
    lambd : lambda (module de Lamé)
    mu    : mu (module de cisaillement)
    amplitude : amplitude du créneau de contrainte
    tmax      : temps max pour la simulation
    num_points: nombre de points pour solve_ivp
    
    Retourne:
    ---------
    xi_vals : tableau des xi
    g_vals  : tableau des g(xi) solution de l'ODE
    """
    c = lambd + 2.0*mu
    d = lambd - 2.0*mu
    
    xi_span = [0.0, tmax + R_out/cp]  # on intègre un peu plus loin que tmax
    
    xi_eval = np.linspace(xi_span[0], xi_span[1], num_points)
    
    # Définition de l'ODE : dy/dxi = ...
    def ode_g(xi, y):
        # y = g(xi)
        # sigma_ext(t) = sigma_ext(xi - R_out/cp)
        t_local = xi - R_out/cp
        rhs = (R_out / c) * (sigma_ext(t_local, amplitude) - (d/(R_out**2)) * y)
        return rhs
    
    # Condition initiale : g(0) = 0 (supposons qu'avant xi=0, pas d'onde)
    y0 = [0.0]
    sol = solve_ivp(ode_g, xi_span, y0, t_eval=xi_eval, vectorized=False, 
                method='RK45', rtol=1e-8, atol=1e-8)
    
    xi_vals = sol.t
    g_vals  = sol.y[0]
    
    return xi_vals, g_vals

def compute_sigma_rr(r_vals, t_vals, R_out, cp, lambd, mu, 
                     xi_vals, g_vals):
    """
    Calcule la contrainte sigma_rr(r,t) = (lambda+2mu)/r*(g'(xi)/cp) + (lambda-2mu)/r^2*g(xi).
    où xi = t + r/cp.
    
    Pour cela, on interpole g'(xi) et g(xi) sur 'xi_vals' et 'g_vals'.
    """
    from scipy.interpolate import interp1d
    
    # On interpole d'abord g(xi)
    g_interp = interp1d(xi_vals, g_vals, kind='linear', bounds_error=False, fill_value=0.0)
    
    # On construit aussi la dérivée g'(xi) par différences ou spline
    # Pour la précision, on peut redériver le spline lui-même, ou faire une diff numérique.
    # Ici, on dérive le spline interpolé (spline derivative) pour avoir g'(xi).
    # Par simplicité, on refait un spline paramétré ou on peut approx. la dérivée par diff.
    # On va plutôt créer un interp1d sur la base d'une dérivation numérique.
    dg_dxi_num = np.gradient(g_vals, xi_vals, edge_order=2)
    dg_interp  = interp1d(xi_vals, dg_dxi_num, kind='linear', 
                           bounds_error=False, fill_value=0.0)
    
    c = lambd + 2.0*mu
    d = lambd - 2.0*mu
    sigma_mat = np.zeros((len(r_vals), len(t_vals)), dtype=float)

    for i, r in enumerate(r_vals):
        for j, t in enumerate(t_vals):
            xi = t + r/cp
            g_val = g_interp(xi)
            dg_val = dg_interp(xi)
            
            # Retrait de la division par cp dans le premier terme
            sig = (c/r)*dg_val + (d/r**2)*g_val
            sigma_mat[i,j] = sig
    
    return sigma_mat

def main_analytique(R_int, R_ext, lmbda, mu, rho, amplitude, Tfin):
    # Vitesse d'onde P
    cp = np.sqrt((lmbda + 2 * mu)  /rho)
    # Discrétisations
    nr = 500  # nb de points en r
    nt = 400 # nb de points en t
    
    r_vals = np.linspace(R_int, R_ext, nr)  # de R_int à R_ext
    t_vals = np.linspace(0, Tfin, nt)
    
    # -------------------------
    # 1) On résout l'ODE pour g(xi)
    # -------------------------
    xi_vals, g_vals = solve_g(R_ext, cp, lmbda, mu, amplitude, 
                              Tfin, num_points=2000)
    
    # -------------------------
    # 2) On calcule sigma_rr(r, t)
    # -------------------------
    sigma_mat = compute_sigma_rr(r_vals, t_vals, R_ext, cp, lmbda, mu,
                                 xi_vals, g_vals)
    
    # Ne tracer que la dernière courbe temporelle
    plt.plot(r_vals, sigma_mat[:, -1], label=f'Analytique t={t_vals[-1]:.2e}ms')