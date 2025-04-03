"""
Created on Wed Jan 15 15:17:40 2025

@author: bouteillerp
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def sigma_ext(t, amplitude):
    """
    Échelon (step function) d'amplitude 'amplitude'
    """
    return amplitude if t >= 0.0 else 0.0

def solve_g(R_out, cp, lambd, mu, amplitude, tmax, num_points):
    """
    Résout l'ODE pour g(xi) en coordonnées cylindriques :
    L'équation est modifiée pour tenir compte de la géométrie cylindrique :
    (lambda + 2mu)/R_out * g'(xi) + (lambda - 2mu)/(2R_out^2) * g(xi) = sigma_ext(xi - R_out/cp)
    
    La principale différence avec le cas sphérique est le facteur 1/2 devant le terme en g(xi)
    dû à la courbure cylindrique différente.
    
    Paramètres:
    -----------
    R_out : rayon externe
    cp    : vitesse onde P = sqrt((lambda+2mu)/rho)
    lambd : lambda (module de Lamé)
    mu    : mu (module de cisaillement)
    amplitude : amplitude de l'échelon de contrainte
    tmax      : temps max pour la simulation
    num_points: nombre de points pour solve_ivp
    
    Retourne:
    ---------
    xi_vals : tableau des xi
    g_vals  : tableau des g(xi) solution de l'ODE
    """
    c = lambd + 2.0*mu
    d = lambd - 2.0*mu
    
    xi_span = [0.0, tmax + R_out/cp]
    
    xi_eval = np.linspace(xi_span[0], xi_span[1], num_points)
    
    def ode_g(xi, y):
        t_local = xi - R_out/cp
        # Modification pour le cas cylindrique : facteur 1/2 et dépendance en 1/sqrt(r)
        rhs = (np.sqrt(R_out) / c) * (sigma_ext(t_local, amplitude) - (d/(2.0*R_out*np.sqrt(R_out))) * y)
        return rhs
    
    y0 = [0.0]
    sol = solve_ivp(ode_g, xi_span, y0, t_eval=xi_eval, vectorized=False, 
                    method='BDF', rtol=1e-8, atol=1e-8)
    
    return sol.t, sol.y[0]

def compute_sigma_rr(r_vals, t_vals, R_out, cp, lambd, mu, xi_vals, g_vals):
    """
    Calcule la contrainte radiale sigma_rr(r,t) en coordonnées cylindriques:
    sigma_rr = (lambda+2mu)/r * (g'(xi)/cp) + (lambda-2mu)/(2r^2) * g(xi)
    
    La différence principale avec le cas sphérique est le facteur 1/2 dans le second terme
    """
    from scipy.interpolate import interp1d
    
    g_interp = interp1d(xi_vals, g_vals, kind='linear', bounds_error=False, fill_value=0.0)
    
    dg_dxi_num = np.gradient(g_vals, xi_vals, edge_order=2)
    dg_interp = interp1d(xi_vals, dg_dxi_num, kind='linear', 
                        bounds_error=False, fill_value=0.0)
    
    c = lambd + 2.0*mu
    d = lambd - 2.0*mu
    sigma_mat = np.zeros((len(r_vals), len(t_vals)), dtype=float)

    for i, r in enumerate(r_vals):
        for j, t in enumerate(t_vals):
            xi = t + r/cp
            g_val = g_interp(xi)
            dg_val = dg_interp(xi)
            
            # Modification pour le cas cylindrique : facteur 1/2 et dépendance en 1/sqrt(r)
            sig = (c/np.sqrt(r))*dg_val + (d/(2.0*r*np.sqrt(r)))*g_val
            sigma_mat[i,j] = sig
    
    return sigma_mat

def main_analytique(R_int, R_ext, lmbda, mu, rho, amplitude, Tfin, num_points=2000):
    """
    Fonction principale pour le cas cylindrique
    """
    cp = np.sqrt((lmbda + 2 * mu)/rho)
    
    nr = 500  # nb de points en r
    nt = 400  # nb de points en t
    
    r_vals = np.linspace(R_int, R_ext, nr)
    t_vals = np.linspace(0, Tfin, nt)
    
    # 1) Résolution de l'ODE pour g(xi)
    xi_vals, g_vals = solve_g(R_ext, cp, lmbda, mu, amplitude, Tfin, num_points)
    
    # 2) Calcul de sigma_rr(r, t)
    sigma_mat = compute_sigma_rr(r_vals, t_vals, R_ext, cp, lmbda, mu, xi_vals, g_vals)
    
    plt.plot(r_vals, sigma_mat[:, -1], label=f't={t_vals[-1]:.2e}ms')