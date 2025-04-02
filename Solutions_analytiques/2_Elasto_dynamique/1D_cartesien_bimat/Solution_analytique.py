import numpy as np
###############################################################################
# 2. Géométrie et pulse
###############################################################################
# L_total = 50.0
# L_int   = L_total/2.0

# sigma0  = 1e3    # Amplitude de contrainte (CHOISISSEZ LA SIGNE). Ici négatif
# T_pulse = (L_total/4)/c1

# Pulse en contrainte : créneau simple
def f_stress_single(t, T_pulse, sigma0):
    """sigma0 pour 0 <= t < T_pulse, sinon 0."""
    return np.where((t >= 0) & (t < T_pulse), sigma0, 0.0)

def g_stress_single(t, T_pulse, sigma0):
    """Intégrale de f_stress_single(t)."""
    return np.where(t < 0, 0.0,
                    np.where(t < T_pulse, sigma0*t, sigma0*T_pulse))

###############################################################################
# 3. Déplacements : incident, réfléchi, transmis
###############################################################################
def u_inc(x, t, T_pulse, c1, Z1, sigma0):
    """Onde incidente, se propageant vers +x."""
    tau = t - x/c1
    return (1.0/Z1)*np.heaviside(tau, 0)*g_stress_single(tau, T_pulse, sigma0)

def u_ref(x, t, T_pulse, c1, L_int, R, Z1, sigma0):
    """
    Onde réfléchie, se propageant vers -x.
    """
    tau = t - ((2*L_int - x)/c1)
    return (R/Z1)*np.heaviside(tau, 0)*g_stress_single(tau, T_pulse, sigma0)

def u_trans(x, t, T_pulse, c1, c2, L_int, Tcoef, Z2, sigma0):
    """Onde transmise dans le milieu 2, se propageant vers +x."""
    tau = t - (L_int/c1 + (x - L_int)/c2)
    return (Tcoef/Z2)*np.heaviside(tau, 0)*g_stress_single(tau, T_pulse, sigma0)

def u_total(x, t, T_pulse, c1, c2, L_int, R, Tcoef, Z1, Z2):
    """Superposition des ondes dans chaque domaine."""
    u = np.zeros_like(x)
    mask1 = (x < L_int)   # Milieu 1
    mask2 = (x >= L_int)  # Milieu 2
    u[mask1] = (u_inc(x[mask1], t, T_pulse, c1, Z1)
                + u_ref(x[mask1], t, T_pulse, c1, L_int, R, Z1))
    u[mask2] = u_trans(x[mask2], t, T_pulse, c1, c2, L_int, Tcoef, Z2)
    return u

###############################################################################
# 4. Contrainte : σ = ρ c^2 ∂u/∂x  (dans chaque domaine)
###############################################################################
def du_inc_dx(x, t, T_pulse, c1, Z1, sigma0):
    tau = t - x/c1
    # dérivée de g_stress_single(tau) par rapport à x => -1/c1 * f_stress_single(tau)
    return -(1/(Z1*c1))*np.heaviside(tau, 0)*f_stress_single(tau, T_pulse, sigma0)

def du_ref_dx(x, t, T_pulse, c1, L_int, R, Z1, sigma0):
    tau = t - ((2*L_int - x)/c1)
    # ICI on a un 'moins' qui apparaît à cause de d/dx[t - (2L_int - x)/c1] = +1/c1
    # On obtient donc un "-" final en combinant le signe R et le - de la dérivation:
    return -(R/(Z1*c1))*np.heaviside(tau, 0)*f_stress_single(tau, T_pulse, sigma0)

def du_trans_dx(x, t, T_pulse, c1, c2, L_int, Tcoef, Z2, sigma0):
    tau = t - (L_int/c1 + (x - L_int)/c2)
    return -(Tcoef/(Z2*c2))*np.heaviside(tau, 0)*f_stress_single(tau, T_pulse, sigma0)

def sigma_total(x, t, T_pulse, c1, c2, L_int, R, Tcoef, rho1, rho2, Z1, Z2, sigma0):
    """
    On calcule ∂u/∂x, puis σ = ρ c^2 (∂u/∂x).
    """
    du = np.zeros_like(x)
    mask1 = (x < L_int)
    mask2 = (x >= L_int)
    du[mask1] = (du_inc_dx(x[mask1], t, T_pulse, c1, Z1, sigma0)
                 + du_ref_dx(x[mask1], t, T_pulse, c1, L_int, R, Z1, sigma0))
    du[mask2] = du_trans_dx(x[mask2], t, T_pulse, c1, c2, L_int, Tcoef, Z2, sigma0)
    
    sigma = np.zeros_like(x)
    sigma[mask1] = rho1*(c1**2)*du[mask1]
    sigma[mask2] = rho2*(c2**2)*du[mask2]
    return sigma

###############################################################################
# 5. Exemple de tracé
###############################################################################


# Choix d'un temps d'observation
# t_obs = 0.9*(L_total/c1)  # juste pour illustrer

def compute_sigma_tot(t_obs, T_pulse, L_tot, L_int, sigma0, rho1, rho2, E1, E2, nu1, nu2):
    def wave_speed(E, nu, rho):
        return np.sqrt(E/rho * (1 - nu)/((1 + nu)*(1 - 2*nu)))
    c1 = wave_speed(E1, nu1, rho1)
    c2 = wave_speed(E2, nu2, rho2)
    Z1 = rho1 * c1
    Z2 = rho2 * c2

    R = (Z2 - Z1)/(Z1 + Z2)
    T = 2*Z2/(Z1 + Z2)
    Nx    = 1000
    x_vals = np.linspace(0, L_tot, Nx)

    print(f"Acier: c1={c1:.3f}, Z1={Z1:.3f}")
    print(f"Alu:   c2={c2:.3f}, Z2={Z2:.3f}")
    print("Coefficient de réflexion R =", R)
    print("Coefficient de transmission T =", T)
    return sigma_total(x_vals, t_obs, T_pulse, c1, c2, L_int, R, T,
                             rho1, rho2, Z1, Z2, sigma0)