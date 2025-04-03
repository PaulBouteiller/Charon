"""
Created on Wed Jan 31 13:28:36 2024

@author: bouteillerp
"""
from math import log, exp

#Comparaison équation d'état hyper-élastique.
def p1(kappa, J):
    return -kappa * (J-1)

def p2(kappa, J):
    return - kappa *log(J)/J

def p3(kappa, J):
    return (p1(J)+p2(J))/2

def p5(kappa, J):
    return -kappa * log(J)

def p7(kappa, J):
    return -kappa/2 * (exp(J-1) - 1/J)

def p8(kappa, J):
    return -kappa/2 * (log(J) - 1/J + 1)

def Vinet(K0, K1, J):
    return 3 * K0 * J**(-2/3) * (1-J**(1/3)) * exp(3./2 * (K1-1)*(1 - J**(1./3)))

def MG(C, D, S, J):
    mu = 1/J - 1
    return C * mu + D * mu**2 + S * mu**3

def JWL(A, B, R1, R2, J):
    return A * exp(-R1 * J) + B * exp(-R2 * J)   

def MACAW(A, B, C, eta, theta0, a0, m, n, gammainf, gamma0, C_mass, J, T):
    def theta(J, theta0, eta, gamma0, m, gammainf):
        return theta0 * (J * eta)**(-gamma0) * \
                (1 + (J * eta)**(-m))**((gammainf-gamma0)/m)
    
    def a(J, a0, eta, n):
        return a0 / (1 + (J * eta)**(-n))
    
    def dadJ(J, a0, eta, n):
        return (a0 * n * (eta * J)**n) / (J * (1 + (J * eta)**n)**2) 
    
    def dthetadJ(J, theta0, eta, gamma0, m, gammainf):
        numerateur = - theta(J, theta0, eta, gamma0, m, gammainf) * \
                        (gammainf + gamma0 * (J * eta)**m)
        denominateur = J * (1 + (J * eta)**m)
        return numerateur / denominateur

    p_cold = A*J**(-B-1) * exp(2./3*C*(1.-J**(3./2))) * (C * J**(3./2) + B) - A*(B+C)
    
    thetaJ = theta(J, theta0, eta, gamma0, m, gammainf)
    aJ = a(J, a0, eta, n)
    dthetadJJ = dthetadJ(J, theta0, eta, gamma0, m, gammainf)
    dadJJ = dadJ(J, a0, eta, n)
        
    q0 = 1. / 3 * dadJJ - dthetadJJ / thetaJ
    q1 = 5. / 6 * thetaJ * dadJJ - aJ * dthetadJJ / 6.
    q2 = thetaJ**2 * dadJJ / 2. - aJ * thetaJ * dthetadJJ / 2.
    pref = C_mass / (T + thetaJ)**3 #Pour compenser ma dérivation par rapport à J
    P_th = pref * (q0 * T**4 + q1 *T **3 + q2 * T **2)
    return p_cold + P_th

def tabulated(kappa, J):
    return p1(kappa, J)

def HPP_devia(mu, J):
    eps = J - 1
    return 2 * mu * 2./3 * eps

def NeoHookean(mu, J):
    eps = J - 1
    return mu * (2 * eps + eps**2) / 3. * 2 / J**(5./3)

def MooneyRivlin(mu, mu_quad, J):
    eps = J - 1
    return mu * (2 * eps + eps**2) / 3. * 2 / J**(5./3) + mu_quad * (2 * eps + eps**2) / 3. * 2 / J**(7./3)