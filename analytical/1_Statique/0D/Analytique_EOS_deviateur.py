"""
Implémentation de différentes équations d'état (EOS) et lois de comportement déviatorique.

Ce module contient des fonctions analytiques permettant de calculer la pression hydrostatique
et les contraintes déviatoriques pour différents modèles constitutifs en mécanique des solides.

Équations d'état implémentées:
    - Différentes formes d'EOS hyperélastiques (p1-p8)
    - Vinet
    - Mie-Grüneisen (MG)
    - JWL (Jones-Wilkins-Lee)
    - MACAW
    - Formulation tabulée

Modèles déviatoriques implémentés:
    - Hyperélastique isotrope (HPP_devia)
    - Néo-Hookéen (NeoHookean)
    - Mooney-Rivlin (MooneyRivlin)

Auteur: bouteillerp
Date de création: 31 Janvier 2024
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

def MACAW(A, B, C, V0, Vinf, theta0, a0, m, n, gammainf, gamma0, Cv_inf, J, T):
    V = J * V0
    
    def theta(V):
        ratio = V / Vinf
        return theta0 * ratio**(-gamma0) * (ratio**(-m) + 1)**((gammainf - gamma0) / m)
      
    def a(V):
        return a0 / (1 + (V / Vinf)**(-n))
    
    def dadV(V):
        return a0 * n * (V / Vinf)**n / (V * (1 + (V / Vinf)**n)**2)
    
    def dthetadV(V):
        return -theta(V) * (gammainf + gamma0 * (V / Vinf)**m) / (V * (1 + (V / Vinf)**m))
    
    p_cold = A*J**(-B-1) * exp(2./3*C*(1.-J**(3./2))) * (C * J**(3./2) + B) - A*(B+C)
    
    # Appeler les nouvelles fonctions avec V
    thetaV = theta(V)
    aV = a(V)
    dthetadVV = dthetadV(V)
    dadVV = dadV(V)
        
    q0 = 1./3 * dadVV - dthetadVV / thetaV
    q1 = 5./6 * thetaV * dadVV - aV * dthetadVV / 6.
    q2 = thetaV**2 * dadVV / 2. - aV * thetaV * dthetadVV / 2.
    pref = Cv_inf / (T + thetaV)**3
    P_th = pref * (q0 * T**4 + q1 * T**3 + q2 * T**2)
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