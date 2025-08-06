"""
Solution analytique pour une plaque trouée soumise à une traction.

Ce module implémente la solution analytique de Kirsch pour le champ de déplacement
dans une plaque infinie avec un trou circulaire soumise à une contrainte de traction
uniforme à l'infini.

Fonctions:
    prefactor(nu, E, sig_infty): Calcule le préfacteur commun aux expressions de déplacement
    ur(a, r, nu, E, sig_infty, theta): Calcule le déplacement radial
    utheta(a, r, nu, E, sig_infty, theta): Calcule le déplacement tangentiel

Paramètres:
    - a: Rayon du trou
    - r: Distance radiale depuis le centre du trou
    - nu: Coefficient de Poisson
    - E: Module d'Young
    - sig_infty: Contrainte appliquée à l'infini
    - theta: Angle (en radians) par rapport à l'axe horizontal

Auteur: bouteillerp
Date de création: 7 Février 2024
"""
from math import cos, sin

def prefactor(nu, E, sig_infty):
    return sig_infty * (1 + nu) / (2 * E)

def ur(a, r, nu, E, sig_infty, theta):
    print("Le rayon vaut", r)
    print("Le point est à", a)
    print("Le coefficient de Poisson vaut", nu)
    print("Le module D'young vaut", E)
    print("La contrainte à l'infty vaut", sig_infty)
    print("L'angle vaut", theta)

    
    pref = prefactor(nu, E, sig_infty)
    u_r = pref * ((1 - 2 * nu) * r + a**2 / r + \
                  (r - a**4 / r**3 + 4 * a**2 / r * (1 - nu)) * cos(2 * theta))
    return u_r

def utheta(a, r, nu, E, sig_infty, theta):
    pref = prefactor(nu, E, sig_infty)
    u_thet = -pref * (r + a**4 / r**3 + 2 * a**2 / r * (1- 2 * nu)) * sin(2 * theta)
    return u_thet
    