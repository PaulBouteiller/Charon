"""
Solution analytique pour un problème de traction uniaxiale en coordonnées cartésiennes 1D.

Ce module fournit des fonctions permettant de calculer les champs de contraintes et de déformations
pour différents modèles constitutifs en hyperélasticité (IsotropicHPP, NeoHook).

Fonctions:
    J(eps): Calcule le jacobien de la transformation (déterminent du gradient de déformation)
    eps_vector(eps): Retourne le vecteur de déformation pour une traction 1D
    dev_eps_vector(eps): Calcule le vecteur déviateur des déformations
    dev_B_vector(eps): Calcule le vecteur déviateur du tenseur de Cauchy-Green gauche
    p(eps, kappa, eos_type): Calcule la pression hydrostatique selon l'équation d'état
    s(eps, mu, devia_type): Calcule la partie déviatorique du tenseur des contraintes
    sigma_xx(eps, kappa, mu, eos_type, devia_type): Calcule la contrainte axiale totale

Auteur: bouteillerp
Date de création: 7 Septembre 2023
"""

from numpy import array

dev_vec = array([2./3, -1./3, -1./3])

def J(eps):
    return 1 + eps

def eps_vector(eps):
    return [eps, 0, 0]

def dev_eps_vector(eps):
    return eps * dev_vec

def dev_B_vector(eps):
    return (2 * eps + eps**2) * dev_vec

def p(eps, kappa, eos_type):
    if eos_type == "IsotropicHPP" or eos_type == "U1":
        return kappa * (J(eps) - 1)

def s(eps, mu, devia_type):
    if devia_type == "IsotropicHPP":
        return 2 * mu * dev_eps_vector(eps)
    elif devia_type == "NeoHook":
        return mu / J(eps)**(5./3) * dev_B_vector(eps)
    elif devia_type == "NeoHook":
        return mu / J(eps)**(5./3) * dev_B_vector(eps)

def sigma_xx(eps, kappa, mu, eos_type, devia_type):
    return p(eps, kappa, eos_type) + s(eps, mu, devia_type)[0]