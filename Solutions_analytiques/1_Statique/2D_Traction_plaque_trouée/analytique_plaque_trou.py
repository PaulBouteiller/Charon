#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 15:19:52 2024

@author: bouteillerp
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
    