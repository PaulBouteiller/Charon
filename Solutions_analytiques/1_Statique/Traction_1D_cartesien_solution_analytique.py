#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 09:06:27 2023

@author: bouteillerp
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