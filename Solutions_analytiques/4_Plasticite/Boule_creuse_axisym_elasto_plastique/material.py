#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 09:45:54 2025

@author: bouteillerp
"""
from CharonX import Material
import numpy as np
###### Modèle mécanique ######
E = 200e3
nu = 0.3
K = E/(3 * (1 - 2 * nu))
G = E/(2 * (1 + nu))

###### Paramètre géométrique ######
Re = 600
Ri = 300.0

#Paramètre élasto-plastique
sig0 = 300  # yield strength in MPa
Et = E / 100.0  # tangent modulus
H = E * Et / (E - Et)  # hardening modulus
q_lim = float(2 * np.log(Re / Ri) * sig0)
