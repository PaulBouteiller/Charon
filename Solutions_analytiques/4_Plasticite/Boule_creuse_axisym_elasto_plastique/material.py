"""
Module de définition des paramètres matériaux et géométriques pour l'analyse élasto-plastique.

Paramètres matériaux:
    - Module d'Young (E): 200 GPa
    - Coefficient de Poisson (nu): 0.3
    - Module de compressibilité (K): calculé selon la théorie élastique
    - Module de cisaillement (G): calculé selon la théorie élastique

Paramètres géométriques:
    - Rayon externe (Re): 600 mm
    - Rayon interne (Ri): 300 mm

Paramètres élasto-plastiques:
    - Contrainte d'écoulement (sig0): 300 MPa
    - Module tangent (Et): E/100
    - Module d'écrouissage (H): calculé selon la théorie de l'écrouissage
    - Pression limite élastique (q_lim): 2·ln(Re/Ri)·sig0

Remarque: Les valeurs sont définies pour un acier structural typique et peuvent
être modifiées selon les besoins de l'analyse.

Auteur: bouteillerp
Date de création: 28 Mai 2025
"""
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
