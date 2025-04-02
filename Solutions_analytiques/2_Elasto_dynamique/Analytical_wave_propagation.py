"""
Created on Thu Nov 30 10:04:52 2023

@author: bouteillerp
"""

from numpy import array, linspace
from math import sqrt, pi, sin
import matplotlib.pyplot as plt
eps = 1e-3
    
def creneau(y, gauche, droit):
    """
    Défini un créneau unitaire qui se trouver initialement entre les points
    gauche et droit

    Parameters
    ----------
    y : float, position.
    gauche : Float, limite gauche du créneau.
    droit : Float, limite droite du créneau

    """
    if y <=droit and y>=gauche:
        return 1
    else:
        return 0
    
def cartesian1D_progressive_wave(amplitude, gauche, droit, wave_speed, position_list, time):
    return array([amplitude * creneau(x - wave_speed * time, gauche, droit) for x in position_list])

def cylindrical1D_convergent_wave(amplitude, gauche, droit, wave_speed, r_list, time):
    """
    Solution analytique pour une onde convergente en coordonnées cylindriques
    L'onde se propage de l'extérieur vers l'intérieur avec atténuation en 1/sqrt(r)
    """
    return array([amplitude * creneau(r + wave_speed * time, gauche, droit) / 
                 sqrt(max(r, eps)) for r in r_list])

def spherical1D_convergent_wave(amplitude, gauche, droit, wave_speed, r_list, time):
    """
    Solution analytique pour une onde convergente en coordonnées sphériques
    L'onde se propage de l'extérieur vers l'intérieur avec atténuation en 1/r
    """
    return array([amplitude * creneau(r + wave_speed * time, gauche, droit) / 
                 max(r**2, eps) for r in r_list])