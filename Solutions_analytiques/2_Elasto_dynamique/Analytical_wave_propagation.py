"""
Module de propagation analytique d'ondes en élastodynamique

Ce module fournit des fonctions pour calculer la propagation d'ondes élastiques
en coordonnées cartésiennes, cylindriques et sphériques. Il permet notamment de 
modéliser des ondes progressives et convergentes avec des atténuations géométriques
appropriées selon la géométrie considérée.

Fonctions principales:
    - creneau: Définit un créneau unitaire entre deux positions
    - cartesian1D_progressive_wave: Calcule une onde progressive en coordonnées cartésiennes
    - cylindrical1D_convergent_wave: Calcule une onde convergente en coordonnées cylindriques
    - spherical1D_convergent_wave: Calcule une onde convergente en coordonnées sphériques

Author: bouteillerp
Created on: Thu Nov 30 10:04:52 2023
"""
from numpy import array
from math import sqrt
eps = 1e-3
    
def creneau(y, gauche, droit):
    """
    Définit un créneau unitaire qui se trouve initialement entre les positions
    gauche et droit.
    
    Parameters
    ----------
    y : float
        Position d'évaluation du créneau.
    gauche : float
        Limite gauche du créneau.
    droit : float
        Limite droite du créneau.
        
    Returns
    -------
    float
        1.0 si y est dans l'intervalle [gauche, droit], 0.0 sinon.
    """
    if y <=droit and y>=gauche:
        return 1
    else:
        return 0
    
def cartesian1D_progressive_wave(amplitude, gauche, droit, wave_speed, position_list, time):
    """
    Calcule une onde progressive en coordonnées cartésiennes.
    
    L'onde se propage vers la droite à la vitesse wave_speed sans atténuation
    géométrique, en conservant sa forme originale.
    
    Parameters
    ----------
    amplitude : float
        Amplitude de l'onde.
    gauche : float
        Position initiale de la limite gauche du créneau.
    droit : float
        Position initiale de la limite droite du créneau.
    wave_speed : float
        Vitesse de propagation de l'onde.
    position_list : array-like
        Liste des positions où évaluer l'onde.
    time : float
        Temps auquel l'onde est évaluée.
        
    Returns
    -------
    numpy.ndarray
        Valeurs de l'onde aux positions demandées.
    """
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