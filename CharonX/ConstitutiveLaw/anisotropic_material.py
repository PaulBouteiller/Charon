"""
Created on Fri Mar 10 09:28:55 2024

@author: bouteillerp
"""
from scipy.linalg import block_diag
from math import cos, sin
from numpy import array, diag
from numpy.linalg import inv

class Orthotropic:
    """Rentourne la souplesse tridimensionnelle d'un matériau orthotrope"""
    def __init__(self, ET, EL, EN, nuLT, nuLN, nuTN, muLT, muLN, muTN):
        print("Le module d'Young longitudinal est", EL)
        print("Le module d'Young transverse est", ET)
        print("Le module d'Young normal est", EN)
        print("Le coefficient de Poisson nu_LT est", nuLT)
        print("Le coefficient de Poisson nu_LN est", nuLN)
        print("Le coefficient de Poisson nu_TN est", nuTN)
        print("Le module de cisaillement mu_LT est", muLT)
        print("Le module de cisaillement mu_LN est", muLN)
        print("Le module de cisaillement mu_TN est", muTN)

        Splan = array([[1. / EL, -nuLT / EL, -nuLN / EL],
                       [-nuLT / EL, 1. / ET, -nuTN / ET],
                       [-nuLN / EL, -nuTN / ET, 1. / EN]])
        S = block_diag(Splan, diag([1 / muLN, 1 / muLT, 1 / muTN]))
        self.C = inv(S)
        
    def rotate(self, C, alpha):
        """ Rotate elasticity matrix in the 1-2 plane by an angle alpha """
        c = cos(alpha)
        s = sin(alpha)
        R = array([[c**2, s**2, 0, 2*s*c, 0, 0],
                       [s**2, c**2, 0, -2*s*c, 0, 0],
                       [0, 0, 1, 0, 0, 0],
                       [-c*s, c*s, 0, c**2 - s**2, 0, 0],
                       [0, 0, 0, 0,  c, s],
                       [0, 0, 0, 0, -s, c]])
        return R.dot(C.dot(R.T))
        # self.S = R.T.dot(self.S.dot(R))
    
class TransverseIsotropic(Orthotropic):
    def __init__(self, ET, EL, nuT, nuL, muL):
        muT = ET / (2 * (1 + nuT))
        Orthotropic.__init__(self, ET, EL, ET, nuL, nuL, nuT, muL, muL, muT)

class Isotropic(TransverseIsotropic):
    def __init__(self, E, nu):
        mu = E / 2. / (1 + nu)
        Orthotropic.__init__(self, E, E, E, nu, nu, nu, mu, mu, mu)