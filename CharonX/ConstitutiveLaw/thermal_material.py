"""
Created on Tue Dec 20 17:36:38 2022

@author: bouteillerp
"""

class LinearThermal:
    """
    Défini un matériau possédant des caractéristiques thermique linéaire isotrope
    """
    def __init__(self, lmbda):
        """
        Parameters
        ----------
        lmbda : Float, coefficient de diffusion thermique.
        """
        self.lmbda = lmbda
        print("Le coefficient de diffusion est", self.lmbda)
        self.type = "LinearIsotropic"
        
class NonLinearThermal:
    """
    Défini un matériau possédant des caractéristiques thermique non linéaire isotrope
    """
    def __init__(self, lmbda, a1, a2):
        """
        Parameters
        ----------
        lmbda : Float, coefficient de diffusion thermique.
        """
        self.lmbda = lmbda
        self.a1 = a1
        self.a2 = a2
        print("Le coefficient de diffusion est", self.lmbda)
        print("Le coefficient de dépendance en température est", self.a1)
        print("Le coefficient de dépendance en pression est", self.a2)
        self.type = "NonLinearIsotropic"