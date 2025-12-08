"""
Birch-Murnaghan Third-Order Equation of State for Solids
"""

from ufl import sqrt
from .base_eos import BaseEOS

class BirchMurnaghanEOS(BaseEOS):
    """Birch-Murnaghan third-order equation of state.
    
    P = (3/2) * B0 * (η^(-7/3) - η^(-5/3)) * [1 + (3/4)(B0' - 4)(η^(-2/3) - 1)]
    
    où η = V/V0 = J
    
    Attributes
    ----------
    B0 : float 
        Bulk modulus at zero pressure (Pa)
    B0_prime : float 
        Pressure derivative of bulk modulus (dimensionless)
    """
    
    def required_parameters(self):
        return ["B0", "B0_prime"]
    
    def __init__(self, params):
        super().__init__(params)
        self.B0 = params["B0"]
        self.B0_prime = params["B0_prime"]
        
        print(f"Bulk modulus B0: {self.B0}")
        print(f"Pressure derivative B0': {self.B0_prime}")
    
    def celerity(self, rho_0):
        return sqrt(self.B0 / rho_0)
    
    def pressure(self, J, T, T0, material, quadrature):
        """
        P = (3/2) * B0 * (J^(-7/3) - J^(-5/3)) * [1 + (3/4)(B0' - 4)(J^(-2/3) - 1)]
        """
        eta_m7_3 = J**(-7./3)
        eta_m5_3 = J**(-5./3)
        eta_m2_3 = J**(-2./3)
        
        return 1.5 * self.B0 * (eta_m7_3 - eta_m5_3) * (1 + 0.75 * (self.B0_prime - 4) * (eta_m2_3 - 1))