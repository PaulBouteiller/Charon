"""
Created on Fri Oct 18 10:11:07 2024

@author: bouteillerp
"""
from CharonX import Material
###### Modèle matériau Acier ######
E = 210e3
nu = 0.3
rho = 7.8e-3
C = 500
mu = E / 2. / (1 + nu)
kappa = E / (3. * (1 - 2 * nu))
lmbda = E * nu / (1 - 2 * nu) / (1 + nu)
dico_eos = {"E" : E, "nu" : nu, "alpha" : 1}
dico_devia = {"mu" : mu}
eos_type = "IsotropicHPP"
devia_type = "IsotropicHPP"

#Paramètre élasto-plastique
sig0 = 300  # yield strength in MPa
Et = E / 100.0  # tangent modulus
H = E * Et / (E - Et)  # hardening modulus


Acier = Material(rho, C, eos_type, devia_type, dico_eos, dico_devia)

###### Modèle matériau Acier ######
ratio = 3
E_alu = E / ratio
nu_alu = nu
rho_alu = 2.7e-3
C = 500
mu_alu = E_alu / 2. / (1 + nu_alu)
dico_eos_alu = {"E" : E_alu, "nu" : nu_alu, "alpha" : 12e-6}
dico_devia_alu = {"E" : E_alu, "nu" : nu_alu}
eos_type_alu = "IsotropicHPP"
devia_type_alu = "IsotropicHPP"
Alu = Material(rho_alu, C, eos_type_alu, devia_type_alu, dico_eos_alu, dico_devia_alu)