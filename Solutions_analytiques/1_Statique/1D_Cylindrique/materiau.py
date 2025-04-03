"""
Définition des propriétés matérielles pour les tests de cylindre.

Ce module définit les constantes matérielles utilisées dans les tests de
compression de cylindre à paroi épaisse.

Propriétés du matériau:
    - Module d'Young (E): 210e3 MPa
    - Coefficient de Poisson (nu): 0.3
    
Ces propriétés sont utilisées pour les dictionnaires d'équation d'état (dico_eos)
et de comportement déviatorique (dico_devia) nécessaires à la définition du matériau.

Auteur: bouteillerp
Date de création: 2 Septembre 2024
"""
###### Modèle mécanique ######
E = 210e3
nu = 0.3
dico_eos = {"E" : E, "nu" : nu, "alpha" : 1}
dico_devia = {"E" : E, "nu" : nu}
# ###### Paramètre géométrique ######
# L = 2
# Nx = 20
# Rint = 9
# Rext = Rint + L

# ###### Chargement ######
# Pext = -10
# Pint = -5