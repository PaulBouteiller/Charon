"""
Test de compression d'un cylindre en coordonn�es cylindriques 1D.

Ce script simule la compression d'un cylindre creux soumis � des pressions interne et externe,
puis compare la solution num�rique au champ de d�placement radial analytique.

Param�tres g�om�triques:
    - Rayon interne (Rint): 9
    - Rayon externe (Rext): Rint + e
    - Épaisseur (e): 2
    - Discr�tisation (Nx): 20 �l�ments radiaux

Chargement:
    - Pression interne (Pint): -5
    - Pression externe (Pext): -10

La solution analytique utilise les �quations de Lam� pour un cylindre � paroi �paisse.
Une assertion v�rifie que l'erreur relative entre les solutions est inf�rieure � 0.1%.
"""
from CharonX import CylindricalUD, Solve, create_1D_mesh
import pytest
from depouillement import validation_analytique
import sys
sys.path.append("../../")
from Generic_isotropic_material import Acier

###### Param�tre g�om�trique ######
e = 2
Nx = 20
Rint = 9
Rext = Rint + e

###### Chargement ######
Pext = -10
Pint = -5

###### Maillage ######
mesh = create_1D_mesh(Rint, Rext, Nx)

###### Param�tre du probl�me ######
dictionnaire = {"mesh" : mesh,
                "boundary_setup": 
                    {"tags": [1, 2],
                     "coordinate": ["x", "x"], 
                     "positions": [Rint, Rext]
                     },
                "loading_conditions": 
                    [{"type": "surfacique", "component" : "F", "tag": 1, "value" : -Pint},
                     {"type": "surfacique", "component" : "F", "tag": 2, "value" : Pext}
                    ],
                "analysis" : "static",
                "isotherm" : True
                }

pb = CylindricalUD(Acier, dictionnaire)

###### Param�tre de la r�solution ######
dictionnaire_solve = {
    "Prefix" : "Compression_cylindrique_1D",
    "csv_output" : {"U" : True}
    }

solve_instance = Solve(pb, dictionnaire_solve, compteur=1, npas=10)
solve_instance.solve()
validation_analytique(Pint, Pext, Rint, Rext)