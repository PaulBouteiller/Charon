"""
Setup file for the CharonX package

@author: Paul Bouteiller, CEA DAM/DIF
@email: paul.bouteiller@ecea.fr
"""

from setuptools import setup, find_packages
import os

# Création automatique du pyproject.toml s'il n'existe pas
if not os.path.exists('pyproject.toml'):
    with open('pyproject.toml', 'w') as f:
        f.write('[build-system]\nrequires = ["setuptools"]\nbuild-backend = "setuptools.build_meta"')


setup(name="Charon",
      description="Charon Python module.",
      version = '0.0.2',
      author="Bouteiller Paul",
      author_email="paul.bouteiller@cea.fr",
      packages = find_packages(),
)

