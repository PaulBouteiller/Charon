# Guide de Développement de Lois Déviatoriques

Ce guide détaille comment créer et intégrer une nouvelle loi déviatorique dans le framework Charon.

## Vue d'ensemble

Une loi déviatorique définit la relation entre les contraintes déviatoriques (cisaillement) et les déformations d'un matériau.

## Structure du module deviator

```
deviator/
├── __init__.py              # Point d'entrée et classe Deviator principale
├── base_deviator.py         # Classe abstraite de base
├── isotropic_hpp.py         # Exemple : Élasticité linéaire isotrope
├── neo_hook.py              # Exemple : Hyperélasticité Neo-Hookéenne
├── hypoelastic.py           # Exemple : Formulation hypoélastique
├── votre_deviator.py        # Votre nouvelle loi déviatorique
└── README.md                # Ce guide
```

## Étape 1 : Créer votre classe déviatorique

### 1.1 Structure de base

Créez un nouveau fichier `votre_deviator.py` dans le dossier `deviator/` :

```python
# Copyright 2025 CEA
# ... (header de licence standard)

"""
Votre Loi Déviatorique
======================

Description de votre modèle déviatorique, ses applications et références.

Exemples d'applications :
- Matériaux hyperélastiques
- Élasticité anisotrope
- Viscoplasticité
- Modèles de fluides

Classes:
--------
VotreDeviator : Implémentation de votre loi déviatorique
"""

from .base_deviator import BaseDeviator
# Importez les modules UFL nécessaires
from ufl import sym, dev, tr, dot, Identity, grad  # exemples

class VotreDeviator(BaseDeviator):
    """Description de votre loi déviatorique.
    
    Expliquez ici :
    - Le type de comportement modélisé
    - Les hypothèses (petites/grandes déformations)
    - La formulation mathématique
    - Le domaine d'application
    
    Attributes
    ----------
    param1 : float
        Description du premier paramètre (ex: module de cisaillement)
    param2 : float
        Description du deuxième paramètre
    """
```

### 1.2 Méthodes obligatoires

Votre classe doit implémenter ces deux méthodes :

#### `required_parameters()`
```python
def required_parameters(self):
    """Retourne la liste des paramètres requis.
    
    Returns
    -------
    list
        Liste des noms de paramètres obligatoires
    """
    return ["mu"]  # Exemple : module de cisaillement
```

#### `calculate_stress(u, v, J, T, T0, kinematic)`
```python
def calculate_stress(self, u, v, J, T, T0, kinematic):
    """Calcule le tenseur des contraintes déviatoriques.
    
    Parameters
    ----------
    u : Function
        Champ de déplacement
    v : Function
        Champ de vitesse
    J : Function
        Jacobien de la transformation (déformation volumétrique)
    T : Function
        Température actuelle
    T0 : Function
        Température initiale
    kinematic : Kinematic
        Gestionnaire cinématique pour les opérations tensorielles
        
    Returns
    -------
    ufl.core.expr.Expr
        Tenseur des contraintes déviatoriques 3D
    """
    # Votre implémentation ici
    pass
```

## Étape 2 : Exemples par catégorie

### 2.1 Élasticité linéaire (petites déformations)

```python
class LinearElasticDeviator(BaseDeviator):
    """Élasticité linéaire isotrope : s = 2μ * dev(ε)"""
    
    def required_parameters(self):
        return ["mu"]  # Module de cisaillement
    
    def __init__(self, params):
        super().__init__(params)
        self.mu = params["mu"]
        print(f"Module de cisaillement: {self.mu}")
    
    def calculate_stress(self, u, v, J, T, T0, kinematic):
        # Tenseur des déformations linéarisé
        epsilon = sym(kinematic.grad_3D(u))
        
        # Contrainte déviatorique de Hooke
        return 2 * self.mu * dev(epsilon)
```

### 2.2 Hyperélasticité (grandes déformations)

```python
class CustomHyperelasticDeviator(BaseDeviator):
    """Hyperélasticité personnalisée basée sur B (left Cauchy-Green)"""
    
    def required_parameters(self):
        return ["mu", "alpha"]  # Paramètres du modèle
    
    def __init__(self, params):
        super().__init__(params)
        self.mu = params["mu"]
        self.alpha = params["alpha"]
    
    def calculate_stress(self, u, v, J, T, T0, kinematic):
        # Tenseur de Cauchy-Green gauche
        B = kinematic.B_3D(u)
        
        # Modèle hyperélastique personnalisé
        term1 = self.mu / J**(5./3) * dev(B)
        term2 = self.alpha / J**(7./3) * dev(dot(B, B))
        
        return term1 + term2
```
## Étape 3 : Support avancé

### 3.1 Énergie libre de Helmholtz (pour l'endommagement)

Pour les modèles d'endommagement par champ de phase, ajoutez :

```python
def isochoric_helmholtz_energy(self, u, kinematic):
    """Énergie libre de Helmholtz isochore.
    
    Parameters
    ----------
    u : Function
        Champ de déplacement
    kinematic : Kinematic
        Gestionnaire cinématique
        
    Returns
    -------
    Expression
        Énergie déviatorique
    """
    # Exemple pour l'élasticité linéaire
    dev_eps = dev(sym(kinematic.grad_3D(u)))
    return self.mu * tr(dot(dev_eps, dev_eps))
```
```

## Étape 4 : Intégration dans le framework

### 4.1 Mise à jour de `__init__.py`

Ajoutez votre déviateur dans `deviator/__init__.py` :

```python
# Ajoutez l'import
from .votre_deviator import VotreDeviator

# Ajoutez dans __all__
__all__ = [
    'BaseDeviator', 
    'Deviator', 
    'NoneDeviator',
    'IsotropicHPPDeviator', 
    'NeoHookDeviator',
    # ... autres déviateurs
    'VotreDeviator',  # Votre nouveau déviateur
]
```

### 4.2 Mise à jour de `material.py`

Dans `material.py`, ajoutez votre déviateur au dictionnaire :

```python
deviatoric_classes = {
    "IsotropicHPP": IsotropicHPPDeviator,
    "NeoHook": NeoHookDeviator,
    # ... autres mappings
    "VotreNom": VotreDeviator,  # Ajoutez cette ligne
}
```

## Étape 5 : Utilisation

Votre loi déviatorique peut maintenant être utilisée :

```python
from ConstitutiveLaw.material import Material

# Paramètres de votre déviateur
deviator_params = {
    "mu": 80e9,
    "param2": 0.3
}

# Paramètres EOS
eos_params = {"E": 210e9, "nu": 0.3, "alpha": 1e-5}

# Création du matériau
material = Material(
    rho_0=7800,
    C_mass=460,
    eos_type="IsotropicHPP",
    dev_type="VotreNom",      # Nom de votre déviateur
    eos_params=eos_params,
    deviator_params=deviator_params
)
```