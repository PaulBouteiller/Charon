# Guide de Développement d'Équations d'État (EOS)

Ce guide détaille comment créer et intégrer une nouvelle équation d'état dans le framework Charon.

## Vue d'ensemble

Une équation d'état (EOS) établit la relation entre la pression, la déformation volumétrique et la température d'un matériau. Dans notre framework, chaque EOS hérite de la classe abstraite `BaseEOS` et doit implémenter certaines méthodes obligatoires.

## Structure du module EOS

```
eos/
├── __init__.py           # Point d'entrée et classe EOS principale
├── base_eos.py          # Classe abstraite de base
├── isotropic_hpp.py     # Exemple : EOS élastique linéaire
├── jwl.py               # Exemple : EOS pour explosifs
├── votre_eos.py         # Votre nouvelle EOS
└── README.md            # Ce guide
```

## Étape 1 : Créer votre classe EOS

### 1.1 Structure de base

Créez un nouveau fichier `votre_eos.py` dans le dossier `eos/` :

```python
# Copyright 2025 CEA
# ... (header de licence standard)

"""
Votre Équation d'État
====================

Description de votre modèle EOS, ses applications et références.

Classes:
--------
VotreEOS : Implémentation de votre équation d'état
"""

from .base_eos import BaseEOS
# Importez les modules UFL nécessaires
from ufl import exp, ln, sqrt  # exemples

class VotreEOS(BaseEOS):
    """Description de votre équation d'état.
    
    Expliquez ici :
    - Le domaine d'application
    - Les hypothèses du modèle
    - La forme mathématique
    
    Attributes
    ----------
    param1 : float
        Description du premier paramètre
    param2 : float
        Description du deuxième paramètre
    """
```

### 1.2 Méthodes obligatoires

Votre classe doit implémenter ces trois méthodes :

#### `required_parameters()`
```python
def required_parameters(self):
    """Retourne la liste des paramètres requis.
    
    Returns
    -------
    list
        Liste des noms de paramètres obligatoires
    """
    return ["param1", "param2", "param3"]
```

#### `__init__(params)`
```python
def __init__(self, params):
    """Initialise l'équation d'état.
    
    Parameters
    ----------
    params : dict
        Dictionnaire contenant les paramètres du modèle
    """
    super().__init__(params)  # Validation automatique des paramètres
    
    # Stockage des paramètres
    self.param1 = params["param1"]
    self.param2 = params["param2"]
    self.param3 = params["param3"]
    
    # Affichage pour débogage
    print(f"Paramètre 1: {self.param1}")
    print(f"Paramètre 2: {self.param2}")
```

#### `celerity(rho_0)`
```python
def celerity(self, rho_0):
    """Calcule la vitesse de propagation des ondes.
    
    Parameters
    ----------
    rho_0 : float
        Densité initiale (kg/m³)
        
    Returns
    -------
    float
        Vitesse des ondes (m/s)
    """
    # Exemple : vitesse du son dans un fluide
    return sqrt(self.bulk_modulus / rho_0)
```

#### `pressure(J, T, T0, material, quadrature)`
```python
def pressure(self, J, T, T0, material, quadrature):
    """Calcule la pression selon votre EOS.
    
    Parameters
    ----------
    J : Function
        Jacobien de la transformation (déformation volumétrique)
    T : Function
        Température actuelle
    T0 : Function
        Température initiale
    material : Material
        Propriétés du matériau
    quadrature : QuadratureHandler
        Gestionnaire de quadrature
        
    Returns
    -------
    Expression
        Pression calculée
    """
    # Exemple d'implémentation
    return self.param1 * (J - 1) + self.param2 * (T - T0)
```

## Étape 2 : Exemples concrets

### 2.1 EOS simple : Loi linéaire

```python
class LinearEOS(BaseEOS):
    """EOS linéaire simple : P = K * (J - 1)"""
    
    def required_parameters(self):
        return ["K"]  # Module de compressibilité
    
    def __init__(self, params):
        super().__init__(params)
        self.K = params["K"]
        print(f"Module de compressibilité: {self.K}")
    
    def celerity(self, rho_0):
        return sqrt(self.K / rho_0)
    
    def pressure(self, J, T, T0, material, quadrature):
        return -self.K * (J - 1)
```

### 2.2 EOS avec température : Gaz parfait

```python
class IdealGasEOS(BaseEOS):
    """Gaz parfait : P = (γ-1) * ρ * e"""
    
    def required_parameters(self):
        return ["gamma"]
    
    def __init__(self, params):
        super().__init__(params)
        self.gamma = params["gamma"]
    
    def celerity(self, rho_0):
        # Estimation pour un gaz
        return sqrt(self.gamma * 287 * 300)  # R*T typique
    
    def pressure(self, J, T, T0, material, quadrature):
        return (self.gamma - 1) * material.rho_0 / J * material.C_mass * T
```

### 2.3 EOS complexe : Avec fonctions UFL

```python
from ufl import exp, ln

class ExponentialEOS(BaseEOS):
    """EOS exponentielle : P = A * exp(B * η) où η = 1/J - 1"""
    
    def required_parameters(self):
        return ["A", "B"]
    
    def __init__(self, params):
        super().__init__(params)
        self.A = params["A"]
        self.B = params["B"]
    
    def celerity(self, rho_0):
        return sqrt(self.A * self.B / rho_0)
    
    def pressure(self, J, T, T0, material, quadrature):
        eta = 1/J - 1  # Déformation de compression
        return self.A * exp(self.B * eta)
```

## Étape 3 : Support de l'énergie libre de Helmholtz (optionnel)

Pour les modèles d'endommagement par champ de phase, ajoutez :

```python
def volumetric_helmholtz_energy(self, u, J, kinematic):
    """Énergie libre de Helmholtz volumétrique.
    
    Parameters
    ----------
    u : Function
        Champ de déplacement
    J : Function
        Jacobien
    kinematic : Kinematic
        Gestionnaire cinématique
        
    Returns
    -------
    Expression
        Énergie volumétrique
    """
    # Exemple pour une EOS hyperélastique
    return self.K * (J * ln(J) - J + 1)
```

## Étape 4 : Intégration dans le framework

### 4.1 Mise à jour de `__init__.py`

Ajoutez votre EOS dans `eos/__init__.py` :

```python
# Ajoutez l'import
from .votre_eos import VotreEOS

# Ajoutez dans __all__
__all__ = [
    'BaseEOS',
    'EOS',
    'IsotropicHPPEOS',
    # ... autres EOS existantes
    'VotreEOS',  # Votre nouvelle EOS
]
```

### 4.2 Mise à jour de `material.py`

Dans `material.py`, ajoutez votre EOS au dictionnaire :

```python
eos_classes = {
    "IsotropicHPP": IsotropicHPPEOS,
    "U1": UEOS, 
    # ... autres mappings
    "VotreNom": VotreEOS,  # Ajoutez cette ligne
}
```

## Étape 5 : Utilisation

Votre EOS peut maintenant être utilisée dans la définition d'un matériau :

```python
from ConstitutiveLaw.material import Material

# Définition des paramètres
eos_params = {
    "param1": 1000,
    "param2": 0.3,
    "param3": 2.5
}

deviator_params = {"mu": 80e9}

# Création du matériau
material = Material(
    rho_0=7800,           # Densité initiale
    C_mass=460,           # Capacité thermique
    eos_type="VotreNom",  # Nom de votre EOS
    dev_type="IsotropicHPP",
    eos_params=eos_params,
    deviator_params=deviator_params
)
```