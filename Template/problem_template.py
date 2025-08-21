"""
Template Complet - Problèmes Mécaniques Charon
===============================================
Ce template présente toutes les options possibles pour configurer un problème
mécanique avec le code Charon utilisant FEniCSx.

Classes de problèmes disponibles :
- 1D : CartesianUD, CylindricalUD, SphericalUD
- 2D : PlaneStrain, Axisymmetric  
- 3D : Tridimensional

Auteur: Template généré pour Charon
"""

from Charon import (
    # Classes de problèmes
    CartesianUD, CylindricalUD, SphericalUD,  # 1D
    PlaneStrain, Axisymmetric,                # 2D  
    Tridimensional,                           # 3D
    # Utilitaires
    Material, LinearThermal, MeshManager, Solve, MyConstant
)
from ufl import SpatialCoordinate, conditional, And, lt, gt
import numpy as np

# =============================================================================
# PRÉREQUIS : MATÉRIAU ET MESH_MANAGER DÉJÀ DÉFINIS
# =============================================================================

# Supposons que vous avez déjà :
# - mat : Material ou [Material1, Material2, ...] pour multiphase
# - mesh_manager : MeshManager configuré avec votre maillage

#==============================================================================
# TEMPLATE DICTIONNAIRE_PROBLEM - TOUTES LES OPTIONS
#==============================================================================

dictionnaire_problem = {
    
    # =========================================================================
    # OBLIGATOIRE : MESH_MANAGER
    # =========================================================================
    "mesh_manager": mesh_manager,  # REQUIS : votre MeshManager configuré
    
    # =========================================================================
    # TYPE D'ANALYSE (Choisir UNE option)
    # =========================================================================
    
    # Option 1 : Analyse statique
    "analysis": "static",
    
    # Option 2 : Dynamique explicite (défaut)
    # "analysis": "explicit_dynamic",
    
    # Option 3 : Diffusion pure thermique
    # "analysis": "Pure_diffusion",
    
    # Option 4 : Pilotage utilisateur (déplacements imposés programmatiquement)
    # "analysis": "User_driven",
    
    # =========================================================================
    # COMPORTEMENT THERMIQUE (Choisir UNE option)
    # =========================================================================
    
    # Option 1 : Analyse isotherme (température constante)
    "isotherm": True,
    
    # Option 2 : Analyse adiabatique (pas d'échange thermique)
    # "adiabatic": True,
    
    # Option 3 : Analyse thermique couplée (nécessite Thermal_material)
    # "adiabatic": False,
    # "Thermal_material": thermal_material,  # Instance de LinearThermal
    
    # =========================================================================
    # CONDITIONS AUX LIMITES DE DÉPLACEMENT
    # =========================================================================
    "boundary_conditions": [
        
        # === CONDITIONS POUR PROBLÈMES 1D ===
        
        # Déplacement bloqué
        {"component": "U", "tag": 1},  # U = 0 sur région 1
        
        # Déplacement imposé constant
        {"component": "U", "tag": 2, "value": 0.001},
        
        # Déplacement imposé variable dans le temps
        {"component": "U", "tag": 3, "value": {"type": "rampe", "amplitude": 0.001}},
        {"component": "U", "tag": 4, "value": {"type": "creneau", "t_crit": 1e-3, "amplitude": 0.002}},
        {"component": "U", "tag": 5, "value": {"type": "chapeau", "t_crit": 2e-3, "amplitude": 0.001}},
        {"component": "U", "tag": 6, "value": {"type": "smooth_creneau", "t_load": 1e-3, "t_plateau": 2e-3, "amplitude": 0.001}},
        
        # === CONDITIONS POUR PROBLÈMES 2D PLAN (PlaneStrain) ===
        
        # Composantes X et Y
        {"component": "Ux", "tag": 1},  # Ux = 0 sur région 1
        {"component": "Uy", "tag": 2},  # Uy = 0 sur région 2
        {"component": "Ux", "tag": 3, "value": 0.001},  # Ux imposé
        {"component": "Uy", "tag": 4, "value": {"type": "rampe", "amplitude": 0.002}},
        
        # === CONDITIONS POUR PROBLÈMES 2D AXISYMÉTRIQUES ===
        
        # Composantes radiale et axiale
        {"component": "Ur", "tag": 1},  # Ur = 0 sur région 1 (axe de symétrie)
        {"component": "Uz", "tag": 2},  # Uz = 0 sur région 2
        {"component": "Ur", "tag": 3, "value": {"type": "rampe", "amplitude": 0.001}},
        
        # === CONDITIONS POUR PROBLÈMES 3D ===
        
        # Composantes X, Y, Z
        {"component": "Ux", "tag": 1},
        {"component": "Uy", "tag": 2},
        {"component": "Uz", "tag": 3},
        {"component": "Ux", "tag": 4, "value": {"type": "rampe", "amplitude": 0.001}},
        {"component": "Uy", "tag": 5, "value": 0.002},
        {"component": "Uz", "tag": 6, "value": {"type": "creneau", "t_crit": 1e-3, "amplitude": 0.003}},
        
        # === CONDITIONS SPÉCIALES ===
        
        # Encastrement complet (toutes composantes bloquées)
        {"component": "clamped", "tag": 7},  # Fonctionne pour toutes dimensions
        
        # Conditions axisymétriques (pour modèles 2D axi)
        {"component": "axi", "tag": 8},  # Condition sur l'axe de symétrie
        
    ],
    
    # =========================================================================
    # CONDITIONS DE CHARGEMENT
    # =========================================================================
    "loading_conditions": [
        
        # === FORCES SURFACIQUES (appliquées sur des frontières) ===
        
        # Forces par composante (1D)
        {"type": "surfacique", "component": "F", "tag": 10, "value": 1000},
        {"type": "surfacique", "component": "F", "tag": 11, "value": {"type": "rampe", "amplitude": 2000}},
        
        # Forces par composante (2D Plan)
        {"type": "surfacique", "component": "Fx", "tag": 12, "value": 1500},
        {"type": "surfacique", "component": "Fy", "tag": 13, "value": {"type": "creneau", "t_crit": 1e-3, "amplitude": 1000}},
        
        # Forces par composante (2D Axi)
        {"type": "surfacique", "component": "Fr", "tag": 14, "value": 800},
        {"type": "surfacique", "component": "Fz", "tag": 15, "value": {"type": "chapeau", "t_crit": 2e-3, "amplitude": 1200}},
        
        # Forces par composante (3D)
        {"type": "surfacique", "component": "Fx", "tag": 16, "value": 2000},
        {"type": "surfacique", "component": "Fy", "tag": 17, "value": 1800},
        {"type": "surfacique", "component": "Fz", "tag": 18, "value": {"type": "rampe", "amplitude": 2500}},
        
        # Pression (normale à la surface)
        {"type": "surfacique", "component": "pressure", "tag": 19, "value": 1e6},
        {"type": "surfacique", "component": "pressure", "tag": 20, "value": {"type": "rampe", "amplitude": 2e6}},
        
        # === FORCES VOLUMIQUES (appliquées dans tout le domaine) ===
        
        # Forces volumiques par composante
        {"type": "volumique", "component": "F", "tag": 0, "value": 1e3},      # 1D
        {"type": "volumique", "component": "Fx", "tag": 0, "value": 1e4},     # 2D/3D
        {"type": "volumique", "component": "Fy", "tag": 0, "value": 2e4},     # 2D/3D
        {"type": "volumique", "component": "Fz", "tag": 0, "value": 3e4},     # 3D
        {"type": "volumique", "component": "Fr", "tag": 0, "value": 1.5e4},   # 2D Axi
        
    ],
    
    # =========================================================================
    # ANALYSE MULTIPHASE
    # =========================================================================
    "multiphase": {
        
        # OBLIGATOIRE : Conditions définissant les régions de chaque phase
        "conditions": [
            # Exemples pour définir les phases spatiales
            # x[0] < 5.0,        # Phase 1 : x < 5
            # x[0] >= 5.0,       # Phase 2 : x >= 5
            # (x[0]**2 + x[1]**2) < 1.0,  # Phase circulaire
        ],
        
        # OPTIONNEL : Lois d'évolution pour transitions entre phases
        "evolution_laws": [
            # Loi d'Arrhenius
            {"type": "Arrhenius", "params": {
                "kin_pref": 1e6,       # Facteur pré-exponentiel [s⁻¹]
                "e_activation": 50000   # Énergie d'activation [J/mol]
            }},
            
            # Loi Forest Fire (simple)
            {"type": "ForestFire", "params": {
                "kin_pref": 1e5,
                "e_activation": 40000
            }},
            
            # Loi KJMA (nucléation-croissance)
            {"type": "KJMA", "params": {
                "melt_param": [1000, 0.5],      # [a, b] pour T_fusion = a * rho^b
                "gamma_param": 1e-3,            # Paramètre vitesse interface
                "alpha_param": [10, 0.01, -0.001],  # [a0, a1, a2] pour taux nucléation
                "tau_param": [5, 0.001, -0.0001]    # [t0, t1, t2] pour temps induction
            }},
            
            # Loi WGT (explosifs)
            {"type": "WGT", "params": {
                "SG1": 25.0,           # Paramètre forme 1
                "SG2": 0.92,           # Paramètre forme 2
                "TALL": 400.0,         # Température d'allumage [K]
                "TADIM": 1035.0,       # Température dimensionnante [K]
                "KDG": 1.6e7,          # Constante diffusion-croissance [s⁻¹]
                "NDG": 2.0,            # Exposant diffusion-croissance
                "KB": 2.8e5,           # Constante combustion [s⁻¹]
                "NB": 1.0              # Exposant combustion
            }},
            
            # Loi Desbiens (explosifs avancé)
            {"type": "Desbiens", "params": {
                "Tadim": 1035.0,       # Température dimensionnante [K]
                "Tall": 510.0,         # Température d'allumage [K]
                "Tc": 1090.0,          # Température critique [K]
                "T0": 293.0,           # Température référence [K]
                "kI": 0.1e-6,          # Constante initiation [μs⁻¹]
                "nI": 1.0,             # Exposant initiation
                "kIG": 6.8e-6,         # Constante croissance-allumage [μs⁻¹]
                "nIG": 1.5,            # Exposant croissance-allumage
                "kDG": 120.0e-6,       # Constante diffusion-croissance [μs⁻¹]
                "nDG": 0.5,            # Exposant diffusion-croissance
                "kB": 0.7e-6,          # Constante combustion [μs⁻¹]
                "nB": 1.0,             # Exposant combustion
                "W1": 8.0,             # Paramètre commutation
                "SI1": 200.0,          # Paramètre forme initiation 1
                "SI2": 0.025,          # Paramètre forme initiation 2
                "SG1": 40.0,           # Paramètre forme croissance 1
                "SG2": 0.835           # Paramètre forme croissance 2
            }},
            
            # Transition instantanée lisse
            {"type": "SmoothInstantaneous", "params": {
                "trigger_variable": "rho",     # "rho", "T", ou "P"
                "trigger_value": 1.2,          # Valeur critique
                "width": 0.1                   # Largeur transition
            }},
            
            # Aucune évolution pour cette phase
            None
        ],
        
        # OPTIONNEL : Libération d'énergie chimique par phase
        "chemical_energy_release": [
            None,           # Phase 1 : pas de libération
            5e6,            # Phase 2 : 5 MJ/kg libérés
            None,           # Phase 3 : pas de libération
            # Ou utiliser une liste pour chaque phase
        ]
    },
    
    # =========================================================================
    # PLASTICITÉ
    # =========================================================================
    "plasticity": {
        
        # === MODÈLES DISPONIBLES ===
        
        # Modèle HPP (Hypoelastic-Plastic, petites déformations)
        "model": "HPP_Plasticity",
        "sigY": 250e6,                    # Limite élastique [Pa]
        "Hardening": "Isotropic",         # "Isotropic", "LinearKinematic", "NonLinear"
        "Hardening_modulus": 1e9,         # Module d'écrouissage [Pa]
        
        # # Modèle déformations finies
        # "model": "Finite_Plasticity", 
        # "sigY": 250e6,
        # "Hardening": "Isotropic",
        # "Hardening_modulus": 1e9,
        
        # # Modèle J2 optimisé JAX
        # "model": "J2_JAX",
        # "sigY": 250e6,
        # "Hardening": "NonLinear",
        # "Hardening_func": lambda p: 250e6 + 100e6 * (1 - np.exp(-10*p)),  # Fonction Python
        
        # # Modèle Gurson pour matériaux poreux
        # "model": "JAX_Gurson",
        # "sigY": 250e6,
        # "q1": 1.5,                      # Paramètre Tvergaard
        # "q2": 1.0,                      # Paramètre Tvergaard
        # "fc": 0.15,                     # Porosité critique
        # "ff": 0.25,                     # Porosité à rupture
        # "fN": 0.04,                     # Fraction nucléation
        # "eN": 0.3,                      # Déformation nucléation
        # "sN": 0.1,                      # Écart-type nucléation
        
        # # Modèle Gurson simplifié
        # "model": "HPP_Gurson",
        # "sigY": 250e6,
        # "q1": 1.5,
        # "q2": 1.0,
        # "fc": 0.15,
        # "ff": 0.25,
        
    },
    
    # =========================================================================
    # ENDOMMAGEMENT
    # =========================================================================
    "damage": {
        
        # === MODÈLES PHASE FIELD ===
        
        # Modèle AT2 (Ambrosio-Tortorelli)
        "model": "PhaseField",
        "PF_model": "AT2",               # "AT1", "AT2", "wu"
        "Gc": 100,                       # Énergie de rupture critique [J/m²]
        "l0": 0.1,                       # Longueur de régularisation [m]
        
        # # Modèle Wu (cohésif)
        # "model": "PhaseField",
        # "PF_model": "wu",
        # "Gc": 100,
        # "l0": 0.05,
        # "sigma_c": 10e6,               # Contrainte critique [Pa]
        # "wu_softening": "exp",         # "exp", "lin", "bilin"
        # "E": 210e9,                    # Module d'Young [Pa]
        
        # === MODÈLES JOHNSON (POROSITÉ) ===
        
        # # Johnson statique
        # "model": "StaticJohnson",
        # "eta": 1e-6,                   # Viscosité [Pa·s]
        # "sigma_0": 1e6,                # Contrainte référence [Pa]
        # "f0": 0.001,                   # Porosité initiale
        # "regularization": False,        # Régularisation
        # "l0": 0.1,                     # Longueur régularisation (si régularisation=True)
        
        # # Johnson dynamique
        # "model": "DynamicJohnson",
        # "eta": 1e-6,
        # "sigma_0": 1e6,
        # "b": 1e-3,                     # Distance inter-pores [m]
        # "material": mat,               # Référence matériau
        
        # # Johnson inertiel
        # "model": "InertialJohnson",
        # "sigma_0": 1e6,
        # "b": 1e-3,
        # "material": mat,
        
    },
    
    # =========================================================================
    # AMORTISSEMENT (pour analyses dynamiques)
    # =========================================================================
    "damping": {
        "damping": True,                 # Activer amortissement
        "linear_coeff": 0.1,             # Coefficient amortissement linéaire
        "quad_coeff": 0.01,              # Coefficient amortissement quadratique
        "correction": True               # Correction des modes rigides
    },
    
}

# =============================================================================
# EXEMPLES D'UTILISATION PAR TYPE DE PROBLÈME
# =============================================================================

# ================================
# 1D CARTÉSIEN - Barre en traction
# ================================
dictionnaire_1d_cartesian = {
    "mesh_manager": mesh_manager,
    "analysis": "static",
    "isotherm": True,
    "boundary_conditions": [
        {"component": "U", "tag": 1},  # Extrémité gauche bloquée
        {"component": "U", "tag": 2, "value": {"type": "rampe", "amplitude": 0.001}}  # Traction droite
    ]
}
# pb_1d_cart = CartesianUD(mat, dictionnaire_1d_cartesian)

# ================================
# 1D CYLINDRIQUE - Tube sous pression
# ================================
dictionnaire_1d_cylindrical = {
    "mesh_manager": mesh_manager,
    "analysis": "explicit_dynamic",
    "isotherm": True,
    "loading_conditions": [
        {"type": "surfacique", "component": "F", "tag": 2, "value": {"type": "creneau", "t_crit": 1e-3, "amplitude": -1e6}}
    ]
}
# pb_1d_cyl = CylindricalUD(mat, dictionnaire_1d_cylindrical)

# ================================
# 1D SPHÉRIQUE - Sphère sous pression
# ================================
dictionnaire_1d_spherical = {
    "mesh_manager": mesh_manager,
    "analysis": "static",
    "isotherm": True,
    "loading_conditions": [
        {"type": "surfacique", "component": "F", "tag": 2, "value": -5e5}
    ],
    "plasticity": {
        "model": "HPP_Plasticity",
        "sigY": 300e6,
        "Hardening": "Isotropic",
        "Hardening_modulus": 2e9
    }
}
# pb_1d_sph = SphericalUD(mat, dictionnaire_1d_spherical)

# ================================
# 2D DÉFORMATION PLANE - Plaque trouée
# ================================
dictionnaire_2d_plane = {
    "mesh_manager": mesh_manager,
    "analysis": "static",
    "isotherm": True,
    "boundary_conditions": [
        {"component": "Ux", "tag": 1},  # Bord gauche bloqué en X
        {"component": "Uy", "tag": 3}   # Bord bas bloqué en Y
    ],
    "loading_conditions": [
        {"type": "surfacique", "component": "Fx", "tag": 2, "value": 1e6}  # Traction droite
    ],
    "damage": {
        "model": "PhaseField",
        "PF_model": "AT2",
        "Gc": 100,
        "l0": 0.05
    }
}
# pb_2d_plane = PlaneStrain(mat, dictionnaire_2d_plane)

# ================================
# 2D AXISYMÉTRIQUE - Sphère creuse
# ================================
dictionnaire_2d_axi = {
    "mesh_manager": mesh_manager,
    "analysis": "static",
    "isotherm": True,
    "boundary_conditions": [
        {"component": "Uz", "tag": 1},  # Axe de symétrie
        {"component": "Ur", "tag": 2}   # Axe de symétrie
    ],
    "loading_conditions": [
        {"type": "surfacique", "component": "pressure", "tag": 3, "value": 2e6}
    ]
}
# pb_2d_axi = Axisymmetric(mat, dictionnaire_2d_axi)

# ================================
# 3D - Cube en compression
# ================================
dictionnaire_3d = {
    "mesh_manager": mesh_manager,
    "analysis": "explicit_dynamic",
    "isotherm": True,
    "boundary_conditions": [
        {"component": "Ux", "tag": 1},  # Face X- bloquée
        {"component": "Uy", "tag": 3},  # Face Y- bloquée
        {"component": "Uz", "tag": 5}   # Face Z- bloquée
    ],
    "loading_conditions": [
        {"type": "surfacique", "component": "Fz", "tag": 6, "value": {"type": "rampe", "amplitude": -1e7}}
    ],
    "plasticity": {
        "model": "J2_JAX",
        "sigY": 400e6,
        "Hardening": "Isotropic",
        "Hardening_modulus": 5e9
    }
}
# pb_3d = Tridimensional(mat, dictionnaire_3d)

# ================================
# MULTIPHASE - Combustion explosive
# ================================
# Nécessite une liste de matériaux : [reactif, intermediaire, produit]
dictionnaire_multiphase = {
    "mesh_manager": mesh_manager,
    "analysis": "explicit_dynamic",
    "isotherm": False,
    "Thermal_material": thermal_material,
    "boundary_conditions": [
        {"component": "U", "tag": 1},
        {"component": "U", "tag": 2}
    ],
    "multiphase": {
        "conditions": [
            # x[0] <= x[0],  # Phase 1 partout initialement
            # x[0] < x[0],   # Phase 2 nulle part initialement
            # x[0] < x[0]    # Phase 3 nulle part initialement
        ],
        "evolution_laws": [
            {"type": "Arrhenius", "params": {"kin_pref": 1e8, "e_activation": 60000}},
            {"type": "Arrhenius", "params": {"kin_pref": 5e7, "e_activation": 40000}},
            None
        ],
        "chemical_energy_release": [None, None, 8e6]  # 8 MJ/kg libérés par la phase 3
    }
}
# pb_multiphase = CartesianUD([mat_reactif, mat_intermediaire, mat_produit], dictionnaire_multiphase)

# =============================================================================
# GUIDE DE RÉFÉRENCE DES PARAMÈTRES
# =============================================================================

"""
GUIDE DE RÉFÉRENCE :

TYPES D'ANALYSE :
- "static" : Analyse statique (quasi-statique)
- "explicit_dynamic" : Dynamique explicite (défaut)
- "Pure_diffusion" : Diffusion thermique seule
- "User_driven" : Contrôle programmatique des déplacements

CHARGEMENTS TEMPORELS :
- "rampe" : Croissance linéaire de 0 à amplitude
- "creneau" : Palier de 0 à t_crit, puis 0
- "chapeau" : Triangle de 0 à t_crit/2 à t_crit
- "smooth_creneau" : Trapèze avec montée/plateau/descente

COMPOSANTES PAR DIMENSION :
- 1D : "U" (déplacement), "F" (force)
- 2D Plan : "Ux", "Uy", "Fx", "Fy"
- 2D Axi : "Ur", "Uz", "Fr", "Fz", "pressure"
- 3D : "Ux", "Uy", "Uz", "Fx", "Fy", "Fz", "pressure"

MODÈLES DE PLASTICITÉ :
- "HPP_Plasticity" : Petites déformations classique
- "Finite_Plasticity" : Grandes déformations
- "J2_JAX" : Performance optimisée
- "JAX_Gurson" : Matériaux poreux avancé
- "HPP_Gurson" : Matériaux poreux simplifié

MODÈLES D'ENDOMMAGEMENT :
- "PhaseField" : Fissuration diffuse (AT1, AT2, wu)
- "StaticJohnson" : Porosité quasi-statique
- "DynamicJohnson" : Porosité avec effets visqueux
- "InertialJohnson" : Porosité avec effets inertiels

LOIS D'ÉVOLUTION MULTIPHASE :
- "Arrhenius" : Cinétique thermique
- "ForestFire" : Feu de forêt simplifié
- "KJMA" : Nucléation-croissance
- "WGT" : Combustion explosive
- "Desbiens" : Combustion explosive avancée
- "SmoothInstantaneous" : Transition instantanée

UNITÉS COHÉRENTES :
- Longueur : [m]
- Temps : [s]
- Force : [N]
- Contrainte : [Pa]
- Énergie : [J]
- Température : [K]
"""