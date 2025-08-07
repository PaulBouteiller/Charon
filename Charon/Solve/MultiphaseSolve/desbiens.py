import numpy as np
from math import tanh, exp, log
from typing import Tuple, Optional

class DesbiensTempReactiveModel:
    """
    Implémentation du modèle de combustion réactive basé sur la température
    de Nicolas Desbiens (2017) - "Modeling of the Jack Rabbit Series of Experiments 
    with a Temperature Based Reactive Burn Model"
    
    Équations principales (1-6) du modèle à 4 régimes:
    r = rI * SI(λ) + {rIG * W(Tshock) + rDG * [1-W(Tshock)]} * SG(λ) + rB * [1-SG(λ)]
    """
    
    def __init__(self, material_params: Optional[dict] = None):
        """
        Initialisation avec les paramètres du modèle pour LX-17
        Paramètres tirés du Tableau 1 de l'article
        """
        if material_params is None:
            # Paramètres par défaut pour LX-17 (Tableau 1)
            self.params = {
                # Températures de référence
                'Tadim': 1035.0,     # K - Température de dimensionnement  
                'Tall': 510.0,       # K - Température d'allumage
                'Tc': 1090.0,        # K - Température critique
                'T0': 293.0,         # K - Température de référence
                
                # Paramètres de vitesse de réaction
                'kI': 0.1e-6,        # μs^-1 - Constante d'initiation 
                'nI': 1.0,           # Exposant d'initiation
                'kIG': 6.8e-6,       # μs^-1 - Constante ignition-croissance
                'nIG': 1.5,          # Exposant ignition-croissance
                'kDG': 120.0e-6,     # μs^-1 - Constante diffusion-croissance
                'nDG': 0.5,          # Exposant diffusion-croissance
                'kB': 0.7e-6,        # μs^-1 - Constante de combustion
                'nB': 1.0,           # Exposant de combustion
                
                # Paramètres de transition
                'W1': 8.0,           # Paramètre de forme fonction W
                
                # Paramètres des fonctions de forme
                'SI1': 200.0,        # Paramètre SI
                'SI2': 0.025,        # Paramètre SI  
                'SG1': 40.0,         # Paramètre SG
                'SG2': 0.835,        # Paramètre SG
                
                # Propriétés matériau
                'rho0': 1895.0,      # kg/m³ - Densité initiale
            }
        else:
            self.params = material_params.copy()
    
    def switching_function_W(self, T_shock: float) -> float:
        """
        Fonction de commutation W(Tshock) - Équation (2)
        Transition entre régimes rIG et rDG
        """
        Tc = self.params['Tc']
        W1 = self.params['W1']
        
        arg = W1 * (T_shock / Tc - 1.0)
        return 0.5 * (1.0 - tanh(arg))
    
    def shape_function_SI(self, lambda_burn: float) -> float:
        """
        Fonction de forme SI(λ) pour le régime d'initiation
        Basée sur le concept de hot spots
        """
        SI1 = self.params['SI1'] 
        SI2 = self.params['SI2']
        
        if lambda_burn < SI2:
            return SI1 * lambda_burn
        else:
            return SI1 * SI2 * (2.0 - lambda_burn / SI2)
    
    def shape_function_SG(self, lambda_burn: float) -> float:
        """
        Fonction de forme SG(λ) pour les régimes de croissance
        Transition douce basée sur tanh
        """
        SG1 = self.params['SG1']
        SG2 = self.params['SG2'] 
        
        arg = SG1 * (lambda_burn - SG2)
        return 0.5 * (1.0 - tanh(arg))
    
    def rate_initiation(self, T_shock: float, lambda_burn: float) -> float:
        """
        Taux de réaction d'initiation rI - Équation (3)
        Dépend de la température de choc
        """
        kI = self.params['kI']
        nI = self.params['nI'] 
        Tall = self.params['Tall']
        
        if T_shock <= Tall:
            return 0.0
            
        temp_ratio = (T_shock - Tall) / Tall
        return kI * (temp_ratio ** nI) * ((1.0 - lambda_burn) ** (2.0/3.0))
    
    def rate_ignition_growth(self, T_shock: float, lambda_burn: float) -> float:
        """
        Taux de réaction ignition-croissance rIG - Équation (4)
        Mécanisme de nucléation et croissance
        """
        kIG = self.params['kIG']
        nIG = self.params['nIG']
        Tall = self.params['Tall']
        
        if T_shock <= Tall or lambda_burn >= 1.0:
            return 0.0
            
        temp_ratio = (T_shock - Tall) / Tall
        
        # Protection contre log(0)
        if lambda_burn >= 0.9999:
            log_term = 0.0
        else:
            log_term = -log(1.0 - lambda_burn)
            
        return kIG * (temp_ratio ** nIG) * (1.0 - lambda_burn) * (log_term ** (2.0/3.0))
    
    def rate_diffusion_growth(self, T: float, lambda_burn: float) -> float:
        """
        Taux de réaction diffusion-croissance rDG - Équation (5)  
        Dépend de la température locale
        """
        kDG = self.params['kDG']
        nDG = self.params['nDG']
        Tall = self.params['Tall']
        Tadim = self.params['Tadim']
        
        if T <= Tall or lambda_burn >= 1.0:
            return 0.0
            
        temp_ratio = (T - Tall) / Tadim
        
        return kDG * (temp_ratio ** nDG) * (lambda_burn ** (2.0/3.0)) * ((1.0 - lambda_burn) ** (2.0/3.0))
    
    def rate_burn(self, T: float, lambda_burn: float) -> float:
        """
        Taux de réaction de combustion rB - Équation (6)
        Phase finale de combustion
        """
        kB = self.params['kB']
        nB = self.params['nB']
        Tall = self.params['Tall']
        Tadim = self.params['Tadim']
        
        if T <= Tall or lambda_burn >= 1.0:
            return 0.0
            
        temp_ratio = (T - Tall) / Tadim
        
        return kB * (temp_ratio ** nB) * ((1.0 - lambda_burn) ** 0.5)
    
    def total_reaction_rate(self, T_shock: float, T: float, lambda_burn: float) -> float:
        """
        Taux de réaction total - Équation (1)
        Combinaison des 4 régimes avec fonctions de forme
        
        Args:
            T_shock: Température de choc (K)
            T: Température locale (K) 
            lambda_burn: Fraction brûlée (0 ≤ λ ≤ 1)
            
        Returns:
            Taux de réaction total (s^-1)
        """
        # Limitation de λ entre 0 et 1
        lambda_burn = max(0.0, min(1.0, lambda_burn))
        
        # Calcul des taux individuels
        rI = self.rate_initiation(T_shock, lambda_burn)
        rIG = self.rate_ignition_growth(T_shock, lambda_burn)
        rDG = self.rate_diffusion_growth(T, lambda_burn)
        rB = self.rate_burn(T, lambda_burn)
        
        # Fonctions de forme et de commutation
        SI = self.shape_function_SI(lambda_burn)
        SG = self.shape_function_SG(lambda_burn) 
        W = self.switching_function_W(T_shock)
        
        # Combinaison finale - Équation (1)
        rate_total = (rI * SI + 
                     (rIG * W + rDG * (1.0 - W)) * SG + 
                     rB * (1.0 - SG))
        
        return max(0.0, rate_total)  # Assurer une vitesse positive
    
    def integrate_reaction(self, T_shock: float, T: float, lambda_initial: float, 
                          dt: float, method: str = 'euler') -> float:
        """
        Intégration de l'évolution de la fraction brûlée
        
        Args:
            T_shock: Température de choc (K)
            T: Température locale (K)
            lambda_initial: Fraction brûlée initiale
            dt: Pas de temps (s)
            method: Méthode d'intégration ('euler' ou 'rk4')
            
        Returns:
            Nouvelle fraction brûlée
        """
        if method == 'euler':
            # Méthode d'Euler explicite
            rate = self.total_reaction_rate(T_shock, T, lambda_initial)
            lambda_new = lambda_initial + rate * dt
            
        elif method == 'rk4':
            # Méthode Runge-Kutta d'ordre 4
            k1 = self.total_reaction_rate(T_shock, T, lambda_initial)
            k2 = self.total_reaction_rate(T_shock, T, lambda_initial + 0.5 * dt * k1)
            k3 = self.total_reaction_rate(T_shock, T, lambda_initial + 0.5 * dt * k2)
            k4 = self.total_reaction_rate(T_shock, T, lambda_initial + dt * k3)
            
            lambda_new = lambda_initial + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        else:
            raise ValueError("Méthode non supportée. Utilisez 'euler' ou 'rk4'")
            
        return max(0.0, min(1.0, lambda_new))
    
    def get_regime_contributions(self, T_shock: float, T: float, lambda_burn: float) -> dict:
        """
        Diagnostic: contributions de chaque régime
        
        Returns:
            Dictionnaire avec les contributions individuelles
        """
        lambda_burn = max(0.0, min(1.0, lambda_burn))
        
        # Taux individuels
        rI = self.rate_initiation(T_shock, lambda_burn)
        rIG = self.rate_ignition_growth(T_shock, lambda_burn) 
        rDG = self.rate_diffusion_growth(T, lambda_burn)
        rB = self.rate_burn(T, lambda_burn)
        
        # Fonctions de forme
        SI = self.shape_function_SI(lambda_burn)
        SG = self.shape_function_SG(lambda_burn)
        W = self.switching_function_W(T_shock)
        
        # Contributions effectives
        contrib_I = rI * SI
        contrib_IG = rIG * W * SG  
        contrib_DG = rDG * (1.0 - W) * SG
        contrib_B = rB * (1.0 - SG)
        
        total = contrib_I + contrib_IG + contrib_DG + contrib_B
        
        return {
            'initiation': contrib_I,
            'ignition_growth': contrib_IG,
            'diffusion_growth': contrib_DG, 
            'burn': contrib_B,
            'total': total,
            'switching_W': W,
            'shape_SI': SI,
            'shape_SG': SG
        }


# Exemple d'utilisation et test du modèle
if __name__ == "__main__":
    
    # Création du modèle avec paramètres LX-17
    model = DesbiensTempReactiveModel()
    
    # Conditions d'exemple
    T_shock = 1200.0  # K - Température de choc
    T_local = 1100.0  # K - Température locale
    lambda_0 = 0.1    # Fraction brûlée initiale
    
    print("=== Test du modèle de Desbiens (2017) ===")
    print(f"Température de choc: {T_shock} K")
    print(f"Température locale: {T_local} K") 
    print(f"Fraction brûlée initiale: {lambda_0}")
    print()
    
    # Calcul du taux de réaction total
    rate_total = model.total_reaction_rate(T_shock, T_local, lambda_0)
    print(f"Taux de réaction total: {rate_total:.2e} s^-1")
    
    # Analyse des contributions
    contrib = model.get_regime_contributions(T_shock, T_local, lambda_0)
    print("\n=== Contributions par régime ===")
    print(f"Initiation (rI): {contrib['initiation']:.2e} s^-1")
    print(f"Ignition-Growth (rIG): {contrib['ignition_growth']:.2e} s^-1")
    print(f"Diffusion-Growth (rDG): {contrib['diffusion_growth']:.2e} s^-1")
    print(f"Burn (rB): {contrib['burn']:.2e} s^-1")
    print()
    print(f"Fonction de commutation W: {contrib['switching_W']:.3f}")
    print(f"Fonction de forme SI: {contrib['shape_SI']:.3f}")  
    print(f"Fonction de forme SG: {contrib['shape_SG']:.3f}")
    
    # Test d'intégration temporelle
    print("\n=== Évolution temporelle ===")
    dt = 1e-8  # 10 ns
    lambda_current = lambda_0
    
    for i in range(5):
        time = i * dt * 1e6  # Conversion en μs
        print(f"t = {time:.3f} μs, λ = {lambda_current:.4f}")
        lambda_current = model.integrate_reaction(T_shock, T_local, lambda_current, dt)