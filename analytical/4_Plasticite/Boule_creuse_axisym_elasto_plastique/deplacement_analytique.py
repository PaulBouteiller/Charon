import numpy as np
def deplacement_elastique(p_int, a, b, G, K, p_ext=0):
    E = 9 * K * G / (3 * K + G)
    nu = (3 * K - 2 * G) / (2 * (3 * K + G))
    ratio = b/a
    prefacteur = a / (ratio**3 - 1)
    parenthese = (1 - 2 * nu) + (1 + nu) * ratio**3 / 2
    return prefacteur * parenthese * p_int / E

def deplacement_ilyushin(p_int, a, b, G, K, sigma_y, lambda_param, p_ext=0):
    """
    Calcule le déplacement radial de la surface intérieure d'une sphère creuse
    selon la solution d'Ilyushin pour un état complètement plastique.
    Formule de l'article Vaziri (1992) équation (22):
    
    w_a = (ab³)/(4G(1-λ)(b³-a³)) × [P_a - P_b - 2ΨλσY ln(b/a)] + (a(a³P_a - b³P_b))/(3K(b³-a³))
    
    Paramètres:
    -----------
    p_int : float ou array (P_a dans l'article)
        Pression interne (Pa)
    a : float
        Rayon interne (m)
    b : float
        Rayon externe (m)
    G : float
        Module de cisaillement (Pa)
    K : float
        Module de compressibilité (Pa)
    sigma_y : float
        Contrainte d'écoulement (Pa)
    lambda_param : float
        Paramètre d'écrouissage
    p_ext : float (P_b dans l'article)
        Pression externe (Pa), par défaut 0
    
    Retourne:
    ---------
    w_a : float ou array
        Déplacement radial à r=a (m)
    """
    
    # Détermination du signe Psi selon l'équation (18) de Vaziri
    Psi = np.where(p_int - p_ext >= 0, 1.0, -1.0)
    
    # Premier terme : contribution plastique selon équation (22)
    terme1 = (a * b**3) / (4 * G * (1 - lambda_param) * (b**3 - a**3))
    terme1 *= (p_int - p_ext - 2 * Psi * lambda_param * sigma_y * np.log(b/a))
    
    # Deuxième terme : contribution élastique volumique
    terme2 = (a * (a**3 * p_int - b**3 * p_ext)) / (3 * K * (b**3 - a**3))
    
    return terme1 + terme2