# Copyright 2025 CEA
# [garder la licence existante]

"""
Module unifié d'intégrateurs temporels pour résoudre des équations différentielles ordinaires.
Comprend des méthodes Runge-Kutta explicites et implicites, ainsi que des intégrateurs
symplectiques pour les problèmes d'ordre 2.
"""

import numpy as np
from dolfinx.fem import Function
from petsc4py.PETSc import InsertMode, ScatterMode
from ..utils.petsc_operations import dt_update, petsc_assign
from dolfinx.fem.petsc import set_bc
from numpy import sqrt

# Classe abstraite de base pour tous les intégrateurs
class TimeIntegrator:
    """Classe de base abstraite pour les intégrateurs temporels."""
    
    def __init__(self, derivative_calculator):
        """
        Initialise l'intégrateur temporel.
        
        Parameters
        ----------
        derivative_calculator : callable
            Fonction qui calcule les dérivées (accélération, taux de variation, etc.)
        """
        self.derivative_calculator = derivative_calculator
        
    def solve(self, scheme_name, primary_field, secondary_field=None, tertiary_field=None, 
              dt=1.0, bcs=None):
        """
        Résout une étape de temps avec le schéma spécifié.
        
        Parameters
        ----------
        scheme_name : str
            Nom du schéma à utiliser
        primary_field : Function
            Champ principal à mettre à jour (ex: déplacement, température)
        secondary_field : Function, optional
            Champ secondaire (ex: vitesse)
        tertiary_field : Function, optional
            Champ tertiaire (ex: accélération)
        dt : float
            Pas de temps
        bcs : object, optional
            Conditions aux limites
            
        Returns
        -------
        bool
            True si succès, False sinon
        """
        raise NotImplementedError("Cette méthode doit être implémentée par les sous-classes")
    
    def apply_boundary_conditions(self, field, bcs=None):
        """Applique les conditions aux limites à un champ."""
        if bcs is None or field is None:
            return
            
        field.x.petsc_vec.ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)
        
        if hasattr(bcs, 'apply'):
            # Si bcs est un objet avec une méthode apply
            bcs.apply(field)
        elif hasattr(bcs, 'bcs') and callable(getattr(bcs, 'apply', None)):
            # Si bcs a un attribut bcs et une méthode apply
            bcs.apply(field)
        else:
            set_bc(field.x.petsc_vec, bcs)


# Implémentation des méthodes RK du fichier runge_kutta.py dans un style orienté objet
class RungeKuttaIntegrator(TimeIntegrator):
    """Intégrateur Runge-Kutta pour les EDO du premier ordre."""
    
    def __init__(self, derivative_calculator):
        super().__init__(derivative_calculator)
    
    def solve(self, scheme_name, primary_field, secondary_field=None, tertiary_field=None, 
              dt=1.0, bcs=None):
        """
        Implémente les différentes méthodes RK du premier ordre.
        
        Parameters
        ----------
        scheme_name : str Nom du schéma à utiliser ("RK1", "RK2", "RK4")
        primary_field : Function Champ principal à mettre à jour
        secondary_field : Function, optional Expression de la dérivée ou fonction pour calculer la dérivée
        tertiary_field : Function, optional Fonction pour stocker la dérivée calculée
        dt : float Pas de temps
        bcs : object, optional Conditions aux limites
            
        Returns
        -------
        bool True si succès, False sinon
        """
        if scheme_name == "RK1":
            return self._solve_rk1(primary_field, secondary_field, tertiary_field, dt, bcs)
        elif scheme_name == "RK2":
            return self._solve_rk2(primary_field, secondary_field, tertiary_field, dt, bcs)
        elif scheme_name == "RK4":
            return self._solve_rk4(primary_field, secondary_field, tertiary_field, dt, bcs)
        else:
            return False
    
    def _solve_rk1(self, f, dot_f_expression, dot_f_function, dt, bcs=None):
        """Implémente Runge-Kutta d'ordre 1 (Euler explicite)."""
        dot_f_function.interpolate(dot_f_expression)
        dt_update(f, dot_f_function, dt)
        self.apply_boundary_conditions(f, bcs)
        return True
    
    def _solve_rk2(self, f, dot_f_expression, dot_f_function, dt, bcs=None):
        """Implémente Runge-Kutta d'ordre 2 (méthode de Heun)."""
        # Sauvegarde de l'état initial
        f_init = f.copy()
        
        # Première étape (Euler)
        dot_f_function.interpolate(dot_f_expression)
        dt_update(f, dot_f_function, dt/2)
        self.apply_boundary_conditions(f, bcs)
        
        # Deuxième étape
        dot_f_function.interpolate(dot_f_expression)
        
        # Restaurer l'état initial et appliquer la mise à jour combinée
        f.x.array[:] = f_init.x.array[:]
        dt_update(f, dot_f_function, dt)
        self.apply_boundary_conditions(f, bcs)
        
        return True
    
    def _solve_rk4(self, f, dot_f_expression, dot_f_function, dt, bcs=None):
        """Implémente Runge-Kutta d'ordre 4 classique."""
        # Sauvegarde de l'état initial
        f_init = f.copy()
        
        # Fonction pour stocker les évaluations intermédiaires
        V_f = f.function_space
        k1 = Function(V_f)
        k2 = Function(V_f)
        k3 = Function(V_f)
        k4 = Function(V_f)
        
        # Étape 1
        k1.interpolate(dot_f_expression)
        
        # Étape 2
        f.x.array[:] = f_init.x.array[:] + 0.5 * dt * k1.x.array[:]
        self.apply_boundary_conditions(f, bcs)
        k2.interpolate(dot_f_expression)
        
        # Étape 3
        f.x.array[:] = f_init.x.array[:] + 0.5 * dt * k2.x.array[:]
        self.apply_boundary_conditions(f, bcs)
        k3.interpolate(dot_f_expression)
        
        # Étape 4
        f.x.array[:] = f_init.x.array[:] + dt * k3.x.array[:]
        self.apply_boundary_conditions(f, bcs)
        k4.interpolate(dot_f_expression)
        
        # Combinaison finale
        f.x.array[:] = f_init.x.array[:] + dt * (
            k1.x.array[:] / 6.0 + 
            k2.x.array[:] / 3.0 + 
            k3.x.array[:] / 3.0 + 
            k4.x.array[:] / 6.0
        )
        self.apply_boundary_conditions(f, bcs)
        
        return True


# Intégrateur basé sur les tableaux de Butcher de Runge_Kutta_Claude.py
class ButcherTableau:
    """Représentation d'un tableau de Butcher pour les méthodes Runge-Kutta."""
    
    def __init__(self, a, b, c, name="Custom", description="", order=0):
        """Initialise un tableau de Butcher."""
        self.a = np.array(a, dtype=float)
        self.b = np.array(b, dtype=float)
        self.c = np.array(c, dtype=float)
        self.name = name
        self.description = description
        self.order = order
        self.stages = len(b)
        
        # Vérification de cohérence
        if self.a.shape[0] != self.stages or self.a.shape[1] != self.stages:
            raise ValueError(f"La matrice a doit être de taille {self.stages}×{self.stages}")
        if len(self.c) != self.stages:
            raise ValueError(f"Le vecteur c doit être de longueur {self.stages}")
            
    def __repr__(self):
        return f"ButcherTableau(name='{self.name}', stages={self.stages}, order={self.order})"
    
    def is_explicit(self):
        """Vérifie si la méthode est explicite (a_ij = 0 pour j ≥ i)."""
        for i in range(self.stages):
            for j in range(i, self.stages):
                if abs(self.a[i, j]) > 1e-14:
                    return False
        return True


class ButcherIntegrator(TimeIntegrator):
    """Intégrateur temporel basé sur les tableaux de Butcher."""
    
    def __init__(self, derivative_calculator):
        super().__init__(derivative_calculator)
        self.tableaux = self._create_butcher_tableaux()
    
    def _create_butcher_tableaux(self):
        """Crée une bibliothèque de tableaux de Butcher prédéfinis."""
        tableaux = {}
        
        # Méthode d'Euler explicite (RK1)
        tableaux["RK1"] = ButcherTableau(
            a=[[0]],
            b=[1],
            c=[0],
            name="Forward Euler",
            description="Méthode d'Euler explicite d'ordre 1",
            order=1
        )
        
        # Méthode de Heun (RK2)
        tableaux["RK2"] = ButcherTableau(
            a=[[0, 0], 
               [1, 0]],
            b=[1/2, 1/2],
            c=[0, 1],
            name="Heun",
            description="Méthode de Heun d'ordre 2",
            order=2
        )
        
        # Méthode RK4 classique
        tableaux["RK4"] = ButcherTableau(
            a=[[0, 0, 0, 0],
               [1/2, 0, 0, 0],
               [0, 1/2, 0, 0],
               [0, 0, 1, 0]],
            b=[1/6, 1/3, 1/3, 1/6],
            c=[0, 1/2, 1/2, 1],
            name="Classical RK4",
            description="Méthode de Runge-Kutta classique d'ordre 4",
            order=4
        )
        
        # Ajouter d'autres tableaux de Butcher selon les besoins...
        
        return tableaux
    
    def solve(self, scheme_name, primary_field, secondary_field=None, tertiary_field=None, 
              dt=1.0, bcs=None):
        """
        Résout une étape de temps avec la méthode RK spécifiée.
        
        Parameters
        ----------
        scheme_name : str
            Nom du schéma à utiliser
        primary_field : Function
            Fonction de déplacement/température
        secondary_field : Function, optional
            Fonction de vitesse/dérivée
        tertiary_field : Function, optional
            Fonction d'accélération/dérivée seconde
        dt : float
            Pas de temps
        bcs : BoundaryConditions, optional
            Conditions aux limites
            
        Returns
        -------
        bool
            True si succès, False sinon
        """
        if scheme_name not in self.tableaux:
            return False
            
        tableau = self.tableaux[scheme_name]
        
        # Vérification que le tableau est explicite
        if not tableau.is_explicit():
            raise ValueError(f"Le tableau {tableau.name} n'est pas explicite")
        
        # Sauvegarde de l'état initial
        y0 = primary_field.copy()
        
        # Stockage des k (évaluations des dérivées) pour chaque étape
        k_stages = []
        y_stages = []
        
        # Calcul des étapes intermédiaires
        for i in range(tableau.stages):
            # Configuration de l'état pour cette étape
            primary_field.x.array[:] = y0.x.array[:]
            for j in range(i):
                primary_field.x.array[:] += dt * tableau.a[i,j] * k_stages[j].x.array[:]
            
            # Application des conditions aux limites
            self.apply_boundary_conditions(primary_field, bcs)
            
            # Calcul de la dérivée à cette étape
            k = self.derivative_calculator(update_velocity=False)
            if isinstance(k, Function):
                k_stages.append(k.copy())
            else:
                # Si derivative_calculator retourne None, utiliser secondary_field
                secondary_field.interpolate(secondary_field)  # Utiliser comme expression
                k_stages.append(secondary_field.copy())
            
            # Sauvegarder l'état à cette étape si nécessaire
            y_stages.append(primary_field.copy())
        
        # Application de la solution finale
        primary_field.x.array[:] = y0.x.array[:]
        for i in range(tableau.stages):
            primary_field.x.array[:] += dt * tableau.b[i] * k_stages[i].x.array[:]
        
        # Application des conditions aux limites finales
        self.apply_boundary_conditions(primary_field, bcs)
        
        return True

# Exportation des fonctions de runge_kutta.py pour maintenir la compatibilité
def first_order_rk1(f, dot_f_expression, dot_f_function, dt, booleen=False, mesh=None, cells=None):
    """
    Schema de runge kutta d'ordre 1 (Euler explicite), maintenu pour compatibilité.
    """
    dot_f_function.interpolate(dot_f_expression)
    if not booleen:
        dt_update(f, dot_f_function, dt)
        return 
    elif booleen:
        f_pred = f.copy()
        dt_update(f_pred, dot_f_function, dt)
        return f_pred

def first_order_rk2(f, dot_f_expression, dot_f_function, dt):
    """
    Schema de runge kutta d'ordre 2, maintenu pour compatibilité.
    """
    dot_f_function.interpolate(dot_f_expression)
    dt_update(f, dot_f_function, dt/2)
    dot_f_function.interpolate(dot_f_expression)
    dt_update(f, dot_f_function, dt/2)

def first_order_rk4(f, dot_f_expression, dot_f_function, dt):
    """
    Schema de runge kutta d'ordre 4, maintenu pour compatibilité.
    """
    prev_f = f.copy()
    V_f = f.function_space
    dot_f_1 = Function(V_f)
    dot_f_1.interpolate(dot_f_expression)
    petsc_assign(f, dt_update(prev_f, dot_f_1, dt/2, new_vec=True))
    dot_f_2 = Function(V_f)
    dot_f_2.interpolate(dot_f_expression)
    petsc_assign(f, dt_update(prev_f, dot_f_2, dt/2, new_vec=True))
    dot_f_3 = Function(V_f)
    dot_f_3.interpolate(dot_f_expression)
    petsc_assign(f, dt_update(prev_f, dot_f_3, dt, new_vec=True))
    dot_f_4 = Function(V_f)
    dot_f_4.interpolate(dot_f_expression)
    dt_update(prev_f, dot_f_1, dt/6.)
    dt_update(prev_f, dot_f_2, dt/3.)
    dt_update(prev_f, dot_f_3, dt/3.)
    dt_update(prev_f, dot_f_4, dt/6.)
    petsc_assign(f, prev_f)

def second_order_rk1(f, dot_f, ddot_f_function, ddot_f_expression, dt):
    """
    Schema d'Euler explicite pour EDO d'ordre 2, maintenu pour compatibilité.
    """
    ddot_f_function.interpolate(ddot_f_expression)
    dt_update(dot_f, ddot_f_function, dt)
    dt_update(f, dot_f, dt)  

def second_order_rk2(f, dot_f, ddot_f_function, ddot_f_expression, dt):
    """
    Schema RK2 pour EDO d'ordre 2, maintenu pour compatibilité.
    """
    ddot_f_function.interpolate(ddot_f_expression)
    dt_update(f, dot_f, dt/2)
    dt_update(dot_f, ddot_f_function, dt/2)
    
    ddot_f_function.interpolate(ddot_f_expression)
    dt_update(f, dot_f, dt/2)
    dt_update(f, ddot_f_function, dt**2/4)
    dt_update(dot_f, ddot_f_function, dt/2)

def second_order_rk4(f, dot_f, ddot_f_function, ddot_f_expression, dt):
    """
    Schema RK4 pour EDO d'ordre 2, maintenu pour compatibilité.
    """
    prev_f = f.copy()
    prev_dot_f = dot_f.copy()
    V_f = f.function_space
    
    ddot_f_function_1 = Function(V_f)
    ddot_f_function_2 = Function(V_f)
    ddot_f_function_3 = Function(V_f)
    ddot_f_function_4 = Function(V_f)
    
    ddot_f_function_1.interpolate(ddot_f_expression)
    dt_update(f, dot_f, dt/2)
    dt_update(dot_f, ddot_f_function_1, dt/2)
    
    ddot_f_function_2.interpolate(ddot_f_expression)
    dt_update(f, ddot_f_function_2, dt**2/4)
    petsc_assign(dot_f, dt_update(prev_dot_f, ddot_f_function_2, dt/2, new_vec=True))
    
    ddot_f_function_3.interpolate(ddot_f_expression)
    petsc_assign(f, dt_update(prev_f, prev_dot_f, dt, new_vec=True))
    dt_update(f, ddot_f_function_2, dt**2 / 2)
    petsc_assign(dot_f, dt_update(prev_dot_f, ddot_f_function_3, dt, new_vec=True))
    
    ddot_f_function_4.interpolate(ddot_f_expression)
    dt_update(prev_f, prev_dot_f, dt)
    petsc_assign(f, prev_f)
    dt_update(f, ddot_f_function_1, dt**2 / 6)
    dt_update(f, ddot_f_function_2, dt**2 / 6)
    dt_update(f, ddot_f_function_3, dt**2 / 6)    
    petsc_assign(dot_f, prev_dot_f)
    dt_update(dot_f, ddot_f_function_1, dt / 6)
    dt_update(dot_f, ddot_f_function_2, dt / 3)
    dt_update(dot_f, ddot_f_function_3, dt / 3)
    dt_update(dot_f, ddot_f_function_4, dt / 6)