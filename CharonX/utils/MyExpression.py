"""
Created on Thu Jul 21 09:52:08 2022

@author: bouteillerp
"""
from dolfinx.fem import Constant
from petsc4py.PETSc import ScalarType
from scipy import interpolate

def interpolation_lin(temps_originaux, valeurs_originales, nouveaux_temps):
    """
    Fonction interpolant 

    Parameters
    ----------
    temps_originaux : array, Liste des temps d'origine.
    valeurs_originales : array, Valeurs aux temps originaux.
    nouveaux_temps : array, Liste de nouveaux temps auxquels on souhaite 
                            interpoler les valeurs_originales
    """
    f = interpolate.interp1d(temps_originaux, valeurs_originales)
    return f(nouveaux_temps)

class MyConstant:
    def __init__(self, mesh, *args, **kwargs):
        if kwargs.get("Type") == "Creneau":
            self.Expression = self.Creneau(mesh, *args)
        elif kwargs.get("Type") == "SmoothCreneau":
            self.Expression = self.SmoothCreneau(mesh, *args)
        elif kwargs.get("Type") == "Chapeau":
            self.Expression = self.Chapeau(mesh, *args)
        elif kwargs.get("Type") == "Rampe":
            self.Expression = self.Rampe(mesh, *args)
        elif kwargs.get("Type") == "UserDefined":
            self.Expression = self.UserDefined(mesh, *args)
        else: 
            raise ValueError("Wrong definition")
        
    class Creneau:  
        def __init__(self, mesh, t_crit, amplitude):
            """
            Définition d'un créneau d'amplitude "amplitude" commençant à
            t=0 et se terminant à t= t_crit

            Parameters
            ----------
            mesh : Mesh, maillage du domaine.
            t_crit : Float, temps de fin du créneau.
            amplitude : Float, amplitude du créneau.
            """
            self.t_crit = t_crit 
            self.amplitude = amplitude
            self.constant = Constant(mesh, float(amplitude))
            
        def set_time_dependant_array(self, load_steps):
            self.value_array = []
            for i in range(len(load_steps)):
                if load_steps[i] <=self.t_crit:
                    self.value_array.append(self.amplitude)
                else:
                    self.value_array.append(0)
                    
    class SmoothCreneau:  
        def __init__(self, mesh, t_load, t_plateau, amplitude):
            """
            Définition d'un créneau d'amplitude "amplitude" commençant à
            t=0 et se terminant à t= t_crit

            Parameters
            ----------
            mesh : Mesh, maillage du domaine.
            t_crit : Float, temps de fin du créneau.
            amplitude : Float, amplitude du créneau.
            """
            self.t_load = t_load
            self.t_plateau = t_plateau 
            self.amplitude = amplitude
            self.constant = Constant(mesh, float(amplitude))
            
        def set_time_dependant_array(self, load_steps):
            self.value_array = []
            t_fin_plate = self.t_load + self.t_plateau
            t_fin = 2 * self.t_load + self.t_plateau
            for i in range(len(load_steps)):
                load_step = load_steps[i]
                if load_step<=self.t_load:
                    self.value_array.append(self.amplitude * load_step/self.t_load)
                elif load_step >=self.t_load and load_step <=t_fin_plate:
                    self.value_array.append(self.amplitude)
                elif load_step >= t_fin_plate and load_step <= t_fin:
                    self.value_array.append(-self.amplitude / self.t_load * (load_step - t_fin_plate) + self.amplitude)
                else:
                    self.value_array.append(0)

    class Chapeau:  
        def __init__(self, mesh, t_crit, amplitude):
            """
            Définition d'un chapeau d'amplitude "amplitude" commençant à
            t=0 et se terminant à t= t_crit

            Parameters
            ----------
            mesh : Mesh, maillage du domaine.
            t_crit : Float, temps de fin du créneau.
            amplitude : Float, amplitude du créneau.
            """
            self.t_crit = t_crit 
            self.amplitude = amplitude
            self.constant = Constant(mesh, float(amplitude))
            
        def set_time_dependant_array(self, load_steps):
            self.value_array = []
            for i in range(len(load_steps)):
                if load_steps[i] <=self.t_crit/2:
                    self.value_array.append(2 * self.amplitude * load_steps[i]/self.t_crit)
                elif load_steps[i] >=self.t_crit/2 and load_steps[i] <=self.t_crit:
                    self.value_array.append(2 * self.amplitude * (1 - load_steps[i]/self.t_crit))
                else:
                    self.value_array.append(0)
            
    class Rampe:  
        def __init__(self, mesh, pente):
            """
            Défini chargement rampe.

            Parameters
            ----------
            mesh : Mesh, maillage du domaine.
            t : Float, instant courant.
            amplitude : Float, pente du chargement.
            Le chargement sera donc T^{D} = pente * t
            """
            self.pente = pente
            self.constant = Constant(mesh, ScalarType(0))
            self.v_constant = Constant(mesh, ScalarType(0))
            self.a_constant = Constant(mesh, ScalarType(0))
        
        def set_time_dependant_array(self, load_steps):
            self.value_array = []
            self.speed_array = []
            self.acceleration_array = []
            for i in range(len(load_steps)):
                self.value_array.append(load_steps[i] * self.pente)
                self.speed_array.append(self.pente)
                self.acceleration_array.append(0)
        
    class UserDefined:
        def __init__(self, mesh, value_array, speed_array = None):
            """
            Défini un chargement uniforme qui sera appliqué au bord.
            Il s'agit d'un chargement de la forme T^{D} = f(t).
            L'utilisateur doit donner en argument une liste ou un numpy
            array, qui contient les valeurs de la contrainte au bord,
            et de même longueur que le nombre de pas de temps.

            Parameters
            ----------
            mesh : Mesh, maillage du domaine.
            value_array : numpy.array ou list, list des valeurs de la contrainte
                                                à appliquer.
            """
            self.constant = Constant(mesh, ScalarType(0))
            self.value_array = value_array
            self.speed_array = speed_array
        
        def set_time_dependant_array(self, load_steps):
            pass
        
class MyConstantExpression(MyConstant):
    def __init__(self, function, mesh, *args, **kwargs):
        self.function = function
        MyConstant.__init__(mesh, *args, **kwargs)
        
class Tabulated_BCs:
    def __init__(self, mesh, *args, **kwargs):
        if kwargs.get("Type") == "Creneau":
            self.Expression = self.Creneau(mesh, *args)
        elif kwargs.get("Type") == "Rampe":
            self.Expression = self.Rampe(mesh, *args)
        elif kwargs.get("Type") == "UserDefined":
            self.Expression = self.UserDefined(mesh, *args)
        
    class Creneau:  
        def __init__(self, mesh, t_crit, amplitude):
            """
            Définition d'un créneau d'amplitude "amplitude" commençant à
            t=0 et se terminant à t= t_crit

            Parameters
            ----------
            mesh : Mesh, maillage du domaine.
            t_crit : Float, temps de fin du créneau.
            amplitude : Float, amplitude du créneau.
            """
            self.t_crit = t_crit 
            self.amplitude = amplitude
            self.constant = Constant(mesh, amplitude)
            
        def set_time_dependant_array(self, load_steps):
            self.value_array = []
            for i in range(len(load_steps)):
                if load_steps[i] <=self.t_crit:
                    self.value_array.append(self.amplitude)
                else:
                    self.value_array.append(0)