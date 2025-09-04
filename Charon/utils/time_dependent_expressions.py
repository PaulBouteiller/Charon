# Copyright 2025 CEA
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Custom Expression Module
=====================

This module provides custom time-dependent expressions for boundary conditions
and loading in finite element simulations. It includes various predefined loading
patterns like step functions, ramps, and custom user-defined functions.

Key components:
- MyConstant: Base class for time-dependent expressions
- Various loading patterns (step, smooth step, hat, ramp)
- User-defined arbitrary loading functions
- Tabulated boundary conditions
"""

from dolfinx.fem import Constant
from petsc4py.PETSc import ScalarType
from scipy import interpolate

def interpolation_lin(temps_originaux, valeurs_originales, nouveaux_temps):
    """
    Linearly interpolate values at new time points.

    Parameters
    ----------
    temps_originaux : array List of original time points
    valeurs_originales : array Values at the original time points
    nouveaux_temps : array List of new time points at which to interpolate
        
    Returns
    -------
    array Interpolated values at the new time points
    """
    f = interpolate.interp1d(temps_originaux, valeurs_originales)
    return f(nouveaux_temps)

class MyExpression:
    def __init__(self, dictionnaire):
        self.ufl_expression = dictionnaire["expression"]

class MyConstant:
    """
    Base class for time-dependent expressions.
    
    This class creates different types of time-dependent expressions based on
    the "Type" argument, such as step functions, ramps, etc.
    
    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh Computational mesh
    *args : tuple Additional arguments for the specific expression type
    **kwargs : dict Keyword arguments, including "Type" to specify the expression type
    """
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
    
    @classmethod
    def from_dict(cls, mesh, constant_dict):
        """
        Create a MyConstant object from a dictionary specification.
        
        Parameters
        ----------
        mesh : dolfinx.mesh.Mesh
            Computational mesh
        constant_dict : dict
            Dictionary with keys:
            - 'type' : str, type of constant ('rampe', 'creneau', 'chapeau', etc.)
            - 'amplitude' : float, amplitude value
            - Additional parameters based on type
        
        Returns
        -------
        MyConstant object
        
        Examples
        --------
        >>> rampe = MyConstant.from_dict(mesh, {"type": "rampe", "amplitude": 1.0})
        >>> creneau = MyConstant.from_dict(mesh, {"type": "creneau", "amplitude": 2.0, "t_crit": 0.5})
        """
        constant_type = constant_dict["type"].lower()

        
        # Map user-friendly names to MyConstant Type names
        type_mapping = {
            "rampe": "Rampe",
            "creneau": "Creneau", 
            "chapeau": "Chapeau",
            "smooth_creneau": "SmoothCreneau",
            "user_defined": "UserDefined"
        }
        
        if constant_type not in type_mapping:
            raise ValueError(f"Unknown constant type: {constant_type}")
        
        myconst_type = type_mapping[constant_type]
        
        # Handle different parameter requirements for each type
        if constant_type == "rampe":
            pente = constant_dict["pente"]
            return cls(mesh, pente, Type=myconst_type)
        
        elif constant_type in ["creneau", "chapeau"]:
            amplitude = constant_dict["amplitude"]
            t_crit = constant_dict["t_crit"]
            return cls(mesh, t_crit, amplitude, Type=myconst_type)
        
        elif constant_type == "smooth_creneau":
            amplitude = constant_dict["amplitude"]
            t_load = constant_dict["t_load"]
            t_plateau = constant_dict["t_plateau"]
            return cls(mesh, t_load, t_plateau, amplitude, Type=myconst_type)
        
        elif constant_type == "user_defined":
            value_array = constant_dict["value_array"]
            speed_array = constant_dict["speed_array"]
            if speed_array:
                return cls(mesh, value_array, speed_array, Type=myconst_type)
            else:
                return cls(mesh, value_array, Type=myconst_type)
        
        else:
            raise ValueError(f"Unsupported constant type: {constant_type}")
        
    class Creneau:  
        """
        Step function (rectangular pulse).
        
        Creates a step function with amplitude "amplitude" starting at t=0 
        and ending at t=t_crit.
        """
        def __init__(self, mesh, t_crit, amplitude):
            """
            Initialize a step function.

            Parameters
            ----------
            mesh : dolfinx.mesh.Mesh Computational mesh
            t_crit : float End time of the step
            amplitude : float Amplitude of the step
            """
            self.t_crit = t_crit 
            self.amplitude = amplitude
            self.constant = Constant(mesh, float(amplitude))
            
        def set_time_dependant_array(self, load_steps):
            """
            Precompute values for all time steps.
            
            Parameters
            ----------
            load_steps : array Array of time points
            """
            self.value_array = []
            for i in range(len(load_steps)):
                if load_steps[i] <= self.t_crit:
                    self.value_array.append(self.amplitude)
                else:
                    self.value_array.append(0)
                    
    class SmoothCreneau:  
        """
        Smooth step function with linear ramps.
        
        Creates a trapezoidal pulse with linear rise and fall.
        """
        def __init__(self, mesh, t_load, t_plateau, amplitude):
            """
            Initialize a smooth step function.

            Parameters
            ----------
            mesh : dolfinx.mesh.Mesh Computational mesh
            t_load : float Duration of the rise/fall
            t_plateau : float Duration of the plateau
            amplitude : float Amplitude of the step
            """
            self.t_load = t_load
            self.t_plateau = t_plateau 
            self.amplitude = amplitude
            self.constant = Constant(mesh, float(amplitude))
            
        def set_time_dependant_array(self, load_steps):
            """
            Precompute values for all time steps.
            
            Parameters
            ----------
            load_steps : array Array of time points
            """
            self.value_array = []
            t_fin_plate = self.t_load + self.t_plateau
            t_fin = 2 * self.t_load + self.t_plateau
            for i in range(len(load_steps)):
                load_step = load_steps[i]
                if load_step <= self.t_load:
                    self.value_array.append(self.amplitude * load_step/self.t_load)
                elif load_step >= self.t_load and load_step <= t_fin_plate:
                    self.value_array.append(self.amplitude)
                elif load_step >= t_fin_plate and load_step <= t_fin:
                    self.value_array.append(-self.amplitude / self.t_load * (load_step - t_fin_plate) + self.amplitude)
                else:
                    self.value_array.append(0)

    class Chapeau:  
        """
        Hat function (triangular pulse).
        
        Creates a hat function with amplitude "amplitude" starting at t=0
        and ending at t=t_crit, with the peak at t=t_crit/2.
        """
        def __init__(self, mesh, t_crit, amplitude):
            """
            Initialize a hat function.

            Parameters
            ----------
            mesh : dolfinx.mesh.Mesh Computational mesh
            t_crit : float End time of the hat
            amplitude : float Amplitude of the hat
            """
            self.t_crit = t_crit 
            self.amplitude = amplitude
            self.constant = Constant(mesh, float(amplitude))
            
        def set_time_dependant_array(self, load_steps):
            """
            Precompute values for all time steps.
            
            Parameters
            ----------
            load_steps : array Array of time points
            """
            self.value_array = []
            for i in range(len(load_steps)):
                if load_steps[i] <= self.t_crit/2:
                    self.value_array.append(2 * self.amplitude * load_steps[i]/self.t_crit)
                elif load_steps[i] >= self.t_crit/2 and load_steps[i] <= self.t_crit:
                    self.value_array.append(2 * self.amplitude * (1 - load_steps[i]/self.t_crit))
                else:
                    self.value_array.append(0)
            
    class Rampe:  
        """
        Ramp function.
        
        Creates a linear ramp function with slope "pente".
        """
        def __init__(self, mesh, pente):
            """
            Define a ramp loading.

            Parameters
            ----------
            mesh : dolfinx.mesh.Mesh Computational mesh
            pente : float Slope of the ramp
                
            Notes
            -----
            The loading will be T^{D} = pente * t
            """
            self.pente = pente
            self.constant = Constant(mesh, ScalarType(0))
            self.v_constant = Constant(mesh, ScalarType(0))
            self.a_constant = Constant(mesh, ScalarType(0))
        
        def set_time_dependant_array(self, load_steps):
            """
            Precompute values for all time steps.
            
            Parameters
            ----------
            load_steps : array Array of time points
            """
            self.value_array = []
            self.speed_array = []
            self.acceleration_array = []
            for i in range(len(load_steps)):
                self.value_array.append(load_steps[i] * self.pente)
                self.speed_array.append(self.pente)
                self.acceleration_array.append(0)
        
    class UserDefined:
        """
        User-defined loading function.
        
        Creates a loading from an array of values provided by the user.
        """
        def __init__(self, mesh, value_array, speed_array=None):
            """
            Define a uniform loading to be applied at the boundary.
            
            This is a loading of the form T^{D} = f(t).
            The user must provide a list or numpy array containing the values
            of the boundary stress, with the same length as the number of time steps.

            Parameters
            ----------
            mesh : dolfinx.mesh.Mesh Computational mesh
            value_array : array List of stress values to apply
            speed_array : array, optional List of rate values, by default None
            """
            self.constant = Constant(mesh, ScalarType(0))
            self.value_array = value_array
            self.speed_array = speed_array
        
        def set_time_dependant_array(self, load_steps):
            """
            This method is a no-op for user-defined arrays.
            
            Parameters
            ----------
            load_steps : array Array of time points
            """
            pass