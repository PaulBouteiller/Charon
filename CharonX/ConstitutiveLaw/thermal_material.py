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
Created on Tue Dec 20 17:36:38 2022

@author: bouteillerp
"""

class LinearThermal:
    """
    Défini un matériau possédant des caractéristiques thermique linéaire isotrope
    """
    def __init__(self, lmbda):
        """
        Parameters
        ----------
        lmbda : Float, coefficient de diffusion thermique.
        """
        self.lmbda = lmbda
        print("Le coefficient de diffusion est", self.lmbda)
        self.type = "LinearIsotropic"
        
class NonLinearThermal:
    """
    Défini un matériau possédant des caractéristiques thermique non linéaire isotrope
    """
    def __init__(self, lmbda, a1, a2):
        """
        Parameters
        ----------
        lmbda : Float, coefficient de diffusion thermique.
        """
        self.lmbda = lmbda
        self.a1 = a1
        self.a2 = a2
        print("Le coefficient de diffusion est", self.lmbda)
        print("Le coefficient de dépendance en température est", self.a1)
        print("Le coefficient de dépendance en pression est", self.a2)
        self.type = "NonLinearIsotropic"