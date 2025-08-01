from .gradients import Gradients
from .conversions import Conversions
from .deformations import Deformations
from .contractions import Contractions

from ufl import grad, as_vector, dot, inv, Identity, div

class Kinematic(Gradients, Conversions, Deformations, Contractions):
    """
    Encapsulates kinematic operations for different dimensions and geometries.
    
    This class provides methods for calculating gradients, tensors, 
    transformations, and other kinematic quantities, automatically adapting
    the calculations to the problem dimension and geometry.
    """
    def __init__(self, name, r):
        """
        Initialize a Kinematic object.

        Parameters
        ----------
        name : str
            Name of the mechanical model
        r : Function or None
            Radial coordinate in axisymmetric, cylindrical, and spherical cases
        """
        self.name = name
        self.r = r
        self._is_1d = self.name in ["CartesianUD", "CylindricalUD", "SphericalUD"]
        
    def grad_vector_3d(self, vector_field, symmetric=False):
        """Return the three-dimensional gradient of a vector field."""
        return self.compact_to_tensor_3d(self.grad_vector_compact(vector_field, symmetric=symmetric), symmetric=symmetric)
    
    def grad_eulerian_compact(self, velocity_field, displacement_field):
        """Return the spatial velocity gradient in compact form."""
        inv_F_compact = self.inv_deformation_gradient_compact(displacement_field)
        grad_compact = self.grad_vector_compact(velocity_field)
        
        if self._is_1d:
            if self.name == "CartesianUD":
                return grad_compact * inv_F_compact[0]
            else:  # CylindricalUD or SphericalUD
                return as_vector([grad_compact[i] * inv_F_compact[i] for i in range(len(grad_compact))])
        elif self.name == "PlaneStrain":
            return self.tensor_2d_to_compact(dot(self.compact_to_tensor_2d(grad_compact), 
                                               inv(Identity(2) + grad(displacement_field))))
        elif self.name == "Axisymmetric":
            return self.tensor_3d_to_compact(dot(self.compact_to_tensor_3d(grad_compact), inv_F_compact))
        else:  # Tridimensional
            return dot(grad_compact, inv_F_compact)
        
    def measure(self, a, dx):
        """Return the integration measure adapted to the geometry."""
        if self.name in ["CartesianUD", "PlaneStrain", "Tridimensional"]:
            return a * dx
        elif self.name in ["CylindricalUD", "Axisymmetric"]:
            return a * self.r * dx
        elif self.name == "SphericalUD":
            return a * self.r**2 * dx
    
    def div(self, v):
        """Return the divergence of vector field v."""
        if self._is_1d:
            return v.dx(0)
        else:
            return div(v)
    
    def push_forward(self, tensor, displacement_field):
        """Return the push-forward of a second-order twice covariant tensor."""
        F = self.deformation_gradient_3d(displacement_field)
        return dot(dot(F, tensor), F.T)
    
    def pull_back(self, tensor, displacement_field):
        """Return the pull-back of a second-order twice covariant tensor."""
        F = self.deformation_gradient_3d(displacement_field)
        inv_F = inv(F)
        return dot(dot(inv_F, tensor), inv_F.T)