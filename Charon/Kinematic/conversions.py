from ufl import as_tensor, as_vector
from math import sqrt
from numpy import array

class Conversions:
    """Conversion between different tensor representations."""
    
    def compact_to_tensor_3d(self, compact_tensor, symmetric=False):
        """Convert a tensor in compact form to its full three-dimensional form."""
        # 1D models
        if self.name == "CartesianUD":
            return as_tensor([[compact_tensor, 0, 0], 
                             [0, 0, 0], [0, 0, 0]])
        elif self.name == "CylindricalUD":
            return as_tensor([[compact_tensor[0], 0, 0], 
                             [0, compact_tensor[1], 0], [0, 0, 0]])
        elif self.name == "SphericalUD":
            return as_tensor([[compact_tensor[0], 0, 0], 
                             [0, compact_tensor[1], 0], 
                             [0, 0, compact_tensor[1]]])
        
        # 2D models
        elif self.name == "PlaneStrain":
            if symmetric:
                return as_tensor([[compact_tensor[0], compact_tensor[2], 0], 
                                 [compact_tensor[2], compact_tensor[1], 0], 
                                 [0, 0, 0]])
            else:
                return as_tensor([[compact_tensor[0], compact_tensor[2], 0], 
                                 [compact_tensor[3], compact_tensor[1], 0], 
                                 [0, 0, 0]])
        elif self.name == "Axisymmetric":
            if symmetric:
                return as_tensor([[compact_tensor[0], 0, compact_tensor[3]], 
                                 [0, compact_tensor[1], 0], 
                                 [compact_tensor[3], 0, compact_tensor[2]]])
            else:
                return as_tensor([[compact_tensor[0], 0, compact_tensor[3]], 
                                 [0, compact_tensor[1], 0], 
                                 [compact_tensor[4], 0, compact_tensor[2]]])
        
        # 3D model
        else:
            if symmetric:
                return as_tensor([[compact_tensor[0], compact_tensor[5], compact_tensor[4]], 
                                 [compact_tensor[5], compact_tensor[1], compact_tensor[3]], 
                                 [compact_tensor[4], compact_tensor[3], compact_tensor[2]]])
            else:
                return compact_tensor
    
    def compact_to_tensor_2d(self, compact_tensor):
        """Convert a tensor in compact form to its two-dimensional form."""
        return as_tensor([[compact_tensor[0], compact_tensor[2]], 
                          [compact_tensor[3], compact_tensor[1]]])
    
    def tensor_2d_to_compact(self, tensor_2d):
        """Convert a 2D tensor to its compact form."""
        return as_vector([tensor_2d[0, 0], tensor_2d[1, 1], tensor_2d[0, 1], tensor_2d[1, 0]])
    
    def tensor_3d_to_compact(self, tensor_3d, symmetric=False):
        """Convert a 3D tensor to its compact form."""
        # 1D models
        if self.name == "CartesianUD":
            return tensor_3d[0, 0]
        elif self.name == "CylindricalUD":
            return as_vector([tensor_3d[0, 0], tensor_3d[1, 1]])
        elif self.name == "SphericalUD":
            return as_vector([tensor_3d[0, 0], tensor_3d[1, 1], tensor_3d[2, 2]])
        
        # 2D models
        elif self.name == "PlaneStrain":
            if symmetric:
                return as_vector([tensor_3d[0, 0], tensor_3d[1, 1], tensor_3d[0, 1]])
            else:
                return as_vector([tensor_3d[0, 0], tensor_3d[1, 1], tensor_3d[0, 1], tensor_3d[1, 0]])
        elif self.name == "Axisymmetric":
            if symmetric:
                return as_vector([tensor_3d[0, 0], tensor_3d[1, 1], tensor_3d[2, 2], tensor_3d[0, 2]])
            else:
                return as_vector([tensor_3d[0, 0], tensor_3d[1, 1], tensor_3d[2, 2], 
                                tensor_3d[0, 2], tensor_3d[2, 0]])
        
        # 3D model
        elif self.name == "Tridimensional":
            if symmetric:
                return as_vector([tensor_3d[0, 0], tensor_3d[1, 1], tensor_3d[2, 2], 
                                tensor_3d[1, 2], tensor_3d[0, 2], tensor_3d[0, 1]])
            else:
                return tensor_3d
    
    def tensor_3d_to_mandel_compact(self, tensor_3d):
        """
        Convert a 3D tensor to its Mandel representation.
        
        The components order is: [11, 22, 33, 23, 13, 12]
    
        Parameters
        ----------
        tensor_3d : Tensor 3D tensor
    
        Returns
        -------
        Vector Mandel representation
        """
        sq2 = sqrt(2)
        
        # 1D models
        if self._is_1d:
            return as_vector([tensor_3d[0, 0], tensor_3d[1, 1], tensor_3d[2, 2]])
        
        # 2D models
        elif self.name == "PlaneStrain":
            return as_vector([tensor_3d[0, 0], tensor_3d[1, 1], tensor_3d[2, 2], sq2 * tensor_3d[0, 1]])
        elif self.name == "Axisymmetric":
            return as_vector([tensor_3d[0, 0], tensor_3d[1, 1], tensor_3d[2, 2], sq2 * tensor_3d[0, 2]])
        
        # 3D model
        elif self.name == "Tridimensional":
            return as_vector([tensor_3d[0, 0], tensor_3d[1, 1], tensor_3d[2, 2], 
                            sq2 * tensor_3d[1, 2], sq2 * tensor_3d[0, 2], sq2 * tensor_3d[0, 1]])
        
    def mandel_compact_to_tensor_3d(self, mandel_compact):
        """Convert a compact Mandel representation to a 3D tensor."""
        sq2 = sqrt(2)
        # 1D models
        if self._is_1d:
            return as_tensor([[mandel_compact[0], 0, 0], 
                             [0, mandel_compact[1], 0], 
                             [0, 0, mandel_compact[2]]])
        
        # 2D models
        elif self.name == "PlaneStrain":
            return as_tensor([[mandel_compact[0], mandel_compact[3] / sq2, 0], 
                             [mandel_compact[3] / sq2, mandel_compact[1], 0], 
                             [0, 0, mandel_compact[2]]])
        elif self.name == "Axisymmetric":
            return as_tensor([[mandel_compact[0], 0, mandel_compact[3] / sq2], 
                             [0, mandel_compact[1], 0], 
                             [mandel_compact[3] / sq2, 0, mandel_compact[2]]])
        
        # 3D model
        elif self.name == "Tridimensional":
            return as_tensor([[mandel_compact[0], mandel_compact[5] / sq2, mandel_compact[4] / sq2], 
                             [mandel_compact[5] / sq2, mandel_compact[1], mandel_compact[3] / sq2], 
                             [mandel_compact[4] / sq2, mandel_compact[3] / sq2, mandel_compact[2]]])
        
        
    def tensor_3d_to_voigt(self, tensor_3d):
        """
        Convert a 3D tensor to its Voigt representation.
        
        The components order is: [11, 22, 33, 23, 13, 12]
    
        Parameters
        ----------
        tens : Tensor 3D tensor
    
        Returns
        -------
        Vector Voigt representation
        """
        return as_vector([tensor_3d[0, 0], tensor_3d[1, 1], tensor_3d[2, 2], 
                            2 * tensor_3d[1, 2], 2 * tensor_3d[0, 2], 2 * tensor_3d[0, 1]])
    
    def voigt_to_tensor_3d(self, voigt, numpy = False):
        """
        Convert a Voigt representation to a 3D tensor.
        
        Note: Doesn't work for strains due to the factor 2.
    
        Parameters
        ----------
        Voigt : Vector Voigt representation
    
        Returns
        -------
        Tensor Corresponding 3D tensor, see tensor_3d_to_voigt for convention used
        """
        if numpy:
            return array([[voigt[0], voigt[5], voigt[4]],
                          [voigt[5], voigt[1], voigt[3]],
                          [voigt[4], voigt[3], voigt[2]]])
        else:
            return as_tensor([[voigt[0], voigt[5], voigt[4]],
                              [voigt[5], voigt[1], voigt[3]],
                              [voigt[4], voigt[3], voigt[2]]])