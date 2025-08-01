from ufl import dot, inner, as_vector

class Contractions:
    """Tensor contraction operations."""
    
    def contract_scalar_gradients(self, grad_1, grad_2):
        """Compute the dot product between two scalar gradients."""
        # 1D models - gradients are scalars
        if self._is_1d:
            return grad_1 * grad_2
        
        # 2D and 3D models - use UFL dot product
        else:  # PlaneStrain, Axisymmetric, Tridimensional
            return dot(grad_1, grad_2)
        
    def contract_simple(self, compact_tensor_1, compact_tensor_2):
        """Compute the dot product between two compact tensor representations."""
        # 1D models
        if self.name == "CartesianUD":
            return compact_tensor_1 * compact_tensor_2
        elif self.name == "CylindricalUD":
            return as_vector([compact_tensor_1[i] * compact_tensor_2[i] for i in range(2)])
        elif self.name == "SphericalUD":
            return as_vector([compact_tensor_1[i] * compact_tensor_2[i] for i in range(3)])
        
        # 2D and 3D models - use UFL dot product
        else:  # PlaneStrain, Axisymmetric, Tridimensional
            return dot(compact_tensor_1, compact_tensor_2)
       
    def contract_double(self, compact_tensor_1, compact_tensor_2):
        """Compute the double contraction between two tensors in compact form."""
        # 1D models
        if self.name == "CartesianUD":
            return compact_tensor_1 * compact_tensor_2
        elif self.name in ["CylindricalUD", "SphericalUD"]:
            return dot(compact_tensor_1, compact_tensor_2)
        
        # 2D models
        elif self.name == "PlaneStrain":
            # Handle different compact formats: [σxx, σyy, σxy] vs [εxx, εyy, εxy, εyx]
            shape_1 = compact_tensor_1.ufl_shape[0]
            shape_2 = compact_tensor_2.ufl_shape[0]
            if shape_1 == 3 and shape_2 == 4:
                return compact_tensor_1[0] * compact_tensor_2[0] + \
                       compact_tensor_1[1] * compact_tensor_2[1] + \
                       compact_tensor_1[2] * (compact_tensor_2[2] + compact_tensor_2[3])
            else:
                return dot(compact_tensor_1, compact_tensor_2)
                
        elif self.name == "Axisymmetric":
            # Handle different compact formats: [σrr, σθθ, σzz, σrz] vs [εrr, εθθ, εzz, εrz, εzr]
            shape_1 = compact_tensor_1.ufl_shape[0]
            shape_2 = compact_tensor_2.ufl_shape[0]
            if shape_1 == 4 and shape_2 == 5:
                return compact_tensor_1[0] * compact_tensor_2[0] + \
                       compact_tensor_1[1] * compact_tensor_2[1] + \
                       compact_tensor_1[2] * compact_tensor_2[2] + \
                       compact_tensor_1[3] * (compact_tensor_2[3] + compact_tensor_2[4])
            else:
                return dot(compact_tensor_1, compact_tensor_2)
        
        # 3D model
        else:  # Tridimensional
            return inner(compact_tensor_1, compact_tensor_2)