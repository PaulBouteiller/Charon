from ufl import Identity, as_vector, as_tensor, grad, det, inv, cofac, dot

class Deformations:
    """Deformation gradient, Jacobian, and related transformations."""
    
    def deformation_gradient_compact(self, displacement_field):
        """Return the deformation gradient in compact form."""
        # 1D models
        F_3d = self.deformation_gradient_3d(displacement_field)
        if self._is_1d:
            return as_vector([F_3d[0, 0], F_3d[1, 1], F_3d[2, 2]])
        elif self.name == "PlaneStrain":
            return as_vector([F_3d[0, 0], F_3d[1, 1], F_3d[0, 1], F_3d[1, 0], F_3d[2, 2]])
        elif self.name == "Axisymmetric":
            return as_vector([F_3d[0, 0], F_3d[1, 1], F_3d[0, 1], F_3d[1, 0], F_3d[2, 2]]) 
        else:
            return Identity(3) + self.grad_vector_3d(displacement_field)
    
    def deformation_gradient_3d(self, displacement_field):
        """Return the full 3D deformation gradient."""
        return Identity(3) + self.grad_vector_3d(displacement_field)
    
    def jacobian(self, displacement_field):
        """Return the Jacobian of the transformation (measure of local dilatation)."""
        # 1D models
        if self.name == "CartesianUD":
            return 1 + displacement_field.dx(0)
        elif self.name == "CylindricalUD":
            return (1 + displacement_field.dx(0)) * (1 + displacement_field / self.r)
        elif self.name == "SphericalUD":
            return (1 + displacement_field.dx(0)) * (1 + displacement_field / self.r)**2
        
        # 2D models
        elif self.name == "PlaneStrain":
            return det(Identity(2) + grad(displacement_field))
        elif self.name == "Axisymmetric":
            F_compact = self.deformation_gradient_compact(displacement_field)
            return F_compact[1, 1] * (F_compact[0, 0] * F_compact[2, 2] - F_compact[0, 2] * F_compact[2, 0])
        
        # 3D model
        else:
            return det(Identity(3) + grad(displacement_field))
    
    def inv_deformation_gradient_compact(self, displacement_field):
        """Return the inverse of the deformation gradient in compact form."""
        # 1D models
        if self.name == "CartesianUD":
            return as_vector([1 / (1 + displacement_field.dx(0)), 1, 1])
        elif self.name == "CylindricalUD":
            return as_vector([1 / (1 + displacement_field.dx(0)), 1/(1 + displacement_field / self.r), 1])
        elif self.name == "SphericalUD":
            return as_vector([1 / (1 + displacement_field.dx(0)), 1 /(1 + displacement_field / self.r), 1 / (1 + displacement_field / self.r)])
        
        # 2D models
        elif self.name == "PlaneStrain":
            inv_F_2d = inv(Identity(2) + grad(displacement_field))
            return as_tensor([[inv_F_2d[0, 0], inv_F_2d[0, 1], 0], 
                             [inv_F_2d[1, 0], inv_F_2d[1, 1], 0], 
                             [0, 0, 1]])
        elif self.name == "Axisymmetric":
            grad_u = grad(displacement_field)
            prefactor = (1 + grad_u[0, 0]) * (1 + grad_u[1, 1]) - grad_u[0, 1] * (1 + grad_u[1, 0])
            return as_tensor([[(1 + grad_u[1, 1]) / prefactor, 0, -grad_u[0, 1] / prefactor],
                             [0, 1 / (1 + displacement_field[0] / self.r), 0],
                             [-grad_u[1, 0] / prefactor, 0, (1 + grad_u[0, 0]) / prefactor]])
        
        # 3D model
        else:
            return inv(Identity(3) + grad(displacement_field))
    
    def relative_deformation_gradient_3d(self, current_displacement, previous_displacement):
        """Return the relative deformation gradient between two configurations."""
        F_current = self.deformation_gradient_3d(current_displacement)
        inv_F_previous = inv(self.deformation_gradient_3d(previous_displacement))
        return dot(F_current, inv_F_previous)
   
    def cofactor_compact(self, displacement_field):
        """Compute the cofactor in compact form adapted to geometry."""
        # 1D models
        if self.name == "CartesianUD":
            return 1
        elif self.name == "CylindricalUD":               
            return as_vector([1 + displacement_field / self.r, 1 + displacement_field.dx(0)])
        elif self.name == "SphericalUD":
            return as_vector([(1 + displacement_field / self.r)**2, 
                             (1 + displacement_field.dx(0)) * (1 + displacement_field / self.r),
                             (1 + displacement_field.dx(0)) * (1 + displacement_field / self.r)])
        
        # 2D models
        elif self.name == "PlaneStrain":
            return cofac(Identity(2) + grad(displacement_field))
        elif self.name == "Axisymmetric":
            F_3d = self.deformation_gradient_3d(displacement_field)
            return as_tensor([[F_3d[2, 2] * F_3d[1, 1], 0, -F_3d[0, 2] * F_3d[1, 1]],
                             [0, F_3d[0, 0] * F_3d[2, 2] - F_3d[0, 2] * F_3d[2, 0], 0],
                             [-F_3d[2, 0] * F_3d[1, 1], 0, F_3d[0, 0] * F_3d[1, 1]]])
        
        # 3D model
        else:  # Tridimensional
            return cofac(self.deformation_gradient_3d(displacement_field))
    
    def left_cauchy_green_compact(self, displacement_field):
        """Return the left Cauchy-Green tensor in compact form."""
        if self._is_1d:
            F_compact = self.deformation_gradient_compact(displacement_field)
            return as_vector([F_compact[0]**2, F_compact[1]**2, F_compact[2]**2])
        else:
            B_3d = self.left_cauchy_green_3d(displacement_field)
            return self.tensor_3d_to_compact(B_3d, symmetric=True)
    
    def left_cauchy_green_3d(self, displacement_field):
        """Return the full 3D left Cauchy-Green tensor."""
        F_3d = self.deformation_gradient_3d(displacement_field)
        return dot(F_3d, F_3d.T)
    
    def reduced_det(self, tensor):
        """Compute determinant for 1D reduced tensors."""
        if self._is_1d:
            return tensor[0, 0] * tensor[1, 1] * tensor[2, 2]