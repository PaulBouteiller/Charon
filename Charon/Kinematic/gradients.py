from ufl import grad, as_vector, div

class Gradients:
    """Gradient calculations for scalar and vector fields."""
    
    def grad_scalar_compact(self, scalar_field):
        """Return the appropriate representation of a scalar field gradient."""
        if self._is_1d:
            return scalar_field.dx(0)
        else: 
            return grad(scalar_field)
    
    def grad_scalar_3d(self, scalar_field):
        """Return the 3D gradient of a scalar field in vector form."""
        if self._is_1d:
            return as_vector([scalar_field.dx(0), 0, 0])
        elif self.name == "PlaneStrain": 
            return as_vector([scalar_field.dx(0), scalar_field.dx(1), 0])
        elif self.name == "Axisymmetric": 
            return as_vector([scalar_field.dx(0), 0, scalar_field.dx(1)])
        else:  # Tridimensional
            return grad(scalar_field)
    
    def grad_vector_compact(self, vector_field, symmetric=False):
        """Return the compact gradient of a vector field."""
        if self.name == "CartesianUD":
            return vector_field.dx(0)
        elif self.name == "CylindricalUD":
            return as_vector([vector_field.dx(0), vector_field / self.r])
        elif self.name == "SphericalUD":
            return as_vector([vector_field.dx(0), vector_field / self.r, vector_field / self.r])
        elif self.name == "PlaneStrain":
            grad_u = grad(vector_field)
            if symmetric:
                return as_vector([grad_u[0, 0], grad_u[1, 1], grad_u[0, 1]])
            else:
                return as_vector([grad_u[0, 0], grad_u[1, 1], grad_u[0, 1], grad_u[1, 0]])
        elif self.name == "Axisymmetric":
            grad_u = grad(vector_field)
            if symmetric:
                return as_vector([grad_u[0, 0], vector_field[0] / self.r, grad_u[1, 1], grad_u[0, 1]])
            else:
                return as_vector([grad_u[0, 0], vector_field[0] / self.r, grad_u[1, 1], 
                                 grad_u[0, 1], grad_u[1, 0]])
        else:  # Tridimensional
            return grad(vector_field)
        
    def div(self, v):
        """Return the divergence of vector field v."""
        if self._is_1d:
            return v.dx(0)
        else:
            return div(v)