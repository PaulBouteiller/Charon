# Copyright 2025 CEA
"""
Modèle de plasticité de Gurson avec JAX

@author: bouteillerp
"""

from .base_plastic import Plastic
from dolfinx.fem import functionspace, Function, Expression
from ufl import dot, det, sqrt, tr, exp
from math import pi


class JAXGursonPlasticity(Plastic):
    """Gurson-Tvergaard-Needleman model for porous plasticity.
    
    Implements GTN model for porous ductile materials,
    accounting for void growth, nucleation, and coalescence.
    
    Attributes
    ----------
    V_Be : FunctionSpace Function space for elastic left Cauchy-Green tensor
    Be_Bar_trial_func : Function Trial elastic left Cauchy-Green tensor
    Be_Bar_old : Function Previous elastic left Cauchy-Green tensor
    len_plas : int Length of plastic variable array
    Be_bar_old_3D : Expression 3D representation of previous tensor
    Be_Bar_trial : Expression Expression for trial tensor
    V_p : FunctionSpace Function space for cumulated plasticity
    p : Function Cumulated plasticity
    V_f : FunctionSpace Function space for porosity
    f : Function Porosity (void volume fraction)
    q1, q2, q3 : float Tvergaard parameters
    f0, fc, ff : float Initial, critical, and failure porosity
    """
    def _set_function(self, element, quadrature):
        """Initialize functions for Gurson plasticity.
        
        Parameters
        ----------
        quadrature : QuadratureHandler Handler for quadrature integration
        """
        self.V_Be = functionspace(self.mesh, element)
        self.Be_Bar_trial_func = Function(self.V_Be)
        self.Be_Bar_old = Function(self.V_Be)

        # Initialisation de Be_Bar_old avec l'identité
        len_plas = len(self.Be_Bar_old.x.array)
        self.len_plas = len_plas
        self.Be_Bar_old.x.array[::len_plas] = 1
        self.Be_Bar_old.x.array[1::len_plas] = 1
        self.Be_Bar_old.x.array[2::len_plas] = 1
        self.Be_bar_old_3D = self.kin.mandel_to_tridim(self.Be_Bar_old)
        
        # Champ de déplacement ancien
        self.u_old = Function(self.V, name="old_displacement")
        F_rel = self.kin.relative_gradient_3D(self.u, self.u_old)
        
        # Prédicteur élastique
        expr = det(F_rel)**(-2./3) * dot(dot(F_rel.T, self.Be_bar_old_3D), F_rel.T)
        self.Be_Bar_trial = Expression(self.kin.tridim_to_mandel(expr), self.V_Be.element.interpolation_points())
        
        # Déformation plastique cumulée
        self.V_p = quadrature.quadrature_space(["Scalar"])
        self.p = Function(self.V_p, name="Cumulated_plasticity")
        
        # Porosité
        self.V_f = quadrature.quadrature_space(["Scalar"])
        self.f = Function(self.V_f, name="Porosity")
        
        # Paramètres du modèle de Gurson-Tvergaard-Needleman (GTN)
        self.q1 = 1.5  # Paramètre de Tvergaard
        self.q2 = 1.0  # Paramètre de Tvergaard
        self.q3 = self.q1**2  # Généralement q3 = q1^2
        
        # Paramètres d'évolution de la porosité
        self.f0 = 0.001  # Porosité initiale
        self.fc = 0.15   # Porosité critique
        self.ff = 0.25   # Porosité à rupture
        
        # Initialisation de la porosité
        self.f.x.array[:] = self.f0
        
    def compute_f_star(self, f):
        """Calculate effective porosity according to GTN model.
        
        Parameters
        ----------
        f : float or Function Current porosity
            
        Returns
        -------
        float or Function Effective porosity accounting for void coalescence
        """
        if f <= self.fc:
            return f
        else:
            fu = 1/self.q1  # Porosité ultime
            return self.fc + (fu - self.fc)*(f - self.fc)/(self.ff - self.fc)
            
    def update_porosity(self, be_bar, dp, f_old, p_old):
        """Update porosity according to GTN model.
        
        Accounts for void growth and nucleation of new voids.
        
        Parameters
        ----------
        be_bar : Expression Elastic left Cauchy-Green tensor
        dp : float or Function Increment of plastic strain
        f_old : float or Function Previous porosity
        p_old : float or Function Previous cumulated plastic strain
            
        Returns
        -------
        float or Function Updated porosity
        """
        # Croissance des vides
        tr_D = dp * tr(self.normal(be_bar))  # Trace du taux de déformation plastique
        f_growth = (1 - f_old) * tr_D
        
        # Nucléation contrôlée par la déformation
        eN = 0.3  # Déformation moyenne de nucléation
        sN = 0.1  # Écart-type
        fN = 0.04  # Fraction volumique de vides nucléés
        
        f_nucleation = fN/(sN*sqrt(2*pi)) * \
                      exp(-0.5*((p_old - eN)/sN)**2) * dp
        
        return f_old + f_growth + f_nucleation