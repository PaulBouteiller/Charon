"""
Created on Thu Mar 17 13:32:46 2022

@author: bouteillerp

"""
from ..utils.generic_functions import (set_correction, petsc_assign, ppart,
                                       over_relaxed_predictor, set_min, petsc_div)
from ..utils.default_parameters import default_PhaseField_solver_parameters, default_damage_solver_type

from .NL_problem import TAOProblem, SNESProblem
from .hybrid_solver import create_linear_solver

from ufl import (TrialFunction, TestFunction, dot, grad, inner, derivative)
from dolfinx.fem.petsc import assemble_vector, create_matrix

from petsc4py.PETSc import TAO, SNES, COMM_SELF
from dolfinx.fem import (form, Function)
from dolfinx.la import create_petsc_vector

try:
    import jax.numpy as jnp
    from jax import vmap, jit
    from diffrax import (diffeqsolve, ODETerm, Tsit5, SaveAt, Kvaerno3, 
                        PIDController, Dopri5, Euler)
except Exception:
    print("JAX or Diffrax has not been loaded therefore Johnson model can not be used")

class DamageSolve:
    """
    La classe damage solve implémente les routines pour le calcul des variables
    d'endommagement (ou de la porosité si l'on utilise le modèle de Johnson)
    """
    def __init__(self, Damage_object, dt):
        """
        Initialisation du solveur d'endommagement.

        Parameters
        ----------
        Damage_object : Objet de la classe Damage.
        dt : Float, pas de temps.
        """
        self.tol = 1e-2
        self.dam = Damage_object
        self.dt = dt
        
    def inf_damage(self):
        """
        Mise à jour de la borne inférieur de l'endommagement au début de chaque
        nouveau incrément.
        """
        petsc_assign(self.dam.inf_d, self.dam.d)
    
    def damage_evolution(self):
        """
        Compare l'endommagement à sa précédente valeur, si la différence est
        supérieure à une tolérance self.tol, on considère que l'endommagement évolue.

        Returns
        -------
        evol_dam : Booléen, True si l'endommagement évolue.
        """
        prev_d = self.dam.d.copy()
        self.compute_damage()
        diff = Function(self.dam.V_d)
        diff.x.array[:] = self.dam.d.x.array - prev_d.x.array
        evol_dam =  diff.x.norm() > self.tol
        return evol_dam
        
class JohnsonSolve(DamageSolve):
    def __init__(self, Damage_object, dt):
        DamageSolve.__init__(self, Damage_object, dt)
        self.method = default_damage_solver_type()
        solver_dict = {"Tsit5": Tsit5(),"Kvaerno3": Kvaerno3(), 
                       "Dopri5": Dopri5(), "Euler": Euler()}
        self.solver = solver_dict.get(self.method)
        self.set_Diffrax_solve()
        
    def explicit_damage(self):
        self.inf_damage()
        self.compute_damage()
        set_correction(self.dam.d, self.dam.inf_d, self.dam.max_d)
    
    def solve_diff_eq(self, term, dt, y0, args):
        if self.method == "Euler":
            # Pour Euler, nous n'utilisons pas de contrôleur de pas de temps adaptatif
            solution = diffeqsolve(term, self.solver, t0=0., t1=dt, dt0=dt, y0=y0, 
                                   args=args, saveat=SaveAt(t1=True),
                                   max_steps=1)
        else:
            solution = diffeqsolve(term, self.solver, t0=0., t1=dt, dt0=dt,
                                   y0=y0, args=args, saveat=SaveAt(t1=True),
                                   stepsize_controller=PIDController(rtol=1e-4, atol=1e-4),
                                   max_steps = int(1e5))
        return solution.ys[-1]
        
class StaticJohnsonSolve(JohnsonSolve):
    def __init__(self, Damage_object, kinematic, comm, dx, dt):
        JohnsonSolve.__init__(self, Damage_object, dt)
        if self.dam.regularization:
            self.set_gaussian_regularization(dx, kinematic)
            self.set_regularization_projection(dx, kinematic)
        
    def set_Diffrax_solve(self):
        @jit    
        def solve_ode(f0, dt, p, p0, eta):
            def func(t, f, args):
                p, p0, eta = args
                F_mot = 2./3 * jnp.log(f) - (1 - f) * p / p0
                return ppart(3 * f * p0 / (4 * eta) * F_mot)

            return self.solve_diff_eq(ODETerm(func), dt, f0, (p, p0, eta))
        # Vectorisation de solve_ode pour prendre en compte plusieurs composantes
        self.solve_ode_step_vmap = vmap(solve_ode, in_axes=(0, None, 0, None, None))
    
    def compute_damage(self):
        """
        Calcul et met à jour la variable d'endommagement.
        """    
        self.dam.p_func.interpolate(self.dam.p_mot)
        self.dam.d.x.array[:] = self.solve_ode_step_vmap(jnp.array(self.dam.d.x.array), 
                                                         self.dt, jnp.array(self.dam.p_func.x.array),
                                                         self.dam.sigma_0, self.dam.eta)
        if self.dam.regularization:
            self.gaussian_regularization.solve()
            petsc_div(assemble_vector(self.b_proj), assemble_vector(self.a_proj), self.f_proj.x.petsc_vec)
            petsc_assign(self.dam.d, self.f_proj)
            
    def set_gaussian_regularization(self, dx, kinematic):
        """
        Régularisation par une Gaussienne du champ de porosité, équivalent à prendre
        le champ de porosité localisé en input d'un solver de diffusion.
        Parameters
        ----------
        dx : Measure, mesure d'intégration.
        kinematic : Objet de la classe kinématic.
        """
        self.dd = TrialFunction(self.dam.V_d_regul)
        self.d_ = TestFunction(self.dam.V_d_regul)
        self.f = Function(self.dam.V_d_regul)
        a_form_regul = kinematic.measure(self.dd * self.d_ + self.dam.lc**2 * dot(grad(self.dd), grad(self.d_)), dx)
        L_form_regul = kinematic.measure(inner(self.dam.d, self.d_), dx)
        self.gaussian_regularization = create_linear_solver(a_form_regul, L_form_regul, self.f, 
                                                            bcs=[], solver_type="hybrid")
            
    def set_regularization_projetion(self, dx, kinematic):
        self.f_proj = Function(self.dam.V_d)
        self.f_ = TestFunction(self.dam.V_d)
        self.f0 = Function(self.dam.V_d)
        self.f0.x.petsc_vec.set(1)
        self.a_proj = form(kinematic.measure(self.f0 * self.f_, dx))
        self.b_proj = form(kinematic.measure(self.f * self.f_, dx))

class JohnsonDynViscSolve(JohnsonSolve):
    def __init__(self, Damage_object, kinematic, comm, dx, dt):
        self.dt_tilde = dt / Damage_object.tau
        print("Le pas de temps adimensionné vaut", self.dt_tilde)
        JohnsonSolve.__init__(self, Damage_object, dt)

    def set_Diffrax_solve(self):
        def func(t, y, args):
            p, p0, f = args
            tilde_a, tilde_a_dot = y
            
            # Ajout d'une protection contre la division par zéro
            epsilon = 1e-10
            tilde_a_safe = jnp.maximum(tilde_a, epsilon)
            
            term1 = (1 - f**(1/3)) * tilde_a_safe
            F_steady = (3/2 - 2*f**(1/3) + 1/2*f**(4/3)) * tilde_a_dot**2
            F_mot = 2/3 * jnp.log(f) - (1 - f)**2 * p / p0
            F_visc = 4 * ppart(tilde_a_dot) / tilde_a_safe * (1 - f)
            
            tilde_a_ddot = jnp.maximum((F_mot - F_visc - F_steady), 0)/ term1
            return jnp.array([tilde_a_dot, tilde_a_ddot])

        @jit    
        def solve_ode(a0, dot_a0, dt, p, p0, f):
            return self.solve_diff_eq(ODETerm(func), dt, jnp.array([a0, dot_a0]), (p, p0, f))
        self.solve_ode_step_vmap = vmap(solve_ode, in_axes=(0, 0, None, 0, None, 0))
        
    def compute_damage(self):
        self.dam.p_func.interpolate(self.dam.p_mot)
        sol = self.solve_ode_step_vmap(jnp.array(self.dam.a_tilde.x.array),
                                       jnp.array(self.dam.dot_a_tilde.x.array),
                                       self.dt_tilde, jnp.array(self.dam.p_func.x.array),
                                       self.dam.sigma_0, jnp.array(self.dam.d.x.array))
        self.dam.a_tilde.x.array[:] = jnp.asarray(sol[:, 0])
        self.dam.dot_a_tilde.x.array[:] = jnp.asarray(sol[:, 1])

        self.dam.d.interpolate(self.dam.d_expr)
        
class JohnsonInerSolve(JohnsonSolve):
    def __init__(self, Damage_object, kinematic, comm, dx, dt):
        JohnsonSolve.__init__(self, Damage_object, dt)
        
    def set_Diffrax_solve(self):
        def func(t, y, args):
            a, dot_a = y
            p, f, p0, rho_0 = args
            
            # Protection contre la division par zéro
            epsilon = 1e-10
            a_safe = jnp.maximum(a, epsilon) 
            F_mot = 2./3 * p0 * jnp.log(f) - (1 - f)**2 * p
            F_steady = rho_0 * (3/2. - 2 * f**(1./3) + 1./2 * f**(4./3)) * dot_a**2
            parenthese = F_mot - F_steady
            ddot_a = jnp.maximum(parenthese, 0) / (rho_0 * a_safe * (1 - f**(1./3)))
            return jnp.array([dot_a, ddot_a])

        @jit    
        def solve_ode(a0, dot_a0, dt, p, f, p0, rho_0):
            return self.solve_diff_eq(ODETerm(func), dt, jnp.array([a0, dot_a0]), (p, f, p0, rho_0))

        self.solve_ode_step_vmap = vmap(solve_ode, in_axes=(0, 0, None, 0, 0, None, None))
    
    def compute_damage(self):
        self.dam.p_func.interpolate(self.dam.p_mot)
        sol = self.solve_ode_step_vmap(jnp.array(self.dam.a.x.array),
                                       jnp.array(self.dam.dot_a.x.array),
                                       self.dt, jnp.array(self.dam.p_func.x.array),
                                       jnp.array(self.dam.d.x.array), 
                                       self.dam.sigma_0, self.dam.rho_0)
        self.dam.a.x.array[:] = jnp.asarray(sol[:, 0])
        self.dam.dot_a.x.array[:] = jnp.asarray(sol[:, 1])
        
        self.dam.d.interpolate(self.dam.d_expr)
      
class PhaseFieldSolve(DamageSolve):
    def __init__(self, Damage_object, kinematic, comm, dx, dt):
        DamageSolve.__init__(self, Damage_object, dt)
        self.dam_compteur = 0
        self.dam_iter = 1
        self.solver_parameters = default_PhaseField_solver_parameters()
        self.dam_solver_type = self.solver_parameters.get("type")
        self.set_damage_problem(kinematic, dx)
        self.solver = self.set_damage_solver()
        self.nb_max_calc_d = 10
        
    def compute_damage(self):
        prev_d = self.dam.inf_d
        if self.dam_solver_type == "TAO":
            self.solver.solve(self.dam.d.x.petsc_vec)
        elif self.dam_solver_type == "SNES":
            self.solver.solve(None, self.dam.d.x.petsc_vec)
        # if self.over_relaxed:
        #     over_relaxed_predictor(self.dam.d, self.dam.prev_d, 1.4)
            
        set_correction(self.dam.d, prev_d, self.dam.max_d)
    
    def explicit_damage(self):
        self.dam_compteur+=1
        if self.dam_compteur>=self.dam_iter:
            self.dam_compteur = 0
            self.inf_damage()
            evol_dam = self.dam_evolution()
            if evol_dam :
                self.dam_iter = 1
            else:
                self.dam_iter = min(self.nb_max_calc_d, self.dam_iter*2)
                print("l'endommagement sera calculé tous les " + str(self.dam_iter) + " pas de temps")              
        else:
            evol_dam =False
        return evol_dam
        
    def set_damage_problem(self, kinematic, dx):
        """
        Définition de la fonctionnelle énergie totale à minimiser ainsi que
        de ses dérivées. Puis définition du problème non linéaire sous-contraintes
        à résoudre.

        Parameters
        ----------
        kinematic : Objet de la classe kinematic.
        dx : Measure, mesure d'intégration.
        """
        energy_tot = kinematic.measure(self.dam.energy + self.dam.fracture_energy, dx)
        # first derivative of energy with respect to d
        F_dam = derivative(energy_tot, self.dam.d, self.dam.d_)
        # second derivative of energy with respect to d
        J_dam = derivative(F_dam, self.dam.d, self.dam.dd)
        if self.dam_solver_type == "TAO":
            self.problem = TAOProblem(energy_tot, F_dam, J_dam, self.dam.d, [])
        elif self.dam_solver_type == "SNES":
            self.problem = SNESProblem(F_dam, J_dam, self.dam.d, [])     
        
    def set_damage_solver(self):
        """
        Initialisation du solveur non linéaire d'optimisation sous-contrainte
        pour le problème de rupture par champ de phase.
        """

        tol = self.solver_parameters.get("tol")
        if self.dam_solver_type == "TAO":
            solver_d = TAO().create(COMM_SELF)
            solver_d.setType("bnls")
            solver_d.getKSP().setType("preonly")
            solver_d.getKSP().getPC().setType("lu")
            solver_d.setObjective(self.problem.f)
            solver_d.setGradient(self.problem.F)
            solver_d.setHessian(self.problem.J, self.problem.A)
            solver_d.setTolerances(gatol = tol, grtol=tol, gttol=tol)

        elif self.dam_solver_type == "SNES":
            V = self.dam.V_d
            b = create_petsc_vector(V.dofmap.index_map, V.dofmap.index_map_bs)
            J = create_matrix(self.problem.a)
            solver_d = SNES().create()
            solver_d.setType("vinewtonrsls")
            solver_d.setFunction(self.problem.F, b)
            solver_d.setJacobian(self.problem.J, J)
            solver_d.setTolerances(rtol = tol, max_it = 50)
            solver_d.getKSP().setType("preonly")
            solver_d.getKSP().setTolerances(rtol = tol)
            solver_d.getKSP().getPC().setType("lu")
            
        solver_d.setVariableBounds(self.dam.inf_d.x.petsc_vec, self.dam.max_d.x.petsc_vec)
        return solver_d