"""
Created on Fri Apr 28 13:30:39 2023

@author: bouteillerp


La méthode de Newton-Raphson est largement utilisée pour résoudre les problèmes non linéaire. 

Pour une fonction $F:\mathbb{R}^{M}\mapsto\mathbb{R}^{M}$, elle constiste à 
construire une suite $(x_{k})\in\mathbb{R}^{M}$ définie par la relation de réccurence:
    
$$x_{k+1}=x_{k}-J_{F}^{-1}(x_{k})F(x_{k})$$

où $J_{F}$ désigne la matrice Jacobienne de $F$. 
Nous pouvons réécrire cette équation en se donnant l'incrément $\delta x_{k}=x_{k+1}-x_{k}$:
$$J_{F}(x_{k})\delta x_{k}=-F(x_{k})$.

"""
from ufl import derivative

from petsc4py import PETSc
from mpi4py import MPI
from dolfinx.fem import form, Function
from dolfinx.fem.petsc import (create_matrix, create_vector, assemble_matrix, 
                               assemble_vector, apply_lifting, set_bc)
from dolfinx.io import XDMFFile
from os import remove, path

class CustomNewtonSolver:
    def __init__(self, Residual, u, bcs, solveur = "default_linear", intermediate_output = True):
        """
        Implemente un solveur de Newton personnalisé

        Parameters
        ----------
        Residual : Form, a(u,v)-L(v) form à annuler.
        u : Function, champ de déplacement.
        bcs : DirichletBC, conditions aux limites de Dirichlet.
        """
        comm = u.function_space.mesh.comm
        self.F_form = form(Residual)
        self.J = derivative(Residual, u)
        self.J_form = form(self.J)  # form of the jacobian
        self.u = u
        self.bcs = bcs
    
        self.du = Function(self.u.function_space)  # increment of the solution
        # holds previous solution for reverting in case of convergence failure
        
        self.A = create_matrix(self.J_form)  # preallocating sparse jacobian matrix
        self.L = create_vector(self.F_form)  # preallocating residual vector

        #On créé un objet ksp, grosso modo un solveur Ax = b
        if solveur == "default_linear":
            self.linear_solver = PETSc.KSP().create(comm)
            
            # On demande que le solveur soit direct
            self.linear_solver.setType(PETSc.KSP.Type.PREONLY)
            # Un préconditionneur de type LU est utilisé
            self.linear_solver.getPC().setType(PETSc.PC.Type.LU)
            opts = PETSc.Options()
            # prefix = f"solver_{id(self.solver)}"
            prefix = f"projector_{id(self)}"
            self.linear_solver.setOptionsPrefix(prefix)
            option_prefix = self.linear_solver.getOptionsPrefix()
            opts[f"{option_prefix}ksp_type"] = "preonly"
            opts[f"{option_prefix}pc_type"] = "lu"
            opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
            self.linear_solver.setFromOptions()
            #La matrice A utilisée pour résoudre le système Ax=b est self.A
            self.linear_solver.setOperators(self.A)
        elif solveur == "hybrid":
            self.linear_solver = HybridSolver(comm, self.A, is_symmetric = False)
            
        # if intermediate_output:
        #     file_name = "intermediate" + "-" + "results"
        #     # if path.isfile(file_name):
        #     #     remove(file_name)
        #     #     remove(file_name.replace(".xdmf", ".h5"))
        #     #     print("A new intermediate file is created")
        #     self.intermediate_result = XDMFFile(u.function_space.mesh.comm, file_name, "a")
        #     self.intermediate_result.write_mesh(u.function_space.mesh)
    
    ### iteration adopted from Custom Newton Solver tutorial ###
    def solve(self, print_steps = True, print_solution = False):
        # print("Initial guess", self.u.vector.array)
        # iteration parameters
        max_it = int(1e3)
        tol = 1e-8
        i = 0  # number of iterations of the Newton solver
        converged = False
        while i < max_it:
            # Assemble Jacobian and residual
            with self.L.localForm() as loc_L:
                loc_L.set(0)
            #On constuit ici la matrice A
            self.A.zeroEntries()
            assemble_matrix(self.A, self.J_form, bcs = self.bcs)

            self.A.assemble()
            assemble_vector(self.L, self.F_form)
            self.L.ghostUpdate(addv=PETSc.InsertMode.ADD,
                          mode=PETSc.ScatterMode.REVERSE)
            
            #Dans la méthode de Newton-Raphson on trouve le vecteur du 
            # tel que A*du = -b il faut donc multiplier le vecteur de droite par -1
            self.L.scale(-1)

            # Compute b - J(u_D-u_(i-1))
            apply_lifting(self.L, [self.J_form], [self.bcs], x0 = [self.u.vector], scale=1)
            # Set dx|_bc = u_{i-1}-u_D
            set_bc(self.L, self.bcs, self.u.vector, 1.0)
            self.L.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES,
                          mode=PETSc.ScatterMode.FORWARD)

            # Solve linear problem, A*self.du = self.L
            self.linear_solver.solve(self.L, self.du.vector)
            self.du.x.scatter_forward()

            self.u.x.array[:] += self.du.x.array[:]
            i += 1
            # Compute norm of update
            correction_norm = self.du.vector.norm(0)
            error_norm = self.L.norm(0)
            if print_steps:
                print(f"Iteration {i}: Correction norm {correction_norm}, Residual {error_norm}")

            if correction_norm < tol:
                converged = True
                break
            # self.intermediate_result.write_function(self.u)

        if print_solution:
            if converged:
                # (Residual norm {error_norm})")
                print(f"Solution reached in {i} iterations.")
            else:
                print(
                    f"No solution found after {i} iterations. Revert to previous solution and adjust solver parameters.")

        return converged, i
    
iteration_switch = 10

class HybridSolver:
    """
    A generic class implementing a hybrid solver.

    A direct solver is used for a first matrix factorization.
    subsequent solves are the obtained using an iterative solver with the
    previously factorized matrix used as a preconditioner. this preconditioner
    is kept even if the operator changes, except if the number of ksp iterations
    exceeds the user-defined threshold `iteration_switch`. in this case,
    the next solve will update the operator and perform a new factorization
    """

    def __init__(self, comm, PETSc_Matrix, is_symmetric = True):
        if is_symmetric:
            _direct_solver_default = {"solver": "mumps", "type": "cholesky", "bwr": True}
            _iterative_solver_default = {"solver": "cg", "maximum_iterations": 5}
        else:
            _direct_solver_default = {"solver": "mumps", "type": "lu", "bwr": True}
            _iterative_solver_default = {"solver": "gmres", "maximum_iterations": 5}

        self.direct_solver = _direct_solver_default
        self.iterative_solver = _iterative_solver_default
        self.ksp = PETSc.KSP().create(comm)
        self.pc = self.ksp.getPC()
        self.update_options()

        self.reuse_preconditioner = True
        self.ksp.setOperators(PETSc_Matrix)


    def update_options(self):
        #Par défaut il s'agit d'un solveur iteratif
        self.ksp.setType(self.iterative_solver["solver"])
        #On définit ensuite le type de pré-conditionneur.
        self.ksp.getPC().setType(self.direct_solver["type"])

        self.ksp.setFromOptions()
        
    def _update_infos(self):
        print("Converged in  {} iterations.".format(self._it))
        if self.reuse_preconditioner:
            print("Preconditioner will be reused on next solve.")
        else:
            print("Next solve will be a direct one with matrix factorization.")

    def preconditioner_choice(self):
        """
        Set precondition_reuse depending on the number of iterations.
        """
        self._it = self.ksp.getIterationNumber()
        self._converged = self.ksp.getConvergedReason()
        if self._converged > 0:
            self.reuse_preconditioner = self._it < iteration_switch
        else:
            self.reuse_preconditioner = False
            
    def update_pc(self):
        """Impose preconditioner to be reused or not."""
        self.preconditioner_choice()
        self.pc.setReusePreconditioner(self.reuse_preconditioner)
        # self._update_infos()

    def solve(self, *args):
        try:
            self.ksp.solve(*args)
        except RuntimeError:  # force direct solver if previous solve fails
            print("Error has been catched")
            #Ici on vient lui dire qu'il faudra résoudre le problème avec un solveur direct
            self.ksp.setType(PETSc.KSP.Type.PREONLY)
            self.ksp.setFromOptions()
            self.ksp.solve(*args)
            self.ksp.setType(self.iterative_solver["solver"])
            self.reuse_preconditioner = True
            self.pc.setReusePreconditioner(self.reuse_preconditioner)
            self.ksp.setFromOptions()
         
        self.update_pc()