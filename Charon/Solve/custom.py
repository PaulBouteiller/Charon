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
from petsc4py import PETSc
from dolfinx import fem
import dolfinx.fem.petsc

class CustomLinearProblem(fem.petsc.LinearProblem):
    def assemble_rhs(self, u=None):
        """Assemble right-hand side and lift Dirichlet bcs.

        Parameters
        ----------
        u : dolfinx.fem.Function, optional
            For non-zero Dirichlet bcs u_D, use this function to assemble rhs with the value u_D - u_{bc}
            where u_{bc} is the value of the given u at the corresponding. Typically used for custom Newton methods
            with non-zero Dirichlet bcs.
        """

        # Assemble rhs
        with self._b.localForm() as b_loc:
            b_loc.set(0)
        fem.petsc.assemble_vector(self._b, self._L)

        # Apply boundary conditions to the rhs
        x0 = [] if u is None else [u.x.petsc_vec]
        fem.petsc.apply_lifting(self._b, [self._a], bcs=[self.bcs], x0=x0, scale=1.0)
        self._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        x0 = None if u is None else u.x.petsc_vec
        fem.petsc.set_bc(self._b, self.bcs, x0, scale=1.0)

    def assemble_lhs(self):
        self._A.zeroEntries()
        fem.petsc.assemble_matrix_mat(self._A, self._a, bcs=self.bcs)
        self._A.assemble()

    def solve_system(self):
        # Solve linear system and update ghost values in the solution
        self._solver.solve(self._b, self._x)
        self.u.x.scatter_forward()


class CustomNewtonSolver:
    def __init__(self, tangent_problem):
        self.tangent_problem = tangent_problem
        self.du = self.tangent_problem.u

    def callback(self, u, *args):
        pass

    def solve(self, u, *args, tol=1e-8, Nitermax=200):
        self.callback(u, *args)

        self.tangent_problem.assemble_rhs()
        nRes0 = self.tangent_problem._b.norm()
        nRes = nRes0

        niter = 0
        converged = False
        while nRes > tol * nRes0 and niter < Nitermax:
            # solve for the displacement correction
            self.tangent_problem.assemble_lhs()
            self.tangent_problem.solve_system()

            # update the displacement with the current correction
            u.x.petsc_vec.axpy(1, self.du.x.petsc_vec)  # u = u + 1*du
            u.x.scatter_forward()

            self.callback(u, *args)
            # compute the new residual (pass u to handle non homogeneous bcs)
            self.tangent_problem.assemble_rhs(u=u)
            nRes = self.tangent_problem._b.norm()

            # print(nRes)
            # converged = True
            # break
            niter += 1
        else:
            converged = niter < Nitermax
        return niter, converged