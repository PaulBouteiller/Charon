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
Created on Mon Mar 27 09:38:24 2023

@author: bouteillerp
"""

from petsc4py.PETSc import InsertMode, ScatterMode
from dolfinx.fem import form, assemble_scalar
from dolfinx.fem.petsc import (create_matrix, create_vector, assemble_vector,\
                               apply_lifting, set_bc, assemble_matrix)

class NLProblem:
    def __init__(self, F, J, u, bcs): 
        self.L = form(F)
        self.a = form(J)
        self.bcs = bcs
        self.u = u
        
        # Create matrix and vector to be used for assembly of the non-linear problem
        self.A = create_matrix(self.a)
        self.b = create_vector(self.L)

    def F(self, PETSc_object, x, b):
        """Assemble the residual F into the vector b. 

        Parameters
        ==========
        PETSc_object: PETSc.SNES or PETSc.TAO, the snes or tao object
        x: PETSc.Vec Vector containing the latest solution.
        b: PETSc.Vec Vector to assemble the residual into.
        """
        # We need to assign the vector to the function

        x.ghostUpdate(addv = InsertMode.INSERT, mode = ScatterMode.FORWARD)
        # self.u.vector.ghostUpdate(addv=InsertMode.INSERT, mode = ScatterMode.FORWARD)
        x.copy(self.u.vector)
        self.u.vector.ghostUpdate(addv=InsertMode.INSERT, mode = ScatterMode.FORWARD)
        
        # Zero the residual vector
        with b.localForm() as b_local:
            b_local.set(0.0)
        assemble_vector(b, self.L)
        
        # Apply boundary conditions
        apply_lifting(b, [self.a], [self.bcs], [x], -1.0)
        b.ghostUpdate(addv=InsertMode.ADD,mode = ScatterMode.REVERSE)
        set_bc(b, self.bcs, x, -1.0)     
        
    def J(self, PETSc_object, x, A, P):
        """Assemble the Jacobian matrix.

        Parameters
        ==========
        x: PETSc.Vec Vector containing the latest solution.
        A: PETSc.Mat Matrix to assemble the Jacobian into.
        P : PETSc.Mat
        """
        A.zeroEntries()
        assemble_matrix(A, self.a, self.bcs)
        A.assemble()
        
class SNESProblem(NLProblem):
    """Nonlinear problem class compatible with PETSC.SNES solver.
    """

    def __init__(self, F, J, u, bcs):
        """This class set up structures for solving a non-linear problem using Newton's method.
"""
        NLProblem.__init__(self, F, J, u, bcs)
        
        
        
class TAOProblem(NLProblem):
    """Nonlinear problem class compatible with PETSC.TAO solver.
    """

    def __init__(self, f, F, J, u, bcs):
        """This class set up structures for solving a non-linear problem using Newton's method.

        Parameters
        ==========
        f: ufl.form.Form Objective.
        F: ufl.form.Form Residual.
        J: ufl.form.Form Jacobian.
        u: dolfinx.fem.Function Solution.
        bcs: List[dolfinx.fem.dirichletbc] Dirichlet boundary conditions.
        """
        self.obj = form(f)
        NLProblem.__init__(self, F, J, u, bcs)
        
    def f(self, PETSc_object, x):
        """Assemble the objective f. 

        Parameters
        ==========
        PETSc_object: the tao object
        x: PETSc.Vec Vector containing the latest solution.
        """

        """Assemble residual vector."""
        x.ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)
        x.copy(self.u.vector)
        self.u.vector.ghostUpdate(addv=InsertMode.INSERT, mode=ScatterMode.FORWARD)
        return assemble_scalar(self.obj)