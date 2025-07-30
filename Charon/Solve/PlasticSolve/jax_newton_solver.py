

import jax
import jax.numpy as jnp
from functools import partial


def newton_solve(x, r, dr_dx, rtol, atol, niter_max):
    res = r(x)
    norm_res0 = jnp.linalg.norm(res)

    def cond_fun(state):
        norm_res, niter, _ = state
        return jnp.logical_and(
            jnp.logical_and(norm_res > atol, norm_res > rtol * norm_res0),
            niter < niter_max
        )

    def body_fun(state):
        norm_res, niter, (x, res) = state
        dx = jnp.linalg.solve(dr_dx(x), -res)
        x += dx
        res = r(x)
        return jnp.linalg.norm(res), niter + 1, (x, res)

    _, niter, (x_sol, res_sol) = jax.lax.while_loop(
        cond_fun, body_fun, (norm_res0, 0, (x, res))
    )
    
    return x_sol, (niter, norm_res0, jnp.linalg.norm(res_sol), res_sol)


class JAXNewton:
    def __init__(self, rtol=1e-8, atol=1e-8, niter_max=2000):
        self.rtol = rtol
        self.atol = atol
        self.niter_max = niter_max

    def set_residual(self, r, dr_dx=None):
        if isinstance(r, (list, tuple)):
            self.r = lambda x: jnp.concatenate([jnp.atleast_1d(ri(x)) for ri in r])
        else:
            self.r = r
        self.dr_dx = jax.jacfwd(self.r) if dr_dx is None else dr_dx

    @partial(jax.jit, static_argnums=(0,))
    def solve(self, x):
        solve = lambda f, x: newton_solve(x, f, jax.jacfwd(f), self.rtol, self.atol, self.niter_max)
        tangent_solve = lambda g, y: jnp.linalg.solve(jax.jacfwd(g)(y), y)
        return jax.lax.custom_root(self.r, x, solve, tangent_solve, has_aux=True)