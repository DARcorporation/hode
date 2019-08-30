#!/usr/bin/env python
import numpy as np
import openmdao.api as om
import sys
import time

from mpi4py import MPI
from scipy.optimize import rosen

if not MPI:
    run_parallel = False
    rank = 0
else:
    run_parallel = True
    rank = MPI.COMM_WORLD.rank


def slow_rosen(x):
    """Slow Rosenbrock function.

    Parameters
    ----------
    x : array_like
        1-D array of points at which the Rosenbrock function is to be computed.

    Returns
    -------
    f : float
        The value of the Rosenbrock function.

    See Also
    --------
    scipy.optimize.rosen
    """
    time.sleep(0.0001)
    return rosen(x)


class Rosen(om.ExplicitComponent):
    """OpenMDAO Component representation of the Rosenbrock function."""

    def _declare_options(self):
        super()._declare_options()
        self.options.declare('dim', default=2, lower=1)

    def setup(self):
        self.add_input('x', shape=self.options['dim'])
        self.add_output('f')

    def compute(self, inputs, outputs):
        outputs['f'] = slow_rosen(inputs['x'])


def solve(dim=2, bits=31):
    """Solve the Rosenbrock optimization problem using a genetic algorithm and a specified number of bits for x.

    Parameters
    ----------
    dim : int
        Dimensionality of the Rosenbrock problem. Defaults is 2.
    bits : int
        Number of bits to use to encode each dimension of the design variable, x.
        Note: Values of bits > 31 result in exceptions due to Numpy's integer representation as 32-bit integers.

    Returns
    -------
    x : numpy.ndarray
        Solution vector
    f : float
        Value of Rosenbrock at x
    dt : float
        Wall-clock time it took to perform the optimization in seconds
    """
    prob = om.Problem()
    prob.model.add_subsystem('coor', om.IndepVarComp('x', -2 + np.random.rand(dim) * 4.), promotes=['*'])
    prob.model.add_subsystem('obj', Rosen(dim=dim), promotes=['*'])

    prob.model.add_design_var('x', lower=-2, upper=2)
    prob.model.add_objective('f')

    prob.driver = om.SimpleGADriver(bits={'x': bits}, run_parallel=run_parallel)

    prob.set_solver_print(2)
    prob.setup()
    prob.run_model()

    t0 = time.time()
    prob.run_driver()
    dt = time.time() - t0

    res = prob['x'], prob['f'][0], dt
    prob.cleanup()
    return res


def main():
    """Interpret console arguments, call solve, and display optimization results."""
    if len(sys.argv) != 3:
        dim, bits = 2, 31
    else:
        dim, bits = int(sys.argv[1]), int(sys.argv[2])

    x, f, dt = solve(dim, bits)
    if rank == 0:
        print(x, f, dt)


if __name__ == '__main__':
    main()
