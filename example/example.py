#!/usr/bin/env python
import numpy as np
import openmdao.api as om
import os
import sys
import time

from scipy.optimize import rosen

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

if not MPI:
    run_parallel = False
    rank = 0
else:
    run_parallel = True
    rank = MPI.COMM_WORLD.rank


class Rosen(om.ExplicitComponent):
    """OpenMDAO Component representation of the Rosenbrock function."""

    def _declare_options(self):
        super()._declare_options()
        self.options.declare('sleep_time', default=1e-4, lower=0)
        self.options.declare('dim', default=2, lower=1)
        self.options.declare('nan_points', default=None, types=list, allow_none=True)
        self.options.declare('nan_range', default=1e-4, lower=0)

    def setup(self):
        self.add_input('x', shape=self.options['dim'])
        self.add_output('f')

    def compute(self, inputs, outputs):
        time.sleep(self.options['sleep_time'])

        nan_points = self.options['nan_points']
        if nan_points is not None:
            for nan_point in nan_points:
                if np.linalg.norm(inputs['x'] - np.asarray(nan_point)) <= self.options['nan_range']:
                    outputs['f'] = 1e27
                    return

        outputs['f'] = rosen(inputs['x'])


def solve(dim=2, bits=31, n_nan_points=100, nan_range=5e-2, plot=False):
    """Solve a modified Rosenbrock optimization problem using a genetic algorithm and a specified number of bits for x.

    Parameters
    ----------
    dim : int
        Dimensionality of the Rosenbrock problem. Defaults is 2.
    bits : int
        Number of bits to use to encode each dimension of the design variable, x.
        Note: Values of bits > 31 result in exceptions due to Numpy's integer representation as 32-bit integers.
    n_nan_points : int
        Number of regions where the Rosenbrock function is NaN to insert at random per dimension. Default is 100.
    nan_range : float
        Distance from NaN points at which function is considered NaN. Default is 1e4.
    plot : bool
        True if a contour plot should be made. Only matters if dim == 2.

    Returns
    -------
    x : numpy.ndarray
        Solution vector
    f : float
        Value of Rosenbrock at x
    dt : float
        Wall-clock time it took to perform the optimization in seconds
    """
    nan_points = []
    while len(nan_points) < n_nan_points * dim:
        nan_point = -2. + np.random.rand(dim) * 4.
        if np.linalg.norm(nan_point - np.ones(dim)) > nan_range:
            nan_points += [nan_point]

    x0 = -2 + np.random.rand(dim) * 4.

    prob = om.Problem()
    prob.model.add_subsystem('coor', om.IndepVarComp('x', x0), promotes=['*'])
    prob.model.add_subsystem('obj', Rosen(dim=dim, nan_points=nan_points, nan_range=nan_range), promotes=['*'])

    prob.model.add_design_var('x', lower=-2, upper=2)
    prob.model.add_objective('f')

    prob.driver = om.SimpleGADriver(bits={'x': bits}, run_parallel=run_parallel)

    # prob.model.approx_totals()
    # prob.driver = om.ScipyOptimizeDriver(optimizer='SLSQP', debug_print=['objs', 'desvars'], disp=True)

    prob.set_solver_print(2)
    prob.setup()
    prob.model.obj.options['sleep_time'] = 0.
    prob.run_model()

    t0 = time.time()
    prob.run_driver()
    dt = time.time() - t0

    res = [np.copy(prob['x']), np.copy(prob['f'])[0], dt, None, None, None]

    if plot and dim == 2 and rank == 0:
        prob.model.obj.options['sleep_time'] = 0.

        n_contour = 100
        x = np.linspace(-2, 2, n_contour)
        x, y = np.meshgrid(x, x)

        f = np.zeros_like(x)
        for i in range(n_contour):
            for j in range(n_contour):
                prob['x'] = [x[i, j], y[i, j]]
                prob.run_model()
                f[i, j] = prob['f'][0]

        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        fig, ax = plt.subplots()
        cs = ax.contourf(x, y, f, levels=np.logspace(-1, np.log10(3000)), norm=LogNorm())
        ax.plot([x0[0], res[0][0], 1], [x0[1], res[0][1], 1], 'ro-')
        res[3:] = [fig, ax, cs]

    prob.cleanup()
    return tuple(res)


def main():
    """Interpret console arguments, call solve, and display optimization results."""
    if len(sys.argv) != 3:
        dim, bits = 2, 31
    else:
        dim, bits = int(sys.argv[1]), int(sys.argv[2])

    plot = True if os.environ.get('PLOT_CONTOUR') else False
    x, f, dt, fig, ax, cs = solve(dim, bits, plot=plot)
    if rank == 0:
        print(x, f, dt)

        if plot:
            fig.show()


if __name__ == '__main__':
    main()
