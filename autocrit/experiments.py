"""Provides Experiment objects, which apply an optimization algorithm
or a critical point-finding algorithm to a function.

If the function is autograd-differentiable, the gradient oracle is
computed automatically. If that's insufficient, a gradient oracle can be
directly provided as the grad_f argument.
"""
import json
import random

import autograd
import autograd.numpy as np

from . import finders
from . import nn
from . import optimizers

SEED = 14

_NETWORK_INITS = {"fullyconnected": nn.networks.FullyConnected}

_OPTIMIZERS = {"gd": optimizers.GradientDescentOptimizer,
               "momentum": optimizers.MomentumOptimizer,
               "btls": optimizers.BackTrackingLineSearchOptimizer}

_FINDER_INITS = {"newtonMR": finders.newtons.FastNewtonMR,
                 "newtonTR": finders.newtons.FastNewtonTR,
                 "gnm": finders.gradnormmin.GradientNormMinimizer}

DEFAULT_LOG_KWARGS = {"track_theta": True, "track_f": True, "track_grad_f": False}


class Experiment(object):
    """Abstract base class for OptimizationExperiments and CritFinderExperiments.

    Concrete classes should implement a .run method that executes the experiment
    and stores the results of runs in self.runs, a list. These should be save-able
    into .npz format by np.savez.

    They should further implement a construct_dictionary method that saves
    all of the relevant arguments necessary for a constructor call as a dictionary
    that can be written to a .json file. These .json files are used to reconstruct
    experiments and their components.
    """

    def __init__(self, seed=None):
        """
        Parameters
        ----------

        seed : int or None, default is None
            Seeding value for random and np.random.
            If None, defaults to global variable SEED.
        """
        if seed is None:
            self.seed = SEED
        else:
            self.seed = seed

        self.runs = []

    def to_json(self, filename):
        dictionary = self.construct_dictionary()

        with open(filename, "w") as f:
            json.dump(dictionary, f)

    def save_results(self, filename):
        results_dict = self.runs[-1]
        np.savez(filename, **results_dict)

    def construct_dictionary(self):
        raise NotImplementedError


class OptimizationExperiment(Experiment):
    """Concrete Experiment that performs optimization on a function.
    """

    def __init__(self, f, grad_f=None, optimizer_str="gd", optimizer_kwargs=None,
                 log_kwargs=None, seed=None):
        """Create an OptimizationExperiment on callable f according to kwargs.

        Parameters
        ----------

        f : callable
            Function to optimize. Should require only parameters as input.
            For stochastic functions, e.g. for stochastic gradient descent,
            function must perform batching.

        grad_f : callable or None, default is None
            A gradient oracle for f. If None, autograd.grad is called on f.

        optimizer_str : str
            String to key into _OPTIMIZERS. Default is "gd", which is
            optimizers.gradient_descent.

        optimizer_kwargs : dict or None, default is None
            A dictionary of keyword arguments for the optimizer selected with
            optimizer_str. See optimizers for call signatures.

        log_kwargs : dict or None, default is None
            A dictionary of keyword arguments for the log_run method, which
            determines which features of the run are saved. If None,
            DEFAULT_LOG_KWARGS is used. See log_run for details.

        seed : int or None, default is None
            Seeding value for random and np.random.
            If None, defaults to global variable SEED.
        """
        Experiment.__init__(self, seed=seed)

        if log_kwargs is None:
            self.log_kwargs = DEFAULT_LOG_KWARGS.copy()
        else:
            self.log_kwargs = log_kwargs

        self.f = f
        self.grad_f = grad_f

        if self.grad_f is None:
            self.grad_f = autograd.grad(f)

        self.optimizer_str = optimizer_str
        self.optimizer = _OPTIMIZERS[self.optimizer_str]

        if optimizer_kwargs is None:
            self.optimizer_kwargs = {}
        else:
            self.optimizer_kwargs = optimizer_kwargs

        self.optimizer = _OPTIMIZERS[self.optimizer_str](
            self.f, self.grad_f, **self.optimizer_kwargs)

    def run(self, init_theta, num_iters=1, seed=None):
        """Execute optimizer on self.f, starting with init_theta, for num_iters.

        Includes optional SEED argument to allow for stochastic behavior
        of stochastic functions f.
        Warning: this does not guarantee that f is non-stochastic across calls.
        """
        if seed is None:
            seed = self.seed
        np.random.seed(seed)
        random.seed(seed)

        empty_run = {"theta": [],
                     "f_theta": [],
                     "grad_f_theta": [],
                     "g_theta": []}
        self.runs.append(empty_run)

        theta = init_theta
        self.log_step(theta, **self.log_kwargs)

        for _ in range(num_iters):
            theta = theta + self.optimizer.update(theta)
            self.log_step(theta, **self.log_kwargs)

        return theta

    def log_step(self, theta,
                 track_theta=False, track_f=False, track_grad_f=False, track_g=False):
        """Append selected values to run dictionary
        """
        run = self.runs[-1]
        if track_theta:
            run["theta"].append(theta)
        if track_f:
            run["f_theta"].append(self.f(theta))
        if track_grad_f:
            run["grad_f_theta"].append(self.grad_f(theta))
        if track_g:
            run["g_theta"].append(0.5 * np.sum(np.square(self.grad_f(theta))))

    @classmethod
    def from_json(cls, f, filename, grad_f=None):
        """Given a function and possibly a gradient oracle and the path to a .json file,
        creates an OptimizationExperiment on f using kwargs in the .json file.
        """
        with open(filename) as fn:
            dictionary = json.load(fn)

        return cls(f, grad_f, **dictionary)

    def construct_dictionary(self):
        """Construct a dictionary containing necessary information for
        reconstructing OptimizationExperiment when combined with self.f.

        See OptimizationExperiment.from_json for details.
        """
        return {"optimizer_str": self.optimizer_str,
                "optimizer_kwargs": self.optimizer_kwargs,
                "log_kwargs": self.log_kwargs,
                "seed": self.seed}


class CritFinderExperiment(Experiment):
    """Concrete Experiment that finds critical points on a function.
    """

    def __init__(self, f, finder_str, finder_kwargs=None):
        """

        Parameters
        ----------

        f : callable
            Function to search on. Should require only parameters as input.
            For stochastic functions, function must perform batching.

        finder_str : str
            String to key into _FINDER_INITS. Identifies the critical point-
            finding algorithm to use.

        finder_kwargs: dict or None, default is None
            Dictionary with keyword arguments to provide to self.finder_init.
            If None, an empty dictionary is used.

        seed : int or None, default is None
            Seeding value for random and np.random.
            If None, defaults to global variable SEED.
        """
        Experiment.__init__(self)
        self.f = f

        self.finder_str = finder_str

        if finder_kwargs is None:
            self.finder_kwargs = {}
        else:
            self.finder_kwargs = finder_kwargs

        if "log_kwargs" not in self.finder_kwargs.keys():
            self.finder_kwargs.update({"log_kwargs": DEFAULT_LOG_KWARGS.copy()})

        self.finder_init = _FINDER_INITS[self.finder_str]

        self.finder = self.finder_init(self.f, **self.finder_kwargs)

    def run(self, init_theta, num_iters=1, seed=None):
        """Execute finder on self.f, starting with init_theta, for num_iters.
        """
        if seed is None:
            seed = self.seed
        np.random.seed(seed)
        random.seed(seed)

        self.finder.log = {}
        thetas = self.finder.run(init_theta, num_iters)
        self.runs.append(self.finder.log)
        return thetas

    @classmethod
    def from_json(cls, f, filename):
        """Given a function f and the path to a .json file,
        creates a CritFinderExperiment for f using kwargs in the .json file.
        """
        with open(filename) as fn:
            dictionary = json.load(fn)

        return cls(f, **dictionary)

    def construct_dictionary(self):
        """Construct a dictionary containing necessary information for
        reconstructing CritFinderExperiment when combined with self.f.

        See CritFinderExperiment.from_json for details.
        """
        dictionary = {"finder_kwargs": self.finder_kwargs,
                      "finder_str": self.finder_str}
        return dictionary

    def uniform(self, thetas):
        """Select a theta at random from list thetas.
        """
        return random.choice(thetas)

    def uniform_f(self, thetas):
        """Select a theta from thetas uniformly across values of self.f.

        This can be slow. Overwrite this method by calling freeze_uniform_f
        if this function needs to be called multiple times.
        """
        return self.uniform_cd(*self.sort_and_calculate_cds(thetas, self.f))

    def freeze_uniform_f(self, thetas):
        """Overwrites self.uniform_f with a function that has pre-computed
        the sorted version of thetas and the cumulative densities, supporting
        much faster random selection.
        """
        sorted_thetas, cds = self.sort_and_calculate_cds(thetas, self.f)
        self.uniform_f = lambda thetas: self.uniform_cd(sorted_thetas, cds)

    @staticmethod
    def sort_and_calculate_cds(thetas, f):
        f_thetas = [f(theta) for theta in thetas]
        min_f, max_f = min(f_thetas), max(f_thetas)
        cds = [(f_theta - min_f) / (max_f - min_f) for f_theta in f_thetas]
        thetas, cds = zip(*sorted(zip(thetas, cds), key=lambda tup: tup[1]))
        return thetas, cds

    @staticmethod
    def uniform_cd(sorted_thetas, cds):
        """Select randomly from sorted_thetas with respect to the cumulative
        density implied by cds, an equal-length list of cumulative density values
        for each element in sorted_thetas.
        """
        rand_cd = random.uniform(0, 1)
        idx = next(filter(lambda tup: tup[1] >= rand_cd, enumerate(cds)))[0]
        return sorted_thetas[idx]
