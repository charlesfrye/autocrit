"""Provides a Finder that performs gradient norm minimization to find critical points.
"""
import json

import autograd
import autograd.numpy as np

from .base import Finder, Logger
from ..defaults import DEFAULT_ALPHA
from ..optimizers import GradientDescentOptimizer, MomentumOptimizer
from ..optimizers import BackTrackingLineSearchOptimizer

DEFAULT_MINIMIZER_PARAMS = {"lr": DEFAULT_ALPHA}


class GradientNormMinimizer(Finder):
    r"""Find critical points of function f by minimizing
    auxiliary function g where
    $$
    g(theta) = \frac{1]{2}\lvert\nabla f(theta)\rvert^2
    $$

    The gradient of g is the product of the hessian with the gradient.
    This can be more efficiently computed as a hessian-vector product.
    """

    def __init__(self, f, log_kwargs=None, minimizer_str="gd", minimizer_params=None):
        Finder.__init__(self, f, log_kwargs=log_kwargs)

        def g(theta):
            return 0.5 * np.sum(np.square(self.grad_f(theta)))

        self.g = g
        self.grad_g = autograd.grad(g)
        self.hvp = autograd.hessian_vector_product(self.f)
        self.fast_grad_g = lambda x: self.hvp(x, self.grad_f(x))

        self.minimizer_str = minimizer_str
        self.minimizer_params = minimizer_params or DEFAULT_MINIMIZER_PARAMS.copy()
        self.set_minimizer(minimizer_str)

    def run(self, init_theta, num_iters=1):
        theta = init_theta
        self.update_logs({"theta": theta})

        for ii in range(num_iters):
            theta_new = self.minimizer.update(theta)
            self.update_logs({"theta": theta_new})

            if np.array_equal(theta, theta_new):
                return theta

            theta = theta_new

        return theta

    def setup_log(self, track_thetas=False, track_f_thetas=False, track_g_thetas=False):

        if track_thetas:
            self.loggers.append(Logger("theta", lambda step_info: step_info["theta"]))

        if track_f_thetas:
            self.loggers.append(Logger("f_theta", lambda step_info: self.f(step_info["theta"])))

        if track_g_thetas:
            self.loggers.append(Logger("g_theta", lambda step_info: self.g(step_info["theta"])))

    def set_minimizer(self, minimizer_str):
        if minimizer_str == "gd":
            self.minimizer = GradientDescentOptimizer(
                self.g, self.fast_grad_g, **self.minimizer_params)
        elif minimizer_str == "momentum":
            self.minimizer = MomentumOptimizer(
                self.g, self.fast_grad_g, **self.minimizer_params)
        elif minimizer_str == "btls":
            self.minimizer = BackTrackingLineSearchOptimizer(
                self.g, self.fast_grad_g, **self.minimizer_params)
        else:
            raise NotImplementedError

    def to_json(self, json_path):
        dictionary = self.construct_dictionary()
        with open(json_path, "w") as fp:
            json.write(dictionary, fp)

    @classmethod
    def from_json(cls, f, json_path):
        with open(json_path) as fp:
            dictionary = json.load(fp)
        return cls(f, **dictionary)

    def construct_dictionary(self):
        dictionary = {"log_kwargs": self.log_kwargs,
                      "minimizer_str": self.minimizer_str,
                      "minimzer_params": self.minimzer_params}
        return dictionary
