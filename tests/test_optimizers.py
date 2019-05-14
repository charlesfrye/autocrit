"""Tests GradientDescentOptimizer, MomentumOptimizer, and BacktrackingLineSearchOptimizer
for convergence in function value, gradient norm, and solution value for linear least squares
linear regression, and linear classification with a shallow network.
Additionally tests MomentumOptimizer on a linear convolutional problem.
"""
import warnings

import autocrit
import autocrit.utils.random_matrix

import tests.utils.shared as shared


def test_GradientDescentOptimizer():
    optimizer = autocrit.optimizers.GradientDescentOptimizer
    optimizer_str = "GradientDescentOptimizer"

    problem_str = "least squares"
    optimizer_kwargs = {}
    num_iters = 1000

    random_least_squares_problem, random_init = \
        shared.generate_random_least_squares()

    shared.convergence_test(optimizer, optimizer_str, optimizer_kwargs,
                            random_least_squares_problem.loss, random_least_squares_problem,
                            problem_str, random_init, num_iters)

    problem_str = "shallow regression"
    optimizer_kwargs = {}
    num_iters = 10000

    random_regression_problem, network, random_init = \
        shared.generate_random_shallow_regression()

    shared.convergence_test(optimizer, optimizer_str, optimizer_kwargs,
                            network.loss, random_regression_problem, problem_str,
                            random_init, num_iters)

    problem_str = "shallow classification"
    optimizer_kwargs = {}
    num_iters = 12500

    random_classification_problem, network, random_init = \
        shared.generate_random_shallow_classification()

    shared.convergence_test(optimizer, optimizer_str, optimizer_kwargs,
                            network.loss, random_classification_problem, problem_str,
                            random_init, num_iters, test_soln_converge=False)


def test_MomentumOptimizer():
    warnings.filterwarnings("ignore")
    optimizer = autocrit.optimizers.MomentumOptimizer
    optimizer_str = "MomentumOptimizer"

    problem_str = "least squares"
    optimizer_kwargs = {}
    num_iters = 1000

    random_least_squares_problem, random_init = \
        shared.generate_random_least_squares()

    shared.convergence_test(optimizer, optimizer_str, optimizer_kwargs,
                            random_least_squares_problem.loss, random_least_squares_problem,
                            problem_str, random_init, num_iters)

    problem_str = "shallow regression"
    optimizer_kwargs = {}
    num_iters = 1000

    random_regression_problem, network, random_init = \
        shared.generate_random_shallow_regression()

    shared.convergence_test(optimizer, optimizer_str, optimizer_kwargs,
                            network.loss, random_regression_problem, problem_str,
                            random_init, num_iters)

    problem_str = "shallow classification"
    optimizer_kwargs = {}
    num_iters = 1000

    random_classification_problem, network, random_init = \
        shared.generate_random_shallow_classification()

    shared.convergence_test(optimizer, optimizer_str, optimizer_kwargs,
                            network.loss, random_classification_problem, problem_str,
                            random_init, num_iters, test_soln_converge=False)

    problem_str = "convolutional classification"
    optimizer_kwargs = {"momentum": 0.99}
    num_iters = 1000

    test_classification_problem, network, random_init = \
        shared.generate_test_conv_classification()

    shared.convergence_test(optimizer, optimizer_str, optimizer_kwargs,
                            network.loss, test_classification_problem, problem_str,
                            random_init, num_iters, test_soln_converge=False)


def test_BackTrackingLineSearchOptimizer(dim=25):
    optimizer = autocrit.optimizers.BackTrackingLineSearchOptimizer
    optimizer_str = "BackTrackingLineSearchOptimizer"

    problem_str = "least squares"
    optimizer_kwargs = {}
    num_iters = 1000

    random_least_squares_problem, random_init = \
        shared.generate_random_least_squares()

    shared.convergence_test(optimizer, optimizer_str, optimizer_kwargs,
                            random_least_squares_problem.loss, random_least_squares_problem,
                            problem_str, random_init, num_iters)

    problem_str = "shallow regression"
    optimizer_kwargs = {"gamma": 1 - 1e-3}
    num_iters = 100

    random_regression_problem, network, random_init = \
        shared.generate_random_shallow_regression()

    shared.convergence_test(optimizer, optimizer_str, optimizer_kwargs,
                            network.loss, random_regression_problem, problem_str,
                            random_init, num_iters)

    problem_str = "shallow classification"
    optimizer_kwargs = {"gamma": 1 - 1e-3}
    num_iters = 100

    random_classification_problem, network, random_init = \
        shared.generate_random_shallow_classification()

    shared.convergence_test(optimizer, optimizer_str, optimizer_kwargs,
                            network.loss, random_classification_problem, problem_str,
                            random_init, num_iters, test_soln_converge=False)
