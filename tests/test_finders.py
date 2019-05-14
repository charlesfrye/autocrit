import warnings

import autocrit
import autocrit.utils.random_matrix
import pytest

import tests.utils.shared as shared


def test_NewtonMethod():
    warnings.filterwarnings("ignore")
    finder = autocrit.finders.newtons.NewtonMethod
    finder_str = "NewtonMethod"

    problem_str = "least squares"
    finder_kwargs = {}
    num_iters = 1

    random_least_squares_problem, random_init = \
        shared.generate_random_least_squares()

    shared.convergence_test(finder, finder_str, finder_kwargs,
                            random_least_squares_problem.loss, random_least_squares_problem,
                            problem_str, random_init, num_iters)


def test_FastNewtonMR():
    warnings.filterwarnings("ignore")
    finder = autocrit.FastNewtonMR
    finder_str = "FastNewtonMR"

    problem_str = "least squares"
    finder_kwargs = {"alpha": 0.5, "beta": 0.99}
    num_iters = 500

    random_least_squares_problem, random_init = \
        shared.generate_random_least_squares()

    shared.convergence_test(finder, finder_str, finder_kwargs,
                            random_least_squares_problem.loss, random_least_squares_problem,
                            problem_str, random_init, num_iters)

    problem_str = "shallow regression"
    finder_kwargs = {"alpha": 0.5, "beta": 0.9, "rho": 1e-6}
    num_iters = 250

    random_regression_problem, network, random_init = \
        shared.generate_random_shallow_regression()

    shared.convergence_test(finder, finder_str, finder_kwargs,
                            network.loss, random_regression_problem, problem_str,
                            random_init, num_iters,
                            test_soln_converge=False)


def test_FastNewtonTR():
    warnings.filterwarnings("ignore")
    finder = autocrit.FastNewtonTR
    finder_str = "FastNewtonTR"

    problem_str = "least squares"
    finder_kwargs = {"step_size": 0.5}
    num_iters = 25

    random_least_squares_problem, random_init = \
        shared.generate_random_least_squares()

    shared.convergence_test(finder, finder_str, finder_kwargs,
                            random_least_squares_problem.loss, random_least_squares_problem,
                            problem_str, random_init, num_iters)

    problem_str = "shallow regression"
    finder_kwargs = {"step_size": 0.1}
    num_iters = 250

    random_regression_problem, network, random_init = \
        shared.generate_random_shallow_regression()

    shared.convergence_test(finder, finder_str, finder_kwargs,
                            network.loss, random_regression_problem, problem_str,
                            random_init, num_iters,
                            test_soln_converge=False)


@pytest.mark.slow
def test_deep_classification():
    warnings.filterwarnings("ignore")

    problem_str = "deep classification"

    finder = autocrit.FastNewtonMR
    finder_str = "FastNewtonMR"

    finder_kwargs = {"alpha": 0.1, "beta": 0.9, "rho": 1e-6}
    num_iters = 250

    random_classification_problem, network, random_init = \
        shared.generate_random_deep_classification()

    shared.convergence_test(finder, finder_str, finder_kwargs,
                            network.loss, random_classification_problem, problem_str,
                            random_init, num_iters,
                            test_function_converge=False,
                            test_soln_converge=False)

    finder = autocrit.FastNewtonTR
    finder_str = "FastNewtonTR"

    finder_kwargs = {"step_size": 0.05}
    num_iters = 250

    random_classification_problem, network, random_init = \
        shared.generate_random_deep_classification(seed=shared.SEED + 1)

    shared.convergence_test(finder, finder_str, finder_kwargs,
                            network.loss, random_classification_problem, problem_str,
                            random_init, num_iters,
                            test_function_converge=False,
                            test_soln_converge=False)


def test_GradientNormMinimizer():
    warnings.filterwarnings("ignore")
    finder = autocrit.GradientNormMinimizer
    finder_str = "GradientNormMinimizer"

    problem_str = "least squares"
    finder_kwargs = {"minimizer_str": "momentum",
                     "minimizer_params": {"lr": 5e-3,
                                          "momentum": 1 - 1e-10}}
    num_iters = 1000

    random_least_squares_problem, random_init = \
        shared.generate_random_least_squares()

    shared.convergence_test(finder, finder_str, finder_kwargs,
                            random_least_squares_problem.loss, random_least_squares_problem,
                            problem_str, random_init, num_iters)
