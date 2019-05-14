import random

import autograd
import autograd.numpy as np

import autocrit.nn as nn
from autocrit.utils.random_matrix import generate_random_unit_vector
import autocrit.utils.math as math
import tests.problems.linear as linear
import tests.problems.convolutional as convolutional

SEED = 14

CRITERION_STRS = ["gradient norm", "function val", "solution val"]
CRITERION_VALS = [1e-5, 5e-5, 1e-5]

DIM = 25

K = 10
L = 5
M = 2
N = 100

FAIL_MSG = "{0} failed to converge on {1} in {2}:\n\t{3} > {4}"


def convergence_test(algorithm, algorithm_str, algorithm_kwargs,
                     loss, problem, problem_str, init, num_iters,
                     test_function_converge=True, test_soln_converge=True):

    _, _, errors = evaluate(algorithm, loss, problem, init, num_iters,
                            kwargs=algorithm_kwargs,
                            calc_func_error=test_function_converge,
                            calc_soln_rms_error=test_soln_converge)

    for criterion_str, error_val, criterion_val in zip(CRITERION_STRS, errors, CRITERION_VALS):
        assert_pass(algorithm_str, problem_str, criterion_str, error_val, criterion_val)


def evaluate(algorithm_constructor, loss,
             problem, init, num_iters,
             kwargs=None,
             calc_func_error=True,
             calc_soln_rms_error=True):
    if kwargs is None:
        kwargs = {}

    algorithm = algorithm_constructor(loss, **kwargs)
    solution = algorithm.run(init, num_iters)
    exact_solution = problem.exact_solution

    grad_rms_error = math.rms(autograd.grad(loss)(solution))
    errors = [grad_rms_error]

    if calc_func_error:
        func_error = loss(solution) - problem.loss(exact_solution)
        errors.append(func_error)

    if calc_soln_rms_error:
        soln_rms_error = math.rms(solution.ravel() - exact_solution.T.ravel())
        errors.append(soln_rms_error)

    return solution, exact_solution, errors


def assert_pass(algorithm_str, problem_str, criterion_str, error_val, criterion_val):
    fail_msg = FAIL_MSG.format(
        algorithm_str, problem_str, criterion_str, error_val, criterion_val)

    assert error_val <= criterion_val, fail_msg


def generate_random_least_squares(dim=DIM, seed=SEED):
    np.random.seed(seed)
    random_least_squares_problem = linear.LeastSquares.\
        generate_random_problem(dim=dim)
    random_init = generate_random_unit_vector(dim=dim)

    return random_least_squares_problem, random_init


def generate_random_shallow_regression(k=K, l=L, n=N, seed=SEED):
    np.random.seed(seed)
    random_regression_problem = linear.Regression.\
        generate_random_problem(k=k, l=l, n=n)

    shallow_network = nn.networks.FullyConnected(
        (random_regression_problem.X, random_regression_problem.Y),
        layer_sizes=[l],
        nonlinearity_str="none",
        has_biases=False)

    random_init = shallow_network.initialize()

    return random_regression_problem, shallow_network, random_init


def generate_random_shallow_classification(k=K, m=M, n=N, seed=SEED):
    np.random.seed(seed)
    random.seed(seed)
    random_classification_problem = linear.Classification.\
        generate_random_problem(k=k, m=m, n=n)

    shallow_network = nn.networks.FullyConnected(
        (random_classification_problem.X, random_classification_problem.Y),
        layer_sizes=[m],
        nonlinearity_str="none",
        has_biases=False,
        cost_str="softmax_cross_entropy")

    random_init = shallow_network.initialize()

    return random_classification_problem, shallow_network, random_init


def generate_random_deep_classification(k=K, m=M, n=N, seed=SEED):
    np.random.seed(seed)
    random.seed(seed)
    random_classification_problem = linear.Classification.\
        generate_random_problem(k=k, m=m, n=n)

    p = min(k, m)
    deep_network = nn.networks.FullyConnected(
        (random_classification_problem.X, random_classification_problem.Y),
        layer_sizes=[p, m],
        nonlinearity_str="none",
        has_biases=False,
        regularizer_str="l2",
        regularization_parameter=0.1,
        cost_str="softmax_cross_entropy")

    random_init = deep_network.initialize()

    return random_classification_problem, deep_network, random_init


def generate_test_conv_classification(n=N, seed=SEED):
    np.random.seed(seed)
    random.seed(seed)
    test_classification_problem = convolutional.Classification.\
        generate_test_problem(n=n)

    conv_network = nn.networks.Network(
        (test_classification_problem.X, test_classification_problem.Y),
        layer_specs=[nn.layers.ConvLayer((4, 4), 2),
                     nn.layers.MaxPoolLayer((2, 2)),
                     nn.layers.GlobalAvgPoolLayer(),
                     nn.layers.SqueezeLayer()],
        cost_str="softmax_cross_entropy")

    random_init = conv_network.initialize()

    return test_classification_problem, conv_network, random_init
