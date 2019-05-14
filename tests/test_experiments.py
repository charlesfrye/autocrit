import autograd.numpy as np

import autocrit

import tests.utils.shared as shared


def test_OptimizationExperiment(tmpdir):
    """test saving, execution, and loading for default kwargs
    """
    num_iters = 10

    _, network, init_theta = shared.generate_random_shallow_regression()

    experiment = autocrit.OptimizationExperiment(network.loss)

    outfile = tmpdir / "optexpt.json"
    experiment_test(experiment, network.loss,
                    init_theta, num_iters, outfile)


def test_CritFinderExperiment(tmpdir):
    """test saving, execution, and loading for default kwargs
    """
    num_iters = 10

    _, network, init_theta = shared.generate_random_shallow_regression()

    experiment = autocrit.CritFinderExperiment(network.loss, "newtonMR")

    outfile = tmpdir / "cfexpt.json"
    experiment_test(experiment, network.loss,
                    init_theta, num_iters, outfile)


def experiment_test(experiment, f, init_theta, num_iters, outfile):
    thetas = experiment.run(init_theta, num_iters=num_iters)

    experiment.to_json(outfile)

    reloaded_expt = experiment.from_json(f, outfile)
    assert experiment.construct_dictionary() == reloaded_expt.construct_dictionary()

    reloaded_thetas = reloaded_expt.run(init_theta, num_iters=num_iters)
    assert np.array_equal(thetas, reloaded_thetas)
