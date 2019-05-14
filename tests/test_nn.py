import autograd.numpy as np
import pytest

import autocrit.nn as nn


def test_id_net():
    none_vector = np.asarray([[None]])
    id_net = nn.networks.Network(
        (none_vector, none_vector),
        [nn.layers.PointwiseNonlinearLayer("none")])

    assert id_net.forward_pass(None, np.asarray([None])) is None


def test_equivalence_fc():

    network_style, fc_style, data = make_network_and_fc_style()
    shared_theta = network_style.initialize()

    assert network_style.loss(shared_theta) == fc_style.loss(shared_theta)

    loss_val = network_style.loss(shared_theta)

    network_style_dict = network_style.construct_dict()
    fc_style_dict = fc_style.construct_dict()

    network_style_rebuild = nn.networks.Network(data, **network_style_dict)
    fc_style_rebuild = nn.networks.FullyConnected(data, **fc_style_dict)

    assert loss_val == network_style_rebuild.loss(shared_theta)
    assert loss_val == fc_style_rebuild.loss(shared_theta)


def test_regularizer_l2():
    scalmult = make_scalmult(regularizer_str="l2",
                             regularization_parameter=1.)
    theta = scalmult.initialize()
    assert scalmult.loss(theta) == np.square(theta)

    scalmult = make_scalmult(regularizer_str="l2",
                             regularization_parameter=0.5)
    theta = scalmult.initialize()
    assert scalmult.loss(theta) == 0.5 * np.square(theta)


def test_regularizer_l1():
    scalmult = make_scalmult(regularizer_str="l1",
                             regularization_parameter=1.)
    theta = scalmult.initialize()
    assert scalmult.loss(theta) == np.abs(theta)

    scalmult = make_scalmult(regularizer_str="l1",
                             regularization_parameter=-1.)
    theta = scalmult.initialize()
    assert scalmult.loss(theta) == -np.abs(theta)


def test_to_from_json(tmpdir):
    with pytest.raises(NotImplementedError):
        none_vector = np.asarray([[None]])
        lambda_id_net = nn.networks.Network(
            (none_vector, none_vector),
            [nn.layers.LambdaLayer(lambda x: x)])
        path = tmpdir.join("lambda_id_net.json")
        lambda_id_net.to_json(path)

    network_style, fc_style, data = make_network_and_fc_style()

    network_style_path = tmpdir.join("network_style.json")
    network_style.to_json(network_style_path)

    fc_style_path = tmpdir.join("fc_style.json")
    fc_style.to_json(fc_style_path)

    network_style_rebuild = nn.networks.Network.from_json(data, network_style_path)
    fc_style_rebuild = nn.networks.FullyConnected.from_json(data, fc_style_path)

    shared_theta = network_style.initialize()

    assert network_style_rebuild.loss(shared_theta) == network_style.loss(shared_theta)
    assert fc_style_rebuild.loss(shared_theta) == fc_style.loss(shared_theta)


def make_scalmult(**kwargs):
    data = (np.asarray([0]), np.asarray([0]))
    scalar_mult_layer = nn.layers.FCLayer(1, has_biases=False)
    scalmult = nn.networks.Network(
        data, layer_specs=[scalar_mult_layer],
        **kwargs)

    return scalmult


def make_network_and_fc_style():
    layer_sizes = [2, 4]
    data = (np.random.standard_normal((4, 1)), np.random.standard_normal((4, 1)))

    network_style = nn.networks.Network(
        data,
        layer_specs=[nn.layers.FCLayer(layer_size) for layer_size in layer_sizes])

    fc_style = nn.networks.FullyConnected(
        data,
        layer_sizes=layer_sizes,
        nonlinearity_str="none")

    return network_style, fc_style, data
