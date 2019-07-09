"""Provides neural networks composed of layers from nn.layers module.

Generic sequential networks are implemented by the Network class,
while the traditional fully-connected network is provided by FullyConnected.
"""
from collections import namedtuple
import json

import autograd
from autograd import numpy as np

from . import layers as nn_layers
from autocrit.utils import random_matrix
from autocrit.utils import math

_LAYERS = nn_layers._LAYERS

_COSTS = {"mean_squared_error": math.mean_squared_error,
          "softmax_cross_entropy": math.softmax_cross_entropy}


def l2_regularizer(theta):
    return np.mean(np.square(theta))


def l1_regularizer(theta):
    return np.mean(np.abs(theta))


_REGULARIZERS = {"l2": l2_regularizer,
                 "l1": l1_regularizer,
                 "none": lambda x: 0.}


Data = namedtuple("Data", ['x', 'y'])


class Network(object):

    def __init__(self, data, layer_specs, cost_str="mean_squared_error",
                 regularizer_str="none", regularization_parameter=0.,
                 batch_size=None):
        if not isinstance(data, Data):
            try:
                data = Data(x=data[0], y=data[1])
            except IndexError:
                raise("data argument not understood")

        self.data = data

        if batch_size is None:
            self.batch_size = self.data.x.shape[-1]
        else:
            self.batch_size = batch_size

        self.cost_str = cost_str
        self.regularizer_str = regularizer_str

        self.cost = _COSTS[self.cost_str]
        self.regularizer = _REGULARIZERS[self.regularizer_str]

        self.regularization_parameter = regularization_parameter

        self.layer_specs = layer_specs
        self.layers = []
        for layer_spec in self.layer_specs:
            if not isinstance(layer_spec, nn_layers.Layer):
                layer_constructor = _LAYERS[layer_spec["type"]]
                layer = layer_constructor(**layer_spec["params"])
            else:
                layer = layer_spec
            self.layers.append(layer)

        self.N_params, _ = self.build()

        self.grad = autograd.grad(self.loss)
        self.hess = autograd.hessian(self.loss)

    def loss(self, theta):
        return self.loss_on_batch(self.data.x, self.data.y, theta)

    def loss_on_batch(self, batch_x, batch_y, theta):
        return (self.cost(self.forward_pass(batch_x, theta), batch_y) +
                self.regularization_parameter * self.regularizer(theta))

    def loss_on_random_batch(self, theta, batch_size=None):
        """Loss on a randomly selected batch of size batch_size.
        Defaults to self.batch_size, which itself defaults to full-batch.
        """
        if batch_size is None:
            batch_size = self.batch_size
        dataset_size = self.data.x.shape[-1]

        if dataset_size == batch_size:
            batch_x, batch_y = self.data.x, self.data.y
        else:
            batch_idxs = np.random.choice(dataset_size, size=batch_size)
            batch_x, batch_y = self.data.x[..., batch_idxs], self.data.y[..., batch_idxs]

        return self.loss_on_batch(batch_x, batch_y, theta)

    def forward_pass(self, x, theta):
        y = x
        for layer in self.layers:
            params = self.parser.get(theta, layer)
            y = layer.forward_pass(y, params)
        return y

    def build(self):
        self.parser = nn_layers.ParamParser()

        shape = self.data.x.shape
        for layer in self.layers:
            N_params, shape = layer.build(shape)
            self.parser.add_params(layer, (N_params))

        return self.parser.N, shape

    def to_json(self, filename):
        dictionary = self.construct_dict()
        with open(filename, "w") as f:
            json.dump(dictionary, f)

    @classmethod
    def from_json(cls, data, filename):
        with open(filename) as f:
            dictionary = json.load(f)
        return cls(data, **dictionary)

    def construct_dict(self):
        self.layer_dicts = [layer.to_dict() for layer in self.layers]

        return {"layer_specs": self.layer_dicts,
                "cost_str": self.cost_str,
                "regularizer_str": self.regularizer_str,
                "regularization_parameter": self.regularization_parameter,
                "batch_size": self.batch_size}

    def initialize(self):
        return 1 / np.sqrt(self.N_params)  * np.random.standard_normal(size=[self.N_params, 1])


class FullyConnected(Network):

    def __init__(self, data, layer_sizes, cost_str="mean_squared_error", nonlinearity_str="relu",
                 regularizer_str="none", regularization_parameter=0., has_biases=True,
                 batch_size=None):
        self.layer_sizes = layer_sizes
        self.has_biases = has_biases
        self.nonlinearity_str = nonlinearity_str
        layers = []
        for layer_size in self.layer_sizes:
            assert isinstance(layer_size, int)
            layers.append(nn_layers.FCLayer(layer_size, self.has_biases))
            layers.append(_LAYERS["pointwise_nonlinear"](self.nonlinearity_str))

        if self.has_biases:
            self.num_biases = sum(self.layer_sizes)
        else:
            self.num_biases = 0

        Network.__init__(self, data, layers, cost_str, regularizer_str,
                         regularization_parameter, batch_size=batch_size)

    def initialize(self, weight_kwargs=None, bias_kwargs=None):
        if weight_kwargs is None:
            weight_kwargs = {}
        if bias_kwargs is None:
            bias_kwargs = {}

        init_weights = self.initialize_weights(**weight_kwargs)
        init_biases = self.initialize_biases(**bias_kwargs)

        return np.atleast_2d(np.concatenate([init_weights, init_biases])).T

    def initialize_weights(self):
        in_sizes = [self.data.x.shape[0]] + self.layer_sizes[:-1]
        out_sizes = self.layer_sizes
        weight_matrices = [self.initialize_weight_matrix(in_size, out_size)
                           for in_size, out_size in zip(in_sizes, out_sizes)]

        return np.concatenate([weight_matrix.ravel()
                               for weight_matrix in weight_matrices])

    def initialize_biases(self, constant=0.01):
        return np.asarray([constant] * self.num_biases)

    def initialize_weight_matrix(self, in_size, out_size):
        weight_matrix = np.asarray([random_matrix.generate_random_unit_vector(dim=in_size)
                                    for _ in range(out_size)]).squeeze()
        return weight_matrix

    def construct_dict(self):

        return {"layer_sizes": self.layer_sizes,
                "cost_str": self.cost_str,
                "nonlinearity_str": self.nonlinearity_str,
                "regularizer_str": self.regularizer_str,
                "regularization_parameter": self.regularization_parameter,
                "has_biases": self.has_biases}
