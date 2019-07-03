# modified from code in autograd/examples/convnet.py

import autograd.numpy as np

from autocrit.nn.conv import convolve
from autocrit.utils import math

_NONLINEARITIES = {"relu": math.relu,
                   "sigmoid": math.sigmoid,
                   "softplus": math.softplus,
                   "swish": math.swish,
                   "none": lambda x: x}


class ParamParser(object):
    """A helper class to index into a parameter vector."""
    def __init__(self):
        self.idxs_and_shapes = {}
        self.N = 0

    def add_params(self, name, shape):
        start = self.N
        self.N += np.prod(shape)
        self.idxs_and_shapes[name] = (slice(start, self.N), shape)

    def get(self, vect, name):
        idxs, shape = self.idxs_and_shapes[name]
        return np.reshape(vect[idxs], shape)


class Layer(object):
    """A Layer implements two methods:
    forward_pass, which takes inputs and a parameter vector and returns outputs,
    and build, which takes the input_shape and computes the number of
    parameters and the shape of the outputs, optionally also
    using a ParamsParser to track those parameters.
    """

    def __init__(self):
        pass

    def to_batch_major(self, inputs):
        """Reorder [y, x, channels, batch]
           to      [batch, channels, y, x]
        """
        return np.moveaxis(inputs, [0, 1, 2, 3], [2, 3, 1, 0])

    def to_batch_minor(self, inputs):
        """Reorder [batch, channels, y, x]
           to      [y, x, channels, batch]
        """
        return np.moveaxis(inputs, [0, 1, 2, 3], [3, 2, 0, 1])

    def forward_pass(self, inputs, theta):
        raise NotImplementedError

    def build(self, input_shape):
        raise NotImplementedError

    def to_dict(self, str, params):
        """Convert Layer to a dictionary representation.
        """
        return {"type": str, "params": params}


class PointwiseNonlinearLayer(Layer):
    """Layer for applying the same nonlinear function to each node,
    aka pointwise.

    Any callable can be provided as the nonlinearity, but the layer
    can only be represented by a dictionary if the nonlinearity is provided
    as a string, used to key into the _NONLINEARITIES dictionary.
    """
    str = "pointwise_nonlinear"

    def __init__(self, nonlinearity):
        """
        Parameters
        ----------
        nonlinearity: str or callable. pointwise nonlinear transformation.
            if is a str instance, used to key into _NONLINEARITIES dictionary.
            if is callable, directly called as function applied by this layer.
            it is assumed but not checked that this function doesn't change the shape.
        """
        if isinstance(nonlinearity, str):
            self.nonlinearity_str = nonlinearity
            nonlinearity = _NONLINEARITIES[nonlinearity]
        else:
            assert callable(nonlinearity)
        self.nonlinearity = nonlinearity

    def forward_pass(self, inputs, theta):
        return self.nonlinearity(inputs)

    def build(self, input_shape):
        return 0, input_shape

    def to_dict(self):
        assert hasattr(self, "nonlinearity_str"), "can't save nonlinear layer without str"
        params = {"nonlinearity": self.nonlinearity_str}
        return super().to_dict(self.str, params)


class FCLayer(Layer):
    """Layer for applying an affine transformation to the inputs.
    """
    str = "fc"

    def __init__(self, out_nodes, has_biases=True):
        """
        Parameters
        ----------
        out_nodes: int, number of nodes in the output layer.
        has_biases: bool, if False, linear transform. otherwise affine.
        """
        self.out_nodes = out_nodes
        self.has_biases = has_biases

    def forward_pass(self, inputs, theta):
        W = self.parser.get(theta, 'weights')
        if self.has_biases:
            b = self.parser.get(theta, 'biases')
        else:
            b = 0.
        activations = np.dot(W, inputs) + b
        return activations

    def build(self, input_shape):
        self.parser = ParamParser()
        self.parser.add_params('weights', (self.out_nodes, input_shape[0]))
        if self.has_biases:
            self.parser.add_params('biases', (self.out_nodes, 1))
        output_shape = (self.out_nodes, 1)

        return self.parser.N, output_shape

    def to_dict(self):
        params = {"out_nodes": self.out_nodes,
                  "has_biases": self.has_biases}
        return super().to_dict(self.str, params)


class ConvLayer(Layer):
    """Layer for applying a valid 2D convolution to inputs.
    """
    str = "conv"

    def __init__(self, kernel_shape, out_channels):
        """
        Parameters
        ----------
        kernel_shape: tuple of ints, shape of convolutional kernel
        out_channels: int, number of output channels aka convolutional kernels
        """
        self.kernel_shape = kernel_shape
        self.out_channels = out_channels

    def forward_pass(self, inputs, theta):
        weights = self.parser.get(theta, 'weights')
        biases = self.parser.get(theta, 'biases')
        inputs = self.to_batch_major(inputs)
        conv = convolve(inputs, weights,
                        axes=([2, 3], [2, 3]), dot_axes=([1], [0]),
                        mode='valid')
        activations = conv + biases
        activations = self.to_batch_minor(activations)
        return activations

    def build(self, input_shape):
        self.parser = ParamParser()
        self.parser.add_params('weights', (input_shape[-2], self.out_channels) +
                               self.kernel_shape)
        self.parser.add_params('biases', (1, self.out_channels, 1, 1))
        output_shape = self.conv_output_shape(input_shape[:-1], self.kernel_shape) +\
            (self.out_channels, input_shape[-1])
        return self.parser.N, output_shape

    def conv_output_shape(self, A, B):
        return (A[0] - B[0] + 1, A[1] - B[1] + 1)

    def to_dict(self):
        params = {"kernel_shape": self.kernel_shape,
                  "out_channels": self.out_channels}
        return super().to_dict(self.str, params)


class PoolLayer(Layer):
    """Abstract class for Layers that applying pooling: summarizing
    a block of values in a feature map with a single number.

    Pooling shapes must evenly tile inputs.
    """

    def __init__(self, pool_shape):
        """
        Parameters
        ----------
        pool_shape: tuple of ints, shape of pooling kernel
        """
        self.pool_shape = pool_shape

    def forward_pass(self, inputs, theta):
        patches = self.to_patches(inputs)
        patch_means = self.pool_func(patches)
        patch_means = self.to_batch_minor(patch_means)
        return patch_means

    def build(self, input_shape):
        output_shape = self.set_output_shapes(input_shape)
        return 0, output_shape

    def to_patches(self, inputs):
        self.set_patch_shapes(inputs.shape)
        channels, batch = inputs.shape[2:]
        inputs = self.to_batch_major(inputs)

        patched_shape = inputs.shape[:2]
        for patch_ct, pool_shape in zip(self.patch_yx, self.pool_shape):
            patched_shape += (patch_ct, pool_shape)

        patches = inputs.reshape(patched_shape)

        return patches

    def set_patch_shapes(self, input_shape):
        self.input_yx = input_shape[:2]
        self.patch_yx = np.floor_divide(self.input_yx, self.pool_shape)
        self.num_patches = np.prod(self.patch_yx)

    def set_output_shapes(self, input_shape):
        self.output_shape = list(input_shape)
        for i in [0, 1]:
            assert input_shape[i] % self.pool_shape[i] == 0, \
                "pool shape should tile input exactly"
            self.output_shape[i] = input_shape[i] // self.pool_shape[i]
        return self.output_shape

    def pool_func(self, patches):
        return patches

    def to_dict(self, str, params):
        return super().to_dict(str, params)


class AvgPoolLayer(PoolLayer):
    """Applies an average pooling: computes mean of elements in pool kernel.

    Pooling kernel shape must evenly tile inputs.
    """
    str = "avg_pool"

    def __init__(self, pool_shape):
        """
        Parameters
        ----------
        pool_shape: tuple of ints, shape of pooling kernel
        """
        super().__init__(pool_shape)

    def pool_func(self, patches):
        return np.mean(np.mean(patches, axis=3), axis=4)


class MaxPoolLayer(PoolLayer):
    """Applies maximum-based pooling: computes max of elements in pool kernel.

    Pooling kernel shape must evenly tile inputs.
    """
    str = "max_pool"

    def __init__(self, pool_shape):
        """
        Parameters
        ----------
        pool_shape: tuple of ints, shape of pooling kernel
        """
        super().__init__(pool_shape)

    def pool_func(self, patches):
        return np.max(np.max(patches, axis=3), axis=4)

    def to_dict(self):
        params = {"pool_shape": self.pool_shape}
        return super().to_dict(self.str, params)


class GlobalAvgPoolLayer(AvgPoolLayer):
    """Applies global average pooling: computes the average of the
    entire feature map.

    Typically used as the last transformation before classification
    in an all-convolutional classification network.
    """
    str = "global_avg_pool"

    def __init__(self):
        pass

    def build(self, input_shape):
        super().__init__(input_shape[:2])
        output_shape = self.set_output_shapes(input_shape)
        return 0, output_shape

    def to_dict(self):
        params = {}
        return super().to_dict(self.str, params)


class SqueezeLayer(Layer):
    """Removes "dummy" singleton axes from shapes.
    """
    str = "squeeze"

    def __init__(self, squeeze_axes=(0, 1)):
        """
        Parameters
        ----------

        squeeze_axes: tuple of ints, axes to remove
        """
        super().__init__()
        self.squeeze_axes = squeeze_axes

    def build(self, input_shape):
        output_shape = [input_shape[i] for i in range(len(input_shape))
                        if i not in self.squeeze_axes]
        return 0, output_shape

    def forward_pass(self, inputs, theta):
        for axis in self.squeeze_axes:
            assert inputs.shape[axis] == 1
        return np.squeeze(inputs, axis=self.squeeze_axes)

    def to_dict(self):
        params = {"squeeze_axes": self.squeeze_axes}
        return super().to_dict(self.str, params)


class LambdaLayer(Layer):
    """Layer for arbitrary functional transformations.

    Cannot be represented by a dictionary.
    """
    str = "lambda"

    def __init__(self, lam, shape_calculator=lambda shape: shape):
        """
        Parameters
        ----------
        lam: callable. Functional transformation to apply.
        shape_calculator: callable. Computes output shape from input shape.
            Defaults to assuming shape does not change.
        """
        super().__init__()
        self.lam = lam
        self.shape_calculator = shape_calculator

    def build(self, input_shape):
        output_shape = self.shape_calculator(input_shape)
        return 0, output_shape

    def forward_pass(self, inputs, theta):
        return self.lam(inputs)

    def to_dict(self):
        raise NotImplementedError("cannot convert LambdaLayer to dict")


_layer_list = [PointwiseNonlinearLayer,
               FCLayer,
               ConvLayer,
               AvgPoolLayer,
               MaxPoolLayer,
               GlobalAvgPoolLayer,
               SqueezeLayer,
               LambdaLayer]

_LAYERS = {layer.str: layer for layer in _layer_list}
