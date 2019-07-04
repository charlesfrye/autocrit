import warnings

import autograd
import autograd.numpy as np

import autocrit.nn.conv

CONV_KWARGS = {"axes": ([2, 3], [2, 3]),
               "dot_axes": ([1], [0]),
               "mode": "valid"}


def test_accelerated_equivalence():
    warnings.filterwarnings("ignore")
    batch = 10
    in_ch = 3
    out_ch = 16
    k_size = 3

    X = np.random.randn(batch, in_ch, 32, 32)
    w = np.random.randn(out_ch, in_ch, k_size, k_size)
    w = np.ascontiguousarray(np.transpose(w, (1, 0, 2, 3)))

    y = autocrit.nn.conv.convolve(X, w, accelerated=False, **CONV_KWARGS)
    accelerated_y = autocrit.nn.conv.convolve(X, w, accelerated=True, **CONV_KWARGS)

    loss_grads = loss_grad(X, w)
    accelerated_loss_grads = accelerated_loss_grad(X, w)

    assert np.allclose(y, accelerated_y),\
        "accelerated output not equal to autograd output"
    assert np.allclose(loss_grads, accelerated_loss_grads),\
        "accelerated gradients not equal to autograd gradients"


def accelerated_loss(X, w):
    activations = autocrit.nn.conv.convolve(X, w, accelerated=True, **CONV_KWARGS)
    squared_activations = np.square(activations)
    return np.mean(squared_activations)


def loss(X, w):
    activations = autocrit.nn.conv.convolve(X, w, accelerated=False, **CONV_KWARGS)
    squared_activations = np.square(activations)
    return np.mean(squared_activations)


accelerated_loss_grad = autograd.grad(accelerated_loss, argnum=1)
loss_grad = autograd.grad(loss, argnum=1)
