# flake8: noqa E221, E251
"""Adds optional torch acceleration to autograd's convolve function.
"""
import autograd.scipy.signal as _autograd_signal
from functools import partial
import autograd.numpy as np
import numpy as npo  # original numpy
from autograd.extend import primitive, defvjp

try:
    import torch
    import torch.nn.functional as torch_F
    torch_accelerated = True
except ImportError:
    torch_accelerated = False


def convolve(A, B, axes=None, dot_axes=[(), ()], mode='full', accelerated=torch_accelerated):
    args_are_implemented = check_implemented(axes, dot_axes, mode)
    if accelerated and args_are_implemented:
        return _torch_convolve(A, B, axes=axes, dot_axes=dot_axes, mode=mode)
    else:
        return _autograd_signal.convolve(A, B, axes=axes, dot_axes=dot_axes, mode=mode)


@primitive
def _torch_convolve(A, B, axes=None, dot_axes=[(), ()], mode='full'):
    B = np.ascontiguousarray(np.transpose(B[:, :, ::-1, ::-1], (1, 0, 2, 3)))
    At, Bt = torch.tensor(A), torch.tensor(B)
    if tuple(dot_axes) == ([0], [0]):
        At = torch.transpose(At, 0 ,1)
        yt = torch_F.conv2d(Bt, At)
        yt = torch.flip(torch.transpose(yt, 0, 1), (-2, -1))
    else:
        yt = torch_F.conv2d(At, Bt)
    return np.asarray(yt)


def check_implemented(axes, dot_axes, mode):
    """Check whether a fast convolution with these argument values has been implemented."""
    if tuple(axes) != ([2, 3], [2, 3]):
        return False
    if tuple(dot_axes) not in [([1], [0]), ([0], [0])]:
        return False
    if mode != "valid":
        return False

    return True


def _torch_grad_convolve(argnum, ans, A, B, axes=None, dot_axes=[(), ()], mode='full'):
    assert mode in ['valid', 'full'], "Grad for mode {0} not yet implemented".format(mode)
    axes, shapes = _autograd_signal.parse_axes(A.shape, B.shape, axes, dot_axes, mode)
    if argnum == 0:
        _, Y = A, B
        _X_, _Y_ = 'A', 'B'
        ignore_Y = 'ignore_B'
    elif argnum == 1:
        _, Y = B, A
        _X_, _Y_ = 'B', 'A'
        ignore_Y = 'ignore_A'
    else:
        raise NotImplementedError("Can't take grad of convolve w.r.t. arg {0}".format(argnum))

    if mode == 'full':
        new_mode = 'valid'
    else:
        if any([x_size > y_size for x_size, y_size
                in zip(shapes[_X_]['conv'], shapes[_Y_]['conv'])]):
            new_mode = 'full'
        else:
            new_mode = 'valid'

    def vjp(g):
        result = convolve(g, Y[tuple(_autograd_signal.flipped_idxs(Y.ndim, axes[_Y_]['conv']))],
                          axes     = [axes['out']['conv'],   axes[_Y_]['conv']],
                          dot_axes = [axes['out'][ignore_Y], axes[_Y_]['ignore']],
                          mode     = new_mode)
        new_order = npo.argsort(axes[_X_]['ignore'] + axes[_X_]['dot'] + axes[_X_]['conv'])
        return np.transpose(result, new_order)
    return vjp


defvjp(_torch_convolve, partial(_torch_grad_convolve, 0), partial(_torch_grad_convolve, 1))
