import autograd.numpy as np

XH_EPS = 1e-25
RESCALE_EPS = 1e-6


def rms(arr):
    return np.sqrt(np.mean(np.square(arr)))


def relu(x):
    return np.where(x > 0., x, 0.)


def softplus(x, lam=5.):
    return 1 / lam * np.log(1 + np.exp(lam * x))


def sigmoid(x):
    return np.where(x >= 0, _positive_sigm(x), _negative_sigm(x))


def swish(x):
    return np.multiply(x, sigmoid(x))


def _negative_sigm(x):
    expon = np.exp(-x)
    return 1 / (1 + expon)


def _positive_sigm(x):
    expon = np.exp(x)
    return expon / (1 + expon)


def mean_squared_error(x, xhat):
    return np.mean(np.square(x - xhat))


def softmax_cross_entropy(l, p):
    phat = softmax(l)
    return np.mean(cross_entropy(p, phat))


def softmax(x):
    expon = np.exp(x - np.max(x, axis=0))
    return expon / np.sum(expon, axis=0)


def cross_entropy(ps, qs, eps=XH_EPS):
    return np.einsum("ij,ij->j", ps, -np.log(qs + eps))


def logits_to_labels(logits):
    return np.argmax(logits, axis=0)


def accuracy(yhats, ys):
    return np.mean(yhats == ys)


def assess_accuracy(network, theta, X, Y_iis):
    logits = network.forward_pass(X, theta)
    labels = logits_to_labels(logits)
    return accuracy(labels, Y_iis)


def pointwise_nonlinearity(parameters, x, nonlinearity):
    W, b = parameters
    return nonlinearity(np.dot(W, x) + b)


def cossim(x, y):
    return np.dot(x.T, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def rescale(arr, eps=RESCALE_EPS):
    return (arr - np.min(arr)) / max((np.max(arr) - np.min(arr)), eps)
