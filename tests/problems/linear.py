import random

import autograd
import autograd.numpy as np
import sklearn.linear_model

from autocrit.utils import random_matrix
import autocrit.utils.math as math


class LeastSquares(object):
    EPS = 1e-20

    def __init__(self, A, b):
        self.A = A
        self.b = b

        self.grad = autograd.grad(self.loss)

        self.exact_solution = self.solve()

        assert self.loss(self.exact_solution) < self.EPS
        assert math.rms(self.grad(self.exact_solution)) < np.sqrt(self.EPS)

    def loss(self, x):
        return np.sum(np.square(np.dot(self.A, x) - self.b))

    def solve(self):
        return np.dot(np.linalg.pinv(self.A), self.b)

    @classmethod
    def generate_random_problem(cls, dim=25, eps=5e-1):
        random_psd_matrix = random_matrix.Wishart(dim, dim)
        A = np.eye(dim) + eps * random_psd_matrix.M
        b = random_matrix.generate_random_unit_vector(dim=dim)

        return cls(A, b)


class Regression(object):
    EPS = 1e-20

    def __init__(self, X, Y):
        """
        X : array, k x n input observations
        Y : array, l x n output observations

        Attributes:
        -----------
        W : array, l x k parameter matrix
        """
        self.X = X
        self.Y = Y

        self.grad = autograd.grad(self.loss)
        self.H = autograd.hessian(self.loss)

        self.exact_solution = self.solve()

        assert math.rms(self.grad(self.exact_solution)) < np.sqrt(self.EPS)

    def loss(self, W):
        return np.mean(np.square(np.dot(W.T, self.X) - self.Y))

    def solve(self):
        return np.dot(
            np.dot(
                np.linalg.pinv(np.dot(self.X, self.X.T)),
                self.X),
            self.Y.T)

    @classmethod
    def generate_random_problem(cls, k, l, n, sigma=1.):
        """Generate a random linear regression problem.

        Parameters:
        -----------

        k : int, dimension of X
        l : int, dimension of Y
        n : int, number of observations
        sigma: float, expected norm of additive noise on Y

        Returns:
        --------

        regression_problem : LinearRegressionProblem, class combining loss, data, solver
        """

        # independent gaussian vectors with approx unit norm
        X = 1 / np.sqrt(k) * np.random.standard_normal(size=(k, n))
        X -= np.mean(X, axis=1)[:, None]

        # unit norm transformation with uniform orientation
        W = np.asarray(
            [random_matrix.generate_random_unit_vector(dim=k) for _ in range(l)])\
            .T.squeeze()

        Y = np.dot(W.T, X)
        Y += np.sqrt(sigma / l) * np.random.standard_normal(size=Y.shape)

        return cls(X, Y)


class Classification(object):
    EPS = 1e-20

    def __init__(self, X, Y, Y_iis):
        """
        X : array, k x n input observations
        Y : array, l x n output label onehots
        Y_iis : array, 1 x n output label integers
        """
        self.X = X
        self.Y = Y
        self.Y_iis = Y_iis

        self.grad = autograd.grad(self.loss)
        self.H = autograd.hessian(self.loss)

        # minimally regularized LogisticRegression
        self.sklearn_model = sklearn.linear_model.LogisticRegression(
            solver="sag", fit_intercept=False, C=1e4)

        self.exact_solution = self.solve()

    def loss(self, W):
        _W = np.hstack([-W, W])
        logits = np.dot(_W.T, self.X)

        return math.softmax_cross_entropy(logits, self.Y)

    def solve(self):
        self.sklearn_model.fit(self.X.T, self.Y_iis)

        return np.atleast_2d(self.sklearn_model.coef_).T

    @classmethod
    def generate_random_problem(cls, k, m, n):
        """Generate a random linear classification // logistic regression
        problem using a mixture of Gaussians.

        Parameters:
        -----------

        k : int, dimension of X
        m : int, number of labels
        n : int, number of observations

        Returns:
        --------
        X : array, k x n input observations
        Y : array, m x n one_hot labels
        regression_problem : RegressionProblem, class combining loss, data, solver
        """

        X, Y, Y_iis, mus, covs = sample_gaussian_mixture(k, m, n)

        return cls(X, Y, Y_iis)


def sample_gaussian_mixture(k, m, n, mus=None, covs=None):
    mus = [random_matrix.generate_random_unit_vector(dim=k) for _ in range(m)]
    covs = [np.eye(k) / np.sqrt(k) for _ in range(m)]

    labels = list(range(m))

    Y_iis = []
    Y = []
    X = []
    one_hots = np.eye(m)

    for _ in range(n):
        y_ii = random.choice(labels)
        y = one_hots[y_ii]
        x = np.random.multivariate_normal(np.squeeze(mus[y_ii]), covs[y_ii])

        Y_iis.append(y_ii)
        Y.append(y)
        X.append(x)

    X = np.asarray(X).T
    Y = np.atleast_2d(np.asarray(Y)).T
    Y_iis = np.asarray(Y_iis)

    return X, Y, Y_iis, mus, covs
