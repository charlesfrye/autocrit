"""Provides Newton-style methods for finding critical points.
"""
import warnings

import autograd
import autograd.numpy as np

from .minresQLP import MinresQLP as mrqlp

from .base import Finder
from ..defaults import DEFAULT_STEP_SIZE, DEFAULT_RTOL, DEFAULT_MAXIT
from ..defaults import DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_GAMMAS, DEFAULT_RHO


class NewtonMethod(Finder):
    """Base version of Newton method for finding critical points.

    All Newton methods are run the same way: select an update direction or directions,
    and then the current value of theta and the update direction(s) are used to select an update.

    Additional Newton methods are defined by over-riding those two methods.
    """

    def __init__(self, f, step_size=DEFAULT_STEP_SIZE, log_kwargs=None):
        Finder.__init__(self, f, log_kwargs=log_kwargs)

        self.step_size = step_size

        self.parameters = {"step_size": step_size}

    def run(self, init_theta, num_iters=1):
        theta = init_theta
        self.update_logs({"theta": theta,
                          "update_direction": None,
                          "parameters": self.parameters})

        for ii in range(num_iters):

            update_direction = self.get_update_direction(theta)
            theta_new = self.select_update(theta, update_direction)

            self.update_logs({"theta": theta_new,
                              "update_direction": update_direction,
                              "parameters": self.parameters})

            if np.array_equal(theta, theta_new):
                return theta

            theta = theta_new

        return theta

    def get_update_direction(self, theta):
        """Compute an update direction using the classic Newton-Raphson method:
        compute the Hessian at theta, invert it explicitly, and then apply that matrix
        to the negative gradient.
        """
        update_direction = -np.linalg.inv(self.H(theta)).dot(self.grad_f(theta))
        return update_direction

    def select_update(self, theta, update_direction):
        """Select the update along update direction using a fixed step size.
        """
        return theta + self.step_size * update_direction

    def squared_grad_norm(self, theta):
        return np.sum(np.square(self.grad_f(theta)))


class NewtonPI(NewtonMethod):
    """Newton method that uses Moore-Penrose pseudo-inversion of the Hessian instead of
    classic inversion, for use in problems with singular Hessians.
    """

    def __init__(self, f, step_size=DEFAULT_STEP_SIZE, log_kwargs=None):
        NewtonMethod.__init__(self, f, step_size=step_size, log_kwargs=log_kwargs)
        self.pinv = np.linalg.pinv

    def get_update_direction(self, theta):
        update_direction = -self.pinv(self.H(theta)).dot(self.grad_f(theta))
        return update_direction


class NewtonBTLS(NewtonMethod):
    """Newton method that uses back-tracking line search to select the update.
    Convergence is checked using the Roosta criterion.
    """

    def __init__(self, f, alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA, rho=DEFAULT_RHO, log_kwargs=None):
        NewtonMethod.__init__(self, f, log_kwargs=log_kwargs)
        self.alpha = alpha
        self.beta = beta
        self.rho = rho

        self.parameters.update({"alpha": alpha,
                                "beta": beta,
                                "rho": rho})
        self.min_step_size = self.compute_min_step_size(alpha, beta)

    def select_update(self, theta, update_direction):
        converged = self.check_convergence(theta, update_direction, self.alpha, self.rho)
        while not converged:
            self.alpha *= self.beta
            if self.alpha <= self.min_step_size:
                return np.zeros_like(theta)
            converged = self.check_convergence(theta, update_direction, self.alpha, self.rho)
        update = theta + self.alpha * update_direction
        self.alpha /= self.beta
        return update

    def check_convergence(self, theta, update_direction, alpha, rho):
        proposed_update = theta + alpha * update_direction
        updated_squared_gradient_norm = self.squared_grad_norm(self.grad_f(proposed_update))
        current_squared_gradient_norm = self.squared_grad_norm(self.grad_f(theta))
        sufficient_decrease = 2 * rho * alpha * np.dot(self.hvp(theta, update_direction).T,
                                                       self.grad_f(theta))

        return (updated_squared_gradient_norm <=
                current_squared_gradient_norm + sufficient_decrease)

    @staticmethod
    def compute_min_step_size(alpha, beta):
        while alpha * beta != alpha:
            alpha *= beta
        return alpha


class NewtonMR(NewtonBTLS):
    """Newton method that uses MRQLP to approximately compute the update direction
    and back-tracking line search to select the update.
    """

    def __init__(self, f, alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA, rho=DEFAULT_RHO,
                 rtol=DEFAULT_RTOL, maxit=DEFAULT_MAXIT,
                 log_kwargs=None):
        NewtonBTLS.__init__(self, f, alpha, beta, rho, log_kwargs=log_kwargs)
        self.rtol = rtol
        self.maxit = maxit

        self.parameters.update({"rtol": rtol,
                                "maxit": maxit})

    def get_update_direction(self, theta):
        current_hvp = lambda v: self.hvp(theta, v)
        mr_update_direction = mrqlp(current_hvp, -1 * self.grad_f(theta),
                                    rtol=self.rtol, maxit=self.maxit)[0]
        return mr_update_direction


class FastNewtonMR(NewtonMR):
    """Newton method that uses MRQLP to approximately compute the update direction.
    Makes use of fast Hessian-vector products.
    """

    def __init__(self, f, alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA, rho=DEFAULT_RHO,
                 rtol=DEFAULT_RTOL, maxit=DEFAULT_MAXIT,
                 log_kwargs=None):
        NewtonMR.__init__(self, f, alpha, beta, rho, rtol=rtol, maxit=maxit, log_kwargs=log_kwargs)
        self.hvp = autograd.hessian_vector_product(self.f)


class NewtonTR(NewtonPI):
    """Newton method that computes a sequence of proposed updates using the pseudo-inverse of
    a sequence of perturbed versions of the Hessian. The perturbations are diagonal matrices with
    varying values gamma. Equivalent to a trust region approach.
    """

    def __init__(self, f, gammas=DEFAULT_GAMMAS, step_size=DEFAULT_STEP_SIZE, log_kwargs=None):
        NewtonPI.__init__(self, f, step_size=step_size, log_kwargs=log_kwargs)
        self.gammas = gammas
        self.Hs = [lambda theta: self.H(theta) + np.diag([gamma] * theta.shape[0])
                   for gamma in gammas]

        self.parameters.update({"gammas": gammas})

    def get_update_direction(self, theta):
        update_directions = []

        for H in self.Hs:
            update_directions.append(-self.pinv(H(theta))
                                     .dot(self.grad_f(theta)))

        return update_directions

    def select_update(self, theta, update_directions):
        best_update = theta
        best_grad_norm = self.squared_grad_norm(best_update)
        for update_direction in update_directions:
            proposed_update = theta + self.step_size * update_direction
            if self.squared_grad_norm(proposed_update) < best_grad_norm:
                best_update = proposed_update

        return best_update


class FastNewtonTR(NewtonTR):
    """Newton method that computes a sequence of proposed updates by applying MRQLP to
    a sequence of perturbed versions of the Hessian. The perturbations are diagonal matrices with
    varying values gamma. Equivalent to a trust region approach.
    Makes use of fast Hessian-vector products.
    """

    def __init__(self, f, gammas=DEFAULT_GAMMAS, step_size=DEFAULT_STEP_SIZE, log_kwargs=None,
                 rtol=DEFAULT_RTOL, maxit=DEFAULT_MAXIT):
        NewtonTR.__init__(self, f, gammas, step_size=step_size, log_kwargs=log_kwargs)
        self.rtol = rtol
        self.maxit = maxit

        self.hvps = [lambda theta, v: autograd.hessian_vector_product(self.f)(theta, v) +
                     np.sum(gamma * theta) for gamma in gammas]

    def get_update_direction(self, theta):
        update_directions = []
        current_hvps = [lambda v: hvp(theta, v) for hvp in self.hvps]

        for current_hvp in current_hvps:
            mr_update_direction = mrqlp(current_hvp, -1 * self.grad_f(theta),
                                        rtol=self.rtol, maxit=self.maxit)[0]
            update_directions.append(mr_update_direction)

        return update_directions
