"""Provides Newton-style methods for finding critical points.
"""
# import warnings

import autograd
import autograd.numpy as np

from .minresQLP import MinresQLP as mrqlp

from .base import Finder, Logger
from ..defaults import DEFAULT_STEP_SIZE, DEFAULT_RTOL, DEFAULT_MAXIT
from ..defaults import DEFAULT_ALPHA, DEFAULT_BETA, DEFAULT_GAMMAS, DEFAULT_RHO, DEFAULT_RHO_PURE

DEFAULT_ACONDLIM = 1e7
DEFAULT_MAXXNORM = 1e4
DEFAULT_TRANCOND = 1e4


class NewtonMethod(Finder):
    """Base version of Newton method for finding critical points.

    All Newton methods are run the same way: select an update direction or directions,
    and then the current value of theta and the update direction(s) are used to select an update.

    Those two steps are implemented here as the methods get_update_direction,
    which inverts the Hessian and multiplies it with the negative gradient,
    and select_update, which scales the result by the step_size.

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

    def __init__(self, f, alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA, rho=DEFAULT_RHO,
                 check_pure=False, rho_pure=DEFAULT_RHO_PURE, log_kwargs=None):
        NewtonMethod.__init__(self, f, log_kwargs=log_kwargs)
        self.alpha = alpha
        self.beta = beta
        self.rho = rho

        self.check_pure = check_pure
        self.rho_pure = rho_pure
        self.pure_accepted = False

        self.parameters.update({"alpha": self.alpha,
                                "pure_accepted": self.pure_accepted})

        self.loggers.append(
            Logger("alpha",
                   lambda step_info: step_info["parameters"]["alpha"]))

        if self.check_pure:
            self.loggers.append(
                Logger("pure_accepted",
                       lambda step_info: step_info["parameters"]["pure_accepted"]))

        self.min_step_size = self.compute_min_step_size(alpha, beta)

    def select_update(self, theta, update_direction):
        if self.check_pure and self.alpha != 1:
            converged = self.check_convergence(theta, update_direction, 1., self.rho_pure)
            if converged:
                self.alpha = 1.
                self.pure_accepted = True
        else:
            converged = False
            self.pure_accepted = False

        while not converged:
            converged = self.check_convergence(theta, update_direction, self.alpha, self.rho)

            if not converged:
                self.alpha *= self.beta
                if self.alpha <= self.min_step_size:
                    return np.zeros_like(theta)

        update = theta + self.alpha * update_direction

        self.parameters.update(
            {"alpha": self.alpha,
             "pure_accepted": self.pure_accepted})

        self.alpha = min(1., self.alpha / self.beta)
        return update

    def check_convergence(self, theta, update_direction, alpha, rho):
        proposed_update = theta + alpha * update_direction
        updated_squared_gradient_norm = self.squared_grad_norm(proposed_update)
        current_squared_gradient_norm = self.squared_grad_norm(theta)
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
                 check_pure=False, rho_pure=DEFAULT_RHO_PURE,
                 rtol=DEFAULT_RTOL, maxit=DEFAULT_MAXIT,
                 acondlim=DEFAULT_ACONDLIM, trancond=DEFAULT_TRANCOND,
                 maxxnorm=DEFAULT_MAXXNORM,
                 log_mrqlp=False, log_kwargs=None):
        NewtonBTLS.__init__(self, f, alpha, beta, rho, check_pure, rho_pure,
                            log_kwargs=log_kwargs)
        self.rtol = rtol
        self.maxit = maxit
        self.acondlim = acondlim
        self.trancond = trancond
        self.maxxnorm = maxxnorm

        self.parameters.update({"rtol": rtol,
                                "maxit": maxit,
                                "acondlim": acondlim,
                                "trancond": trancond,
                                "maxxnorm": maxxnorm})

        self.log_mrqlp = log_mrqlp

        if self.log_mrqlp:
            self.loggers.append(
                Logger("mrqlp_outputs",
                       lambda step_info: step_info["parameters"]["mrqlp_outputs"]))
            self.parameters.update({"mrqlp_outputs": None})

    def get_update_direction(self, theta):
        current_hvp = lambda v: self.hvp(theta, v)
        mrqlp_outputs = mrqlp(
            current_hvp, -1 * self.grad_f(theta),
            rtol=self.rtol, maxit=self.maxit,
            acondlim=self.acondlim, trancond=self.trancond, maxxnorm=self.maxxnorm)

        self.parameters.update({"mrqlp_outputs": mrqlp_outputs[1:]})
        mr_update_direction = mrqlp_outputs[0]

        return mr_update_direction


class FastNewtonMR(NewtonMR):
    """Newton method that uses MRQLP to approximately compute the update direction.
    Makes use of fast Hessian-vector products.
    """

    def __init__(self, f, alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA, rho=DEFAULT_RHO,
                 check_pure=False, rho_pure=DEFAULT_RHO_PURE,
                 rtol=DEFAULT_RTOL, maxit=DEFAULT_MAXIT,
                 acondlim=DEFAULT_ACONDLIM, trancond=DEFAULT_TRANCOND,
                 maxxnorm=DEFAULT_MAXXNORM,
                 log_mrqlp=False, log_kwargs=None):
        NewtonMR.__init__(self, f, alpha, beta, rho, check_pure, rho_pure,
                          rtol=rtol, maxit=maxit, acondlim=acondlim,
                          maxxnorm=maxxnorm, trancond=trancond,
                          log_mrqlp=log_mrqlp, log_kwargs=log_kwargs)
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
