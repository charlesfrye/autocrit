"""Optimization algorithms using zeroth- and first-order oracles.

Includes gradient descent, momentum, and backtracking line search,
using either the standard Wolfe criterion or the Roosta criterion from
the paper on Newton-MR.
"""
import autograd
import autograd.numpy as np

from autocrit.defaults import DEFAULT_ALPHA, DEFAULT_MOMENTUM
from autocrit.defaults import DEFAULT_BETA, DEFAULT_GAMMA, DEFAULT_RHO


class FirstOrderOptimizer(object):
    """Abstract Base Class for optimizers with a zeroth- and first-order oracle.

    If no first-order oracle is provided, it is computed from the zeroth-order
    oracle with autograd."""

    def __init__(self, f, grad_f):
        self.f = f
        if grad_f is None:
            self.grad_f = autograd.grad(f)
        else:
            self.grad_f = grad_f

    def run(self, init, num_iters):
        solution = np.copy(init)

        for _ in range(num_iters):
            solution += self.update(solution)

        return solution


class GradientDescentOptimizer(FirstOrderOptimizer):
    """FirstOrderOptimizer that uses scaled gradients to update."""

    def __init__(self, f, grad_f=None, lr=DEFAULT_ALPHA):
        super().__init__(f, grad_f)
        self.lr = lr

    def update(self, theta):
        return -self.lr * self.grad_f(theta)


class MomentumOptimizer(FirstOrderOptimizer):
    """FirstOrderOptimizer that maintains a 'velocity' term in addition to scaled gradients.

    If initial velocity is not provided in init_velocity, starts at 0.
    """

    def __init__(self, f, grad_f=None, lr=DEFAULT_ALPHA, momentum=DEFAULT_MOMENTUM,
                 init_velocity=None):
        super().__init__(f, grad_f)
        self.lr = lr
        self.momentum = momentum

        self.velocity = init_velocity

    def update(self, theta):
        if self.velocity is None:
            self.velocity = np.zeros_like(theta)
        self.velocity = self.grad_f(theta) + self.momentum * self.velocity
        update = -self.lr * self.velocity

        return update


class BackTrackingLineSearchOptimizer(FirstOrderOptimizer):
    """FirstOrderOptimizer that uses line search over the gradient direction.

    Can use either the traditional Wolfe criterion for terminating the line search
    or the new critertion from Roosta et al., 2018.
    """

    def __init__(self, f, grad_f=None, hvp=None,
                 alpha=DEFAULT_ALPHA, beta=DEFAULT_BETA,
                 rho=DEFAULT_RHO, gamma=None,
                 criterion="wolfe"):

        super().__init__(f, grad_f)
        self.set_criterion(criterion, gamma)

        self.alpha = alpha
        self.beta = beta
        self.rho = rho

        self.min_step_size = self.compute_min_step_size(self.alpha, self.beta)

        if hvp is None:
            self.hvp = autograd.hessian_vector_product(self.f)
        else:
            self.hvp = hvp

    def update(self, theta):
        update_direction = -self.grad_f(theta)
        converged = self.check_convergence(theta, update_direction)
        while not converged:
            self.alpha *= self.beta
            if self.alpha <= self.min_step_size:
                return np.zeros_like(theta)
            converged = self.check_convergence(theta, update_direction)
        step = self.alpha * update_direction
        self.alpha /= self.beta
        return step

    def set_criterion(self, criterion_str, gamma):
        self.criterion_str = criterion_str

        if self.criterion_str is None:
            return

        if self.criterion_str == "roosta":
            self.check_convergence = self.roosta_criterion
        elif self.criterion_str == "wolfe":
            self.check_convergence = self.wolfe_criterion
            if gamma is None:
                self.gamma = DEFAULT_GAMMA
            else:
                self.gamma = gamma
        else:
            raise NotImplementedError

    def roosta_criterion(self, theta, update_direction):
        proposed_update = theta + self.alpha * update_direction
        updated_f = self.f(proposed_update)
        current_f = self.f(theta)

        sufficient_decrease = 2 * self.rho * self.alpha * np.dot(
            self.hvp(theta, update_direction).T, self.grad_f(theta))

        return (updated_f <=
                current_f + sufficient_decrease)

    def wolfe_criterion(self, theta, update_direction):
        proposed_update = theta + self.alpha * update_direction
        updated_f = self.f(proposed_update)
        current_f = self.f(theta)

        current_grad = self.grad_f(theta)
        grad_update_product = np.dot(update_direction.T, current_grad)

        new_grad = self.grad_f(proposed_update)
        new_grad_update_product = np.dot(update_direction.T, new_grad)

        passed_armijo = updated_f <= current_f + self.rho * self.alpha * grad_update_product

        passed_curvature = -new_grad_update_product <= -self.gamma * grad_update_product

        return passed_armijo and passed_curvature

    @staticmethod
    def compute_min_step_size(alpha, beta):
        while alpha * beta != alpha:
            alpha *= beta
        return alpha
