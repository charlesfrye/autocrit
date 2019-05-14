"""Shared default values for numerical constants/hyperparameters
"""
# ALPHA: learning rate
DEFAULT_ALPHA = 0.1
# BETA: learning rate decrease factor in BTLS
DEFAULT_BETA = 0.5
# RHO: scaling factor for Armijo/sufficient decrease criterion
DEFAULT_RHO = 1e-4
# GAMMA: scaling factor for Wolfe/sufficient curvature decrease criterion
DEFAULT_GAMMA = 0.9
# GAMMAS: "nudge" to add to diagonal in NewtonTR
DEFAULT_GAMMAS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6)

# MOMENTUM: momentum coefficient for MomentumOptimizers
DEFAULT_MOMENTUM = 0.9

# MINIMIZER_PARAMS: default parameters for GNM minimizer
DEFAULT_MINIMIZER_PARAMS = {"lr": DEFAULT_ALPHA}

# STEP_SIZE: "learning rate" for Newton methods
DEFAULT_STEP_SIZE = 1.0

# RTOL, MAXIT: parameters for MRQLP step of NewtonMR
DEFAULT_RTOL = 1e-10
DEFAULT_MAXIT = 25
