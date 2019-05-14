"""Utilities for generating gaussian random matrices
with translation-invariant covariance (toroidal boundaries)
and converting between "vector" and "image" representations of same.

Images constructed in this fashion (with or without channel dependencies)
are the natural targets of convolutional neural networks.
"""
import random

import autograd.numpy as np
import scipy

from autocrit.utils.math import rescale

EPS = 1e-3


def rgb_gauss_random_samples(N, mean_or_means=None, cov_or_covs=None, im_sz=None):
    means, covs, im_sz = _handle_kwargs(mean_or_means, cov_or_covs, im_sz)

    samples_by_channel = np.asarray(
        [np.random.multivariate_normal(mean, cov, N).T
         for mean, cov in zip(means, covs)])

    samples_batch_minor = np.moveaxis(samples_by_channel, [0, 1], [1, 0])

    return samples_batch_minor


def generate_iostropic_circulant_cov_2d(k, autocorr_scale=1.):
    """returns covariance matrix for translation-invariant,
    isotropic multivariate gaussian defined on a discrete torus
    with side length k
    """
    isotropic_circulant_1d = generate_isotropic_circulant_2d_vector(k, autocorr_scale)

    circ_mat = circulant_2d_vector_to_circulant_2d_matrix(isotropic_circulant_1d)

    # check symmetry
    assert np.array_equal(circ_mat, (circ_mat.T + circ_mat) / 2)

    # impose PSD
    cov_mat = apply_damping(circ_mat)

    return cov_mat


def to_im(vals, im_side):
    return np.reshape(vals, (im_side, im_side))


def from_im(im):
    return np.reshape(im, im.shape[0] ** 2)


def to_im_rgb(rgb_vec, im_side):
    return np.asarray([to_im(ch_vec, im_side) for ch_vec in rgb_vec])


def apply_damping(mat, eps=EPS):
    eigvals = np.linalg.eigvalsh(mat)
    damping_coeff = np.abs(min([min(eigvals) - eps, -eps]))
    damped_mat = mat + damping_coeff * np.eye(mat.shape[0])
    return damped_mat


def generate_isotropic_circulant_2d_vector(k, autocorr_scale):
    gaussian = scipy.stats.multivariate_normal(mean=[0, 0]).pdf
    xs = ys = np.linspace(-autocorr_scale, autocorr_scale, k)
    Xs, Ys = np.meshgrid(xs, ys)
    isotropic_circulant_1d = np.asarray(
        [gaussian([x, y]) for x, y in zip(Xs.flatten(), Ys.flatten())])
    isotropic_circulant_1d = np.roll(isotropic_circulant_1d,
                                     -np.argmax(isotropic_circulant_1d))
    return isotropic_circulant_1d


def circulant_2d_vector_to_circulant_2d_matrix(circulant_2d_vector):

    circulant_2d_matrix = np.asarray(
        [np.roll(circulant_2d_vector, ii)
         for ii in range(len(circulant_2d_vector))])

    return circulant_2d_matrix


def _handle_kwargs(mean_or_means, cov_or_covs, im_sz):
    kwargs = [mean_or_means, cov_or_covs, im_sz]
    assert not all([kwarg is None for kwarg in kwargs])

    if im_sz is None:
        assert not (mean_or_means is None and cov_or_covs is None)

    if cov_or_covs is not None:
        if type(cov_or_covs) is not list:
            assert isinstance(cov_or_covs, np.ndarray)
            covs = 3 * [cov_or_covs]
        else:
            covs = cov_or_covs
    else:
        covs = None

    if mean_or_means is not None:
        if type(mean_or_means) is not list:
            assert isinstance(mean_or_means, np.ndarray)
            means = 3 * [mean_or_means]
        else:
            means = mean_or_means
    else:
        means = None

    if im_sz is None:
        if covs is not None:
            im_sz = covs[0].shape[0]
        else:
            im_sz = means.shape[0]

    if means is None:
        means = 3 * [np.zeros(im_sz)]

    if covs is None:
        covs = 3 * [np.eye(im_sz)]

    return means, covs, im_sz


def display_sample_rgb(rgbs_batch_minor, ax, im_side=None):
    random_rgb_vec = random.choice(rgbs_batch_minor.T)
    assert random_rgb_vec.shape[0] == 3
    if im_side is None:
        candidate_im_side = np.sqrt(random_rgb_vec.shape[1])
        assert candidate_im_side == int(candidate_im_side)
        im_side = int(candidate_im_side)

    random_rgb_im = to_im_rgb(random_rgb_vec, im_side)
    if np.min(random_rgb_im) < 0:
        random_rgb_im = rescale(random_rgb_im)

    ax.imshow(random_rgb_im.T)
    ax.axis("off")
