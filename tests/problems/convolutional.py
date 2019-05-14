import autograd.numpy as np

from ..utils import circulant


class Classification(object):

    def __init__(self, X, Y, Y_iis):
        """
        X : array, h x w x ch x n input observations
        Y : array, l x n output label onehots
        Y_iis : array, 1 x n output label integers
        """
        self.X = X
        self.Y = Y
        self.Y_iis = Y_iis

        self.exact_solution = None

    def loss(self, soln):
        if soln is None:
            return 0
        else:
            raise ValueError(
                "loss not implemented for convolutional classification")

    @classmethod
    def generate_test_problem(cls, n, im_side=17, autocorr_scale=5.):
        """Generate a test convolutional linear classification //
        logistic regression problem using a mixture of Gaussians.

        The problem is to separate images with a pink power spectrum
        from images with a white power spectrum.

        Images are square with side length im_side and the pink noise
        images have an autocorrelation scale monotonically increasing
        with autocorr_scale.
        """
        n = n // 2

        X, Y, Y_iis = generate_convolutional_mog_data(n, im_side, autocorr_scale)

        return cls(X, Y, Y_iis)


def generate_convolutional_mog_data(n, im_side=17, autocorr_scale=5.):

    circ_cov_mat = circulant.generate_iostropic_circulant_cov_2d(
        im_side, autocorr_scale=autocorr_scale)
    circ_class_samples = circulant.rgb_gauss_random_samples(
        n, cov_or_covs=circ_cov_mat)

    white_noise_cov_mat = np.eye(im_side ** 2)
    noise_class_samples = circulant.rgb_gauss_random_samples(
        n, cov_or_covs=white_noise_cov_mat)

    # combine batch major samples from each class
    inputs = np.concatenate([circ_class_samples.T, noise_class_samples.T])

    # covert to images, then return to batch minor
    inputs = np.asarray(
        [circulant.to_im_rgb(inpt, im_side) for inpt in inputs]).T

    # generate one_hot and label vectors
    one_hots = np.hstack([np.tile(np.atleast_2d([1, 0]).T, [1, n]),
                          np.tile(np.atleast_2d([0, 1]).T, [1, n])])
    labels = np.argmax(one_hots, axis=0)

    return inputs, one_hots, labels
