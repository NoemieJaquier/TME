#! /usr/bin/env python
import numpy as np


def multi_variate_normal(x, mu, sigma=None, log=True, inv_sigma=None):
	"""
	Multivariatve normal distribution PDF

	:param x: data, np.array([nb_samples, nb_dim])
	:param mu: mean, np.array([nb_dim])
	:param sigma: covariance matrix, np.array([nb_dim, nb_dim])
	:param log: compute the loglikelihood if true
	:return: pdf of the data for the given Gaussian distribution
	"""
	dx = x - mu
	if sigma.ndim == 1:
		sigma = sigma[:, None]
		dx = dx[:, None]
		inv_sigma = np.linalg.inv(sigma) if inv_sigma is None else inv_sigma
		log_lik = -0.5 * np.sum(np.dot(dx, inv_sigma) * dx, axis=1) - 0.5 * np.log(np.linalg.det(2 * np.pi * sigma))
	else:
		inv_sigma = np.linalg.inv(sigma) if inv_sigma is None else inv_sigma
		log_lik = -0.5 * np.einsum('...j,...j', dx, np.einsum('...jk,...j->...k', inv_sigma, dx)) - 0.5 * np.log(np.linalg.det(2 * np.pi * sigma))

	return log_lik if log else np.exp(log_lik)
