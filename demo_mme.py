#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from utils.logistic_regression_utils import sigmoid
from utils.tensor_utils import khatriRaoProd
from utils.regression_methods import RidgeRegression, MatrixRidgeRegression, MixtureLinearExperts, \
	MaxtrixMixtureLinearExperts

'''
This example generates data according to matrix mixture of experts model based on matrix coefficients.
The recovery of these coefficients is tested with Ridge regression, tensor Ridge regression, a mixture of experts model 
and a matrix mixture of experts model.

The user can:
	1. Change the coefficient shapes.
	2. Change the number of training data.
	3. Change the noise in the generated data.
	4. Change the rank of the TRR and TME models.

Author: Noémie Jaquier, 2018
'''

if __name__ == '__main__':

	print('Generating data...')
	# ## Read images
	img = mpimg.imread('data/dummy_square')
	square = np.copy(img[:, :, 0])
	square[square < 200] = 0
	square[square > 0] = 1

	img = mpimg.imread('data/dummy_circle')
	circle = np.copy(img[:, :, 0])
	circle[circle < 200] = 0
	circle[circle > 0] = 1

	img = mpimg.imread('data/dummy_triangle')
	triangle = np.copy(img[:, :, 0])
	triangle[triangle < 200] = 0
	triangle[triangle > 0] = 1

	img = mpimg.imread('data/dummy_cross')
	cross = np.copy(img[:, :, 0])
	cross[cross < 200] = 0
	cross[cross > 0] = 1

	img = mpimg.imread('data/dummy_tshape')
	tshape = np.copy(img[:, :, 0])
	tshape[tshape < 200] = 0
	tshape[tshape > 0] = 1

	img = mpimg.imread('data/dummy_star')
	star = np.copy(img[:, :, 0])
	star[star < 100] = 0
	star[star > 0] = 1

	# Display coefficients
	plt.imshow(square, cmap='Greys')
	frame1 = plt.gca()
	frame1.axes.get_xaxis().set_ticks([])
	frame1.axes.get_yaxis().set_ticks([])
	plt.show()
	plt.imshow(circle, cmap='Greys')
	frame1 = plt.gca()
	frame1.axes.get_xaxis().set_ticks([])
	frame1.axes.get_yaxis().set_ticks([])
	plt.show()
	plt.imshow(triangle, cmap='Greys')
	frame1 = plt.gca()
	frame1.axes.get_xaxis().set_ticks([])
	frame1.axes.get_yaxis().set_ticks([])
	plt.show()
	plt.imshow(cross, cmap='Greys')
	frame1 = plt.gca()
	frame1.axes.get_xaxis().set_ticks([])
	frame1.axes.get_yaxis().set_ticks([])
	plt.show()
	plt.imshow(tshape, cmap='Greys')
	frame1 = plt.gca()
	frame1.axes.get_xaxis().set_ticks([])
	frame1.axes.get_yaxis().set_ticks([])
	plt.show()
	plt.imshow(star, cmap='Greys')
	frame1 = plt.gca()
	frame1.axes.get_xaxis().set_ticks([])
	frame1.axes.get_yaxis().set_ticks([])
	plt.show()

	# ## Generate parameters
	# - bVec
	bVec = [circle.flatten(), tshape.flatten()]  # The chosen shapes can be changed
	# - phipsiVec
	phipsiVec = [cross.flatten()]  # The chosen shapes can be changed

	# ## Generate data
	# Parameters
	Ndata = 500  # The number of data can be changed
	Nclass = len(bVec)
	Ndim = 1
	noise_level = 0.1  # The noise level can be changed

	d1 = square.shape[0]
	d2 = square.shape[1]
	N = Ndata * Nclass
	x = np.random.randn(N, d1*d2)  # random input

	prior = np.zeros((N, Nclass))
	prior = sigmoid(np.sum(phipsiVec * x, axis=1))  # generate prior
	prior = np.vstack((prior, 1-prior))

	y = np.zeros(N)
	for i in range(Nclass):
		y += prior[i] * np.sum(bVec[i] * x, axis=1)  # generate output
	noise = noise_level * np.std(y)
	y += noise * np.random.randn(N)  # add noise to the output
	y = y[:, None]

	y_class = prior.T

	X = np.reshape(x, (N, d1, d2))  # reshape input in matrix form

	# ## Test different methods
	print('Coefficient recovery:')
	# Models parameters MRR
	mrr_rank = 2  # The rank can be changed
	# Models parameters MME
	mme_rank_e = 2  # The rank can be changed
	mme_rank_g = 2  # The rank can be changed

	# Ridge regression
	print('Ridge regression...')
	rr = RidgeRegression()
	rr.training(x, y)
	rr_coeffs = np.reshape(rr.beta, (d1, d2))

	# Matrix ridge regression
	print('Matrix Ridge regression...')
	mrr = MatrixRidgeRegression(mrr_rank)
	mrr.training(X, y, reg=1e-1)
	mrr_coeffs = np.reshape(mrr.bVec[0], (d1, d2))

	# Mixture of experts
	print('Mixture of experts...')
	me = MixtureLinearExperts()
	me_LL = me.training(x, y, y_class, maxiter=20, optmethod='CG', reg=1e-1)
	me_coeffs = []
	me_coeffs_lr = []
	for i in range(0, me.nb_class):
		me_coeffs.append(np.reshape(me.W[i], (d1, d2)))
		me_coeffs_lr.append(np.reshape(me.V[:, i], (d1, d2)))

	rmse_me_coeffs_rr = np.sqrt(np.sum([(np.sum((bVec[i] - me_coeffs[i].flatten()) ** 2))
										for i in range(Nclass)]) / (Nclass * d1 * d2))
	rmse_me_coeffs_lr = np.sqrt(np.sum(np.sum((phipsiVec - me_coeffs_lr[0].flatten()) ** 2)) / (d1 * d2))

	tot_rmse = np.sum([(np.sum((bVec[i] - me_coeffs[i].flatten()) ** 2))
					   for i in range(Nclass)]) + np.sum((np.sum((phipsiVec - me_coeffs_lr[0].flatten()) ** 2)))
	tot_rmse /= Nclass * d1 * d2 + d1 * d2
	rmse_me_coeffs = tot_rmse

	# Matrix-valued mixture of experts
	print('Matrix-valued mixture of experts...')
	mme = MaxtrixMixtureLinearExperts([mme_rank_e, mme_rank_e], mme_rank_g)
	mme_LL = mme.training(X, y, y_class, reg_rr=1e-1, reg_lr=1e-1, maxiter=20, max_diff_ll=5.0, optmethod='CG')
	mme_coeffs = []
	mme_coeffs_lr = []
	for i in range(0, mme.nb_class):
		alpha = mme.alpha[i][:]
		b1tmp = mme.b1[i][:]
		b2tmp = mme.b2[i][:]
		mme_bVec = [np.dot(khatriRaoProd(b2tmp[j], b1tmp[j]), np.ones((mme.rank_e[i], 1))) for j in range(mme.nb_dim)]
		mme_coeffs.append(np.reshape(mme_bVec[0], (d1, d2)))
		mme_phipsiVec = np.dot(khatriRaoProd(mme.psi[i], mme.phi[i]), np.ones((mme.rank_g, 1)))
		mme_coeffs_lr.append(np.reshape(mme_phipsiVec, (d1, d2)))

	rmse_mme_coeffs_rr = np.sqrt(np.sum([(np.sum((bVec[i] - mme_coeffs[i].flatten()) ** 2))
										 for i in range(Nclass)]) / (Nclass * d1 * d2))

	rmse_mme_coeffs_lr = np.sqrt(np.sum((np.sum((phipsiVec - mme_coeffs_lr[0].flatten()) ** 2))) / (d1 * d2))

	tot_rmse = np.sum([(np.sum((bVec[i] - mme_coeffs[i].flatten()) ** 2)) for i in range(Nclass)]) + \
			   np.sum((np.sum((phipsiVec - mme_coeffs_lr[0].flatten()) ** 2)))
	tot_rmse /= Nclass * d1 * d2 + d1 * d2
	rmse_mme_coeffs = np.sqrt(tot_rmse)

	# Show recovered coefficients
	# RR
	plt.imshow(rr_coeffs, cmap='Greys')
	frame1 = plt.gca()
	frame1.axes.get_xaxis().set_ticks([])
	frame1.axes.get_yaxis().set_ticks([])
	plt.title('Coefficient recovered by RR', size=25)
	plt.show()

	# MRR
	plt.imshow(mrr_coeffs, cmap='Greys')
	frame1 = plt.gca()
	frame1.axes.get_xaxis().set_ticks([])
	frame1.axes.get_yaxis().set_ticks([])
	plt.title('Coefficient recovered by MRR', size=20)
	plt.show()

	# ME
	for i in range(Nclass):
		plt.imshow(me_coeffs[i], cmap='Greys')
		frame1 = plt.gca()
		frame1.axes.get_xaxis().set_ticks([])
		frame1.axes.get_yaxis().set_ticks([])
		plt.title('Expert coefficient recovered by ME', size=20)
		plt.show()
	plt.imshow(me_coeffs_lr[0], cmap='Greys')
	frame1 = plt.gca()
	frame1.axes.get_xaxis().set_ticks([])
	frame1.axes.get_yaxis().set_ticks([])
	plt.title('Gate coefficient recovered by MME', size=20)
	plt.show()

	# MME
	for i in range(Nclass):
		plt.imshow(mme_coeffs[i], cmap='Greys')
		frame1 = plt.gca()
		frame1.axes.get_xaxis().set_ticks([])
		frame1.axes.get_yaxis().set_ticks([])
		plt.title('Expert coefficient recovered by MME', size=20)
		plt.show()
	plt.imshow(mme_coeffs_lr[i], cmap='Greys')
	frame1 = plt.gca()
	frame1.axes.get_xaxis().set_ticks([])
	frame1.axes.get_yaxis().set_ticks([])
	plt.title('Gate coefficient recovered by MME', size=20)
	plt.show()

