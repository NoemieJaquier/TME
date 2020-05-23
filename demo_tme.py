#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from utils.logistic_regression_utils import sigmoid
from utils.tensor_utils import khatriRaoProd
from utils.regression_methods import MixtureLinearExperts, TensorRidgeRegression, TensorMixtureLinearExperts

'''
This example generates data according to tensor mixture of experts model based on tensor coefficients.
The recovery of these coefficients is tested with Ridge regression, tensor Ridge regression, a mixture of experts model 
and a tensor mixture of experts model.

The user can:
	1. Change the dimension of the problem.
	2. Change the coefficient shapes.
	3. Change the number of training data.
	4. Change the noise in the generated data.
	5. Change the rank of the TRR and TME models.

Author: Noémie Jaquier, 2018
'''

if __name__ == '__main__':
	np.random.seed(1234)

	print('Generating data...')
	# ## Generate data
	pb_dim = 3
	im_dim = 16  # For dim=2, 64 is fine. For dim > 2, memory issues can arise with ME, therefore prefer 16.

	coeff_param = im_dim/64.

	# Cross
	a1 = int(round(17 * coeff_param))
	a2 = int(round(46 * coeff_param))
	b1 = int(round(28 * coeff_param))
	b2 = int(round(35 * coeff_param))

	cross = np.zeros([im_dim for i in range(pb_dim)])
	if pb_dim is 2:
		cross[a1:a2, b1:b2] = 1
		cross[b1:b2, a1:a2] = 1
	if pb_dim is 3:
		cross[a1:a2, b1:b2, b1:b2] = 1
		cross[b1:b2, a1:a2, b1:b2] = 1
		cross[b1:b2, b1:b2, a1:a2] = 1
	if pb_dim is 4:
		cross[a1:a2, b1:b2, b1:b2, b1:b2] = 1
		cross[b1:b2, a1:a2, b1:b2, b1:b2] = 1
		cross[b1:b2, b1:b2, a1:a2, b1:b2] = 1
		cross[b1:b2, b1:b2, b1:b2, a1:a2] = 1
	if pb_dim is 5:
		cross[a1:a2, b1:b2, b1:b2, b1:b2, b1:b2] = 1
		cross[b1:b2, a1:a2, b1:b2, b1:b2, b1:b2] = 1
		cross[b1:b2, b1:b2, a1:a2, b1:b2, b1:b2] = 1
		cross[b1:b2, b1:b2, b1:b2, a1:a2, b1:b2] = 1
		cross[b1:b2, b1:b2, b1:b2, b1:b2, a1:a2] = 1

	# T-shape
	a1 = int(round(17 * coeff_param))
	a2 = int(round(46 * coeff_param))
	b1 = int(round(17 * coeff_param))
	b2 = int(round(24 * coeff_param))
	c1 = int(round(14 * coeff_param))
	c2 = int(round(49 * coeff_param))
	d1 = int(round(28 * coeff_param))
	d2 = int(round(35 * coeff_param))

	tshape = np.zeros([im_dim for i in range(pb_dim)])
	if pb_dim is 2:
		tshape[a1:a2, d1:d2] = 1
		tshape[b1:b2, c1:c2] = 1
	if pb_dim is 3:
		tshape[a1:a2, d1:d2, d1:d2] = 1
		tshape[b1:b2, c1:c2, d1:d2] = 1
		tshape[b1:b2, d1:d2, c1:c2] = 1
	if pb_dim is 4:
		tshape[a1:a2, d1:d2, d1:d2, d1:d2] = 1
		tshape[b1:b2, c1:c2, d1:d2, d1:d2] = 1
		tshape[b1:b2, d1:d2, c1:c2, d1:d2] = 1
		tshape[b1:b2, d1:d2, d1:d2, c1:c2] = 1
	if pb_dim is 5:
		tshape[a1:a2, d1:d2, d1:d2, d1:d2, d1:d2] = 1
		tshape[b1:b2, c1:c2, d1:d2, d1:d2, d1:d2] = 1
		tshape[b1:b2, d1:d2, c1:c2, d1:d2, d1:d2] = 1
		tshape[b1:b2, d1:d2, d1:d2, c1:c2, d1:d2] = 1
		tshape[b1:b2, d1:d2, d1:d2, d1:d2, c1:c2] = 1

	# Circle
	c = int(round(32 * coeff_param))
	r = int(round(10 * coeff_param))

	circle = np.zeros([im_dim for i in range(pb_dim)])
	for index, x in np.ndenumerate(circle):
		if np.linalg.norm(np.array(index)-c) <= r+0.5:
			circle[index] = 1

	# ## Generate parameters
	# - wVec
	bVec = [circle.flatten(), tshape.flatten()]  # The chosen shapes can be changed
	# - vVec
	phipsiVec = [cross.flatten()]  # The chosen shapes can be changed

	# ## Generate data
	# Parameters
	Ndata = 50  # The number of data can be changed
	Nclass = len(bVec)
	Ndim = 1
	noise_level = 0.1  # The noise level can be changed

	N = Ndata * Nclass
	x = np.random.randn(N, np.power(im_dim, pb_dim))  # random input

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

	tuple_data_dim = (N,) + (im_dim,) * pb_dim
	tuple_dim = (im_dim,) * pb_dim
	X = np.reshape(x, tuple_data_dim)  # reshape input in matrix form

	# ## Test different methods
	print('Coefficient recovery:')
	# Models parameters TRR
	trr_rank = 2  # The rank can be changed
	# Models parameters TME
	tme_rank_e = 2  # The rank can be changed
	tme_rank_g = 2  # The rank can be changed

	# Tensor ridge regression
	print('Tensor Ridge regression...')
	trr = TensorRidgeRegression(trr_rank)
	trr.training(X, y, reg=1e-1)
	trr_coeffs = np.reshape(trr.wVec[0], tuple_dim)

	# Mixture of experts
	print('Mixture of experts...')
	me = MixtureLinearExperts()
	me_LL = me.training(x, y, y_class, maxiter=20, optmethod='CG', reg=1e-1)
	me_coeffs = []
	me_coeffs_lr = []
	for i in range(0, me.nb_class):
		me_coeffs.append(np.reshape(me.W[i], tuple_dim))
		me_coeffs_lr.append(np.reshape(me.V[:, i], tuple_dim))

	rmse_me_coeffs_rr = np.sqrt(np.sum([(np.sum((bVec[i] - me_coeffs[i].flatten()) ** 2))
										for i in range(Nclass)]) / (Nclass * np.power(im_dim, pb_dim)))
	rmse_me_coeffs_lr = np.sqrt(np.sum(np.sum((phipsiVec - me_coeffs_lr[0].flatten()) ** 2)) / np.power(im_dim, pb_dim))

	tot_rmse = np.sum([(np.sum((bVec[i] - me_coeffs[i].flatten()) ** 2))
					   for i in range(Nclass)]) + np.sum((np.sum((phipsiVec - me_coeffs_lr[0].flatten()) ** 2)))
	tot_rmse /= (Nclass+1) * np.power(im_dim, pb_dim)
	rmse_me_coeffs = tot_rmse
	print('Total RMSE for ME: ', rmse_me_coeffs)

	# Tensor-valued mixture of experts
	print('Tensor-valued mixture of experts...')
	tme = TensorMixtureLinearExperts([tme_rank_e, tme_rank_e], tme_rank_g)
	tme_LL = tme.training(X, y, y_class, reg_rr=1e-1, reg_lr=1e-1, maxiter=20, max_diff_ll=5.0, optmethod='CG')

	tme_coeffs = []
	tme_coeffs_lr = []
	for c in range(0, tme.nb_class):
		# TRR part
		alpha = tme.alpha[c][:]
		wmsTmp = tme.W[c][0]
		# Compute vec(W)
		tme_bVec = []
		for j in range(tme.nb_dim):
			wTmp = khatriRaoProd(wmsTmp[1], wmsTmp[0])
			for k in range(2, tme.nb_dim_x):
				wTmp = khatriRaoProd(wmsTmp[k], wTmp)

			tme_bVec.append(np.dot(wTmp, np.ones((tme.rank_e[c], 1))))

		tme_coeffs.append(np.reshape(tme_bVec[0], tuple_dim))

		# TLR part
		vTmp = khatriRaoProd(tme.V[1][c], tme.V[0][c])
		for j in range(2, tme.nb_dim_x):
			vTmp = khatriRaoProd(tme.V[j][c], vTmp)
		tme_vVec = np.dot(vTmp, np.ones((tme.rank_g, 1)))
		tme_coeffs_lr.append(np.reshape(tme_vVec, tuple_dim))

	rmse_tme_coeffs_rr = np.sqrt(np.sum([(np.sum((bVec[i] - tme_coeffs[i].flatten()) ** 2))
										 for i in range(Nclass)]) / (Nclass * np.power(im_dim, pb_dim)))

	rmse_tme_coeffs_lr = \
		np.sqrt(np.sum((np.sum((phipsiVec - tme_coeffs_lr[0].flatten()) ** 2))) / np.power(im_dim, pb_dim))

	tot_rmse = np.sum([(np.sum((bVec[i] - tme_coeffs[i].flatten()) ** 2)) for i in range(Nclass)]) \
			   + np.sum((np.sum((phipsiVec - tme_coeffs_lr[0].flatten()) ** 2)))
	tot_rmse /= (Nclass+1) * np.power(im_dim, pb_dim)
	rmse_tme_coeffs = np.sqrt(tot_rmse)
	print('Total RMSE for TME: ', rmse_tme_coeffs)

