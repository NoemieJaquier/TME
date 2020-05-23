#! /usr/bin/env python
import numpy as np
import scipy.optimize as sc_opt
from utils.tensor_utils import khatriRaoProd, tensor2mat
from utils.utils import multi_variate_normal
from utils.logistic_regression_utils import sigmoid, softmax, optLogReg, optMatLogReg, optTensLogReg


class RidgeRegression:
	"""
	Ridge regression class
	"""
	def __init__(self):
		self.beta = 0

	def training(self, x, y, reg=1e-2):
		"""
		Train RR model
		:param x: input data (nb_data, dim_x)
		:param y: output data (nb_data, dim_y)
		:param reg: regularization term
		"""
		# Define regularization
		reg_mat = np.eye(x.shape[1])*reg
		# Compute beta
		self.beta = np.linalg.solve(x.T.dot(x) + reg_mat, x.T)
		self.beta = self.beta.dot(y)

	def testing(self, x):
		"""
		Estimate the output corresponding to a new input point
		:param x: input data (dim_x)
		:return: corresponding output prediction (dim_y)
		"""
		# print(x.T.dot(self.beta))
		return x[:, -1].T.dot(self.beta)

	def testing_multiple(self, x):
		"""
		Estimate the output corresponding to new input points
		:param x: input data (nb_data, dim_x)
		:return: corresponding output predictions (nb_data, dim_y)
		"""
		return x.dot(self.beta)


class MatrixRidgeRegression:
	"""
	Ridge regression for matrix-valued data
	Based on [Guo, Kotsia, Patras. 2012] and [Zhou, Li, Zhu. 2013]
	"""
	def __init__(self, rank):
		"""
		Initialize the model
		:param rank: rank of the weight matrices b1, b2
		"""
		self.b1 = []  # Left multiplying matrix
		self.b2 = []  # Right multiplying matrix
		self.alpha = []  # Constant term
		self.bVec = []
		self.rank = rank
		self.dY = 0

	def training(self, x, y, reg=1e-2, maxDiffCrit=1e-4, maxIter=200):
		"""
		Train the parameters of the MRR model
		:param x: input matrices (nb_data, dim1, dim2)
		:param y: output vectors (nb_data, dim_y)
		:param reg: regularization term
		:param maxDiffCrit: stopping criterion for the alternative least squares procedure
		:param maxIter: maximum number of iterations for the alternative least squares procedure
		"""
		# Dimensions
		N = x.shape[0]
		d1 = x.shape[1]
		d2 = x.shape[2]
		self.dY = y.shape[1]

		for dim in range(0, self.dY):
			# Initialization
			# self.b1.append(np.random.randn(d1, self.rank))
			# self.b2.append(np.random.randn(d2, self.rank))
			# self.alpha.append(np.random.randn(1))
			self.b1.append(np.ones((d1, self.rank)))
			self.b2.append(np.ones((d2, self.rank)))
			self.alpha.append(np.zeros(1))
			self.bVec.append(np.random.randn(d1 * d2, 1))

			# Optimization of parameters (ALS procedure)
			nbIter = 1
			prevRes = 0

			while nbIter < maxIter:
				# Update b1
				zVec1 = np.zeros((N, d1*self.rank))
				for n in range(0, N):
					zVec1[n] = np.dot(x[n], self.b2[-1]).flatten()
				b1 = np.linalg.solve(zVec1.T.dot(zVec1) + np.eye(d1*self.rank)*reg, zVec1.T).dot(y[:, dim] - self.alpha[-1])
				self.b1[-1] = np.reshape(b1, (d1, self.rank))

				# Update b2
				zVec2 = np.zeros((N, d2*self.rank))
				for n in range(0, N):
					zVec2[n] = np.dot(x[n].T, self.b1[-1]).flatten()
				b2 = np.linalg.solve(zVec2.T.dot(zVec2) + np.eye(d2 * self.rank) * reg, zVec2.T).dot(y[:, dim] - self.alpha[-1])
				self.b2[-1] = np.reshape(b2, (d2, self.rank))

				# Update alpha
				self.bVec[-1] = np.dot(khatriRaoProd(self.b2[-1], self.b1[-1]), np.ones((self.rank, 1)))
				alpha = 0
				for n in range(0, N):
					alpha += y[n, dim] - np.dot(self.bVec[-1][:, None].T, x[n].flatten())
				self.alpha[-1] = alpha[0]/N

				# Compute residuals
				res = 0
				for n in range(0, N):
					res += (y[n, dim] - self.alpha[-1] - np.dot(self.bVec[-1][:, None].T, x[n].flatten()))**2

				resDiff = prevRes - res

				# Check convergence
				if resDiff < maxDiffCrit and nbIter > 1:
					print('MRR converged after %d iterations.' % nbIter)
					break
				nbIter += 1
				prevRes = res

			if resDiff > maxDiffCrit:
				print('MRR did not converged after %d iterations.' % nbIter)

	def testing(self, x):
		"""
		Estimate the output corresponding to a new input point
		:param x: new input data (dim1, dim2)
		:return: corresponding output prediction (dim_y)
		"""
		y = np.zeros(self.dY)
		for dim in range(0, self.dY):
			y[dim] = self.alpha[dim] + np.dot(self.bVec[dim][:, None].T, x.flatten())
		return y

	def testing_multiple(self, x):
		"""
		Estimate the outputs corresponding to new input points
		:param x: new input data (nb_data, dim1, dim2)
		:return: corresponding output predictions (nb_data, dim_y)
		"""
		N = x.shape[0]
		y = np.zeros((N, self.dY))
		for n in range(0, N):
			for dim in range(0, self.dY):
				y[n, dim] = self.alpha[dim] + np.dot(self.bVec[dim][:, None].T, x[n].flatten())
		return y


class TensorRidgeRegression:
	"""
	Ridge regression for tensor-valued data
	Based on [Guo, Kotsia, Patras. 2012] and [Zhou, Li, Zhu. 2013]
	"""
	def __init__(self, rank):
		"""
		Initialize the model
		:param rank: rank of the weight matrices b1, b2
		"""
		self.W = []  # Regression coefficients
		self.alpha = []  # Constant term
		self.wVec = []
		self.rank = rank
		self.dY = 0

	def training(self, x, y, reg=1e-2, maxDiffCrit=1e-4, maxIter=200):
		"""
		Train the parameters of the MRR model
		:param x: input matrices (nb_data, dim1, dim2, ...)
		:param y: output vectors (nb_data, dim_y)
		:param reg: regularization term
		:param maxDiffCrit: stopping criterion for the alternative least squares procedure
		:param maxIter: maximum number of iterations for the alternative least squares procedure
		"""
		# Dimensions
		N = x.shape[0]
		dX = x.shape[1:]
		self.dY = y.shape[1]

		for dim in range(0, self.dY):
			# Initialization
			wms = []
			for m in range(len(dX)):
				wms.append(np.ones((dX[m], self.rank)))

			self.alpha.append(np.zeros(1))
			self.wVec.append(np.reshape(np.zeros(dX), -1))

			# Optimization of parameters (ALS procedure)
			nbIter = 1
			prevRes = 0

			while nbIter < maxIter:
				for m in range(len(dX)):
					# Compute Wm complement (WM o ... o Wm+1 o Wm-1 o ... o W1)
					if m is 0:
						wmComplement = wms[1]
						for i in range(2, len(dX)):
							wmComplement = khatriRaoProd(wms[i], wmComplement)
					else:
						wmComplement = wms[0]
						for i in range(1, len(dX)):
							if i != m:
								wmComplement = khatriRaoProd(wms[i], wmComplement)

					# Update Wm
					zVec = np.zeros((N, dX[m] * self.rank))
					for n in range(0, N):
						zVec[n] = np.dot(tensor2mat(x[n], m), wmComplement).flatten()
					wm = np.linalg.solve(zVec.T.dot(zVec) + np.eye(dX[m]*self.rank)*reg, zVec.T).dot(y[:, dim] - self.alpha[-1])
					wms[m] = np.reshape(wm, (dX[m], self.rank))

				# Update alpha
				wTmp = khatriRaoProd(wms[1], wms[0])
				for i in range(2, len(dX)):
					wTmp = khatriRaoProd(wms[i], wTmp)

				self.wVec[-1] = np.dot(wTmp, np.ones((self.rank, 1)))
				alpha = 0
				for n in range(0, N):
					alpha += y[n, dim] - np.dot(self.wVec[-1][:, None].T, x[n].flatten())
				self.alpha[-1] = alpha[0]/N

				# Compute residuals
				res = 0
				for n in range(0, N):
					res += (y[n, dim] - self.alpha[-1] - np.dot(self.wVec[-1][:, None].T, x[n].flatten()))**2

				resDiff = prevRes - res

				# Check convergence
				if resDiff < maxDiffCrit and nbIter > 1:
					print('TRR converged after %d iterations.' % nbIter)
					break
				nbIter += 1
				prevRes = res

			if resDiff > maxDiffCrit:
				print('TRR did not converged after %d iterations.' % nbIter)

			self.W.append(wms)

	def testing(self, x):
		"""
		Estimate the output corresponding to a new input point
		:param x: new input data (dim1, dim2, ...)
		:return: corresponding output prediction (dim_y)
		"""
		y = np.zeros(self.dY)
		for dim in range(0, self.dY):
			y[dim] = self.alpha[dim] + np.dot(self.wVec[dim][:, None].T, x.flatten())
		return y

	def testing_multiple(self, x):
		"""
		Estimate the outputs corresponding to new input points
		:param x: new input data (nb_data, dim1, dim2, ...)
		:return: corresponding output predictions (nb_data, dim_y)
		"""
		N = x.shape[0]
		y = np.zeros((N, self.dY))
		for n in range(0, N):
			for dim in range(0, self.dY):
				y[n, dim] = self.alpha[dim] + np.dot(self.wVec[dim][:, None].T, x[n].flatten())
		return y


class MixtureLinearExperts:
	"""
	Mixture of expert class
	The experts follows a Gaussian model and the gate is defined as the softmax function.
	"""
	def __init__(self):
		"""
		Initialization
		"""
		self.V = None  # Weight matrix of the gate
		self.W = None  # Weight matrix of the experts
		self.sigma = None  # Covariance matrix of the experts
		self.nb_dim = 0
		self.nb_class = 0

	def training(self, x, y, y_class, reg=1e-2, maxiter=100, max_diff_ll=1e-5, optmethod='BFGS'):
		"""
		Train the ME model
		:param x: input data (nb_data, dim_x)
		:param y: output data (nb_data, dim_y)
		:param y_class: classes labels of outputs (nb_data, nb_classes)
		:param reg: regularization term
		:param maxiter: maximum number of iterations for the EM algorithm
		:param max_diff_ll: maximum difference of log-likelihood for the EM algorithm
		:param optmethod: optimization method for the logistic regression (gate)
		"""
		# Copy data
		x = np.copy(x.T)
		y = np.copy(y.T)
		y_class = np.copy(y_class.T)

		# Set dimensions
		nb_data = x.shape[1]
		nb_dim = x.shape[0]
		self.nb_class = y_class.shape[0]
		self.nb_dim = y.shape[0]

		# Initialization
		self.V = optLogReg(x, y_class, optmethod=optmethod, reg_fact=reg)

		W0 = np.linalg.solve(x.dot(x.T) + reg * np.eye(nb_dim), x).dot(y.T)

		self.W = [W0 for i in range(0, self.nb_class)]

		self.sigma = [np.eye(self.nb_dim) for i in range(0, self.nb_class)]

		# EM algorithm
		nb_min_steps = 5  # min num iterations
		nb_max_steps = maxiter  # max iterations

		LL = np.zeros(nb_max_steps)

		for it in range(nb_max_steps):
			# E - step
			Ltmp = np.zeros((self.nb_class, nb_data))
			L = np.zeros((self.nb_class, nb_data))

			priors = softmax(np.dot(self.V.T, x).T).T

			for i in range(0, self.nb_class):
				Ltmp[i] = priors[i] * multi_variate_normal(y.T, np.dot(self.W[i].T, x).T, self.sigma[i], log=False)

			GAMMA = Ltmp / np.sum(Ltmp, axis=0)

			LL[it] = np.sum(np.sum(GAMMA * np.log(Ltmp+1e-100)))

			# M-step
			for i in range(0, self.nb_class):
				r = np.diag(GAMMA[i])
				self.W[i] = np.linalg.solve(x.dot(r).dot(x.T) + reg * np.eye(nb_dim), np.dot(x, r)).dot(y.T)

				self.sigma[i] = np.dot(np.dot(y - np.dot(self.W[i].T, x), r), (y - np.dot(self.W[i].T, x)).T) / sum(GAMMA[i]) + 1e-6 * np.eye(self.nb_dim)

			self.V = optLogReg(x, GAMMA, optmethod=optmethod, reg_fact=reg)
			print(it)

			# Check for convergence
			if it > nb_min_steps:
				if LL[it] - LL[it - 1] < max_diff_ll:
					print('Converged after %d iterations: %.3e' % (it, LL[it]), 'red', 'on_white')
					return LL[it]

		print("ME did not converge before reaching max iteration. Consider augmenting the number of max iterations.")
		return LL[-1]

	def testing(self, x):
		"""
		Estimate the outputs corresponding to new input points
		:param x: new input data (nb_data, dim_x)
		:return: corresponding output predictions (nb_data, dim_y)
		"""
		priors = softmax(np.dot(self.V.T, x.T).T).T

		ytmp = [np.dot(self.W[i].T, x.T)*priors[i] for i in range(len(self.W))]

		return np.sum(ytmp, axis=0).T


class MaxtrixMixtureLinearExperts:
	"""
	Mixture of expert for matrix-valued data class
	The experts follows a matrix-variate Gaussian model and the gate is defined as the matrix-variate softmax function.
	"""
	def __init__(self, rank_e=1, rank_g=1):
		"""
		Initialization
		:param rank_e: expert models ranks (scalar or list with one rank for each expert)
		:param rank_g: gate model rank
		"""
		self.nb_class = 0
		self.nb_dim = 0
		self.rank_e = rank_e
		self.rank_g = rank_g
		self.alpha = None  # Constant terms of the experts
		self.b1 = None  # Left multiplying matrices of the experts
		self.b2 = None  # Right multiplying matrices of the experts
		self.sigma = None  # Covariance matrix of the experts
		self.xhi = None  # Constant term of the gate
		self.phi = None  # Left multiplying matrix of the gate
		self.psi = None  # Right multiplying matrix of the gate

	def training(self, x, y, y_class, reg_rr=1e-2, reg_lr=1e-2, maxiter=100, max_diff_ll=1e-5, optmethod='BFGS'):
		"""
		Training the MME model
		:param x: input data (nb_data, dim_x)
		:param y: output data (nb_data, dim_y)
		:param y_class: classes labels of outputs (nb_data, nb_classes)
		:param reg_rr: regularization term of the experts
		:param reg_lr: regularization term of the gate
		:param maxiter: maximum number of iterations for the EM algorithm
		:param max_diff_ll: maximum difference of log-likelihood for the EM algorithm
		:param optmethod: optimization method for the logistic regression (gate)
		:return:
		"""
		nb_data = x.shape[0]
		d1 = x.shape[1]
		d2 = x.shape[2]
		self.nb_class = y_class.shape[1]
		self.nb_dim = y.shape[1]

		if type(self.rank_e) is not list:
			self.rank_e = [self.rank_e for i in range(self.nb_class)]

		# Initialization with sklearn
		# ysk = np.sum(y.T * np.array(range(0, self.nb_class))[:, None], axis=0)
		# mul_lr = sk.linear_model.LogisticRegression(multi_class='multinomial', solver='newton-cg').fit(xvec, ysk)
		# self.V = mul_lr.coef_.T
		# self.V = optLogReg(xvec.T, y.T)

		# matRR = MatrixRidgeRegression(rank=self.rank_e)
		# matRR.training(x, y)
		#
		# self.alpha = [matRR.alpha[:] for i in range(0, self.nb_class)]
		# self.b1 = [matRR.b1[:] for i in range(0, self.nb_class)]
		# self.b2 = [matRR.b2[:] for i in range(0, self.nb_class)]

		# Experts initialization
		self.alpha = []
		self.b1 = []
		self.b2 = []

		for rank_e in self.rank_e:
			matRR = MatrixRidgeRegression(rank=rank_e)
			matRR.training(x, y)
			self.alpha.append(matRR.alpha[:])
			self.b1.append(matRR.b1[:])
			self.b2.append(matRR.b2[:])

		self.sigma = [np.eye(self.nb_dim) for i in range(0, self.nb_class)]

		# Gating initialization
		matRR = MatrixRidgeRegression(rank=self.rank_g)
		matRR.training(x, y_class)
		xhi_init = np.array(matRR.alpha).T
		phi_init = np.reshape(np.array(matRR.b1), (self.nb_class, d1*self.rank_g)).T
		psi_init = np.reshape(np.array(matRR.b2), (self.nb_class, d2*self.rank_g)).T
		vinit = np.vstack((xhi_init, phi_init, psi_init)).flatten()
		self.xhi, self.phi, self.psi = optMatLogReg(x, y_class, rank=self.rank_g, v_init=vinit, optmethod=optmethod, reg_fact=reg_lr)

		# EM algorithm
		nb_min_steps = 2  # min num iterations
		nb_max_steps = maxiter  # max iterations

		LL = np.zeros(nb_max_steps)

		for it in range(nb_max_steps):
			# E - step
			Ltmp = np.zeros((self.nb_class, nb_data))

			# Compute gate probabilities
			inProdEst = np.zeros((nb_data, self.nb_class))
			for n in range(0, nb_data):
				for dim in range(0, self.nb_class):
					phipsiVec = np.dot(khatriRaoProd(self.psi[dim], self.phi[dim]), np.ones((self.rank_g, 1)))
					inProdEst[n, dim] = self.xhi[dim] + np.dot(phipsiVec[:, None].T, x[n].flatten())

			priors = softmax(inProdEst).T

			# Compute experts distributions weighted by gate probabilities
			for i in range(0, self.nb_class):
				alpha = self.alpha[i][:]
				b1tmp = self.b1[i][:]
				b2tmp = self.b2[i][:]
				bVec = [np.dot(khatriRaoProd(b2tmp[j], b1tmp[j]), np.ones((self.rank_e[i], 1))) for j in range(self.nb_dim)]

				yhat_tmp = np.zeros((nb_data, self.nb_dim))
				for n in range(0, nb_data):
					for dim in range(0, self.nb_dim):
						yhat_tmp[n, dim] = alpha[dim] + np.dot(bVec[dim][:, None].T, x[n].flatten())

				Ltmp[i] = priors[i] * multi_variate_normal(y, yhat_tmp, self.sigma[i], log=False)

			# Compute  responsabilities
			GAMMA = Ltmp / (np.sum(Ltmp, axis=0)+1e-100)

			LL[it] = np.sum(np.sum(GAMMA * np.log(Ltmp+1e-100)))

			# M-step
			# Experts parameters update
			yhat = []
			for i in range(0, self.nb_class):
				r = np.diag(GAMMA[i])
				sqrGAMMA = np.sqrt(GAMMA[i])
				weighted_x = sqrGAMMA[:, None, None] * x
				weighted_y = sqrGAMMA[:, None] * y

				matRR = None
				matRR = MatrixRidgeRegression(rank=self.rank_e[i])
				matRR.training(weighted_x, weighted_y, reg=reg_rr)

				self.alpha[i] = matRR.alpha[:]
				self.b1[i] = matRR.b1[:]
				self.b2[i] = matRR.b2[:]

				yhat_tmp = matRR.testing_multiple(x)
				yhat.append(yhat_tmp * priors[i][:, None])

				self.sigma[i] = np.dot(np.dot((y - yhat_tmp).T, r), (y - yhat_tmp)) / sum(GAMMA[i]) + 1e-6*np.eye(self.nb_dim)

			# Gate parameters update
			xhi_init = np.array(self.xhi).T
			phi_init = np.reshape(np.array(self.phi), (self.nb_class, d1 * self.rank_g)).T
			psi_init = np.reshape(np.array(self.psi), (self.nb_class, d2 * self.rank_g)).T
			vinit = np.vstack((xhi_init, phi_init, psi_init)).flatten()
			self.xhi, self.phi, self.psi = optMatLogReg(x, GAMMA.T, rank=self.rank_g, v_init=vinit, optmethod=optmethod, reg_fact=reg_lr)

			print(it)
			# Check for convergence
			if it > nb_min_steps:
				if LL[it] - LL[it - 1] < max_diff_ll:
					print('Converged after %d iterations: %.3e' % (it, LL[it]), 'red', 'on_white')
					print(LL)
					return LL[it]

		print("MME did not converge before reaching max iteration. Consider augmenting the number of max iterations.")
		print(LL)
		return LL[-1]

	def testing(self, x):
		"""
		Estimate the outputs corresponding to new input points
		:param x: new input data (nb_data, dim_x)
		:return: corresponding output predictions (nb_data, dim_y)
		"""
		nb_data = x.shape[0]

		# Compute gate probabilities
		inProdEst = np.zeros((nb_data, self.nb_class))
		for n in range(0, nb_data):
			for dim in range(0, self.nb_class):
				phipsiVec = np.dot(khatriRaoProd(self.psi[dim], self.phi[dim]), np.ones((self.rank_g, 1)))
				inProdEst[n, dim] = self.xhi[dim] + np.dot(phipsiVec[:, None].T, x[n].flatten())

		priors = softmax(inProdEst).T

		ytmp = []
		for i in range(0, self.nb_class):
			# Compute experts predictions
			alpha = self.alpha[i][:]
			b1tmp = self.b1[i][:]
			b2tmp = self.b2[i][:]
			bVec = [np.dot(khatriRaoProd(b2tmp[j], b1tmp[j]), np.ones((self.rank_e[i], 1))) for j in range(self.nb_dim)]

			yhat_tmp = np.zeros((nb_data, self.nb_dim))
			for n in range(0, nb_data):
				for dim in range(0, self.nb_dim):
					yhat_tmp[n, dim] = alpha[dim] + np.dot(bVec[dim][:, None].T, x[n].flatten())

			# Append expert predictions weighted by the gate
			ytmp.append(yhat_tmp * priors[i][:, None])

		# Compute final predictions
		return np.sum(ytmp, axis=0)


class TensorMixtureLinearExperts:
	"""
	Mixture of expert for tensor-valued data class
	The experts follows a tensor-variate Gaussian model and the gate is defined as the tensor-variate softmax function.
	"""
	def __init__(self, rank_e=1, rank_g=1):
		"""
		Initialization
		:param rank_e: expert models ranks (scalar or list with one rank for each expert)
		:param rank_g: gate model rank
		"""
		self.nb_dim_x = 0
		self.nb_class = 0
		self.nb_dim = 0
		self.rank_e = rank_e
		self.rank_g = rank_g
		self.alpha = None  # Constant terms of the experts
		self.W = None  # Coefficients of the experts
		self.sigma = None  # Covariance matrix of the experts
		self.beta = None  # Constant term of the gate
		self.V = None  # Coefficients of the gate

	def training(self, x, y, y_class, reg_rr=1e-2, reg_lr=1e-2, maxiter=100, max_diff_ll=1e-5, optmethod='BFGS'):
		"""
		Training the MME model
		:param x: input data (nb_data, dim_x)
		:param y: output data (nb_data, dim_y)
		:param y_class: classes labels of outputs (nb_data, nb_classes)
		:param reg_rr: regularization term of the experts
		:param reg_lr: regularization term of the gate
		:param maxiter: maximum number of iterations for the EM algorithm
		:param max_diff_ll: maximum difference of log-likelihood for the EM algorithm
		:param optmethod: optimization method for the logistic regression (gate)
		"""
		nb_data = x.shape[0]
		dX = x.shape[1:]
		self.nb_dim_x = len(dX)
		self.nb_class = y_class.shape[1]
		self.nb_dim = y.shape[1]

		if type(self.rank_e) is not list:
			self.rank_e = [self.rank_e for i in range(self.nb_class)]

		# Experts initialization
		self.alpha = []
		self.W = []

		for rank_e in self.rank_e:
			tensRR = TensorRidgeRegression(rank=rank_e)
			tensRR.training(x, y)
			self.alpha.append(tensRR.alpha[:])
			self.W.append(tensRR.W)

		self.sigma = [np.eye(self.nb_dim) for i in range(0, self.nb_class)]

		# Gating initialization
		tensRR = TensorRidgeRegression(rank=self.rank_g)
		tensRR.training(x, y_class)
		beta_init = np.array(tensRR.alpha).T
		Vall = beta_init
		for m in range(self.nb_dim_x):
			wTmpRR = []
			for c in range(self.nb_class):
				wTmpRR.append(tensRR.W[c][m])
			Vtmp = np.reshape(np.array(wTmpRR), (self.nb_class, dX[m]*self.rank_g)).T
			Vall = np.vstack((Vall, Vtmp))
		vinit = Vall.flatten()

		self.beta, self.V = optTensLogReg(x, y_class, rank=self.rank_g, v_init=vinit, optmethod=optmethod, reg_fact=reg_lr)

		# EM algorithm
		nb_min_steps = 2  # min num iterations
		nb_max_steps = maxiter  # max iterations

		LL = np.zeros(nb_max_steps)

		for it in range(nb_max_steps):
			# E - step
			Ltmp = np.zeros((self.nb_class, nb_data))

			# Compute gate probabilities
			inProdEst = np.zeros((nb_data, self.nb_class))
			for n in range(0, nb_data):
				for c in range(0, self.nb_class):
					vTmp = khatriRaoProd(self.V[1][c], self.V[0][c])
					for i in range(2, self.nb_dim_x):
						vTmp = khatriRaoProd(self.V[i][c], vTmp)

					vVec = np.dot(vTmp, np.ones((self.rank_g, 1)))

					inProdEst[n, c] = self.beta[c] + np.dot(vVec[:, None].T, x[n].flatten())

			priors = softmax(inProdEst).T

			# Compute experts distributions weighted by gate probabilities
			for c in range(0, self.nb_class):
				alpha = self.alpha[c][:]
				wmsTmp = self.W[c][0]

				# Compute vec(W)
				wVec = []
				for j in range(self.nb_dim):
					wTmp = khatriRaoProd(wmsTmp[1], wmsTmp[0])
					for k in range(2, self.nb_dim_x):
						wTmp = khatriRaoProd(wmsTmp[k], wTmp)

					wVec.append(np.dot(wTmp, np.ones((self.rank_e[c], 1))))

				# Compute predictions
				yhat_tmp = np.zeros((nb_data, self.nb_dim))
				for n in range(0, nb_data):
					for dim in range(0, self.nb_dim):
						yhat_tmp[n, dim] = alpha[dim] + np.dot(wVec[dim][:, None].T, x[n].flatten())

				# Likelihood
				Ltmp[c] = priors[c] * multi_variate_normal(y, yhat_tmp, self.sigma[c], log=False)

			# Compute  responsabilities
			GAMMA = Ltmp / (np.sum(Ltmp, axis=0)+1e-100)

			LL[it] = np.sum(np.sum(GAMMA * np.log(Ltmp+1e-100)))

			# M-step
			# Experts parameters update
			yhat = []
			for c in range(0, self.nb_class):
				r = np.diag(GAMMA[c])
				sqrGAMMA = np.sqrt(GAMMA[c])

				weighted_y = sqrGAMMA[:, None] * y
				for d in range(len(x.shape)-1):
					sqrGAMMA = np.expand_dims(sqrGAMMA, axis=-1)
				weighted_x = sqrGAMMA * x

				tensRR = None
				tensRR = TensorRidgeRegression(rank=self.rank_e[c])
				tensRR.training(weighted_x, weighted_y, reg=reg_rr)

				self.alpha[c] = tensRR.alpha[:]
				self.W[c] = tensRR.W

				yhat_tmp = tensRR.testing_multiple(x)
				yhat.append(yhat_tmp * priors[c][:, None])

				self.sigma[c] = np.dot(np.dot((y - yhat_tmp).T, r), (y - yhat_tmp)) / sum(GAMMA[c]) + 1e-6*np.eye(self.nb_dim)

			# Gate parameters update
			beta_init = np.array(self.beta).T
			Vall = beta_init
			for m in range(self.nb_dim_x):
				Vtmp = np.reshape(np.array(self.V[m]), (self.nb_class, dX[m] * self.rank_g)).T
				Vall = np.vstack((Vall, Vtmp))
			vinit = Vall.flatten()

			self.beta, self.V = optTensLogReg(x, GAMMA.T, rank=self.rank_g, v_init=vinit, optmethod=optmethod, reg_fact=reg_lr)

			print(it)
			# Check for convergence
			if it > nb_min_steps:
				if LL[it] - LL[it - 1] < max_diff_ll:
					print('Converged after %d iterations: %.3e' % (it, LL[it]), 'red', 'on_white')
					print(LL)
					return LL[it]

		print("TME did not converge before reaching max iteration. Consider augmenting the number of max iterations.")
		print(LL)
		return LL[-1]

	def testing(self, x):
		"""
		Estimate the outputs corresponding to new input points
		:param x: new input data (nb_data, dim_x)
		:return: corresponding output predictions (nb_data, dim_y)
		"""
		nb_data = x.shape[0]

		# Compute gate probabilities
		inProdEst = np.zeros((nb_data, self.nb_class))
		for n in range(0, nb_data):
			for c in range(0, self.nb_class):
				vTmp = khatriRaoProd(self.V[1][c], self.V[0][c])
				for i in range(2, self.nb_dim_x):
					vTmp = khatriRaoProd(self.V[i][c], vTmp)
				vVec = np.dot(vTmp, np.ones((self.rank_g, 1)))

				inProdEst[n, c] = self.beta[c] + np.dot(vVec[:, None].T, x[n].flatten())

		priors = softmax(inProdEst).T

		ytmp = []
		for c in range(0, self.nb_class):
			# Compute experts predictions
			alpha = self.alpha[c][:]
			wmsTmp = self.W[c][0]

			# Compute vec(W)
			wVec = []
			for j in range(self.nb_dim):
				wTmp = khatriRaoProd(wmsTmp[1], wmsTmp[0])
				for k in range(2, self.nb_dim_x):
					wTmp = khatriRaoProd(wmsTmp[k], wTmp)

				wVec.append(np.dot(wTmp, np.ones((self.rank_e[c], 1))))

			yhat_tmp = np.zeros((nb_data, self.nb_dim))

			for n in range(0, nb_data):
				for dim in range(0, self.nb_dim):
					yhat_tmp[n, dim] = alpha[dim] + np.dot(wVec[dim][:, None].T, x[n].flatten())

			# Append expert predictions weighted by the gate
			ytmp.append(yhat_tmp * priors[i][:, None])

		# Compute final predictions
		return np.sum(ytmp, axis=0)
