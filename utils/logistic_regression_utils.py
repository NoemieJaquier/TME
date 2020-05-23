#! /usr/bin/env python
import numpy as np
import scipy.optimize as sc_opt
from utils.tensor_utils import khatriRaoProd, tensor2mat


def softmax(X, copy=True):
	"""
	FUNCTION FROM SKLEARN
	Calculate the softmax function.

	The softmax function is calculated by
	np.exp(X) / np.sum(np.exp(X), axis=1)

	This will cause overflow when large values are exponentiated.
	Hence the largest value in each row is subtracted from each data
	point to prevent this.

	Parameters
	----------
	X : array-like, shape (M, N)
		Argument to the logistic function

	copy : bool, optional
		Copy X or not.

	Returns
	-------
	out : array, shape (M, N)
		Softmax function evaluated at every point in x
	"""
	if copy:
		X = np.copy(X)
	max_prob = np.max(X, axis=1).reshape((-1, 1))
	X -= max_prob
	np.exp(X, X)
	sum_prob = np.sum(X, axis=1).reshape((-1, 1))
	X /= sum_prob
	return X


def sigmoid(x):
	"""
	Sigmoid function
	:param x: matrix
	:return: softmax function of x
	"""
	return 1 / (1 + np.exp(-x))


def optLogReg(x, y, optmethod='BFGS', reg_fact=0.0):
	"""
	Logistic regression optimization
	:param x: input data, np.array((nb_dim, nb_data))
	:param y: output data, np.array((nb_class, nb_data))
	:param optmethod: scipy optimization method
	:param reg_fact: regularization factor
	:return: weight vector, np.array((nb_dim, nb_class))
	"""
	nb_data = x.shape[1]
	nb_dim = x.shape[0]
	nb_class = y.shape[0]

	# LR function to optimize
	def func(v):
		V = np.reshape(v, (nb_class, nb_dim))
		est = softmax(np.dot(V, x).T).T
		# reg = np.dot(V, np.dot(reg_fact * np.eye(nb_dim), V.T))
		reg = np.dot(V, reg_fact * V.T)
		reg = np.sum(np.diagonal(reg))
		return -np.sum(y * np.log(est)) + reg

	# Gradient of the function to optimize
	def grad(v):
		V = np.reshape(v, (nb_class, nb_dim))
		est = softmax(np.dot(V, x).T).T
		grad = [np.kron((est[:, i] - y[:, i]), x[:, i]) for i in range(0, nb_data)]
		return np.sum(grad, axis=0) + 2 * reg_fact * v

	# Hessian of the function to optimize
	def hess(v):
		V = np.reshape(v, (nb_class, nb_dim))
		est = softmax(np.dot(V, x))
		est /= np.sum(est, axis=0)
		hess = np.zeros((nb_class * nb_dim, nb_class * nb_dim))
		for i in range(0, nb_data):
			hess += np.kron(np.diag(est[:, i]) - np.dot(est[:, i][None].T, est[:, i][None]), np.dot(x[:, i][None].T, x[:, i][None]))

	# Optimization of the weight vector
	opt = sc_opt.minimize(func, np.zeros((nb_dim*nb_class, 1)), jac=grad, method=optmethod, options={'maxiter': 200, 'disp': True})
	# opt = sc_opt.minimize(func, np.random.rand(nb_dim*nb_class, 1), jac=grad, hess=hess)
	v = opt.x

	# Return weight parameter
	return np.reshape(v, (nb_class, nb_dim)).T


def optMatLogReg(x, y, rank, v_init, optmethod='BFGS', reg_fact=0):
	"""
	Matrix logistic regression optimization
	:param x: input data, np.array((nb_data, nb_dim))
	:param y: output data, np.array((nb_data, nb_class))
	:param rank: rank of the weight matrices to optimize
	:param vinit: initial weight vector
	:param optmethod: scipy optimization method
	:param reg_fact: regularization factor
	:return: weight matrices and constant term
	"""
	nb_data = x.shape[0]
	nb_dim1 = x.shape[1]
	nb_dim2 = x.shape[2]
	nb_dim = nb_dim1 + nb_dim2
	nb_class = y.shape[1]

	# MLR function to optimize
	def func(v):
		# Recover parameters from vector v
		# vector v is composed as [ xhi_1, phi_1, psi_1, ..., xhi_c, phi_c, psi_c ]
		V = np.reshape(v, (nb_dim*rank+1, nb_class))
		xhi = V[0]
		phi = [np.reshape(V[1:nb_dim1*rank+1, i], (nb_dim1, rank)) for i in range(nb_class)]
		psi = [np.reshape(V[nb_dim1*rank+1::, i], (nb_dim2, rank)) for i in range(nb_class)]

		# Compute probabilitites
		inProdEst = np.zeros((nb_data, nb_class))
		for n in range(0, nb_data):
			for dim in range(0, nb_class):
				phipsiVec = np.dot(khatriRaoProd(psi[dim], phi[dim]), np.ones((rank, 1)))
				inProdEst[n, dim] = xhi[dim] + np.dot(phipsiVec[:, None].T, x[n].flatten())

		est = softmax(inProdEst)
		est += 1e-308

		# Regularization term
		reg_term = 0.0
		for dim in range(0, nb_class):
			reg_term += np.linalg.norm(psi[dim]) + np.linalg.norm(phi[dim])

		return -np.sum(y * np.log(est)) + reg_fact*reg_term

	# Gradient of the function to optimize
	def grad(v):
		# Recover parameters from vector v
		V = np.reshape(v, (nb_dim * rank + 1, nb_class))
		xhi = V[0]
		phi = [np.reshape(V[1:nb_dim1 * rank + 1, i], (nb_dim1, rank)) for i in range(nb_class)]
		psi = [np.reshape(V[nb_dim1 * rank + 1::, i], (nb_dim2, rank)) for i in range(nb_class)]

		# Compute probabilities
		inProdEst = np.zeros((nb_data, nb_class))
		for n in range(0, nb_data):
			for dim in range(0, nb_class):
				phipsiVec = np.dot(khatriRaoProd(psi[dim], phi[dim]), np.ones((rank, 1)))
				inProdEst[n, dim] = xhi[dim] + np.dot(phipsiVec[:, None].T, x[n].flatten())

		est = softmax(inProdEst)

		# Compute gradients
		grad_xhi = np.sum(est - y, axis=0).flatten()

		grad_phi = np.zeros((nb_dim1 * rank, nb_class))
		grad_psi = np.zeros((nb_dim2 * rank, nb_class))
		for dim in range(0, nb_class):
			xVecPsi = np.zeros((nb_data, nb_dim1 * rank))
			for n in range(0, nb_data):
				xVecPsi[n] = np.dot(x[n], psi[dim]).flatten()

			xVecPhi = np.zeros((nb_data, nb_dim2 * rank))
			for n in range(0, nb_data):
				xVecPhi[n] = np.dot(x[n].T, phi[dim]).flatten()

			grad_phi[:, dim] = np.dot(xVecPsi.T, (est[:, dim] - y[:, dim]))
			grad_psi[:, dim] = np.dot(xVecPhi.T, (est[:, dim] - y[:, dim]))

			# Regularization term
			grad_phi[:, dim] += 2*reg_fact*phi[dim].flatten()
			grad_psi[:, dim] += 2*reg_fact*psi[dim].flatten()

		return np.hstack((grad_xhi.flatten(), grad_phi.flatten(), grad_psi.flatten()))

	# Initialization
	if v_init is None:
		xhi_init = np.zeros((nb_class, 1))  # Constant term
		phi_init = np.ones((nb_dim1 * rank * nb_class, 1))  # Left multiplying matrix
		psi_init = np.ones((nb_dim2 * rank * nb_class, 1))  # Rght multiplying matrix
		v_init = np.vstack((xhi_init, phi_init, psi_init))  # Stack parameters in a vector (for optimization)

	# Optimization of the weight matrices and constant term
	opt = sc_opt.minimize(func, v_init, jac=grad, method=optmethod, options={'maxiter': 200, 'disp': True, 'gtol': 1e-4})
	v = opt.x

	V = np.reshape(v, (nb_dim * rank + 1, nb_class))
	xhi = V[0]
	phi = [np.reshape(V[1:nb_dim1 * rank + 1, i], (nb_dim1, rank)) for i in range(nb_class)]
	psi = [np.reshape(V[nb_dim1 * rank + 1::, i], (nb_dim2, rank)) for i in range(nb_class)]

	return xhi, phi, psi


def optTensLogReg(x, y, rank, v_init, optmethod='BFGS', reg_fact=0):
	"""
	Tensor logistic regression optimization
	:param x: input data, np.array((nb_data, dim 1, ...))
	:param y: output data, np.array((nb_data, nb_class))
	:param rank: rank of the weight matrices to optimize
	:param vinit: initial weight vector
	:param optmethod: scipy optimization method
	:param reg_fact: regularization factor
	:return: weight matrices and constant term
	"""
	nb_data = x.shape[0]
	dims = x.shape[1:]
	nb_dim = np.sum(np.array(dims))
	nb_class = y.shape[1]

	# MLR function to optimize
	def func(v):
		# Recover parameters from vector v
		# vector v is composed as [ beta_1, v1_1, v2_1, ..., beta_c, v1_c, v2_c, ... ]
		Vall = np.reshape(v, (nb_dim*rank+1, nb_class))
		beta = Vall[0]
		V = []
		start = 1
		end = dims[0]*rank+1
		for m in range(len(dims)):
			Vtmp = [np.reshape(Vall[start:end, i], (dims[m], rank)) for i in range(nb_class)]
			V.append(Vtmp)

			start = end
			if m < len(dims)-1:
				end += dims[m+1] * rank

		# Compute probabilitites
		inProdEst = np.zeros((nb_data, nb_class))
		for n in range(0, nb_data):
			for c in range(0, nb_class):
				vTmp = khatriRaoProd(V[1][c], V[0][c])
				for i in range(2, len(dims)):
					vTmp = khatriRaoProd(V[i][c], vTmp)
				vVec = np.dot(vTmp, np.ones((rank, 1)))
				inProdEst[n, c] = beta[c] + np.dot(vVec[:, None].T, x[n].flatten())

		est = softmax(inProdEst)
		est += 1e-308

		# Regularization term
		reg_term = 0.0
		for c in range(0, nb_class):
			for m in range(len(dims)):
				reg_term += np.linalg.norm(V[m][c])

		return -np.sum(y * np.log(est)) + reg_fact*reg_term

	# Gradient of the function to optimize
	def grad(v):
		# Recover parameters from vector v
		Vall = np.reshape(v, (nb_dim * rank + 1, nb_class))
		beta = Vall[0]
		V = []
		start = 1
		end = dims[0] * rank + 1
		for m in range(len(dims)):
			Vtmp = [np.reshape(Vall[start:end, i], (dims[m], rank)) for i in range(nb_class)]
			V.append(Vtmp)

			start = end
			if m < len(dims) - 1:
				end += dims[m + 1] * rank

		# Compute probabilities
		inProdEst = np.zeros((nb_data, nb_class))
		for n in range(0, nb_data):
			for c in range(0, nb_class):
				vTmp = khatriRaoProd(V[1][c], V[0][c])
				for i in range(2, len(dims)):
					vTmp = khatriRaoProd(V[i][c], vTmp)
				vVec = np.dot(vTmp, np.ones((rank, 1)))
				inProdEst[n, c] = beta[c] + np.dot(vVec[:, None].T, x[n].flatten())

		est = softmax(inProdEst)

		# Compute gradients
		grad_beta = np.sum(est - y, axis=0).flatten()

		grad_vec = grad_beta.flatten()

		gradV = [np.zeros((dims[m] * rank, nb_class)) for m in range(len(dims))]
		for c in range(nb_class):
			for m in range(len(dims)):
					# Compute Vm complement (VM o ... o Vm+1 o Vm-1 o ... o V1)
					if m is 0:
						vmComplement = V[1][c]
						for i in range(2, len(dims)):
							vmComplement = khatriRaoProd(V[i][c], vmComplement)
					else:
						vmComplement = V[0][c]
						for i in range(1, len(dims)):
							if i != m:
								vmComplement = khatriRaoProd(V[i][c], vmComplement)

					# Gradient
					zVec = np.zeros((nb_data, dims[m] * rank))
					for n in range(0, nb_data):
						zVec[n] = np.dot(tensor2mat(x[n], m), vmComplement).flatten()

					gradV[m][:, c] = np.dot(zVec.T, (est[:, c] - y[:, c]))

					# Regularization term
					gradV[m][:, c] += 2 * reg_fact * V[m][c].flatten()

		for m in range(len(dims)):
			grad_vec = np.hstack((grad_vec, gradV[m].flatten()))

		return grad_vec

	# Initialization
	if v_init is None:
		v_init = np.ones(((nb_dim * rank + 1)*nb_class, 1))  # Stack parameters in a vector (for optimization)

	# Optimization of the weight tensor and constant term
	opt = sc_opt.minimize(func, v_init, jac=grad, method=optmethod, options={'maxiter': 200, 'disp': True, 'gtol': 1e-4})
	v = opt.x

	Vall = np.reshape(v, (nb_dim * rank + 1, nb_class))
	beta = Vall[0]
	V = []
	start = 1
	end = dims[0] * rank + 1
	for m in range(len(dims)):
		Vtmp = [np.reshape(Vall[start:end, c], (dims[m], rank)) for c in range(nb_class)]
		V.append(Vtmp)

		start = end
		if m < len(dims) - 1:
			end += dims[m + 1] * rank

	return beta, V
