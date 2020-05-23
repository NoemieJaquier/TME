#! /usr/bin/env python
import numpy as np


def tensor2mat(T, rows=None, cols=None):
	"""
	Matricization of a tensor
	:param T: tensor
	:param rows: dimensions that are rows of the final matrix
	:param cols: dimensions that are columns of the final matrix
	:return: matrix
	"""
	sizeT = np.array(T.shape)
	N = len(sizeT)

	if rows is None and cols is None:
		rows = range(0, N)

	if rows is None:
		if not isinstance(cols, list):
			cols = [cols]
		rows = list(set(range(0, N)) - set(cols))

	if cols is None:
		if not isinstance(rows, list):
			rows = [rows]
		cols = list(set(range(0, N)) - set(rows))

	T = np.transpose(T, rows + cols)

	return np.reshape(T, (np.prod(sizeT[rows]), np.prod(sizeT[cols])))


def khatriRaoProd(A, B):
	"""
	Computes the Khatri-Rao product of two matrices
	:param A: matrix (I, F)
	:param B: matrix (J, F)
	:return: Khatri-Rao product (IJ, F)
	"""
	I = A.shape[0]
	J = B.shape[0]
	F = A.shape[1]
	F1 = B.shape[1]

	if F != F1:
		print('A and B must have an equal number of columns.')

	KRprod = np.zeros((I*J, F))
	for f in range(0, F):
		KRprod[:, f] = np.dot(B[:, f][:, None], A[:, f][:, None].T).flatten()

	return KRprod
