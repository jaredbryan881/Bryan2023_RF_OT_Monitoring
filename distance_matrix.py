import numpy as np
from scipy.spatial.distance import cdist

def distance_matrix_1d(arr1, arr2):
	"""
	Calculate pairwise distances between two arrays, using the euclidean distance

	Arguments
	---------
	:param arr1: np.array
		First signal
	:param arr2: np.array
		Second signal

	Returns
	-------
	:return M: np.array
		Distance matrix
	"""
	M = cdist(arr1, arr2, metric='euclidean')
	return M

def distance_matrix_2d(grid, t_weight=1.0, norm=True):
	"""
	Calculate the pairwise distance between each point on the time-amplitude grid.

	Arguments
	---------
	:param grid: list
		meshgrids corresponding to time and amplitude
	:param t_weight: float
		Factor by which to reduce the cost of transport in lag-time

	Returns
	-------
	:return M: np.array
		Pairwise distance between each point.
	"""
	# unpack components of the meshgrid
	T,Y = grid[0][:-1],grid[1][:-1]
	if norm:
		# normalize time and amplitude so transport is equally costly (across the whole image rather than per pixel)
		T=T/np.max(np.abs(T))
		Y=Y/np.max(np.abs(Y))

	T_coords = T.flatten()[...,np.newaxis]
	Y_coords = Y.flatten()[...,np.newaxis]
	M_t = distance_matrix_1d(T_coords, T_coords)
	M_a = distance_matrix_1d(Y_coords, Y_coords)
	M = M_t + t_weight*M_a
	M/=np.sum(M)

	return M