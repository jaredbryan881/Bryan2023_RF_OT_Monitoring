import numpy as np
import ot

def ot_dist_tlp(M, dist1=None, dist2=None):
	"""
	Arguments
	---------
	:param M: np.array
		Distance matrix
	:param dist1: np.array
		Source distribution. If left unspecified, assume uniform distribution.
	:param dist2: np.array
		Target distribution. If left unspecified, assume uniform distribution.

	Returns
	-------
	:return dist: float
		Transport Lp distance
	:return ot_map: np.array
		Optimal transport map
	"""
	# calculate optimal mapping
	if dist1 is None:
		dist1 = np.ones(M.shape[0])
	if dist2 is None:
		dist2 = np.ones(M.shape[1])
	ot_map = ot.emd(dist1, dist2, M, numItermax=500000)

	# calculate distance
	dist = np.sum(ot_map*M,axis=0)

	return dist, ot_map

def partial_ot_dist_tlp(M, dist1=None, dist2=None, m=0.99, numItermax=100000, nb_dummies=1):
	"""
	Arguments
	---------
	:param M: np.array
		Distance matrix
	:param dist1: np.array
		Source distribution. If left unspecified, assume uniform distribution.
	:param dist2: np.array
		Target distribution. If left unspecified, assume uniform distribution.
	:param m: float
		Fraction of mass in the source distribution to transport to the target distribution. Default is 99%.
	:param numItermax: int
		Maximum number of iterations in the optimal transport problem.
	param nb_dummies: int
		Number of dummy points used to allow partial transport.


	Returns
	-------
	:return dist: float
		Transport Lp distance
	:return ot_map: np.array
		Optimal transport map
	"""
	# calculate optimal mapping
	if dist1 is None:
		dist1 = np.ones(M.shape[0])/M.shape[0]
	if dist2 is None:
		dist2 = np.ones(M.shape[1])/M.shape[1]

	ot_map = ot.partial.partial_wasserstein(dist1, dist2, M, m=m, numItermax=numItermax, nb_dummies=nb_dummies)

	return ot_map