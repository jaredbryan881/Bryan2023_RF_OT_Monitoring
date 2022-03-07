import numpy as np
from scipy.spatial.distance import cdist

def create_grid(rf_arr, t_axis, tlim, npts):
	"""
	Create the time-amplitude grid for one component of the receiver function. 

	Arguments
	---------
	:param rf_arr: np.array
		Array of receiver functions with dimensions [time, event]
	:param t_axis: np.array
		Lag-times corresponding to dimension 0 of SS_arrs and nSS_arrs
	:param tlim: list
		Lower and upper time limits for the sliding time window
	:param npts: int
		Number of points in amplitude

	Returns
	-------
	:return T: np.array
		meshgrid for time
	:return Y: list
		meshgrid for amplitude
	"""
	# get points on the t-axis within the bounds set by tlim
	t_inds = (t_axis>=tlim[0]) & (t_axis<tlim[1])
	t_pts = t_axis[t_inds]
	# get points on the y-axis
	y_pts = np.linspace(np.min(rf_arr[:,t_inds]), np.max(rf_arr[:,t_inds]), npts)

	# create the grid
	T,Y=np.meshgrid(t_pts, y_pts)

	return T, Y

def rf_hist(rf_arr, grid):
	"""
	Compute an array of histogram counts for each time sample in an array of RFs

	Arguments
	---------
	:param rf_arr: np.array
		Receiver functions, dimensions are [event, time]
	:param grid: np.array
		Time-amplitude grid used to discretize the recevier functions

	Returns
	-------
	:return hist: np.array
		1D histograms of RF amplitudes for each lag-time
	"""
	T,Y=grid
	hist=np.array([np.histogram(rf_arr[:,i], bins=Y[:,0], density=False)[0] for i in range(rf_arr.shape[1])])
	hist=hist/np.sum(hist)

	return hist

def distance_matrix(grid, t_weight=1.0, norm=True):
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

	# compute pairwise distance between points on a 2d grid
	coords = np.array([t_weight*T.flatten(), Y.flatten()]).T

	T_coords = T.flatten()[...,np.newaxis]
	Y_coords = Y.flatten()[...,np.newaxis]
	M_t = cdist(T_coords,T_coords, metric='euclidean')
	M_a = cdist(Y_coords,Y_coords, metric='euclidean')
	M = M_t + t_weight*M_a
	M/=np.sum(M)

	return M