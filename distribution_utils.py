import numpy as np

def create_grid(rf_arr, t_axis, npts):
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
	# get points on the y-axis
	y_pts = np.linspace(np.min(rf_arr), np.max(rf_arr), npts)

	# create the grid
	T,Y=np.meshgrid(t_axis, y_pts)

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