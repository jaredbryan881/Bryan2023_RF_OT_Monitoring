import numpy as np
import matplotlib.pyplot as plt

from telewavesim import utils as ut
import obspy

def main():
	# define parameters
	modfile = '../velocity_models/model_lohs.txt'
	wvtype = 'P' # incident wave type
	npts = 8193  # Number of samples
	dt = 0.01    # Sample distance in seconds
	baz = 0.0    # Back-azimuth direction in degrees (has no influence if model is isotropic)
	
	t_axis = np.linspace(-(npts//2)*dt, (npts//2)*dt, npts) # time axis
	tlim = [-2.5, 20.0] # time window
	t_inds = (t_axis >= tlim[0]) & (t_axis < tlim[1]) # corresponding indices
	flim = 1.0 # lowpass frequency
	slow = 0.05 # slowness

	savefig=False

	# load model
	model = ut.read_model(modfile)

	# generate displacement time series
	trxyz = ut.run_plane(model, slow, npts, dt, baz=baz, wvtype=wvtype, obs=False)
	# get transfer function
	tfs = ut.tf_from_xyz(trxyz, pvh=False)

	# extract the different components from the time series
	ntr = trxyz[0] # north
	etr = trxyz[1] # east
	ztr = trxyz[2] # vertical

	# copy to radial and transverse, and then rotate
	rtr = ntr.copy()
	ttr = etr.copy()
	rtr.data, ttr.data = ut.rotate_ne_rt(ntr.data, etr.data, baz)

	zrt_str = obspy.Stream(traces=[ztr,rtr,ttr])
	zrt_str.filter("lowpass", freq=flim, corners=2, zerophase=True)
	fig,axs=plt.subplots(3,1, sharex=True, sharey=True)
	axs[0].plot(np.arange(0.0,25.0,0.01), zrt_str[0].data[:2500], lw=2, c='k', label="Z")
	axs[1].plot(np.arange(0.0,25.0,0.01), zrt_str[1].data[:2500], lw=2, c='k', label="R")
	axs[2].plot(np.arange(0.0,25.0,0.01), zrt_str[2].data[:2500], lw=2, c='k', label="T")
	axs[2].set_xlabel("Time [s]")
	axs[2].set_xlim(0.0,25.0)
	axs[0].legend()
	axs[1].legend()
	axs[2].legend()
	plt.show()

	maxv = np.max(np.abs(ztr.data))
	maxr = np.max(np.abs(rtr.data))
	maxtf = np.max(np.abs(tfs[0].data))

	# plot the TF and RF
	fig, ax = plt.subplots(1,1, figsize=(10,5))
	ax.plot(t_axis, tfs[0].data*maxr/maxv/maxtf, lw=2, c='k', label="Radial Transfer Function")

	# bandpass filter to get receiver function
	rf = tfs[0].filter('lowpass', freq=flim, corners=2, zerophase=True)
	maxrf = np.max(np.abs(rf.data))

	ax.plot(t_axis, rf.data*maxr/maxv/maxrf, lw=2, c='crimson', label="Radial Receiver Function")

	ax.set_xlim(tlim[0], tlim[1])
	ax.set_xlabel("Time [s]")
	ax.legend(loc="upper right")
	if savefig:
		plt.savefig("TFandRF.pdf")
	plt.show()

if __name__=="__main__":
	main()