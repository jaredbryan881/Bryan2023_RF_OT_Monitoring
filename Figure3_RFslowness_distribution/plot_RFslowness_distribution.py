import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from telewavesim import utils as ut

import sys
sys.path.append("../")
from sim_synth import simulate_RF
from distribution_utils import create_grid, rf_hist

def main():
	# define parameters
	modfile = '../velocity_models/model_lohs.txt'
	wvtype = 'P' # incident wave type
	npts=8193    # Number of samples
	dt = 0.05     # Sample distance in seconds
	baz = 0.0    # Back-azimuth direction in degrees (has no influence if model is isotropic)
	
	t_axis = np.linspace(-(npts//2)*dt, (npts//2)*dt, npts) # time axis
	tlim = [-1, 10.0] # time window
	t_inds = (t_axis >= tlim[0]) & (t_axis < tlim[1]) # corresponding indices
	flim = 1.0 # lowpass frequency
	plim = [0.04,0.08] # slowness limits
	n_rfs = 1000 # number of RFs in the synthetic distributions
	sigma_pct=0.2
	save_fig=False

	# load model
	ref_model = ut.read_model(modfile)

	# horizontal slowness (ray parameter) in s/km
	slows = np.random.uniform(low=plim[0], high=plim[1], size=n_rfs)

	fig,axs=plt.subplots(2,1,figsize=(10,5),sharex=True,sharey=True)

	# simulate default RFs at a range of slowness values
	rfs = np.empty((n_rfs, npts))
	for (i,slow) in enumerate(slows):
		rf = simulate_RF(ref_model, slow, baz, npts, dt, freq=flim, vels=None).data

		axs[0].plot(t_axis[t_inds], rf[t_inds], alpha=0.2, lw=2, c=cm.inferno((slow-plim[0])/(plim[1]-plim[0])))

		sigma = sigma_pct*np.max(np.abs(rf))
		noise = np.random.normal(loc=0.0, scale=sigma, size=npts)

		rfs[i] = rf + noise

	G = create_grid(rfs, t_axis, tlim, 51)

	# convert RF arrays to RF distributions
	rfs_dist = rf_hist(rfs[:, t_inds], G)

	# mask positions on the time-amplitude grid with zero trace crossings
	alim = [np.min(rfs), np.max(rfs)]
	masked_array1 = np.ma.masked_where(rfs_dist.T==0, rfs_dist.T)
	cmap=copy.copy(cm.inferno)
	cmap.set_bad(color='white')
	im2=axs[1].imshow(masked_array1, origin='lower', cmap=cmap, extent=[tlim[0], tlim[1], alim[0], alim[1]])
	axs[1].set_aspect('auto')

	axs[0].set_xlim(tlim[0], tlim[1])
	axs[1].set_xlabel("Time [s]")

	cax = make_axes_locatable(axs[0]).append_axes('right', size='2%', pad=0.05)
	fig.colorbar(im2, cax=cax, orientation='vertical') # change the limits in inkscape
	cax = make_axes_locatable(axs[1]).append_axes('right', size='2%', pad=0.05)
	fig.colorbar(im2, cax=cax, orientation='vertical')
	if save_fig:
		plt.savefig("RFslowness_distribution.pdf")
	plt.show()

if __name__=="__main__":
	main()