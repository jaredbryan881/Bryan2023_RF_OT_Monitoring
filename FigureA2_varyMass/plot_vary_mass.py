import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from telewavesim import utils as ut

import ot

import sys
sys.path.append("../")
from sim_synth import simulate_RF
from distance_matrix import distance_matrix_1d
from ot_distance import partial_ot_dist_tlp, ot_dist_tlp

def main():
	# Show the transport map for a range of masses

	# define parameters
	modfile = '../velocity_models/model_lohs.txt'
	wvtype = 'P' # incident wave type
	npts = 8193  # Number of samples
	dt = 0.05
	baz = 0.0    # Back-azimuth direction in degrees (has no influence if model is isotropic)
	
	t_axis = np.linspace(-(npts//2)*dt, (npts//2)*dt, npts) # time axis
	tlim = [-1.0, 10.0] # time window
	t_inds = (t_axis >= tlim[0]) & (t_axis < tlim[1]) # corresponding indices
	npts_win = np.sum(t_inds)
	flim = 1.0 # bandpass frequencies
	slow = 0.05 # slowness
	pert = -0.02
	noise_level=0.00

	save_fig=False

	# load model
	ref_model = ut.read_model(modfile)

	# simulate default RFs at a range of slowness values
	print("Generating reference distribution")
	rf_ref_ts = simulate_RF(ref_model, slow, baz, npts, dt, freq=flim, vels=None).data

	# noise parameters
	amax = np.max(np.abs(rf_ref_ts))
	sigma=amax*noise_level
	noise = np.random.normal(loc=0.0, scale=sigma, size=npts)
	rf_ref_ts = simulate_RF(ref_model, slow, baz, npts, dt, freq=flim, vels=None, noise=noise).data

	# time-amplitude scaling
	delta_t = np.max(t_axis[t_inds])-np.min(t_axis[t_inds])
	delta_a = np.max(rf_ref_ts)-np.min(rf_ref_ts)
	t_weight = (delta_t/delta_a)

	pert_model = copy.deepcopy(ref_model)
	pert_model.vs[0]*=(1+pert)
	pert_model.update_tensor()

	# Calculate the OT distance for the full distribution
	# simulate default RFs at a range of slowness values
	noise = np.random.normal(loc=0.0, scale=sigma, size=npts)
	rf_pert_ts = simulate_RF(pert_model, slow, baz, npts, dt, freq=flim, vels=None, noise=noise).data

	rf_ref=np.array([t_axis[t_inds], t_weight*rf_ref_ts[t_inds]]).T
	rf_pert=np.array([t_axis[t_inds], t_weight*rf_pert_ts[t_inds]]).T
	M_tlp=distance_matrix_1d(rf_ref, rf_pert)

	# Individual RFs
	fig,axs=plt.subplots(3,1,sharex=True,sharey=True, figsize=(15,8))

	masses = [0.9,0.97]
	for (m,mass) in enumerate(masses):
		print("--Mass={}".format(mass))

		# Stacked TLp
		ot_map = partial_ot_dist_tlp(M_tlp, m=mass)
		ot_map*=npts_win

		# source distribution and transport vectors
		axs[m].scatter(t_axis[t_inds], rf_ref_ts[t_inds], c='crimson', ec=None, s=20)
		axs[m].scatter(t_axis[t_inds], rf_pert_ts[t_inds], c='steelblue', ec=None, s=20)
		# iterate over times in first RF dist
		for i in range(npts_win):
			if np.sum(ot_map[i]==1)!=0:
				# coords of the source point
				vector_x = t_axis[t_inds][ot_map[i]==1]-t_axis[t_inds][i]
				vector_y = rf_pert_ts[t_inds][ot_map[i]==1]-rf_ref_ts[t_inds][i]
				axs[m].plot([t_axis[t_inds][i], t_axis[t_inds][i]+vector_x], [rf_ref_ts[t_inds][i], rf_ref_ts[t_inds][i]+vector_y], c='k')

	# Full OT for mass=1.0
	print("--Mass=1.0")
	_,ot_map = ot_dist_tlp(M_tlp)
	dists = np.mean(ot_map*M_tlp,axis=0)
	# source distribution and transport vectors
	axs[len(masses)].scatter(t_axis[t_inds], rf_ref_ts[t_inds], c='crimson', ec=None, s=20)
	axs[len(masses)].scatter(t_axis[t_inds], rf_pert_ts[t_inds], c='steelblue', ec=None, s=20)
	# iterate over times in first RF dist
	for i in range(npts_win):
		if np.sum(ot_map[i]==1)!=0:
			# coords of the source point
			vector_x = t_axis[t_inds][ot_map[i]==1]-t_axis[t_inds][i]
			vector_y = rf_pert_ts[t_inds][ot_map[i]==1]-rf_ref_ts[t_inds][i]
			axs[len(masses)].plot([t_axis[t_inds][i], t_axis[t_inds][i]+vector_x], [rf_ref_ts[t_inds][i], rf_ref_ts[t_inds][i]+vector_y], c='k')

	axs[0].annotate("a", (-0.95,0.028),  fontsize=16, weight='bold')
	axs[1].annotate("b", (-0.95,0.028),  fontsize=16, weight='bold')
	axs[2].annotate("c", (-0.95,0.028),  fontsize=16, weight='bold')

	plt.subplots_adjust(hspace=0)
	axs[0].set_xlim(-1,10)
	axs[0].set_yticks([-0.01,0.00,0.01,0.02,0.03])
	axs[1].set_yticks([-0.01,0.00,0.01,0.02,0.03])
	axs[2].set_yticks([-0.01,0.00,0.01,0.02,0.03])
	axs[2].set_xticks([-1,0,1,2,3,4,5,6,7,8,9,10])
	axs[2].set_xlabel("Time [s]", fontsize=12)
	plt.xticks(fontsize=12)
	axs[0].yaxis.set_tick_params(labelsize=12)
	axs[1].yaxis.set_tick_params(labelsize=12)
	axs[2].yaxis.set_tick_params(labelsize=12)

	plt.tight_layout()

	if save_fig:
		plt.savefig("FigureA2_varyMass.pdf")
	plt.show()


if __name__=="__main__":
	main()