import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy
import time

from telewavesim import utils as ut

import sys
sys.path.append("../")
from sim_synth import simulate_RF
from ot_distance import partial_ot_dist_tlp
from distribution_utils import create_grid, rf_hist
from distance_matrix import distance_matrix_1d, distance_matrix_2d

def main():
	plot_pot_vs_prune()

def plot_pot_vs_prune():
	# ----- Define parameters -----
	# Parameters for raw synthetic RFs
	modfile='../velocity_models/model_lohs.txt'
	wvtype='P' # incident wave type
	npts=8193 # Number of samples
	dt=0.05 # time discretization
	baz=0.0 # Back-azimuth direction in degrees (has no influence if model is isotropic)
	slow=0.06 # slowness

	np.random.seed(0)

	# Parameters for processed synthetic RFs
	t_axis = np.linspace(-(npts//2)*dt, (npts//2)*dt, npts) # time axis
	tlim = [-1.0, 10.0] # time window
	t_inds = (t_axis >= tlim[0]) & (t_axis < tlim[1]) # corresponding indices
	npts_win = np.sum(t_inds) # number of points in the time window
	flim = 1.0 # bandpass frequencies

	# Parameters for RF ensemble & noise
	noise_level=0.0 # fraction of the range used for additive Gaussian noise

	# Parameters for optimal transport
	m=0.95

	# ----- Calculate reference RF -----
	# Load model and calculate RF
	ref_model = ut.read_model(modfile)

	pert_s=-0.02
	pert_model = copy.deepcopy(ref_model)
	pert_model.vs[0]*=(1+pert_s)
	pert_model.update_tensor()

	rf_ref=simulate_RF(ref_model, slow, baz, npts, dt, freq=flim, vels=None).data
	amax = np.max(np.abs(rf_ref))
	sigma=amax*noise_level

	delta_t=tlim[1]-tlim[0]
	delta_a=np.max(rf_ref)-np.min(rf_ref)
	t_weight=(delta_t/delta_a)

	# ----- Calculate RFs -----
	noise = np.random.normal(loc=0.0, scale=sigma, size=npts)
	rf_ref_ts = simulate_RF(ref_model, slow, baz, npts, dt, freq=flim, vels=None, noise=noise).data
	noise = np.random.normal(loc=0.0, scale=sigma, size=npts)
	rf_pert_ts = simulate_RF(pert_model, slow, baz, npts, dt, freq=flim, vels=None, noise=noise).data

	rf_ref=np.array([t_axis[t_inds], t_weight*rf_ref_ts[t_inds]]).T
	rf_pert=np.array([t_axis[t_inds], t_weight*rf_pert_ts[t_inds]]).T

	M_tlp=distance_matrix_1d(rf_ref, rf_pert)

	# Full OT
	ot_map=partial_ot_dist_tlp(M_tlp, m=1.0)
	ot_map*=npts_win

	# iterate over times in first RF dist
	fig,axs=plt.subplots(3,1,figsize=(10,5),sharex=True)
	axs[0].scatter(t_axis[t_inds], t_weight*rf_ref_ts[t_inds], c='steelblue', s=5)
	axs[0].scatter(t_axis[t_inds], t_weight*rf_pert_ts[t_inds], c='crimson', s=5)
	axs[0].set_xlim(tlim[0], tlim[1])
	dists=np.zeros(npts_win)
	for i in range(npts_win):
		if np.sum(ot_map[i]==1)!=0:
			# coords of the source point
			vector_x = rf_ref[:,0][ot_map[i]==1]-rf_ref[:,0][i]
			vector_y = rf_pert[:,1][ot_map[i]==1]-rf_ref[:,1][i]

			dists[i]=vector_x**2 + vector_y**2

			axs[0].plot([rf_ref[:,0][i], rf_ref[:,0][i]+vector_x], [rf_ref[:,1][i], rf_ref[:,1][i]+vector_y], c='k')

	# Pruned full OT
	axs[1].scatter(t_axis[t_inds], t_weight*rf_ref_ts[t_inds], c='steelblue', s=5)
	axs[1].scatter(t_axis[t_inds], t_weight*rf_pert_ts[t_inds], c='crimson', s=5)
	axs[1].set_xlim(tlim[0], tlim[1])
	dist_95=np.percentile(dists, 95)
	for i in range(npts_win):
		if np.sum(ot_map[i]==1)!=0:
			if dists[i]>=dist_95:
				continue
			# coords of the source point
			vector_x = rf_ref[:,0][ot_map[i]==1]-rf_ref[:,0][i]
			vector_y = rf_pert[:,1][ot_map[i]==1]-rf_ref[:,1][i]

			axs[1].plot([rf_ref[:,0][i], rf_ref[:,0][i]+vector_x], [rf_ref[:,1][i], rf_ref[:,1][i]+vector_y], c='k')

	# Partial OT
	ot_map=partial_ot_dist_tlp(M_tlp, m=0.96)
	ot_map*=npts_win

	# iterate over times in first RF dist
	axs[2].scatter(t_axis[t_inds], t_weight*rf_ref_ts[t_inds], c='steelblue', s=5)
	axs[2].scatter(t_axis[t_inds], t_weight*rf_pert_ts[t_inds], c='crimson', s=5)
	axs[2].set_xlim(tlim[0], tlim[1])
	for i in range(npts_win):
		if np.sum(ot_map[i]==1)!=0:
			# coords of the source point
			vector_x = rf_ref[:,0][ot_map[i]==1]-rf_ref[:,0][i]
			vector_y = rf_pert[:,1][ot_map[i]==1]-rf_ref[:,1][i]

			axs[2].plot([rf_ref[:,0][i], rf_ref[:,0][i]+vector_x], [rf_ref[:,1][i], rf_ref[:,1][i]+vector_y], c='k')

	axs[2].set_xlabel("Time [s]", fontsize=12)

	axs[0].annotate("a", (-0.97,6.5), fontsize=12, weight='bold')
	axs[1].annotate("b", (-0.97,6.5), fontsize=12, weight='bold')
	axs[2].annotate("c", (-0.97,6.5), fontsize=12, weight='bold')

	axs[0].annotate("Full OT", (4,6.5), fontsize=12, weight='bold')
	axs[1].annotate("Pruned Full OT", (3.5,6.5), fontsize=12, weight='bold')
	axs[2].annotate("Partial OT", (3.8,6.5), fontsize=12, weight='bold')

	plt.tight_layout()
	plt.show()

if __name__=="__main__":
	main()