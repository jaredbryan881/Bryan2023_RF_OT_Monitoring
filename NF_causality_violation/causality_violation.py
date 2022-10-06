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
from ot_distance import partial_ot_dist_tlp

def main():
	# Show a causality-violating transport map
	# Compare to a causality-preserving warping function

	# define parameters
	wvtype = 'P' # incident wave type
	npts = 8193  # Number of samples
	dt = 0.01
	baz = 0.0    # Back-azimuth direction in degrees (has no influence if model is isotropic)
	
	t_axis = np.linspace(-(npts//2)*dt, (npts//2)*dt, npts) # time axis
	tlim = [-1.0, 20.0] # time window
	t_inds = (t_axis >= tlim[0]) & (t_axis < tlim[1]) # corresponding indices
	npts_win = np.sum(t_inds)
	flim = 4.0 # lowpass frequency
	slow = 0.05 # slowness
	pert = -0.03
	mass=0.98

	save_fig=True

	# contrive a model whose perturbation induces a causality violation in the transport map
	ref_model = ut.Model(thickn = [10.0, 10.0, 0.],
						 rho    = [2800., 2800., 3200.],
						 vp     = [3.0, 5.0, 8.0],
						 vs     = [2.0, 3.0, 6.0])

	pert_model = copy.deepcopy(ref_model)
	pert_model.vs[1]*=(1-pert)
	pert_model.update_tensor()
	rf_ref = simulate_RF(pert_model, slow, baz, npts, dt, freq=flim, vels=None).data

	pert_model = copy.deepcopy(ref_model)
	pert_model.vs[1]*=(1+pert)
	pert_model.update_tensor()
	rf_pert = simulate_RF(pert_model, slow, baz, npts, dt, freq=flim, vels=None).data

	# set time-amplitude scaling
	delta_t = np.max(t_axis[t_inds])-np.min(t_axis[t_inds])
	delta_a = np.max(rf_ref)-np.min(rf_ref)
	t_weight = (delta_t/delta_a)
	# calculate distance matrices
	M_t = distance_matrix_1d(t_axis[t_inds,np.newaxis], t_axis[t_inds,np.newaxis])
	M_a = distance_matrix_1d(rf_ref[t_inds,np.newaxis], rf_pert[t_inds,np.newaxis])
	M_tlp = M_t + t_weight*M_a

	# solve OT problem
	ot_map = partial_ot_dist_tlp(M_tlp, m=mass, nb_dummies=20)

	# Initialize plot
	fig,axs=plt.subplots(1,1, figsize=(8,3))
	# plot RFs
	axs.scatter(t_axis[t_inds], rf_ref[t_inds], c='crimson', ec=None, s=20)
	axs.scatter(t_axis[t_inds], rf_pert[t_inds], c='steelblue', ec=None, s=20)

	# plot connections in the transport map
	for i in range(npts_win):
		if np.sum(ot_map[i]==1)!=0:
			# coords of the source point
			vector_x = t_axis[t_inds][ot_map[i]==1]-t_axis[t_inds][i]
			vector_y = rf_pert[t_inds][ot_map[i]==1]-rf_ref[t_inds][i]

			# plot connection
			axs.plot([t_axis[t_inds][i], t_axis[t_inds][i]+vector_x], [rf_ref[t_inds][i], rf_ref[t_inds][i]+vector_y], lw=1, c='k')

	# make the plot nicer
	axs.set_yticks([-0.01,0.00,0.01,0.02,0.03])
	axs.yaxis.set_tick_params(labelsize=12)
	axs.set_xlim(-1,10)
	axs.set_xticks([i for i in range(-1, 11)])
	axs.set_xlabel("Time [s]", fontsize=12)
	plt.xticks(fontsize=12)
	plt.subplots_adjust(hspace=0)
	plt.tight_layout()
	plt.show()


if __name__=="__main__":
	main()