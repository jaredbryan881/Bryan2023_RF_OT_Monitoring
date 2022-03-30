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
	# Show the transport map for a range of time-amplitude scalings

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
	flim = 1.0 # lowpass frequency
	slow = 0.05 # slowness
	pert = -0.02
	mass=0.95

	save_fig=False

	# load model
	ref_model = ut.read_model(modfile)

	# simulate default RFs at a range of slowness values
	print("Generating reference distribution")
	rf_ref = simulate_RF(ref_model, slow, baz, npts, dt, freq=flim, vels=None).data

	delta_t = np.max(t_axis[t_inds])-np.min(t_axis[t_inds])
	delta_a = np.max(rf_ref)-np.min(rf_ref)
	t_weight = delta_t/delta_a

	pert_model = copy.deepcopy(ref_model)
	pert_model.vs[0]*=(1+pert)
	pert_model.update_tensor()

	# Calculate the OT distance for the full distribution
	# simulate default RFs at a range of slowness values
	rf_pert = simulate_RF(pert_model, slow, baz, npts, dt, freq=flim, vels=None).data

	M_t = distance_matrix_1d(t_axis[t_inds,np.newaxis], t_axis[t_inds,np.newaxis])
	M_a = distance_matrix_1d(rf_ref[t_inds,np.newaxis], rf_pert[t_inds,np.newaxis])

	# Individual RFs
	fig,axs=plt.subplots(3,1,sharex=True,sharey=True, figsize=(15,8))

	lambs = t_weight*np.array([0,1,10])
	for (l,lamb) in enumerate(lambs):
		print("Lambda={}".format(lamb))

		M_tlp = M_t + lamb*M_a
		ot_map = partial_ot_dist_tlp(M_tlp, m=mass)

		axs[l].scatter(t_axis[t_inds], rf_ref[t_inds], c='crimson', ec=None, s=20)
		axs[l].scatter(t_axis[t_inds], rf_pert[t_inds], c='steelblue', ec=None, s=20)

		# iterate over times in first RF dist
		for i in range(npts_win):
			if np.sum(ot_map[i]==1)!=0:
				# coords of the source point
				vector_x = t_axis[t_inds][ot_map[i]==1]-t_axis[t_inds][i]
				vector_y = rf_pert[t_inds][ot_map[i]==1]-rf_ref[t_inds][i]
				axs[l].plot([t_axis[t_inds][i], t_axis[t_inds][i]+vector_x], [rf_ref[t_inds][i], rf_ref[t_inds][i]+vector_y], c='k')

	plt.subplots_adjust(hspace=0)
	plt.tight_layout()
	axs[0].set_xlim(-1,10)

	if save_fig:
		plt.savefig("FigureA1_varyScaling.pdf")
	plt.show()


if __name__=="__main__":
	main()