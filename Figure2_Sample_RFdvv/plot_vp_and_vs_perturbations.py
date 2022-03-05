import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from telewavesim import utils as ut

import sys
sys.path.append("../")
from sim_synth import simulate_RF

def main():
	# define parameters
	modfile = '../velocity_models/model_lohs.txt'
	wvtype = 'P' # incident wave type
	npts = 8193  # Number of samples
	dt = 0.05    # sample distance in s
	baz = 0.0    # Back-azimuth direction in degrees (has no influence if model is isotropic)
	
	t_axis = np.linspace(-(npts//2)*dt, (npts//2)*dt, npts) # time axis
	tlim = [-1.0, 10.0] # time window
	t_inds = (t_axis >= tlim[0]) & (t_axis < tlim[1]) # corresponding indices
	npts_win = np.sum(t_inds)
	flim = 1.0 # lowpass frequency
	slow = 0.05 # horizontal slowness (ray parameter) in s/km

	save_fig=False

	perts = np.linspace(-0.02,0.02,11)

	# load model
	ref_model = ut.read_model(modfile)

	# Simulate RFs for a range of Vs perturbations in the upper layer
	rfs_pert_vs2 = np.empty((len(perts), npts))
	for (i,pert) in enumerate(perts):
		pert_model = copy.deepcopy(ref_model)
		pert_model.vs[0]*=(1+pert)
		pert_model.update_tensor()

		rfs_pert_vs2[i] = simulate_RF(pert_model, slow, baz, npts, dt, freq=flim, vels=None).data

	# Simulate RFs for a range of Vp perturbations in the upper layer
	rfs_pert_vp2 = np.empty((len(perts), npts))
	for (i,pert) in enumerate(perts):
		pert_model = copy.deepcopy(ref_model)
		pert_model.vp[0]*=(1+pert)
		pert_model.update_tensor()

		rfs_pert_vp2[i] = simulate_RF(pert_model, slow, baz, npts, dt, freq=flim, vels=None).data

	# Plot RFs
	fig1,axs=plt.subplots(2,1,sharex=True,sharey=True, figsize=(15,16/3))
	for (i,pert) in enumerate(perts):
		# Reference RF is black
		if pert==0:
			color='k'
		else:
			color=cm.coolwarm(i/len(perts))
		axs[0].plot(t_axis[t_inds], rfs_pert_vs2[i,t_inds], c=color, lw=2)
		axs[1].plot(t_axis[t_inds], rfs_pert_vp2[i,t_inds], c=color, lw=2)

	plt.subplots_adjust(hspace=0)
	plt.tight_layout()
	axs[0].set_xlim(-1,10)
	axs[1].set_xticks([-1,0,1,2,3,4,5,6,7,8,9,10])
	axs[1].set_xlabel("Time [s]", fontsize=12)
	if save_fig:
		plt.savefig("RF_Vs_and_Vp_perturbations.pdf")
	plt.show()


if __name__=="__main__":
	main()