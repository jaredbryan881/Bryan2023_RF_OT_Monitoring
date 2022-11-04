import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy

from telewavesim import utils as ut

import sys
sys.path.append("../")
from distance_matrix import distance_matrix_1d
from sim_synth import simulate_RF

import ot

def main():
	# ----- Define parameters -----
	# Parameters for raw synthetic RFs
	modfile='../velocity_models/model_lohs.txt'
	wvtype='P' # incident wave type
	npts=8193 # Number of samples
	dt=0.05 # time discretization
	baz=0.0 # Back-azimuth direction in degrees (has no influence if model is isotropic)
	slow=0.05 # slowness limits
	dvlim=[-0.00,0.02] # Vs perturbation limits

	# Parameters for processed synthetic RFs
	t_axis = np.linspace(-(npts//2)*dt, (npts//2)*dt, npts) # time axis
	tlim = [-1.0, 10.0] # time window
	t_inds = (t_axis >= tlim[0]) & (t_axis < tlim[-1]) # corresponding indices
	npts_win = np.sum(t_inds) # number of points in the time window
	flim = 1.0 # bandpass frequencies
	save_figs=False

	# Parameters for optimal transport
	m=0.97

	sign=False

	# horizontal slowness (ray parameter) in s/km
	perts = np.linspace(dvlim[0], dvlim[1], 6)

	# ----- Calculate reference RF -----
	# Load model and calculate RF
	ref_model = ut.read_model(modfile)
	rf_ref_ts = simulate_RF(ref_model, slow, baz, npts, dt, freq=flim, vels=None).data

	# Turn 1D reference RF into a 2D point cloud via a time-amplitude scaling
	delta_t = np.max(t_axis[t_inds])-np.min(t_axis[t_inds])
	delta_a = np.max(rf_ref_ts)-np.min(rf_ref_ts)
	t_weight = (delta_t/delta_a)
	rf_ref = np.array([t_axis[t_inds], t_weight*rf_ref_ts[t_inds]]).T

	# ----- Calculate ensemble of RFs -----
	rfs_pert_vp = np.empty((len(perts), npts))
	rfs_pert_vs = np.empty((len(perts), npts))
	fig,axs=plt.subplots(5,1,figsize=(10,8), sharex=True)
	axs[0].plot(t_axis[t_inds], rf_ref_ts[t_inds], lw=2, c='k')
	for (i, pert) in enumerate(perts):
		# Get telewavesim model with given Vp perturbation
		pert_model_p = copy.deepcopy(ref_model)
		pert_model_p.vp[0]*=(1+pert)
		pert_model_p.update_tensor()

		# Calculate the RF and add noise
		rfs_pert_vp[i] = simulate_RF(pert_model_p, slow, baz, npts, dt, freq=flim, vels=None).data

		# Get telewavesim model with given Vs perturbation
		pert_model = copy.deepcopy(ref_model)
		pert_model.vs[0]*=(1+pert)
		pert_model.update_tensor()

		# Calculate the RF and add noise
		rfs_pert_vs[i] = simulate_RF(pert_model, slow, baz, npts, dt, freq=flim, vels=None).data

		# Turn 1D RF into 2D point cloud
		rf_cur = np.array([t_axis[t_inds], t_weight*rfs_pert_vs[i,t_inds]]).T

		# ----- Calculate the distance matrix -----
		M_t = distance_matrix_1d(t_axis[t_inds,np.newaxis], t_axis[t_inds,np.newaxis])
		M_a = distance_matrix_1d(rf_cur[:,1,np.newaxis], rf_ref[:,1,np.newaxis])
		M_tlp=distance_matrix_1d(rf_cur, rf_ref)

		# ----- Calculate the OT plan -----
		M=ot.dist(rf_cur, rf_ref) # GSOT distance matrix
		a=np.ones((npts_win,))/float(npts_win) # uniform distribution over reference points
		b=np.ones((npts_win,))/float(npts_win) # uniform distribution over current points
		p=ot.partial.partial_wasserstein(a,b,M_tlp,m=m)
		valid_inds = np.sum(p,axis=0)!=0

		# ----- Calculate the OT dist -----
		# Sign the OT distance depending on the direction of transport
		if sign:
			dt_sign=np.sign(t_axis[t_inds,np.newaxis]-t_axis[np.newaxis,t_inds])
			da_sign=np.sign(rf_cur[:,1,np.newaxis]-rf_ref[np.newaxis,:,1])
			M_t*=dt_sign
			M_a*=da_sign
		# Integrate over one dimension to leave distance time series
		d_t = np.sum(p*M_t,axis=0)
		d_a = np.sum(p*M_a,axis=0)
		d   = np.sum(p*M_tlp,axis=0)

		# plot the distance
		if pert==0:
			# reference RF
			c='k'
		else:
			# perturbed RFs
			if sign:
				c=cm.coolwarm(i/len(perts))
			else:
				c=cm.inferno(i/len(perts))

		# plot receiver functions
		axs[0].plot(t_axis[t_inds], rfs_pert_vp[i,t_inds], c=c, lw=2)
		axs[1].plot(t_axis[t_inds], rfs_pert_vs[i, t_inds], c=c, lw=2)
		# plot OT distances
		axs[2].plot(t_axis[t_inds][valid_inds], d_t[valid_inds], c=c, lw=2)
		axs[3].plot(t_axis[t_inds][valid_inds], d_a[valid_inds], c=c, lw=2)
		axs[4].plot(t_axis[t_inds][valid_inds], d[valid_inds], c=c, lw=2)

	# format axes
	axs[0].set_xlim(tlim[0],tlim[-1])
	axs[4].set_xlabel("Time [s]", fontsize=12)
	axs[2].set_ylabel(r"$\gamma c^t$", fontsize=12)
	axs[3].set_ylabel(r"$\gamma c^a$", fontsize=12)
	axs[4].set_ylabel(r"$\gamma (c^t+\lambda^2 c^a)$", fontsize=12)

	# make subplot annotations
	axs[0].annotate("a", (-0.91,0.025),   fontsize=16, weight='bold')
	axs[1].annotate("b", (-0.91,0.025),   fontsize=16, weight='bold')
	axs[2].annotate("c", (-0.91,1.15e-3), fontsize=16, weight='bold')
	axs[3].annotate("d", (-0.91,1.2e-3),  fontsize=16, weight='bold')
	axs[4].annotate("e", (-0.91,1.22e-3), fontsize=16, weight='bold')

	# get the axes in scientific noation
	axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
	axs[1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
	axs[2].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
	axs[3].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
	axs[4].ticklabel_format(axis="y", style="sci", scilimits=(0,0))

	plt.tight_layout()
	if save_figs:
		plt.savefig("RF_OT_dists_LOHS.pdf")
	plt.show()

if __name__=="__main__":
	main()