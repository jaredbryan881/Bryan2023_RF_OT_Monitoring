import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy

from telewavesim import utils as ut

import sys
sys.path.append("../")
from distribution_utils import distance_matrix_1d
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
	dvlim=[0.00,-0.02] # Vs perturbation limits

	# Parameters for processed synthetic RFs
	t_axis = np.linspace(-(npts//2)*dt, (npts//2)*dt, npts) # time axis
	tlim = [-1.0, 10.0] # time window
	t_inds = (t_axis >= tlim[0]) & (t_axis < tlim[-1]) # corresponding indices
	npts_win = np.sum(t_inds) # number of points in the time window
	flim = 1.0 # bandpass frequencies
	save_figs=False

	# Parameters for optimal transport
	m=0.95

	# horizontal slowness (ray parameter) in s/km
	perts = np.linspace(dvlim[0], dvlim[1], 11)

	# ----- Calculate reference RF -----
	# Load model and calculate RF
	ref_model = ut.read_model(modfile)
	rf_ref_ts = simulate_RF(ref_model, slow, baz, npts, dt, freq=flim, vels=None).data

	# Turn 1D reference RF into a 2D point cloud via a time-amplitude scaling
	delta_t = np.max(t_axis[t_inds])-np.min(t_axis[t_inds])
	delta_a = np.max(rf_ref_ts)-np.min(rf_ref_ts)
	t_weight = delta_t/delta_a
	rf_ref = np.array([t_axis[t_inds], rf_ref_ts[t_inds]]).T

	# ----- Calculate ensemble of RFs -----
	rfs_pert = np.empty((len(perts), npts))
	fig,axs=plt.subplots(4,1,figsize=(10,8), sharex=True)
	axs[0].plot(t_axis[t_inds], rf_ref_ts[t_inds], lw=2, c='k')
	for (i, pert) in enumerate(perts):
		# Get telewavesim model with given perturbation
		pert_model = copy.deepcopy(ref_model)
		pert_model.vs[0]*=(1+pert)
		pert_model.update_tensor()

		# Calculate the RF and add noise
		rfs_pert[i] = simulate_RF(pert_model, slow, baz, npts, dt, freq=flim, vels=None).data

		# Turn 1D RF into 2D point cloud
		rf_cur = np.array([t_axis[t_inds], rfs_pert[i,t_inds]]).T

		# ----- Calculate the distance matrix -----
		M_t = distance_matrix_1d(t_axis[t_inds,np.newaxis], t_axis[t_inds,np.newaxis])
		M_a = distance_matrix_1d(rf_cur[:,1,np.newaxis], rf_ref[:,1,np.newaxis])
		M_tlp = M_t + t_weight*M_a

		# ----- Calculate the OT plan -----
		M=ot.dist(rf_cur, rf_ref) # GSOT distance matrix
		a=np.ones((npts_win,))/float(npts_win) # uniform distribution over reference points
		b=np.ones((npts_win,))/float(npts_win) # uniform distribution over current points
		p=ot.partial.partial_wasserstein(a,b,M_tlp,m=m)
		valid_inds = np.sum(p,axis=0)!=0

		# ----- Calculate the OT dist -----
		d_t=np.sum(p*M_t,axis=0)
		d_a=np.sum(p*M_a,axis=0)
		d=np.sum(p*M_tlp,axis=0)

		# plot the distance
		if pert==0:
			c='k'
		else:
			c=cm.inferno(i/len(perts))

		axs[0].plot(t_axis[t_inds], rfs_pert[i, t_inds], c=c, lw=2)
		axs[1].plot(t_axis[t_inds][valid_inds], d_t[valid_inds], c=c, lw=2)
		axs[2].plot(t_axis[t_inds][valid_inds], d_a[valid_inds], c=c, lw=2)
		axs[3].plot(t_axis[t_inds][valid_inds], d[valid_inds], c=c, lw=2)

	axs[0].set_xlim(tlim[0],tlim[-1])
	axs[3].set_xlabel("Time [s]", fontsize=12)
	axs[1].set_ylabel(r"$\gamma c_t$", fontsize=12)
	axs[2].set_ylabel(r"$\gamma c_a$", fontsize=12)
	axs[3].set_ylabel(r"$\gamma (c_t+\lambda c_a)$", fontsize=12)

	# get the axes in scientific noation
	axs[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
	axs[1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
	axs[2].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
	axs[3].ticklabel_format(axis="y", style="sci", scilimits=(0,0))

	plt.tight_layout()
	if save_figs:
		plt.savefig("RF_OT_dists_LOHS.pdf")
	plt.show()

if __name__=="__main__":
	main()