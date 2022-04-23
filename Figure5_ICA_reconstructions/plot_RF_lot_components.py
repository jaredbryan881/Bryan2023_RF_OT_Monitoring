import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy

from telewavesim import utils as ut

import sys
sys.path.append("../")
from sim_synth import simulate_RF
sys.path.append("../../")
from pyppca.pyppca import ppca
from sklearn.decomposition import FastICA

import ot

def main():
	# ----- Define parameters -----
	# Parameters for raw synthetic RFs
	modfile='../velocity_models/model_lohs.txt'
	wvtype='P' # incident wave type
	npts=8193 # Number of samples
	dt=0.05 # time discretization
	baz=0.0 # Back-azimuth direction in degrees (has no influence if model is isotropic)
	plim=[0.04,0.06] # slowness limits
	dvlim=[-0.02,0.02] # Vs perturbation limits

	# Parameters for processed synthetic RFs
	t_axis = np.linspace(-(npts//2)*dt, (npts//2)*dt, npts) # time axis
	tlim = [-1.0, 10.0] # time window
	t_inds = (t_axis >= tlim[0]) & (t_axis < tlim[1]) # corresponding indices
	npts_win = np.sum(t_inds) # number of points in the time window
	flim = 1.0 # bandpass frequencies

	# Parameters for RF ensemble & noise
	n_rfs = 1000 # number of RFs in the synthetic distributions
	noise_level=0.0 # fraction of the range used for additive Gaussian noise

	# Parameters for optimal transport
	m=0.95

	ncomp=2

	save_figs=False

	# horizontal slowness (ray parameter) in s/km
	slows = np.random.uniform(low=plim[0], high=plim[1], size=n_rfs)
	perts_s = np.random.uniform(low=dvlim[0], high=dvlim[1], size=n_rfs)
	perts_p = np.random.uniform(low=dvlim[0], high=dvlim[1], size=n_rfs)*0

	# ----- Calculate reference RF -----
	# Load model and calculate RF
	ref_model = ut.read_model(modfile)
	rf_ref_ts = simulate_RF(ref_model, np.mean(slows), baz, npts, dt, freq=flim, vels=None).data

	# Turn 1D reference RF into a 2D point cloud via a time-amplitude scaling
	delta_t = np.max(t_axis[t_inds])-np.min(t_axis[t_inds])
	delta_a = np.max(rf_ref_ts)-np.min(rf_ref_ts)
	t_weight = delta_t/delta_a
	rf_ref = np.array([t_axis[t_inds], t_weight*rf_ref_ts[t_inds]]).T

	# ----- Calculate ensemble of RFs -----
	V=list()
	rfs_pert = np.empty((n_rfs, npts))
	for i in range(len(slows)):
		# Get random slowness and Vs perturbation
		slow = slows[i]
		pert_s = perts_s[i]
		pert_p = perts_p[i]

		# Get telewavesim model with given perturbation
		pert_model = copy.deepcopy(ref_model)
		pert_model.vs[0]*=(1+pert_s)
		pert_model.vp[0]*=(1+pert_p)
		pert_model.update_tensor()

		# Calculate the RF and add noise
		rf_pert = simulate_RF(pert_model, slow, baz, npts, dt, freq=flim, vels=None).data
		amax = np.max(np.abs(rf_pert))
		sigma=amax*noise_level
		noise = np.random.normal(loc=0.0, scale=sigma, size=npts)
		rfs_pert[i] = rf_pert + noise

		# Turn 1D RF into 2D point cloud
		rf_cur = np.array([t_axis[t_inds], t_weight*rfs_pert[i,t_inds]]).T

		# ----- Calculate the LOT embedding -----
		C=ot.dist(rf_cur, rf_ref) # GSOT distance matrix
		a=np.ones((npts_win,))/float(npts_win) # uniform distribution over reference points
		b=np.ones((npts_win,))/float(npts_win) # uniform distribution over current points
		p=ot.partial.partial_wasserstein(a,b,C,m=m)
		Vi=np.matmul((npts_win*p).T,rf_cur)-rf_ref

		# Find indices that mapped to the dummy points and set to NaN
		# We'll treat these as missing observations during the LOT section
		pruned_tinds = np.isclose(Vi[:,0]+rf_ref[:,0], 0)
		pruned_ainds = np.isclose(Vi[:,1]+rf_ref[:,1], 0)
		pruned_inds = pruned_tinds & pruned_ainds
		Vi[pruned_inds,:]=np.nan

		V.append(Vi)
	V=np.asarray(V)

	# ----- Perform PPCA in the LOT embedding space -----
	#ppca = PPCA(n_components=ncomp)
	C, ss, M, X, Ye = ppca(V.reshape(n_rfs,-1), d=ncomp, dia=False)

	# visuzalize the modes of variation back in waveform space
	stds = np.std(X,0)
	taus = np.linspace(-2,2,5)

	# ----- Perform ICA in LOT embedding space using reconstructed data -----
	# initialize ica object
	ica = FastICA(n_components=ncomp)
	# reconstruct signals
	S = ica.fit_transform(Ye.reshape(n_rfs,-1).T)
	# get estimated mixing matrix
	A = ica.mixing_

	stds = np.std(A,0)
	taus = np.linspace(-2,2,11)
	fig,axs=plt.subplots(ncomp,1,figsize=(15,2.5*ncomp),sharex=True)
	for i,vpca in enumerate(S.T):
		Vpca=vpca.reshape(npts_win,2)
		axs[i].set_ylabel("IC #{}".format(i+1), fontsize=12)
		for k,tau in enumerate(taus):
			samples=rf_ref+tau*stds[i]*Vpca

			if tau==0:
				color='k'
			else:
				color=cm.coolwarm(k/len(taus))

			axs[i].plot(samples[:,0], samples[:,1]/t_weight, lw=2, c=color)
	axs[0].set_xlim(tlim[0], tlim[1])
	axs[ncomp-1].set_xlabel("Time [s]", fontsize=12)
	plt.tight_layout()
	if save_figs:
		plt.savefig("Reconstructed_ICs.png")
		plt.savefig("Reconstructed_ICs.pdf")
	plt.show()

if __name__=="__main__":
	main()