import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy

from telewavesim import utils as ut

import sys
sys.path.append("../")
from sim_synth import simulate_RF
from distance_matrix import distance_matrix_1d
from ot_distance import partial_ot_dist_tlp
sys.path.append("../../")
from pyppca.pyppca import ppca
from sklearn.decomposition import FastICA

import ot

def main():
	vary_slowness_vs()
	vary_slowness_baz()

def vary_slowness_baz():
	# ----- Define parameters -----
	# Parameters for raw synthetic RFs
	modfile='../velocity_models/model_lohs_aniso.txt'
	wvtype='P' # incident wave type
	npts=8193 # Number of samples
	dt=0.05 # time discretization
	bazlim=[0.0,360.0] # Back-azimuth direction in degrees (has no influence if model is isotropic)
	plim=[0.04,0.08] # slowness limits
	dvlim=[-0.05,0.05] # Vs perturbation limits

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
	m=1.0

	ncomp=2

	save_figs=False

	# horizontal slowness (ray parameter) in s/km
	slows = np.random.uniform(low=plim[0], high=plim[1], size=n_rfs)
	bazs = np.random.uniform(low=bazlim[0], high=bazlim[1], size=n_rfs)

	# ----- Calculate reference RF -----
	# Load model and calculate RF
	ref_model = ut.read_model(modfile)
	rf_ref_ts = simulate_RF(ref_model, np.mean(slows), np.mean(bazs), npts, dt, freq=flim, vels=None).data

	# ----- Plot real forms of variation -----
	rfs_slow = np.zeros((11, npts))
	rfs_baz = np.zeros((11, npts))
	for (i,slow) in enumerate(np.linspace(plim[0], plim[1], 11)):
		rfs_slow[i] = simulate_RF(ref_model, slow, np.mean(bazs), npts, dt, freq=flim, vels=None).data
	for (i,baz) in enumerate(np.linspace(bazlim[0], bazlim[1]/2, 11)):
		rfs_baz[i] = simulate_RF(ref_model, np.mean(slows), baz, npts, dt, freq=flim, vels=None).data
	plot_true_signal_variation(rfs_slow, rfs_baz, t_axis, t_inds, foname="./figs/baz_slow/RF_true_signal_variation_bazslow.pdf")

	amax = np.max(np.abs(rf_ref_ts))
	sigma=amax*noise_level

	# Turn 1D reference RF into a 2D point cloud via a time-amplitude scaling
	delta_t = np.max(t_axis[t_inds])-np.min(t_axis[t_inds])
	delta_a = np.max(rf_ref_ts)-np.min(rf_ref_ts)
	t_weight = 0.1*(delta_t/delta_a)
	rf_ref = np.array([t_axis[t_inds], t_weight*rf_ref_ts[t_inds]]).T

	# ----- Calculate ensemble of RFs -----
	V=list()
	rfs_pert = np.empty((n_rfs, npts))
	for i in range(n_rfs):
		# Get random slowness and Vs perturbation
		slow = slows[i]
		baz  = bazs[i]

		# Calculate the RF and add noise
		noise = np.random.normal(loc=0.0, scale=sigma, size=npts)
		rf_pert = simulate_RF(ref_model, slow, baz, npts, dt, freq=flim, vels=None, noise=noise).data
		rfs_pert[i] = rf_pert + noise

		# Turn 1D RF into 2D point cloud
		rf_cur = np.array([t_axis[t_inds], t_weight*rfs_pert[i,t_inds]]).T

		# ----- Calculate the OT plan -----
		M_tlp = distance_matrix_1d(rf_cur, rf_ref)
		p=partial_ot_dist_tlp(M_tlp, m=m, nb_dummies=20)

		# ----- Calculate the LOT embedding -----
		Vi=np.matmul((npts_win*p).T,rf_cur)-rf_ref

		# Find indices that mapped to the dummy points and set to NaN
		# We'll treat these as missing observations during the LOT section
		pruned_inds=np.sum(p,axis=0)==0
		Vi[pruned_inds,:]=np.nan

		V.append(Vi)
	V=np.asarray(V)

	# ----- Perform PPCA in the LOT embedding space -----
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

	plot_ICA_signal_variation(A, S, t_weight, rf_ref, ncomp=ncomp, npts=npts_win, tlim=tlim, foname="./figs/baz_slow/embedding/RF_ICA_bazslow_embedding.pdf")
	plot_ic_scatter([slows, bazs], ica, ncomp=ncomp, labels=["Slowness [s/deg]", "Backazimuth [deg]"], cmaps=[cm.twilight,cm.inferno])


	# ----- Perform ICA in waveform space -----
	ica = FastICA(n_components=ncomp)
	# reconstruct signals
	S = ica.fit_transform(rfs_pert[:,t_inds].T)
	# get estimated mixing matrix
	A = ica.mixing_

	plot_ICA_signal_variation(A, S, t_weight, rf_ref_ts[t_inds], t_axis=t_axis[t_inds], ncomp=ncomp, npts=npts_win, tlim=tlim, foname="./figs/baz_slow/embedding/RF_ICA_bazslow_waveform.pdf")
	plot_ic_scatter([slows, bazs], ica, ncomp=ncomp, labels=["Slowness [s/deg]", "Backazimuth [deg]"], cmaps=[cm.twilight, cm.inferno])

def vary_slowness_vs():
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

	# ----- Calculate reference RF -----
	# Load model and calculate RF
	ref_model = ut.read_model(modfile)
	rf_ref_ts = simulate_RF(ref_model, np.mean(slows), baz, npts, dt, freq=flim, vels=None).data

	# ----- Plot real forms of variation -----
	rfs_slow = np.zeros((11, npts))
	rfs_pert = np.zeros((11, npts))
	for (i,slow) in enumerate(np.linspace(plim[0], plim[1], 11)):
		rfs_slow[i] = simulate_RF(ref_model, slow, baz, npts, dt, freq=flim, vels=None).data
	for (i,pert_s) in enumerate(np.linspace(dvlim[0], dvlim[1], 11)):
		pert_model = copy.deepcopy(ref_model)
		pert_model.vs[0]*=(1+pert_s)
		pert_model.update_tensor()
		rfs_pert[i] = simulate_RF(pert_model, np.mean(slows), baz, npts, dt, freq=flim, vels=None).data
	plot_true_signal_variation(rfs_slow, rfs_pert, t_axis, t_inds, foname="./figs/pert_slow/RF_true_signal_variation_pertslow.pdf")

	amax = np.max(np.abs(rf_ref_ts))
	sigma=amax*noise_level

	# Turn 1D reference RF into a 2D point cloud via a time-amplitude scaling
	delta_t = np.max(t_axis[t_inds])-np.min(t_axis[t_inds])
	delta_a = np.max(rf_ref_ts)-np.min(rf_ref_ts)
	t_weight = (delta_t/delta_a)
	rf_ref = np.array([t_axis[t_inds], t_weight*rf_ref_ts[t_inds]]).T

	# ----- Calculate ensemble of RFs -----
	V=list()
	rfs_pert = np.empty((n_rfs, npts))
	for i in range(n_rfs):
		# Get random slowness and Vs perturbation
		slow = slows[i]
		pert_s = perts_s[i]

		# Get telewavesim model with given perturbation
		pert_model = copy.deepcopy(ref_model)
		pert_model.vs[0]*=(1+pert_s)
		pert_model.update_tensor()

		# Calculate the RF and add noise
		noise = np.random.normal(loc=0.0, scale=sigma, size=npts)
		rf_pert = simulate_RF(pert_model, slow, baz, npts, dt, freq=flim, vels=None, noise=noise).data
		rfs_pert[i] = rf_pert + noise

		# Turn 1D RF into 2D point cloud
		rf_cur = np.array([t_axis[t_inds], t_weight*rfs_pert[i,t_inds]]).T

		# ----- Calculate the OT map -----
		M_tlp = distance_matrix_1d(rf_cur, rf_ref)
		p=partial_ot_dist_tlp(M_tlp, m=m, nb_dummies=20)

		# ----- Calculate the LOT embedding -----
		Vi=np.matmul((npts_win*p).T,rf_cur)-rf_ref

		# Find indices that mapped to the dummy points and set to NaN
		# We'll treat these as missing observations during the LOT section
		pruned_inds=np.sum(p,axis=0)==0
		Vi[pruned_inds,:]=np.nan

		V.append(Vi)
	V=np.asarray(V)

	# ----- Perform PPCA in the LOT embedding space -----
	C, ss, M, X, Ye = ppca(V.reshape(n_rfs,-1), d=ncomp, dia=False)
	# ----- Perform ICA in LOT embedding space using reconstructed data -----
	# initialize ica object
	ica = FastICA(n_components=ncomp)
	# reconstruct signals
	S = ica.fit_transform(Ye.reshape(n_rfs,-1).T)
	# get estimated mixing matrix
	A = ica.mixing_

	plot_ICA_signal_variation(A, S, t_weight, rf_ref, ncomp=ncomp, npts=npts_win, tlim=tlim, foname="./figs/pert_slow/embedding/RF_ICA_pertslow_embedding.pdf")
	plot_ic_scatter([slows, perts_s], ica, ncomp=ncomp, labels=["Slowness [s/deg]", r"$dV_S/V_S$"], cmaps=[cm.coolwarm,cm.inferno])


	# ----- Perform ICA in waveform space -----
	ica = FastICA(n_components=ncomp)
	# reconstruct signals
	S = ica.fit_transform(rfs_pert[:,t_inds].T)
	# get estimated mixing matrix
	A = ica.mixing_

	plot_ICA_signal_variation(A, S, t_weight, rf_ref_ts[t_inds], t_axis=t_axis[t_inds], ncomp=ncomp, npts=npts_win, tlim=tlim, foname="./figs/pert_slow/embedding/RF_ICA_pertslow_waveform.pdf")
	plot_ic_scatter([slows, perts_s], ica, ncomp=ncomp, labels=["Slowness [s/deg]", r"$dV_S/V_S$"], cmaps=[cm.coolwarm, cm.inferno])

def plot_true_signal_variation(rf1, rf2, t_axis, t_inds, foname=None):
	fig,axs=plt.subplots(2,1,figsize=(15,5), sharex=True)
	for i in range(11):
		if i==5:
			c='k'
		else:
			c=cm.coolwarm(i/11)
		axs[0].plot(t_axis[t_inds], rf1[i,t_inds], c=c, lw=2)
		axs[1].plot(t_axis[t_inds], rf2[i,t_inds], c=c, lw=2)

	for ax in axs:
		for axis in ['top', 'bottom', 'left', 'right']:
			ax.spines[axis].set_linewidth(1.0)
			ax.tick_params(axis='both', which='major', labelsize=12)

	axs[1].set_xlim(t_axis[t_inds][0], t_axis[t_inds][-1])
	axs[1].set_xlabel("Time [s]", fontsize=16)

	if foname is not None:
		plt.savefig(foname)
		plt.close()
	else:
		plt.show()


def plot_ICA_signal_variation(A, S, t_weight, ref_signal, t_axis=None, ncomp=2, npts=220, tlim=[-1,10], foname=None):
	# visuzalize the modes of variation back in waveform space
	stds = np.std(A,0)
	taus = np.linspace(-2,2,11)
	mcs = np.linspace(np.min(A,0), np.max(A,0), 11)
	fig,axs=plt.subplots(ncomp,1,figsize=(15,2.5*ncomp),sharex=True)
	for i,vica in enumerate(S.T):
		if t_axis is not None:
			pass
		else:
			vica=vica.reshape(npts,2)

		axs[i].set_ylabel("IC #{}".format(i+1), fontsize=16)
		for k,tau in enumerate(taus):
			#samples=ref_signal+tau*stds[i]*vica
			samples=ref_signal+mcs[k,i]*vica

			color = 'k' if tau==0 else cm.coolwarm(k/len(taus))

			if t_axis is not None:
				axs[i].plot(t_axis, samples/t_weight, lw=2, c=color)
			else:
				axs[i].plot(samples[:,0], samples[:,1]/t_weight, lw=2, c=color)

	for ax in axs:
		for axis in ['top', 'bottom', 'left', 'right']:
			ax.spines[axis].set_linewidth(1.0)
			ax.tick_params(axis='both', which='major', labelsize=16)

	axs[0].set_xlim(tlim[0], tlim[1])
	axs[-1].set_xlabel("Time [s]", fontsize=16)
	plt.tight_layout()
	if foname is not None:
		plt.savefig(foname)
		plt.close()
	else:
		plt.show()

def plot_ic_scatter(quants, ica, ncomp=2, labels=["Slowness [s/deg]", r"$dV_S/V_S$"], cmaps=[cm.coolwarm, cm.inferno], foname=None):
	# Plot mixing coefficients
	# embedding space
	Q=len(quants)
	fig,axs=plt.subplots(ncomp,2,figsize=(10,5))
	for i in range(ncomp):
		for j in range(Q):
			axs[i,j].scatter(quants[j], ica.components_[i], c=cmaps[j]((quants[Q-j-1]-quants[Q-j-1].min())/(quants[Q-j-1].max()-quants[Q-j-1].min())), s=5, alpha=0.5)
			if j==0:
				axs[i,0].set_ylabel("IC #{}".format(i+1), fontsize=14)
			if i==ncomp:
				axs[-1,0].set_xlabel(labels[j], fontsize=14)

	for ax_row in axs:
		for ax in ax_row:
			for axis in ['top', 'bottom', 'left', 'right']:
				ax.spines[axis].set_linewidth(1.0)
				ax.tick_params(axis='both', which='major', labelsize=12)

	if foname is not None:
		plt.savefig(foname)
		plt.close()
	else:
		plt.show()


if __name__=="__main__":
	main()