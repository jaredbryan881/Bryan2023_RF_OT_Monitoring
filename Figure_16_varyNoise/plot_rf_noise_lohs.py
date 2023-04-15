import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from telewavesim import utils as ut

import ot

import sys
sys.path.append("../")
from sim_synth import simulate_RF

from distance_matrix import distance_matrix_1d,distance_matrix_2d
from distribution_utils import create_grid, rf_hist
from ot_distance import ot_dist_tlp, partial_ot_dist_tlp

def noise_psd(N, psd = lambda f: 1):
	X_white = np.fft.rfft(np.random.randn(N))
	S = psd(np.fft.rfftfreq(N))
	# Normalize S
	S = S / np.sqrt(np.mean(S**2))
	X_shaped = X_white * S

	noise = np.fft.irfft(X_shaped)

	if len(noise)<N:
		noise=np.append(noise,0)

	return noise

def PSDGenerator(f):
	return lambda N: noise_psd(N, f)

@PSDGenerator
def white_noise(f):
	return 1

@PSDGenerator
def pink_noise(f):
	return 1/np.where(f == 0, float('inf'), np.sqrt(f))

def snr(signal, noise):
	srms = np.sqrt(np.mean(np.square(signal)))
	nrms = np.sqrt(np.mean(np.square(noise)))
	return (srms*srms/nrms/nrms)

def main():
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
	flim = 1.0 # lowpass corner
	slow = 0.05 # slowness limits
	n_rfs = 100 # number of RFs in the synthetic distributions
	snr_goals=np.linspace(1,5,10)
	pert=-0.02
	m=0.98
	n_amp=51
	save_figs=False
	f_noise = pink_noise # white_noise

	# load layer over halfspace model
	ref_model = ut.read_model(modfile)
	pert_model = copy.deepcopy(ref_model)
	pert_model.vs[0]*=(1+pert)
	pert_model.update_tensor()

	# initialize plots for distribution of noise RFs
	fig1,axs1=plt.subplots(4,1,sharex=True,figsize=(10,10))

	for (n,snr_goal) in enumerate(snr_goals):
		print("SNR Goal: {}".format(snr_goal))
		# simulate reference RFs with realizations of noise
		rfs_ref = np.empty((n_rfs, npts))
		print("--Generating reference distribution")
		rf_ref_sample = simulate_RF(ref_model, slow, baz, npts, dt, freq=flim, vels=None).data
		amax = np.max(np.abs(rf_ref_sample))
		
		for i in range(n_rfs):
			rf_ref = simulate_RF(ref_model, slow, baz, npts, dt, freq=flim, vels=None, noise=None).data
			noise = f_noise(npts)
			noise*=(amax/noise.max())
			rfs_ref[i] = rf_ref + noise/snr_goal

		# simulate perturbed RFs with realizations of noise
		rfs_pert = np.empty((n_rfs, npts))
		print("--Generating perturbed distribution")
		for i in range(n_rfs):
			rf_pert = simulate_RF(pert_model, slow, baz, npts, dt, freq=flim, vels=None, noise=None).data
			noise = f_noise(npts)
			noise*=(amax/noise.max())
			rfs_pert[i] = rf_pert + noise/snr_goal

		# calculate grid
		G = create_grid(rfs_ref[:,t_inds]*1.1, t_axis[t_inds], n_amp)
		M_t = distance_matrix_1d(t_axis[t_inds,np.newaxis], t_axis[t_inds,np.newaxis])

		# Time-amplitude scaling
		delta_t = np.max(t_axis[t_inds])-np.min(t_axis[t_inds])
		delta_a = np.max(rfs_ref)-np.min(rfs_ref)
		t_weight = (delta_t/delta_a)

		# convert RF arrays to RF distributions
		rfs_ref_hist = rf_hist(rfs_ref[:, t_inds], G)
		rfs_pert_hist = rf_hist(rfs_pert[:, t_inds], G)

		# convert to 1D arrays
		nz_inds=np.where(rfs_ref_hist.T!=0)
		ref_im_times = G[0][nz_inds]
		ref_im_amps = G[1][nz_inds]
		ref_im_mass = rfs_ref_hist.T[nz_inds]
		ref_coords = np.array([ref_im_times, ref_im_amps])

		nz_inds=np.where(rfs_pert_hist.T!=0)
		pert_im_times = G[0][nz_inds]
		pert_im_amps = G[1][nz_inds]
		pert_im_mass = rfs_pert_hist.T[nz_inds]
		pert_coords = np.array([pert_im_times, pert_im_amps])

		# define distance matrices for the full distribution
		M_t_nonstack = distance_matrix_1d(ref_im_times[...,np.newaxis], pert_im_times[...,np.newaxis])
		M_a_nonstack = distance_matrix_1d(ref_im_amps[...,np.newaxis], pert_im_amps[...,np.newaxis])
		M_tlp_nonstack = np.sqrt((M_t_nonstack**2) + (t_weight**2)*(M_a_nonstack**2))

		# Calculate OT distance for a distribution of noisy RFs
		ot_map = partial_ot_dist_tlp(M_tlp_nonstack, dist1=ref_im_mass, dist2=pert_im_mass, m=m, numItermax=500000)
		ot_dist = ot_map*M_tlp_nonstack
		ot_dist_t = ot_map*M_t_nonstack
		ot_dist_a = ot_map*M_a_nonstack
		times = np.unique(ref_im_times)
		dists = np.zeros(times.shape)
		dists_t=np.zeros(times.shape)
		dists_a=np.zeros(times.shape)
		for (t,time) in enumerate(times):
			dists[t] = np.sum(ot_dist[ref_im_times==time])
			dists_t[t] = np.sum(ot_dist_t[ref_im_times==time])
			dists_a[t] = np.sum(ot_dist_a[ref_im_times==time])

		# Stack ensemble of noisy RFs
		rfs_ref_stack = np.mean(rfs_ref, axis=0)
		rfs_pert_stack = np.mean(rfs_pert, axis=0)
		
		color=cm.inferno_r(n/len(snr_goals))
		
		# plot distances from distribution
		axs1[0].fill_between(t_axis[t_inds], (rfs_ref_stack-2*np.std(rfs_ref, axis=0))[t_inds], (rfs_ref_stack+2*np.std(rfs_ref, axis=0))[t_inds], color=color)
		axs1[1].plot(t_axis[t_inds], dists_t, c=color, lw=2)
		axs1[2].plot(t_axis[t_inds], dists_a, c=color, lw=2)
		axs1[3].plot(t_axis[t_inds], dists, c=color, lw=2)
	
	# Format axes for distribution of RFs
	axs1[0].set_xlim(tlim[0],tlim[-1])
	axs1[3].set_xlabel("Time [s]", fontsize=12)
	axs1[1].set_ylabel(r"$\gamma c_t$", fontsize=12)
	axs1[2].set_ylabel(r"$\gamma c_a$", fontsize=12)
	axs1[3].set_ylabel(r"$\gamma (c_t+\lambda^2 c_a)$", fontsize=12)
	# get the axes in scientific noation
	axs1[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
	axs1[1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
	axs1[2].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
	axs1[3].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
	fig1.tight_layout()

	if save_figs:
		fig1.savefig("Distribution_dvv2pct_noise5-30pct.png")
		fig1.savefig("Distribution_dvv2pct_noise5-30pct.pdf")

	plt.show()

if __name__=="__main__":
	main()