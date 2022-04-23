import numpy as np
import copy
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from telewavesim import utils as ut

import ot

import sys
sys.path.append("../../")
sys.path.append("../../../")
from sim_synth import simulate_RF

sys.path.append("../../../distance/")
from tlp_dist import distance_matrix as distance_matrix_1d
from image_dist import distance_matrix as distance_matrix_2d

from tlp_dist import distance_matrix_tlp, ot_dist_tlp, partial_ot_dist_tlp
from image_dist import create_grid, rf_hist, ot_dist_2d

def main():
	# define parameters
	modfile = '../../velocity_model/model_simple.txt'
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
	noise_levels=np.linspace(0.05,0.3,11)
	pert=-0.02
	m=0.95
	save_figs=False

	# load layer over halfspace model
	ref_model = ut.read_model(modfile)
	pert_model = copy.deepcopy(ref_model)
	pert_model.vs[0]*=(1+pert)
	pert_model.update_tensor()

	# initialize plots for
	# individual noisy RF
	fig1,axs1=plt.subplots(4,1,sharex=True,figsize=(10,10))
	# stacked ensemble of noisy RFs
	fig2,axs2=plt.subplots(4,1,sharex=True,figsize=(10,10))
	# distribution of noisy RFs
	fig3,axs3=plt.subplots(4,1,sharex=True,figsize=(10,10))

	for (n,noise_level) in enumerate(noise_levels[::-1]):
		print("Noise Level: {}".format(noise_level))
		# simulate default RFs at a range of slowness values
		rfs_ref = np.empty((n_rfs, npts))
		print("--Generating reference distribution")
		for i in range(n_rfs):
			rf_ref = simulate_RF(ref_model, slow, baz, npts, dt, freq=flim, vels=None).data
			amax = np.max(np.abs(rf_ref))
			sigma=amax*noise_level
			noise = np.random.normal(loc=0.0, scale=sigma, size=npts)

			rfs_ref[i] = rf_ref + noise

		G = create_grid(rfs_ref*1.1, t_axis, tlim, 51)
		M_t = distance_matrix_1d(t_axis[t_inds,np.newaxis], t_axis[t_inds,np.newaxis])

		# Time-amplitude scaling
		delta_t = np.max(t_axis[t_inds])-np.min(t_axis[t_inds])
		delta_a = np.max(rfs_ref)-np.min(rfs_ref)
		t_weight = delta_t/delta_a

		# convert RF arrays to RF distributions
		rfs_ref_hist = rf_hist(rfs_ref[:, t_inds], G)

		# Calculate the OT distance for the full distribution
		# simulate default RFs at a range of slowness values
		rfs_pert = np.empty((n_rfs, npts))
		print("--Generating perturbed distribution")
		for i in range(n_rfs):
			rf_pert = simulate_RF(pert_model, slow, baz, npts, dt, freq=flim, vels=None).data
			noise = np.random.normal(loc=0.0, scale=sigma, size=npts)

			rfs_pert[i] = rf_pert + noise

		# convert RF arrays to RF distributions
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
		M_tlp_nonstack = M_t_nonstack + t_weight*M_a_nonstack

		# Calculate OT distance for a distribution of noisy RFs
		ot_map = partial_ot_dist_tlp(M_tlp_nonstack, dist1=ref_im_mass, dist2=pert_im_mass, m=m)
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
		valid_inds = np.sum(ot_map,axis=0)!=0

		# Stack ensemble of noisy RFs
		rfs_ref_stack = np.mean(rfs_ref, axis=0)
		rfs_pert_stack = np.mean(rfs_pert, axis=0)
		
		color=cm.inferno_r(n/len(noise_levels))
		
		# plot distances from distribution
		axs1[0].fill_between(t_axis[t_inds], (rfs_pert_stack-2*np.std(rfs_pert, axis=0))[t_inds], (rfs_pert_stack+2*np.std(rfs_pert, axis=0))[t_inds], color=color)
		axs1[1].plot(t_axis[t_inds], dists_t, c=color, lw=2)
		axs1[2].plot(t_axis[t_inds], dists_a, c=color, lw=2)
		axs1[3].plot(t_axis[t_inds], dists, c=color, lw=2)

		# Calculate OT distance for a stack of noisy RFs
		M_a = distance_matrix_1d(rfs_ref_stack[t_inds,np.newaxis], rfs_pert_stack[t_inds,np.newaxis])
		M_tlp = M_t + t_weight*M_a
		ot_map = partial_ot_dist_tlp(M_tlp, m=m)
		dists = np.sum(ot_map*M_tlp,axis=0)
		dists_t = np.sum(ot_map*M_t,axis=0)
		dists_a = np.sum(ot_map*M_a,axis=0)
		valid_inds = np.sum(ot_map,axis=0)!=0
		
		# plot distances from stack
		axs2[0].plot(t_axis[t_inds][valid_inds], rfs_pert_stack[t_inds][valid_inds], lw=2, c=color)
		axs2[1].plot(t_axis[t_inds][valid_inds], dists_t[valid_inds], c=color, lw=2)
		axs2[2].plot(t_axis[t_inds][valid_inds], dists_a[valid_inds], c=color, lw=2)
		axs2[3].plot(t_axis[t_inds][valid_inds], dists[valid_inds], c=color, lw=2)

		# Calculate OT distance for a single noisy RF
		M_a = distance_matrix_1d(rfs_ref[0,t_inds,np.newaxis], rfs_pert[0,t_inds,np.newaxis])
		M_tlp = M_t + t_weight*M_a
		ot_map = partial_ot_dist_tlp(M_tlp, m=m)
		dists = np.sum(ot_map*M_tlp,axis=0)
		dists_t = np.sum(ot_map*M_t,axis=0)
		dists_a = np.sum(ot_map*M_a,axis=0)
		valid_inds = np.sum(ot_map,axis=0)!=0

		# plot distances from single noisy RF
		axs3[0].plot(t_axis[t_inds][valid_inds], rfs_pert[0,t_inds][valid_inds], c=color)
		axs3[1].plot(t_axis[t_inds][valid_inds], dists_t[valid_inds], c=color, lw=2)
		axs3[2].plot(t_axis[t_inds][valid_inds], dists_a[valid_inds], c=color, lw=2)
		axs3[3].plot(t_axis[t_inds][valid_inds], dists[valid_inds], c=color, lw=2)
	
	# Format axes for distribution of RFs
	axs1[0].set_xlim(tlim[0],tlim[-1])
	axs1[3].set_xlabel("Time [s]", fontsize=12)
	axs1[1].set_ylabel(r"$\gamma c_t$", fontsize=12)
	axs1[2].set_ylabel(r"$\gamma c_a$", fontsize=12)
	axs1[3].set_ylabel(r"$\gamma (c_t+\lambda c_a)$", fontsize=12)
	# get the axes in scientific noation
	axs1[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
	axs1[1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
	axs1[2].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
	axs1[3].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
	fig1.tight_layout()

	# Format axes for stack of RFs
	axs2[0].set_xlim(tlim[0],tlim[-1])
	axs2[3].set_xlabel("Time [s]", fontsize=12)
	axs2[1].set_ylabel(r"$\gamma c_t$", fontsize=12)
	axs2[2].set_ylabel(r"$\gamma c_a$", fontsize=12)
	axs2[3].set_ylabel(r"$\gamma (c_t+\lambda c_a)$", fontsize=12)
	# get the axes in scientific noation
	axs2[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
	axs2[1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
	axs2[2].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
	axs2[3].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
	fig2.tight_layout()

	# Format axes for single RF
	axs3[0].set_xlim(tlim[0],tlim[-1])
	axs3[3].set_xlabel("Time [s]", fontsize=12)
	axs3[1].set_ylabel(r"$\gamma c_t$", fontsize=12)
	axs3[2].set_ylabel(r"$\gamma c_a$", fontsize=12)
	axs3[3].set_ylabel(r"$\gamma (c_t+\lambda c_a)$", fontsize=12)
	# get the axes in scientific noation
	axs3[0].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
	axs3[1].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
	axs3[2].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
	axs3[3].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
	fig3.tight_layout()

	if save_figs:
		fig1.savefig("Distribution_dvv2pct_noise5-30pct.png")
		fig1.savefig("Distribution_dvv2pct_noise5-30pct.pdf")

		fig2.savefig("Stack_dvv2pct_noise5-30pct.png")
		fig2.savefig("Stack_dvv2pct_noise5-30pct.pdf")

		fig3.savefig("Single_dvv2pct_noise5-30pct.png")
		fig3.savefig("Single_dvv2pct_noise5-30pct.pdf")

	plt.show()

if __name__=="__main__":
	main()