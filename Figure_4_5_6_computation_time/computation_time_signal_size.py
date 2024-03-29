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
	time_full_ensemble_vs_distribution()

def time_full_ensemble_vs_distribution():
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
	noise_level=0.3 # fraction of the range used for additive Gaussian noise

	# Parameters for optimal transport
	m=0.95

	save_figs=False

	# ----- Calculate reference RF -----
	# Load model and calculate RF
	ref_model = ut.read_model(modfile)

	pert_s=-0.05
	pert_model = copy.deepcopy(ref_model)
	pert_model.vs[0]*=(1+pert_s)
	pert_model.update_tensor()

	n_amps=np.logspace(1,2.5,10).astype("int")
	n_rfs_arr = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 26, 32, 40]

	t_full=np.zeros(len(n_rfs_arr))
	d_full=np.zeros(len(n_rfs_arr))
	t_dist=np.zeros(len(n_rfs_arr))
	d_dist=np.zeros(len(n_rfs_arr))

	rf_ref_test=simulate_RF(ref_model, slow, baz, npts, dt, freq=flim, vels=None).data[t_inds]
	amax = np.max(np.abs(rf_ref_test))
	sigma=amax*noise_level

	delta_t=tlim[1]-tlim[0]
	delta_a=np.max(rf_ref_test)-np.min(rf_ref_test)
	t_weight=delta_t/delta_a

	n_iters=100

	t_concat = np.zeros((n_iters, len(n_rfs_arr)))
	d_concat = np.zeros((n_iters, len(n_rfs_arr)))
	t_dist = np.zeros((n_iters, len(n_rfs_arr), len(n_amps)))
	d_dist = np.zeros((n_iters, len(n_rfs_arr), len(n_amps)))
	for i in range(n_iters):
		for (n,n_rfs) in enumerate(n_rfs_arr):
			print(n)
			n_rfs=int(n_rfs)

			# ----- Calculate ensemble of RFs -----
			rfs_ref  = np.empty((n_rfs, npts_win))
			rfs_pert = np.empty((n_rfs, npts_win))
			for j in range(n_rfs):
				noise = np.random.normal(loc=0.0, scale=sigma, size=npts)
				rfs_ref[j] = simulate_RF(ref_model, slow, baz, npts, dt, freq=flim, vels=None, noise=noise).data[t_inds]
				noise = np.random.normal(loc=0.0, scale=sigma, size=npts)
				rfs_pert[j] = simulate_RF(pert_model, slow, baz, npts, dt, freq=flim, vels=None, noise=noise).data[t_inds]

			# concatenate and time/calculate the OT distance
			rfs_ref_concat=np.array([np.tile(t_axis[t_inds], n_rfs), t_weight*rfs_ref.flatten()]).T
			rfs_pert_concat=np.array([np.tile(t_axis[t_inds], n_rfs), t_weight*rfs_pert.flatten()]).T

			M_tlp_concat = distance_matrix_1d(rfs_ref_concat, rfs_pert_concat)

			# Time and calculate full ensemble comparison
			t1=time.time()
			p_concat = partial_ot_dist_tlp(M_tlp_concat, m=m, numItermax=1e6, nb_dummies=20)
			d_concat[i, n] = np.sum(p_concat*M_tlp_concat)
			t2=time.time()
			t_concat[i, n]=t2-t1

			for (a,n_amp) in enumerate(n_amps):
				# create grid
				G = create_grid(rfs_ref*1.1, t_axis[t_inds], n_amp)

				# convert RF arrays to RF distributions
				rfs_ref_dist = rf_hist(rfs_ref, G)
				rfs_pert_dist = rf_hist(rfs_pert, G)

				"""
				if n==18:
					fig,ax=plt.subplots(1,1,figsize=(10,5))
					ax.scatter(rfs_ref_concat[:,0], rfs_ref_concat[:,1], c='k', s=1)
					plt.show()

					print("{} RFs: {} Amps".format(n_rfs, n_amp))
					masked_array1 = np.ma.masked_where(rfs_ref_dist.T==0, rfs_ref_dist.T)
					cmap=copy.copy(cm.inferno)
					cmap.set_bad(color='white')
					fig,ax=plt.subplots(1,1,figsize=(10,5))
					ax.imshow(masked_array1, origin='lower', cmap=cmap, interpolation=None)
					ax.set_aspect("auto")
					plt.show()
				"""

				nz_inds=np.where(rfs_ref_dist.T!=0)
				ref_im_times = G[0][nz_inds]
				ref_im_amps = G[1][nz_inds]
				ref_im_mass = rfs_ref_dist.T[nz_inds]
				ref_coords = np.array([ref_im_times, ref_im_amps])

				nz_inds=np.where(rfs_pert_dist.T!=0)
				pert_im_times = G[0][nz_inds]
				pert_im_amps = G[1][nz_inds]
				pert_im_mass = rfs_pert_dist.T[nz_inds]
				pert_coords = np.array([pert_im_times, pert_im_amps])

				# define distance matrices for the full distribution
				M_t_dist = distance_matrix_1d(ref_im_times[...,np.newaxis], pert_im_times[...,np.newaxis])
				M_a_dist = distance_matrix_1d(ref_im_amps[...,np.newaxis], pert_im_amps[...,np.newaxis])
				M_tlp_dist = np.sqrt((M_t_dist**2) + (t_weight**2)*(M_a_dist**2))

				# Calculate and time distribution comparison
				t1=time.time()
				p_dist = partial_ot_dist_tlp(M_tlp_dist, dist1=ref_im_mass, dist2=pert_im_mass, m=m, numItermax=1e6, nb_dummies=20)
				d_dist[i,n,a] = np.sum(p_dist*M_tlp_dist)
				t2=time.time()
				t_dist[i,n,a] = (t2-t1)

	t_concat=t_concat.mean(axis=0)
	t_dist=t_dist.mean(axis=0)

	# How does compute time change with n_rfs and n_amp
	fig,ax=plt.subplots(1,1,figsize=(10,10))
	ax.plot(n_rfs_arr, t_concat, color='green', marker='o')
	for a in range(len(n_amps)):
		ax.plot(n_rfs_arr, t_dist[:,a], color=cm.inferno(a/len(n_amps)), marker='o')

	quadratic=np.array(n_rfs_arr, dtype=float)**2
	quadratic*=(t_concat[0]/quadratic[0])
	ax.plot(n_rfs_arr, quadratic, lw=2, c='steelblue', linestyle='--')
	cubic=np.array(n_rfs_arr, dtype=float)**3
	cubic*=(t_concat[0]/cubic[0])
	ax.plot(n_rfs_arr, cubic, lw=2, c='crimson', linestyle='--')

	ax.set_ylabel("Time [s]", fontsize=14)
	ax.set_xlabel("Number of RFs", fontsize=14)
	ax.set_yscale('log')
	ax.set_xscale('log')
	plt.show()

	n_rfs_arr=np.array(n_rfs_arr, dtype=float)
	# How does error change with n_rfs and n_amp
	fig,axs=plt.subplots(2,1,figsize=(10,10), gridspec_kw={'height_ratios': [2,1]})
	# Iterate over N_a
	for a in range(len(n_amps)):
		# calculate percentage difference betweeen dist(M, N_a) and dist_exact
		cur_error = (np.abs(d_dist[:,:,a]-d_concat)*100)/d_concat
		cur_error = cur_error.mean(axis=0)

		# plot error as f(M=N_RFs) for current N_a
		axs[0].plot(n_rfs_arr, cur_error, color=cm.inferno(a/len(n_amps)), marker='o')
	
		# plot average error for current N_a
		axs[1].scatter(n_amps[a], np.mean(cur_error), c=cm.inferno(a/len(n_amps)))

	axs[0].plot(n_rfs_arr, n_rfs_arr-n_rfs_arr, c='green', lw=2)
	axs[0].set_ylabel("Error [%]", fontsize=14)
	axs[0].set_xlabel("Number of RFs", fontsize=14)
	axs[0].set_xscale('log')

	axs[1].set_ylabel("Error [%]", fontsize=14)
	axs[1].set_xlabel("Number of Amplitudes", fontsize=14)
	axs[1].set_xscale('log')
	plt.show()

if __name__=="__main__":
	main()