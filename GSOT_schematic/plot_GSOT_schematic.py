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
	plot_single_waveform_schematic()
	plot_distribution_schematic()

def plot_single_waveform_schematic():
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
	noise_level=0.0 # fraction of the range used for additive Gaussian noise

	# Parameters for optimal transport
	m=0.97

	save_figs=False

	# ----- Calculate reference RF -----
	# Load model and calculate RF
	ref_model = ut.read_model(modfile)

	pert_s=-0.02
	pert_model = copy.deepcopy(ref_model)
	pert_model.vs[0]*=(1+pert_s)
	pert_model.update_tensor()

	rf_ref=simulate_RF(ref_model, slow, baz, npts, dt, freq=flim, vels=None).data
	amax = np.max(np.abs(rf_ref))
	sigma=amax*noise_level

	delta_t=tlim[1]-tlim[0]
	delta_a=np.max(rf_ref)-np.min(rf_ref)
	t_weight=(delta_t/delta_a)

	# ----- Calculate RFs -----
	noise = np.random.normal(loc=0.0, scale=sigma, size=npts)
	rf_ref_ts = simulate_RF(ref_model, slow, baz, npts, dt, freq=flim, vels=None, noise=noise).data
	noise = np.random.normal(loc=0.0, scale=sigma, size=npts)
	rf_pert_ts = simulate_RF(pert_model, slow, baz, npts, dt, freq=flim, vels=None, noise=noise).data

	fig,axs=plt.subplots(4,1,figsize=(10,8), sharex=True)
	axs[0].plot(t_axis[t_inds], rf_ref_ts[t_inds], c='steelblue', lw=2)
	axs[0].plot(t_axis[t_inds], rf_pert_ts[t_inds], c='crimson', lw=2)
	axs[0].set_xlim(tlim[0], tlim[1])
	
	axs[1].scatter(t_axis[t_inds], t_weight*rf_ref_ts[t_inds], c='steelblue', s=5)
	axs[1].scatter(t_axis[t_inds], t_weight*rf_pert_ts[t_inds], c='crimson', s=5)
	axs[1].set_xlim(tlim[0], tlim[1])

	rf_ref=np.array([t_axis[t_inds], t_weight*rf_ref_ts[t_inds]]).T
	rf_pert=np.array([t_axis[t_inds], t_weight*rf_pert_ts[t_inds]]).T

	M_tlp=distance_matrix_1d(rf_ref, rf_pert)

	# Full EMD
	ot_map=partial_ot_dist_tlp(M_tlp, m=1.0)
	ot_map*=npts_win

	# iterate over times in first RF dist
	axs[2].scatter(t_axis[t_inds], t_weight*rf_ref_ts[t_inds], c='steelblue', s=5)
	axs[2].scatter(t_axis[t_inds], t_weight*rf_pert_ts[t_inds], c='crimson', s=5)
	axs[2].set_xlim(tlim[0], tlim[1])
	for i in range(npts_win):
		if np.sum(ot_map[i]==1)!=0:
			# coords of the source point
			vector_x = rf_ref[:,0][ot_map[i]==1]-rf_ref[:,0][i]
			vector_y = rf_pert[:,1][ot_map[i]==1]-rf_ref[:,1][i]

			axs[2].plot([rf_ref[:,0][i], rf_ref[:,0][i]+vector_x], [rf_ref[:,1][i], rf_ref[:,1][i]+vector_y], c='k')

	# Partial OT dist
	ot_map=partial_ot_dist_tlp(M_tlp, m=0.96)
	ot_map*=npts_win

	# iterate over times in first RF dist
	axs[3].scatter(t_axis[t_inds], t_weight*rf_ref_ts[t_inds], c='steelblue', s=5)
	axs[3].scatter(t_axis[t_inds], t_weight*rf_pert_ts[t_inds], c='crimson', s=5)
	axs[3].set_xlim(tlim[0], tlim[1])
	for i in range(npts_win):
		if np.sum(ot_map[i]==1)!=0:
			# coords of the source point
			vector_x = rf_ref[:,0][ot_map[i]==1]-rf_ref[:,0][i]
			vector_y = rf_pert[:,1][ot_map[i]==1]-rf_ref[:,1][i]

			axs[3].plot([rf_ref[:,0][i], rf_ref[:,0][i]+vector_x], [rf_ref[:,1][i], rf_ref[:,1][i]+vector_y], c='k')

	axs[3].set_xlabel("Time [s]", fontsize=16)

	axs[0].annotate("a", (-0.91,0.035),  fontsize=16, weight='bold')
	axs[1].annotate("b", (-0.91,6.9),    fontsize=16, weight='bold')
	axs[2].annotate("c", (-0.91,6.9),    fontsize=16, weight='bold')
	axs[3].annotate("d", (-0.91,6.9),    fontsize=16, weight='bold')

	plt.tight_layout()

	plt.show()

def plot_distribution_schematic():
	# ----- Define parameters -----
	# Parameters for raw synthetic RFs
	modfile='../velocity_models/model_lohs.txt'
	wvtype='P' # incident wave type
	npts=8193 # Number of samples
	dt=0.05 # time discretization
	baz=0.0 # Back-azimuth direction in degrees (has no influence if model is isotropic)
	plim=[0.04,0.08] # slowness

	np.random.seed(0)

	# Parameters for processed synthetic RFs
	t_axis = np.linspace(-(npts//2)*dt, (npts//2)*dt, npts) # time axis
	tlim = [-1.0, 10.0] # time window
	t_inds = (t_axis >= tlim[0]) & (t_axis < tlim[1]) # corresponding indices
	npts_win = np.sum(t_inds) # number of points in the time window
	flim = 1.0 # bandpass frequencies

	# Parameters for RF ensemble & noise
	noise_level=0.5 # fraction of the range used for additive Gaussian noise

	# Parameters for optimal transport
	m=0.9

	save_figs=False

	# ----- Calculate reference RF -----
	# Load model and calculate RF
	ref_model = ut.read_model(modfile)

	pert_s=-0.02
	pert_model = copy.deepcopy(ref_model)
	pert_model.vs[0]*=(1+pert_s)
	pert_model.update_tensor()

	n_amp = 51
	n_rfs = 500

	slows=np.random.uniform(plim[0],plim[1],n_rfs)

	rf_ref_test=simulate_RF(ref_model, np.mean(slows), baz, npts, dt, freq=flim, vels=None).data[t_inds]
	amax = np.max(np.abs(rf_ref_test))
	sigma=amax*noise_level

	delta_t=tlim[1]-tlim[0]
	delta_a=np.max(rf_ref_test)-np.min(rf_ref_test)
	t_weight=(delta_t/delta_a)

	# ----- Calculate ensemble of RFs -----
	rfs_ref  = np.empty((n_rfs, npts_win))
	rfs_pert = np.empty((n_rfs, npts_win))
	for i in range(n_rfs):
		slow=slows[i]
		noise = np.random.normal(loc=0.0, scale=sigma, size=npts)
		rfs_ref[i] = simulate_RF(ref_model, slow, baz, npts, dt, freq=flim, vels=None, noise=noise).data[t_inds]
		noise = np.random.normal(loc=0.0, scale=sigma, size=npts)
		rfs_pert[i] = simulate_RF(pert_model, slow, baz, npts, dt, freq=flim, vels=None, noise=noise).data[t_inds]

	# concatenate and time/calculate the OT distance
	rfs_ref_concat=np.array([np.tile(t_axis[t_inds], n_rfs), t_weight*rfs_ref.flatten()]).T
	rfs_pert_concat=np.array([np.tile(t_axis[t_inds], n_rfs), t_weight*rfs_pert.flatten()]).T
	
	G = create_grid(rfs_ref*1.1, t_axis[t_inds], n_amp)

	# convert RF arrays to RF distributions
	rfs_ref_dist = rf_hist(rfs_ref, G)
	rfs_pert_dist = rf_hist(rfs_pert, G)
	
	fig,axs=plt.subplots(4,1,figsize=(10,8),sharex=True)
	for i in range(n_rfs):
		axs[0].plot(t_axis[t_inds], rfs_ref[i], c='k', lw=1, alpha=0.1, rasterized=True)
	axs[0].set_xlim(tlim[0], tlim[1])
	
	axs[1].scatter(rfs_ref_concat[:,0], rfs_ref_concat[:,1], c='k', s=1, alpha=0.1, rasterized=True)
	axs[1].set_xlim(tlim[0], tlim[1])

	masked_array1 = np.ma.masked_where(rfs_ref_dist.T==0, rfs_ref_dist.T)
	cmap=copy.copy(cm.inferno)
	cmap.set_bad(color='white')
	axs[2].imshow(masked_array1, origin='lower', cmap=cmap, interpolation=None, extent=[tlim[0], tlim[1], t_weight*np.min(rfs_ref), t_weight*np.max(rfs_ref)], rasterized=True)
	axs[2].set_aspect("auto")

	nz_inds=np.where(rfs_pert_dist.T!=0)
	pert_im_times = G[0][nz_inds]
	pert_im_amps = G[1][nz_inds]
	pert_im_mass = rfs_pert_dist.T[nz_inds]
	pert_coords = np.array([pert_im_times, pert_im_amps]).T

	nz_inds=np.where(rfs_ref_dist.T!=0)
	ref_im_times = G[0][nz_inds]
	ref_im_amps = G[1][nz_inds]
	ref_im_mass = rfs_ref_dist.T[nz_inds]
	ref_coords = np.array([ref_im_times, ref_im_amps]).T

	print("Calculating OT plan")
	# define distance matrices for the full distribution
	M_t = distance_matrix_1d(pert_im_times[...,np.newaxis], ref_im_times[...,np.newaxis])
	M_a = distance_matrix_1d(pert_im_amps[...,np.newaxis], ref_im_amps[...,np.newaxis])
	M_tlp = np.sqrt((M_t**2) + (t_weight**2)*(M_a**2))
	ot_plan = partial_ot_dist_tlp(M_tlp, dist1=pert_im_mass, dist2=ref_im_mass, m=m, numItermax=500000)

	axs[3].imshow(masked_array1, origin='lower', cmap=cmap, interpolation=None, extent=[tlim[0], tlim[1], t_weight*np.min(rfs_ref), t_weight*np.max(rfs_ref)], rasterized=True)
	axs[3].set_aspect("auto")

	print("Calculating net vectors")
	npts_dist=np.sum(nz_inds)
	net_vectors=np.zeros((ot_plan.shape[1], 2))
	for i in range(ot_plan.shape[0]):
		for j in range(ot_plan.shape[1]):
			if ot_plan[i,j]==0:
				continue
			delta_t=pert_coords[i,0]-ref_coords[j,0]
			delta_a=pert_coords[i,1]-ref_coords[j,1]
			net_vectors[j,0]+=npts_dist*ot_plan[i,j]*delta_t
			net_vectors[j,1]+=npts_dist*ot_plan[i,j]*delta_a

	net_vectors_sparse=np.zeros(net_vectors.shape)
	net_vectors_sparse[::1,:]=net_vectors[::1,:]

	quiver_params = {"scale": 5e2, "color": 'w', "ec": 'k', "width": 0.002, "headwidth": 3, "linewidth": 0.1, "minlength": 0.01, "pivot": "tail"}
	axs[3].quiver(G[0][nz_inds], t_weight*G[1][nz_inds], net_vectors_sparse[:,0], t_weight*net_vectors_sparse[:,1], **quiver_params, rasterized=True)
	axs[3].set_xlabel("Time [s]", fontsize=16)


	axs[0].annotate("a", (-0.91,0.045),  fontsize=16, weight='bold')
	axs[1].annotate("b", (-0.91,10.1),   fontsize=16, weight='bold')
	axs[2].annotate("c", (-0.91,9.95),   fontsize=16, weight='bold')
	axs[3].annotate("d", (-0.91,9.8),    fontsize=16, weight='bold')

	plt.tight_layout()
	plt.show()

if __name__=="__main__":
	main()