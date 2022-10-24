import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy
import time

from scipy.stats import gaussian_kde
from scipy.interpolate import interp1d

from telewavesim import utils as ut

import sys
sys.path.append("../")
from sim_synth import simulate_RF

import ot

def main():
	#linearization_error_RFs()
	compare_transport_maps()

def linearization_error_RFs():
	# ----- Define parameters -----
	# Parameters for raw synthetic RFs
	modfile='../velocity_models/model_lohs.txt'
	wvtype='P' # incident wave type
	npts=8193 # Number of samples
	dt=0.05 # time discretization
	baz=0.0 # Back-azimuth direction in degrees (has no influence if model is isotropic)
	plim=[0.04,0.08] # slowness limits

	np.random.seed(0)

	# Parameters for processed synthetic RFs
	t_axis = np.linspace(-(npts//2)*dt, (npts//2)*dt, npts) # time axis
	tlim = [-1.0, 10.0] # time window
	t_inds = (t_axis >= tlim[0]) & (t_axis < tlim[1]) # corresponding indices
	npts_win = np.sum(t_inds) # number of points in the time window
	flim = 1.0 # bandpass frequencies

	# Parameters for RF ensemble & noise
	n_rfs = 100 # number of RFs in the synthetic distributions
	noise_level=0.0 # fraction of the range used for additive Gaussian noise

	# Parameters for optimal transport
	m=0.95

	save_figs=False

	# horizontal slowness (ray parameter) in s/km
	slows = np.random.uniform(low=plim[0], high=plim[1], size=n_rfs)
	perts_s = np.random.uniform(low=-0.05, high=0.05, size=n_rfs)

	# ----- Calculate reference RF -----
	# Load model and calculate RF
	ref_model = ut.read_model(modfile)
	rf_ref_ts = simulate_RF(ref_model, np.mean(slows), baz, npts, dt, freq=flim, vels=None).data

	# Turn 1D reference RF into a 2D point cloud via a time-amplitude scaling
	delta_t = np.max(t_axis[t_inds])-np.min(t_axis[t_inds])
	delta_a = np.max(rf_ref_ts)-np.min(rf_ref_ts)
	t_weight = (delta_t/delta_a)
	rf_ref = np.array([t_axis[t_inds], t_weight*rf_ref_ts[t_inds]]).T

	# ----- Calculate ensemble of RFs -----
	Vs=list()
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
		rf_pert = simulate_RF(pert_model, slow, baz, npts, dt, freq=flim, vels=None).data
		amax = np.max(np.abs(rf_pert))
		sigma=amax*noise_level
		noise = np.random.normal(loc=0.0, scale=sigma, size=npts)
		rfs_pert[i] = rf_pert + noise

		# Turn 1D RF into 2D point cloud
		rf_cur = np.array([t_axis[t_inds], t_weight*rfs_pert[i,t_inds]]).T

		# ----- Calculate the LOT embedding -----
		C=ot.dist(rf_cur, rf_ref, metric='euclidean') # GSOT distance matrix
		a=np.ones((npts_win,))/float(npts_win) # uniform distribution over reference points
		b=np.ones((npts_win,))/float(npts_win) # uniform distribution over current points
		p=ot.partial.partial_wasserstein(a,b,C,m=m)
		
		Vi=(np.matmul((npts_win*p).T,rf_cur)-rf_ref)#/np.sqrt(npts_win)

		# Find indices that mapped to the dummy points and set to NaN
		# We'll treat these as missing observations during the LOT section
		pruned_inds=np.sum(p,axis=0)==0
		Vi[pruned_inds,:]=np.nan

		Vs.append(Vi)
	Vs=np.asarray(Vs)

	# Calculate the distance matrix
	ind=0
	d_lot=np.zeros(int(n_rfs*(n_rfs-1)/2))
	d_ot=np.zeros(int(n_rfs*(n_rfs-1)/2))

	for i in range(n_rfs):
		print(i)
		rf1=np.array([t_axis[t_inds], t_weight*rfs_pert[i,t_inds]]).T
		for j in range(i+1,n_rfs):
			rf2=np.array([t_axis[t_inds], t_weight*rfs_pert[j,t_inds]]).T

			# calculate the OT distance directly
			C=ot.dist(rf1, rf2, metric='euclidean') # GSOT distance matrix
			a=np.ones((npts_win,))/float(npts_win) # uniform distribution over reference points
			b=np.ones((npts_win,))/float(npts_win) # uniform distribution over current points
			p=ot.partial.partial_wasserstein(a,b,C,m=m)
			d_ot[ind]=np.sum(p*C)

			# calculate the OT distance in the embedding space
			V_diff=Vs[j]-Vs[i]
			d_lot[ind]=np.sqrt(np.nansum((V_diff/np.sqrt(npts_win))**2))

			ind+=1

	fig,axs=plt.subplots(2,1)
	dist_dens=np.vstack([d_ot,d_lot])
	dist_dens_c = gaussian_kde(dist_dens)(dist_dens)
	axs[0].plot(d_ot, d_ot, c='crimson', linestyle='--', lw=2)
	axs[0].scatter(d_ot, d_lot, c=cm.inferno(dist_dens_c/dist_dens_c.max()))
	axs[0].set_xlabel(r"$d_{OT}$", fontsize=12)
	axs[0].set_ylabel(r"$d_{LOT}$", fontsize=12)

	error=(d_lot-d_ot)/d_ot
	error_dens=np.vstack([d_ot, error])
	error_dens_c = gaussian_kde(error_dens)(error_dens)
	axs[1].plot(d_ot, d_ot-d_ot, c='crimson', linestyle='--', lw=2)
	axs[1].scatter(d_ot, error*100, c=cm.inferno(error_dens_c/error_dens_c.max()))
	axs[1].set_xlabel(r"$d_{OT}$", fontsize=12)
	axs[1].set_ylabel("error [%]", fontsize=12)
	plt.show()


def compare_transport_maps():
	# ----- Define parameters -----
	# Parameters for raw synthetic RFs
	modfile='../velocity_models/model_lohs.txt'
	wvtype='P' # incident wave type
	npts=8193 # Number of samples
	dt=0.05 # time discretization
	baz=0.0 # Back-azimuth direction in degrees (has no influence if model is isotropic)
	plim=[0.04,0.08] # slowness limits

	np.random.seed(0)

	# Parameters for processed synthetic RFs
	t_axis = np.linspace(-(npts//2)*dt, (npts//2)*dt, npts) # time axis
	tlim = [-1.0, 10.0] # time window
	t_inds = (t_axis >= tlim[0]) & (t_axis < tlim[1]) # corresponding indices
	npts_win = np.sum(t_inds) # number of points in the time window
	flim = 1.0 # bandpass frequencies

	# Parameters for optimal transport
	m=0.95

	# horizontal slowness (ray parameter) in s/km
	slows = [0.06, 0.05, 0.06]
	perts_s = [0.0, 0.02, -0.02]

	# ----- Calculate reference RF -----
	# Load model and calculate RF
	ref_model = ut.read_model(modfile)
	rf_ref_ts = simulate_RF(ref_model, slows[0], baz, npts, dt, freq=flim, vels=None).data
	# Turn 1D reference RF into a 2D point cloud via a time-amplitude scaling
	delta_t = np.max(t_axis[t_inds])-np.min(t_axis[t_inds])
	delta_a = np.max(rf_ref_ts)-np.min(rf_ref_ts)
	t_weight = (delta_t/delta_a)
	rf_ref = np.array([t_axis[t_inds], t_weight*rf_ref_ts[t_inds]]).T

	# ----- Calculate two perturbed RFs, their LOT embedding, and the OT map between them in the embedding space -----
	# ----- First RF -----
	# Get random slowness and Vs perturbation
	slow = slows[1]
	pert_s = perts_s[1]
	# Get telewavesim model with given perturbation
	pert_model = copy.deepcopy(ref_model)
	pert_model.vs[0]*=(1+pert_s)
	pert_model.update_tensor()
	# Calculate the RF and add noise
	rfs_pert_1 = simulate_RF(pert_model, slow, baz, npts, dt, freq=flim, vels=None).data
	# Turn 1D RF into 2D point cloud
	rf_cur_1 = np.array([t_axis[t_inds], t_weight*rfs_pert_1[t_inds]]).T
	# ----- Calculate the OT map -----
	C_1=ot.dist(rf_cur_1, rf_ref, metric='euclidean') # GSOT distance matrix
	a=np.ones((npts_win,))/float(npts_win) # uniform distribution over reference points
	b=np.ones((npts_win,))/float(npts_win) # uniform distribution over current points
	p_1=ot.partial.partial_wasserstein(a,b,C_1,m=m)
	# ----- Calculate the LOT embedding -----
	Vi_1=(np.matmul((npts_win*p_1).T,rf_cur_1)-rf_ref)
	# Find indices that mapped to the dummy points and set to NaN
	# We'll treat these as missing observations during the LOT section
	pruned_inds=np.sum(npts_win*p_1,axis=0)!=1
	Vi_1[pruned_inds,:]=np.nan

	# ----- Second RF -----
	# Get random slowness and Vs perturbation
	slow = slows[2]
	pert_s = perts_s[2]
	# Get telewavesim model with given perturbation
	pert_model = copy.deepcopy(ref_model)
	pert_model.vs[0]*=(1+pert_s)
	pert_model.update_tensor()
	# Calculate the RF and add noise
	rfs_pert_2 = simulate_RF(pert_model, slow, baz, npts, dt, freq=flim, vels=None).data
	# Turn 1D RF into 2D point cloud
	rf_cur_2 = np.array([t_axis[t_inds], t_weight*rfs_pert_2[t_inds]]).T
	# ----- Calculate the OT map -----
	C_2=ot.dist(rf_cur_2, rf_ref, metric='euclidean') # GSOT distance matrix
	a=np.ones((npts_win,))/float(npts_win) # uniform distribution over reference points
	b=np.ones((npts_win,))/float(npts_win) # uniform distribution over current points
	p_2=ot.partial.partial_wasserstein(a,b,C_2,m=m)
	# ----- Calculate the LOT embedding -----
	Vi_2=(np.matmul((npts_win*p_2).T,rf_cur_2)-rf_ref)
	# Find indices that mapped to the dummy points and set to NaN
	# We'll treat these as missing observations during the LOT section
	pruned_inds=np.sum(npts_win*p_2,axis=0)!=1
	Vi_2[pruned_inds,:]=np.nan
	p_embed = Vi_2-Vi_1

	# ----- Calculate OT map between two perturbed RFs directly -----
	C=ot.dist(rf_cur_1, rf_cur_2, metric='euclidean') # GSOT distance matrix
	a=np.ones((npts_win,))/float(npts_win) # uniform distribution over reference points
	b=np.ones((npts_win,))/float(npts_win) # uniform distribution over current points
	p_direct=ot.partial.partial_wasserstein(a,b,C,m=m)
	d_direct=np.sum(p_direct*C)

	# plot the transport map comparison: direct vs embedding space
	fig,axs=plt.subplots(3,1,figsize=(10,5),sharex=True)
	p_1*=npts_win
	p_2*=npts_win
	axs[0].scatter(rf_ref[:,0], rf_ref[:,1], c='k', s=5)
	axs[0].scatter(rf_cur_1[:,0], rf_cur_1[:,1], c='steelblue', s=5)
	axs[0].scatter(rf_cur_2[:,0], rf_cur_2[:,1], c='crimson', s=5)
	for i in range(npts_win):
		if np.sum(p_1[i]==1)!=0:
			vector_x = rf_cur_1[:,0][p_1[i]==1]-rf_cur_1[:,0][i]
			vector_y = rf_ref[:,1][p_1[i]==1]-rf_cur_1[:,1][i]
			axs[0].plot([rf_cur_1[:,0][i], rf_cur_1[:,0][i]+vector_x], [rf_cur_1[:,1][i], rf_cur_1[:,1][i]+vector_y], c='k')
		if np.sum(p_2[i]==1)!=0:
			vector_x = rf_cur_2[:,0][p_2[i]==1]-rf_cur_2[:,0][i]
			vector_y = rf_ref[:,1][p_2[i]==1]-rf_cur_2[:,1][i]
			axs[0].plot([rf_cur_2[:,0][i], rf_cur_2[:,0][i]+vector_x], [rf_cur_2[:,1][i], rf_cur_2[:,1][i]+vector_y], c='k')

	p_direct*=npts_win
	axs[1].scatter(rf_cur_1[:,0], rf_cur_1[:,1], c='steelblue', s=5)
	axs[1].scatter(rf_cur_2[:,0], rf_cur_2[:,1], c='crimson', s=5)
	for i in range(npts_win):
		if np.sum(p_direct[i]==1)!=0:
			vector_x = rf_cur_1[:,0][p_direct[i]==1]-rf_cur_1[:,0][i]
			vector_y = rf_cur_2[:,1][p_direct[i]==1]-rf_cur_1[:,1][i]
			axs[1].plot([rf_cur_1[:,0][i], rf_cur_1[:,0][i]+vector_x], [rf_cur_1[:,1][i], rf_cur_1[:,1][i]+vector_y], c='k')

	axs[2].scatter(rf_cur_1[:,0], rf_cur_1[:,1], c='steelblue', s=5)
	axs[2].scatter(rf_cur_2[:,0], rf_cur_2[:,1], c='crimson', s=5)
	for i in range(npts_win):
		axs[2].plot([rf_cur_1[:,0][i], rf_cur_1[:,0][i]+p_embed[i,0]], [rf_cur_1[:,1][i], rf_cur_1[:,1][i]+p_embed[i,1]], c='k')

	axs[0].annotate("a", (-0.95, 6.5), fontsize=12, weight='bold')
	axs[1].annotate("b", (-0.95, 6), fontsize=12, weight='bold')
	axs[2].annotate("c", (-0.95, 6.5), fontsize=12, weight='bold')

	axs[2].set_xlabel("Time [s]", fontsize=14)
	axs[2].set_xlim(tlim[0], tlim[1])
	plt.show()

if __name__=="__main__":
	main()