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
	time_dist_matrix_lot_vs_ot()

def time_dist_matrix_lot_vs_ot():
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
	noise_level=0.0 # fraction of the range used for additive Gaussian noise

	# Parameters for optimal transport
	m=0.95

	save_figs=False

	n_rfs_arr = np.logspace(1,3,20)
	t_ot=np.zeros(len(n_rfs_arr))
	t_lot=np.zeros(len(n_rfs_arr))
	for (n,n_rfs) in enumerate(n_rfs_arr):
		n_rfs=int(n_rfs)
		# ----- Calculate ensemble of RFs -----
		Vs=list()
		rfs_pert = np.empty((n_rfs, npts))
		for i in range(n_rfs):
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

		# embed RFs
		t1_lot=time.time()
		for i in range(n_rfs):
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

		# Calculate the distance matrix in the embedding space
		ind=0
		d_lot=np.zeros((n_rfs,n_rfs))
		for i in range(n_rfs):
			print(i)
			rf1=np.array([t_axis[t_inds], t_weight*rfs_pert[i,t_inds]]).T
			for j in range(i+1,n_rfs):
				rf2=np.array([t_axis[t_inds], t_weight*rfs_pert[j,t_inds]]).T

				# calculate the OT distance in the embedding space
				V_diff=Vs[j]-Vs[i]
				d_lot[i,j]=np.sqrt(np.nansum((V_diff/np.sqrt(npts_win))**2))
				d_lot[j,i]=d_lot[i,j]
		t2_lot=time.time()
		t_lot[n]=(t2_lot-t1_lot)


		t1_ot=time.time()
		d_ot=np.zeros((n_rfs,n_rfs))
		# calculate the distance matrix directly
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
				d_ot[i,j]=np.sum(p*C)
				d_ot[j,i]=d_ot[i,j]
		t2_ot=time.time()
		t_ot[n]=(t2_ot-t1_ot)

	fig,ax=plt.subplots(1,1)
	ax.scatter(n_rfs_arr, t_ot, c='k', label=r"$T_{OT}$")
	ax.scatter(n_rfs_arr, t_lot, c='crimson', label=r"$T_{LOT}$")
	ax.set_xlabel("Number of signals", fontsize=12)
	ax.set_ylabel("Time [s]", fontsize=12)
	ax.set_yscale("log")
	ax.set_xscale("log")
	ax.legend(fontsize=12)
	plt.show()

if __name__=="__main__":
	main()