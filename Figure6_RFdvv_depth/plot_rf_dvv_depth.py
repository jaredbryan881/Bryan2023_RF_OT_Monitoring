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
	modfile='../velocity_models/model_10lohs.txt'
	wvtype='P' # incident wave type
	npts=8193 # Number of samples
	dt=0.05 # time discretization
	baz=0.0 # Back-azimuth direction in degrees (has no influence if model is isotropic)
	slow=0.06 # slowness limits
	pert=-0.05

	# Parameters for processed synthetic RFs
	t_axis = np.linspace(-(npts//2)*dt, (npts//2)*dt, npts) # time axis
	tlim = [-1.0, 20.0] # time window
	t_inds = (t_axis >= tlim[0]) & (t_axis < tlim[1]) # corresponding indices
	npts_win = np.sum(t_inds) # number of points in the time window
	flim = 1.0 # bandpass frequencies
	save_figs=False

	# Parameters for optimal transport
	m=0.95

	np.random.seed(0)

	# ----- Calculate reference RF -----
	# Load model and calculate RF
	ref_model = ut.read_model(modfile)
	n_layers = len(ref_model.vs)
	ref_model.rho = np.linspace(2800, 3200, n_layers)
	ref_model.vp[:-1] = np.linspace(5.0, 7.1, n_layers-1) + 0.5*np.random.randn(n_layers-1)
	ref_model.vs[:-1] = np.linspace(2.8, 4.0, n_layers-1) + 0.28*np.random.randn(n_layers-1)
	ref_model.vp[-1]=8.1
	ref_model.vs[-1]=4.55
	ref_model.update_tensor()

	rf_ref_ts = simulate_RF(ref_model, slow, baz, npts, dt, freq=flim, vels=None).data

	# Turn 1D reference RF into a 2D point cloud via a time-amplitude scaling
	delta_t = np.max(t_axis[t_inds])-np.min(t_axis[t_inds])
	delta_a = np.max(rf_ref_ts)-np.min(rf_ref_ts)
	t_weight = (delta_t/delta_a)
	rf_ref = np.array([t_axis[t_inds], t_weight*rf_ref_ts[t_inds]]).T

	# ----- Calculate ensemble of RFs -----
	rfs_pert = np.empty((n_layers, npts))
	dists_im = np.empty((3,n_layers-1,npts_win))
	
	fig1,axs1=plt.subplots(4,1,figsize=(10,10), sharex=True)
	
	fig2,axs2=plt.subplots(1,1,figsize=(4,10))
	vp = ref_model.vp[:]
	vp = np.insert(vp, 0, vp[0])
	vs = ref_model.vs[:]
	vs = np.insert(vs, 0, vs[0])
	depth = np.cumsum(ref_model.thickn)
	depth[-1]=50
	depth = np.insert(depth, 0, 0)

	axs2.plot(vp, depth, drawstyle='steps-post', c='steelblue', lw=2)
	axs2.plot(vs, depth, drawstyle='steps-post', c='k', lw=2)
	for i in range(0,n_layers-1,2):
		# Get telewavesim model with given perturbation
		pert_model = copy.deepcopy(ref_model)
		pert_model.vs[i:i+2]*=(1+pert)
		#pert_model.vp[i]*=(1+pert)
		pert_model.update_tensor()

		# Calculate the RF and add noise
		rfs_pert[i] = simulate_RF(pert_model, slow, baz, npts, dt, freq=flim, vels=None).data

		# Turn 1D RF into 2D point cloud
		rf_cur = np.array([t_axis[t_inds], t_weight*rfs_pert[i,t_inds]]).T

		# ----- Calculate the distance matrix -----
		M_t = distance_matrix_1d(t_axis[t_inds,np.newaxis], t_axis[t_inds,np.newaxis])
		M_a = distance_matrix_1d(rf_cur[:,1,np.newaxis], rf_ref[:,1,np.newaxis])
		M_tlp = distance_matrix_1d(rf_ref, rf_cur)

		# ----- Calculate the OT plan -----
		M=ot.dist(rf_cur, rf_ref) # GSOT distance matrix
		a=np.ones((npts_win,))/float(npts_win) # uniform distribution over reference points
		b=np.ones((npts_win,))/float(npts_win) # uniform distribution over current points
		p=ot.partial.partial_wasserstein(a,b,M_tlp,m=m,nb_dummies=10)
		valid_inds = np.sum(p,axis=0)!=0
		valid_inds=[True for i in range(npts_win)]

		# ----- Calculate the OT dist -----
		d_t=np.sum(p*M_t,axis=0)
		d_a=np.sum(p*M_a,axis=0)
		d=np.sum(p*M_tlp,axis=0)

		# plot the distance
		c=cm.inferno((i+1)/n_layers)
		axs1[0].plot(t_axis[t_inds], rfs_pert[i, t_inds], c=c, lw=2)
		axs1[1].plot(t_axis[t_inds][valid_inds], d_t[valid_inds] - i/1e4, c=c, lw=2)
		axs1[2].plot(t_axis[t_inds][valid_inds], d_a[valid_inds] - i/1e7, c=c, lw=2)
		axs1[3].plot(t_axis[t_inds][valid_inds], d[valid_inds] - i/1e4, c=c, lw=2)

		dists_im[0,i]=d_t
		dists_im[1,i]=d_a
		dists_im[2,i]=d

		# fill in Vs perturbation
		axs2.axvspan(pert_model.vs[i], ref_model.vs[i], ymin=1-(4*i)/depth[-1], ymax=1-(4*(i+1))/depth[-1], facecolor=c, alpha=0.75)
		axs2.axvspan(pert_model.vs[i+1], ref_model.vs[i+1], ymin=1-(4*(i+1))/depth[-1], ymax=1-(4*(i+2))/depth[-1], facecolor=c, alpha=0.75)
		# fill in Vp perturbation
		axs2.axvspan(pert_model.vp[i], ref_model.vp[i], ymin=1-(4*i)/depth[-1], ymax=1-(4*(i+1))/depth[-1], facecolor=c, alpha=0.75)

	axs1[0].plot(t_axis[t_inds], rf_ref_ts[t_inds], lw=2, c='k')
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


	axs2.set_ylim(depth[-1],depth[0])
	axs2.set_ylabel("Depth [km]", fontsize=12)
	axs2.set_xlabel("Velocity [km/s]", fontsize=12)

	plt.tight_layout()
	if save_figs:
		fig1.savefig("RF_OT_dists_depth.pdf")
		fig2.savefig("velocity_model.pdf")
	plt.show()

if __name__=="__main__":
	main()