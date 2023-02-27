import numpy as np
from sklearn.datasets import make_circles
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.decomposition import PCA
import ot
import ot.plot

from telewavesim import utils as ut

import copy

import sys
sys.path.append("../")
from sim_synth import simulate_RF

def main():
	linearization_error_circles()
	#linearization_error_RFs()

def linearization_error_RFs():
	# ----- Define parameters -----
	# Parameters for raw synthetic RFs
	modfile='../velocity_models/model_lohs.txt'
	wvtype='P' # incident wave type
	npts=8193 # Number of samples
	dt=0.05 # time discretization
	baz=0.0 # Back-azimuth direction in degrees (has no influence if model is isotropic)
	plim=[0.04,0.08] # slowness limits
	dvlim=[-0.05,0.05] # Vs perturbation limits

	# Parameters for processed synthetic RFs
	t_axis = np.linspace(-(npts//2)*dt, (npts//2)*dt, npts) # time axis
	tlim = [-1.0, 10.0] # time window
	t_inds = (t_axis >= tlim[0]) & (t_axis < tlim[1]) # corresponding indices
	npts_win = np.sum(t_inds) # number of points in the time window
	flim = 1.0 # bandpass frequencies

	# Parameters for RF ensemble & noise
	n_rfs=50 # number of RFs in the synthetic distributions
	noise_level=0.0 # fraction of the range used for additive Gaussian noise

	# Parameters for optimal transport
	m=0.95

	save_figs=False

	# horizontal slowness (ray parameter) in s/km
	slows = np.random.uniform(low=plim[0], high=plim[1], size=n_rfs)
	perts_s = np.random.uniform(low=dvlim[0], high=dvlim[1], size=n_rfs)

	# ----- Calculate reference RF -----
	# Load model and calculate RF
	ref_model = ut.read_model(modfile)
	rf_ref_ts = simulate_RF(ref_model, np.mean(slows), baz, npts, dt, freq=flim, vels=None).data

	# Turn 1D reference RF into a 2D point cloud via a time-amplitude scaling
	delta_t = np.max(t_axis[t_inds])-np.min(t_axis[t_inds])
	delta_a = np.max(rf_ref_ts)-np.min(rf_ref_ts)
	t_weight = (delta_t/delta_a)
	rf_ref = np.array([t_axis[t_inds], t_weight*rf_ref_ts[t_inds]]).T

	# ----- Calculate ensemble of RFs and their linear embedding -----
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

	dists_lower_bound=np.zeros((n_rfs,n_rfs))
	dists_upper_bound=np.zeros((n_rfs,n_rfs))
	for i in range(n_rfs):
		print(i)
		rf1 = np.array([t_axis[t_inds], t_weight*rfs_pert[i,t_inds]]).T
		for j in range(i,n_rfs):
			rf2 = np.array([t_axis[t_inds], t_weight*rfs_pert[j,t_inds]]).T

			# lower bound distance
			C=ot.dist(rf1, rf2, metric='euclidean') # GSOT distance matrix
			a=np.ones((npts_win,))/float(npts_win) # uniform distribution over reference points
			b=np.ones((npts_win,))/float(npts_win) # uniform distribution over current points
			p=ot.partial.partial_wasserstein(a,b,C,m=m)
			d=np.sum(C*p)

			dists_lower_bound[i,j]=d
			dists_lower_bound[j,i]=d

			# upper bound distance
			# Xi to X0
			N=rf1.shape[0]

			C_i0=ot.dist(rf1, rf_ref, metric='euclidean')
			a=np.ones((N,))/float(N)
			b=np.ones((N,))/float(N)
			p_i0=ot.partial.partial_wasserstein(a,b,C_i0,m=m)

			# X0 to Xj
			C_0j=ot.dist(rf_ref, rf2, metric='euclidean')
			a=np.ones((N,))/float(N)
			b=np.ones((N,))/float(N)
			p_0j=ot.partial.partial_wasserstein(a,b,C_0j,m=m)

			# difference between direct transport map and the composition
			# of two maps with the reference as the midpoint
			direct_map=N*p.T
			composed_map=np.matmul(N*p_0j.T, N*p_i0.T)
			dir_vs_comp=direct_map-composed_map
			dir_vs_comp_mag=np.sqrt(np.sum((dir_vs_comp)**2)/N)

			dists_upper_bound[i,j] = d + dir_vs_comp_mag
			dists_upper_bound[j,i] = d + dir_vs_comp_mag

	# calculate the LOT embedding
	Vs=list()
	for i in range(n_rfs):
		rf_cur = np.array([t_axis[t_inds], t_weight*rfs_pert[i,t_inds]]).T
		C=ot.dist(rf_cur, rf_ref, metric='euclidean')
		b=np.ones((N,))/float(N)
		a=np.ones((N,))/float(N)
		p=ot.emd(a,b,C)

		V=(np.matmul((N*p).T,rf_cur)-rf_ref)/np.sqrt(N)
		Vs.append(V)

	# Calculate the distance matrix with LOT
	dists_LOT=np.zeros((n_rfs,n_rfs))
	colors=np.zeros((n_rfs,n_rfs))
	for i in range(n_rfs):
		for j in range(i,n_rfs):
			cur_dist=np.sqrt(np.sum((Vs[i]-Vs[j])**2))
			dists_LOT[i,j] = cur_dist
			dists_LOT[j,i] = cur_dist

	# full comparison
	fig,ax=plt.subplots(1,1)
	ax.scatter(dists_lower_bound, dists_LOT, s=5, c='k')
	ax.scatter(dists_lower_bound, dists_lower_bound, c='b', s=5)
	ax.scatter(dists_lower_bound, dists_upper_bound, c='r', s=5)
	ax.set_ylabel(r"$d_{LOT}$", fontsize=12)
	ax.set_xlabel(r"$d_{OT}$", fontsize=12)
	plt.show()

	fig,ax=plt.subplots(1,1)
	ax.scatter(dists_lower_bound, 100*(dists_LOT-dists_lower_bound)/dists_lower_bound, s=5, c='k')
	#ax.scatter(dists_lower_bound, dists_upper_bound, c='r', s=5)
	ax.set_ylabel("error [%]", fontsize=12)
	ax.set_xlabel(r"$d_{OT}$", fontsize=12)
	plt.show()

def linearization_error_circles():
	m=0.99

	nsamp=100
	theta=np.linspace(0,2*np.pi,nsamp)

	M=100 # Number of data samples
	X=list()
	scale=np.random.rand(M)+0.5
	shift=2*np.random.randn(M,2)
	for i in range(M):
		data=np.array([np.cos(theta), np.sin(theta)]).T
		#make_circles(n_samples=nsamp, noise=0.0, factor=0.95)
		data=scale[i]*data + shift[i]
		X.append(data)

	N=int(np.asarray([x.shape[0] for x in X]).mean())
	X0=np.array([np.cos(theta), np.sin(theta)]).T
	#X0=np.random.randn(N,2)
	#X0[:,0]*=10

	dists_upper_bound=np.zeros((M,M))
	dists_lower_bound=np.zeros((M,M))
	for i in range(M):
		print(i)
		for j in range(i+1,M):
			# lower bound distance
			Ni=X[i].shape[0]
			Nj=X[j].shape[0]

			C=ot.dist(X[i], X[j], metric='euclidean')
			a=np.ones((Ni,))/float(Ni)
			b=np.ones((Nj,))/float(Nj)
			p=ot.emd(a,b,C)
			d=np.sum(C*p)

			Vi=(np.matmul((N*p), X[j])-X[i])
			for alpha in np.linspace(0,1,5):
				X_cur = X[i] + alpha*Vi
				#X_cur = (1-alpha)*X[i] + alpha*np.matmul(N*p, X[j])
				plt.scatter(X_cur[:,0], X_cur[:,1], c=cm.inferno(alpha))
			plt.scatter(X[i][:,0], X[i][:,1], c='r', s=1)
			plt.scatter(X[j][:,0], X[j][:,1], c='b', s=1)
			plt.show()

			dists_lower_bound[i,j] = d
			dists_lower_bound[j,i] = d

			# upper bound distance
			# Xi to X0
			Ni=X[i].shape[0]
			N0=X0.shape[0]

			C_i0=ot.dist(X[i], X0, metric='euclidean')
			a=np.ones((Ni,))/float(Ni)
			b=np.ones((N0,))/float(N0)
			p_i0=ot.emd(a,b,C_i0)

			# X0 to Xj
			N0=X0.shape[0]
			Nj=X[j].shape[0]

			C_0j=ot.dist(X0, X[j], metric='euclidean')
			a=np.ones((N0,))/float(N0)
			b=np.ones((Nj,))/float(Nj)
			p_0j=ot.emd(a,b,C_0j)

			# difference between direct transport map and the composition
			# of two maps with the reference as the midpoint
			direct_map=N*p.T
			composed_map=np.matmul(N*p_0j.T, N*p_i0.T)
			dir_vs_comp=direct_map-composed_map
			dir_vs_comp_mag=np.sqrt(np.sum(dir_vs_comp**2))

			dists_upper_bound[i,j] = d + dir_vs_comp_mag
			dists_upper_bound[j,i] = d + dir_vs_comp_mag

	# calculate the LOT embedding
	Vs=list()
	for ind in range(M):
		Ni=X[ind].shape[0]

		C=ot.dist(X0, X[ind], metric='euclidean')
		b=np.ones((N,))/float(N)
		a=np.ones((Ni,))/float(Ni)
		p=ot.emd(a,b,C)

		V=(np.matmul(N*p, X[ind])-X0)/np.sqrt(N)
		Vs.append(V)

	"""
	for i in range(M):
		for j in range(M):
			V_diff = Vs[j] - Vs[i]

			# lower bound distance
			Ni=X[i].shape[0]
			Nj=X[j].shape[0]
			C=ot.dist(X[i], X[j], metric='euclidean')
			a=np.ones((Ni,))/float(Ni)
			b=np.ones((Nj,))/float(Nj)
			p=ot.emd(a,b,C)
			d=np.sum(C*p)
			V_dir=np.matmul(Ni*p, X[j])-X[i]

			fig,axs=plt.subplots(3,1,sharex=True,sharey=True,figsize=(3,10))
			axs[0].scatter(X0[:,0], X0[:,1], c='k')
			axs[0].scatter(X[j][:,0], X[j][:,1], c='b', s=1)
			axs[0].scatter(X[i][:,0], X[i][:,1], c='r', s=1)
			axs[1].scatter(Vs[j][:,0], Vs[j][:,1], c='b', s=1)
			axs[1].scatter(Vs[i][:,0], Vs[i][:,1], c='r', s=1)
			axs[2].scatter(V_diff[:,0], V_diff[:,1], c='k')
			axs[2].scatter(V_dir[:,0], V_dir[:,1], c='orange',s=1)

			axs[0].set_xlim(-10,10)
			axs[0].set_ylim(-10,10)
			plt.show()

			fig,axs=plt.subplots(2,1,sharex=True,sharey=True,figsize=(5,10))
			for alpha in np.linspace(0,1,5):
				X_cur=X[i] + alpha*V_diff
				axs[0].scatter(X_cur[:,0], X_cur[:,1], c=cm.inferno(alpha))
			axs[0].scatter(X[i][:,0], X[i][:,1], c='r', s=1)
			axs[0].scatter(X[j][:,0], X[j][:,1], c='r', s=1)

			for alpha in np.linspace(0,1,5):
				X_cur=X[i] + alpha*V_dir
				axs[1].scatter(X_cur[:,0], X_cur[:,1], c=cm.inferno(alpha))
			axs[1].scatter(X[i][:,0], X[i][:,1], c='r', s=1)
			axs[1].scatter(X[j][:,0], X[j][:,1], c='r', s=1)
			axs[0].set_xlim(-5,5)
			axs[0].set_ylim(-5,5)
			plt.show()
	""" 

	# Calculate the distance matrix with LOT
	dists_LOT=np.zeros((M,M))
	colors=np.zeros((M,M))
	for i in range(M):
		for j in range(i,M):
			cur_dist=np.sqrt(np.sum((Vs[i]-Vs[j])**2))
			dists_LOT[i,j] = cur_dist
			dists_LOT[j,i] = cur_dist

			colors[i,j]=np.sqrt(np.sum((shift[i]-shift[j])**2))
			colors[j,i]=colors[i,j]

	# full comparison
	fig,ax=plt.subplots(1,1)
	ax.scatter(dists_lower_bound, dists_LOT, c=(colors-colors.min())/(colors.max()-colors.min()), s=5, cmap='inferno')
	ax.scatter(dists_lower_bound, dists_lower_bound, c='b', s=5)
	#ax.scatter(dists_lower_bound, dists_upper_bound, c='r', s=5)
	#ax.scatter(dists_lower_bound, dists_lower_bound**(1/2), c='k')
	ax.set_ylabel(r"$d_{LOT}$", fontsize=12)
	ax.set_xlabel(r"$d_{OT}$", fontsize=12)
	plt.show()

if __name__=="__main__":
	main()
