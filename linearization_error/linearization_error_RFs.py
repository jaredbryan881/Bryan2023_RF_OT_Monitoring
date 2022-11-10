import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker
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
	linearization_error_RFs()
	#compare_transport_maps()

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
		
		Vi=(np.matmul((npts_win*p).T,rf_cur)-rf_ref)

		# Find indices that mapped to the dummy points and set to NaN
		# We'll treat these as missing observations during the LOT section
		pruned_inds=np.sum(p,axis=0)==0
		Vi[pruned_inds,:]=np.nan
		Vs.append(Vi)
	Vs=np.asarray(Vs)

	# Calculate the distance matrix
	d_lot = np.zeros((n_rfs, n_rfs))
	d_ot = np.zeros((n_rfs, n_rfs))
	d_comp = np.zeros((n_rfs, n_rfs))

	for i in range(n_rfs):
		print(i)
		rf1=np.array([t_axis[t_inds], t_weight*rfs_pert[i,t_inds]]).T

		# calculate the transport map from rf1 to rf_ref
		C_10=ot.dist(rf_ref, rf1, metric='euclidean') # GSOT distance matrix
		#C_10=ot.dist(rf1, rf_ref, metric='euclidean')
		a=np.ones((npts_win,))/float(npts_win) # uniform distribution over reference points
		b=np.ones((npts_win,))/float(npts_win) # uniform distribution over current points
		p_10=ot.partial.partial_wasserstein(a,b,C_10,m=m) # transport plan

		for j in range(i+1,n_rfs):
			rf2=np.array([t_axis[t_inds], t_weight*rfs_pert[j,t_inds]]).T

			# calculate the OT plan from rf1 to rf2
			C_12=ot.dist(rf1, rf2, metric='euclidean') # GSOT distance matrix
			a=np.ones((npts_win,))/float(npts_win) # uniform distribution over reference points
			b=np.ones((npts_win,))/float(npts_win) # uniform distribution over current points
			p_12=ot.partial.partial_wasserstein(a,b,C_12,m=m) # transport plan
			# calculate the associated distance
			d_ot[i,j]=np.sum(p_12*C_12)
			d_ot[j,i]=d_ot[i,j]
			# approximate the Monge map
			f_12=np.matmul((npts_win*p_12).T, rf1)
			f_12[np.sum(p_12, axis=0)==0]=np.nan

			# calculate the OT distance in the embedding space
			V_diff=Vs[j]-Vs[i]
			d_lot[i,j]=np.sqrt(np.nansum((V_diff/np.sqrt(npts_win))**2))
			d_lot[j,i]=d_lot[i,j]

			# calculate the transport map from rf_ref to rf2
			C_02=ot.dist(rf2, rf_ref, metric='euclidean') # GSOT distance matrix
			#C_02=ot.dist(rf_ref, rf2, metric='euclidean')
			p_02=ot.partial.partial_wasserstein(a,b,C_02,m=m) # transport plan
			# compose the two transport plans rf1->rf_ref->rf2
			p_12_comp=np.matmul((npts_win*p_02).T, (npts_win*p_10).T)
			f_12_comp=np.matmul(p_12_comp.T, rf1)
			f_12_comp[np.sum(p_12_comp, axis=0)==0]=np.nan

			dir_vs_comp=f_12-f_12_comp
			dir_vs_comp_mag=np.sqrt(np.nansum((dir_vs_comp)**2)/npts_win)

			d_comp[i,j] = d_ot[i,j] + dir_vs_comp_mag
			d_comp[j,i] = d_comp[i,j]

	fig,axs=plt.subplots(2,1,sharex=True, figsize=(8,8))
	# raw distances
	# 1:1
	axs[0].plot(d_ot, d_ot, c='k', lw=2)

	# transport map distortion
	dist_dens=np.vstack([d_ot.flatten(),d_comp.flatten()])
	dist_dens_c = gaussian_kde(dist_dens)(dist_dens)
	axs[0].scatter(d_ot.flatten(), d_comp.flatten(), c=cm.Reds(dist_dens_c/dist_dens_c.max()), rasterized=True)

	# LOT distances
	dist_dens=np.vstack([d_ot.flatten(),d_lot.flatten()])
	dist_dens_c = gaussian_kde(dist_dens)(dist_dens)
	axs[0].scatter(d_ot.flatten(), d_lot.flatten(), c=cm.Blues(dist_dens_c/dist_dens_c.max()), rasterized=True)

	# error
	# 1:1
	axs[1].plot(d_ot, np.ones(len(d_ot)), c='k', lw=2)
	# LOT distances
	error=(d_comp-d_lot)/(d_comp-d_ot)
	error=error.flatten()
	d_ot=d_ot.flatten()
	d_ot=d_ot[error>-1]
	error=error[error>-1]
	error_dens=np.vstack([d_ot, error])
	error_dens_c = gaussian_kde(error_dens)(error_dens)
	axs[1].scatter(d_ot, error, c=cm.Blues(error_dens_c/error_dens_c.max()))

	# format axes
	axs[1].set_xlabel(r"$d_{OT}$", fontsize=12)

	ybox1 = TextArea(r"$d_{LOT}  $", textprops=dict(color="b", size=12, rotation='vertical'))
	ybox2 = TextArea(r"$  d_{comp}$", textprops=dict(color="r", size=12, rotation='vertical'))
	ybox = VPacker(children=[ybox2, ybox1], align="center", pad=0, sep=5)
	anchored_ybox = AnchoredOffsetbox(loc=8, child=ybox, pad=0., frameon=False,
                                      bbox_to_anchor=(-0.08, 0.2),
                                      bbox_transform=axs[0].transAxes, borderpad=0.)
	axs[0].add_artist(anchored_ybox)

	axs[1].set_ylabel(r"$\frac{d_{comp}-d_{LOT}}{d_{comp}-d_{OT}}$", fontsize=15)
		
	axs[0].set_xlim(0,d_ot.max())
	axs[0].set_ylim(0, 1.4)
	axs[1].set_ylim(-0.2, 1.1)

	axs[0].annotate("a", (0.0025,1.3),   fontsize=16, weight='bold')
	axs[1].annotate("b", (0.0025,1.01),   fontsize=16, weight='bold')

	plt.tight_layout()

	plt.savefig("./figs/OTmap_linearization_error_withcomp.pdf")
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

	axs[0].annotate(r"$\mu_0 \rightarrow \mu_1$, $\mu_0 \rightarrow \mu_2$", (3.8, 6.5), fontsize=12)
	axs[1].annotate(r"$\mu_1 \rightarrow \mu_2$", (4.2, 6), fontsize=12)
	axs[2].annotate(r"$\phi(\mu_2) - \phi(\mu_1)$", (3.9, 6), fontsize=12)

	axs[2].set_xlabel("Time [s]", fontsize=14)
	axs[2].set_xlim(tlim[0], tlim[1])
	plt.tight_layout()
	plt.show()

if __name__=="__main__":
	main()