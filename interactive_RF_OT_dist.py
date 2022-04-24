import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.widgets import Slider

from telewavesim import utils as ut
import numpy as np
import copy

from distance_matrix import distance_matrix_1d
from sim_synth import simulate_RF

import ot

def main():
	# define parameters
	modfile = './velocity_models/model_lohs.txt' # layer-over-halfspace geometry
	wvtype  = 'P'
	npts    = 8193 # Number of samples
	dt      = 0.05   # Sample distance in seconds
	slow    = 0.06 # Horizontal slowness (or ray parameter) in s/km
	baz     = 0.0   # Back-azimuth direction in degrees (has no influence if model is isotropic)
	t_axis  = np.linspace(-(npts//2)*dt, (npts//2)*dt, npts) # time axis
	tlim    = [-1.0, 10.0] # time window
	t_inds  = (t_axis >= tlim[0]) & (t_axis < tlim[1])
	flim    = 1.0 # bandpass frequencies
	m       = 0.90 # partial OT mass
	cmap    = cm.inferno

	# load model
	ref_model = ut.read_model(modfile)
	# simulate reference RF
	ref_rf = simulate_RF(ref_model, slow, baz, npts, dt, freq=flim, vels=None).data

	# initialize plot
	fig, axs = plt.subplots(3,1,figsize=(12, 12), gridspec_kw={'height_ratios':[2,2,5]})
	fig.subplots_adjust(bottom=0.1, top=0.83)

	# Create axes for sliders
	ax_p = fig.add_axes([0.3, 0.93, 0.4, 0.03])
	ax_p.spines['top'].set_visible(True)
	ax_p.spines['right'].set_visible(True)

	ax_vp = fig.add_axes([0.3, 0.91, 0.4, 0.03]) #0.89
	ax_vp.spines['top'].set_visible(True)
	ax_vp.spines['right'].set_visible(True)

	ax_vs = fig.add_axes([0.3, 0.89, 0.4, 0.03]) #0.85
	ax_vs.spines['top'].set_visible(True)
	ax_vs.spines['right'].set_visible(True)

	ax_lamb = fig.add_axes([0.3, 0.87, 0.4, 0.03])
	ax_lamb.spines['top'].set_visible(True)
	ax_lamb.spines['right'].set_visible(True)

	ax_m = fig.add_axes([0.3, 0.85, 0.4, 0.03])
	ax_m.spines['top'].set_visible(True)
	ax_m.spines['right'].set_visible(True)

	# create sliders
	s_p    = Slider(ax=ax_p,    label='Slowness', valmin=0.04, valmax=0.08, valinit=0.06, valfmt=' %1.3f s/deg', facecolor='k', alpha=0.5)
	s_vp   = Slider(ax=ax_vp,   label='Layer Vp Perturbation', valmin=-5, valmax=5, valinit=0.0, valfmt='%1.1f %%', facecolor='crimson', alpha=0.5)
	s_vs   = Slider(ax=ax_vs,   label='Layer Vs Perturbation', valmin=-5, valmax=5, valinit=0.0, valfmt='%1.1f %%', facecolor='steelblue', alpha=0.5)
	s_lamb = Slider(ax=ax_lamb, label=r"Time-Amplitude Scaling $c$, for $c\lambda_0$", valmin=0, valmax=10, valinit=1, valfmt="%1.0f", facecolor='green', alpha=0.5)
	s_m    = Slider(ax=ax_m,    label=r"Partial OT Mass $m$", valmin=0.8, valmax=0.99, valinit=0.9, valfmt='%1.2f', facecolor='orange', alpha=0.5)

	# initialize plotting cache
	current_plots_rf   = []
	current_plots_dist = []

	# -----------------
	# PLOT REFERENCE RF
	# -----------------
	f_d, = axs[0].plot(t_axis, ref_rf, linewidth=2.5, alpha=1.0, c=cmap(0.0))
	current_plots_rf.append(f_d)
	axs[0].set_xlim(tlim[0], tlim[1])
	axs[0].set_ylim(-0.02, 0.05)

	axs[1].set_ylim(-0.001,0.002)

	# ------------------------
	# PLOT MEDIUM AND RECEIVER
	# ------------------------
	layer_thickness = 0.5
	recv_loc = [0.8, 1.02]

	axs[2].fill_between([0,1], [layer_thickness, layer_thickness], [1, 1], facecolor='steelblue', alpha=0.3)
	axs[2].fill_between([0,1], [0.0, 0.0], [layer_thickness, layer_thickness], facecolor='steelblue', alpha=0.5)
	axs[2].scatter([recv_loc[0]], [recv_loc[1]], marker='v', s=200, facecolor='k')
	axs[2].set_xlim(0,1)
	axs[2].set_ylim(0,1.1)

	# ---------------------
	# PLOT PLANAR WAVEFRONT
	# ---------------------
	plane_right = np.tan(np.arcsin(s_p.val * ref_model.vp[1]))
	plane, = axs[2].plot([0,1], [layer_thickness, layer_thickness-plane_right], alpha=1.0, c=cmap(0.0), lw=2.5)

	# ------------------------
	# PLOT RAYS IN UPPER LAYER
	# ------------------------
	# lateral distance traveled by P and S rays in the upper layer 
	dx_p = layer_thickness*np.tan(np.arcsin(s_p.val * ref_model.vp[0]))
	dx_s = layer_thickness*np.tan(np.arcsin(s_p.val * ref_model.vs[0]))
	# Pp
	Pp_up,        = axs[2].plot([recv_loc[0]-dx_p,recv_loc[0]], [layer_thickness,1], c='k')
	# Ps
	Ps_up,        = axs[2].plot([recv_loc[0]-dx_s,recv_loc[0]], [layer_thickness,1], c='r')
	# PpPs
	PpPs_up_up1,  = axs[2].plot([recv_loc[0]-dx_s-2*dx_p,recv_loc[0]-dx_s-dx_p], [layer_thickness,1], c='k')
	PpPs_up_down, = axs[2].plot([recv_loc[0]-dx_s-dx_p,recv_loc[0]-dx_s], [1,layer_thickness], c='k')
	PpPs_up_up2,  = axs[2].plot([recv_loc[0]-dx_s,recv_loc[0]], [layer_thickness,1], c='r')
	# PsPs
	PsPs_up_up1,  = axs[2].plot([recv_loc[0]-2*dx_s-dx_p,recv_loc[0]-dx_s-dx_p], [layer_thickness,1], c='r')
	PsPs_up_down, = axs[2].plot([recv_loc[0]-dx_s-dx_p,recv_loc[0]-dx_s], [1,layer_thickness], c='k')
	PsPs_up_up2,  = axs[2].plot([recv_loc[0]-dx_s,recv_loc[0]], [layer_thickness,1], c='r')
	# PpSs
	PpSs_up_up1,  = axs[2].plot([recv_loc[0]-2*dx_s-dx_p,recv_loc[0]-2*dx_s], [layer_thickness,1], c='k')
	PpSs_up_down, = axs[2].plot([recv_loc[0]-2*dx_s,recv_loc[0]-dx_s], [1,layer_thickness], c='r')
	PpSs_up_up2,  = axs[2].plot([recv_loc[0]-dx_s,recv_loc[0]], [layer_thickness,1], c='r')
	# PsSs
	PsSs_up_up1,  = axs[2].plot([recv_loc[0]-3*dx_s,recv_loc[0]-2*dx_s], [layer_thickness,1], c='r')
	PsSs_up_down, = axs[2].plot([recv_loc[0]-2*dx_s,recv_loc[0]-dx_s], [1,layer_thickness], c='r')
	PsSs_up_up2,  = axs[2].plot([recv_loc[0]-dx_s,recv_loc[0]], [layer_thickness,1], c='r')

	# ------------------------
	# PLOT RAYS IN LOWER LAYER
	# ------------------------
	# Pp
	Pp_low_x      = ((1/plane_right)/((1/plane_right) + plane_right))*(recv_loc[0]-dx_p)
	Pp_low_y      = 0.5-plane_right*Pp_low_x
	Pp_low,       = axs[2].plot([Pp_low_x,recv_loc[0]-dx_p], [Pp_low_y,layer_thickness], c='k')
	# Ps
	Ps_low_x      = ((1/plane_right)/((1/plane_right) + plane_right))*(recv_loc[0]-dx_s)
	Ps_low_y      = 0.5-plane_right*Ps_low_x
	Ps_low,       = axs[2].plot([Ps_low_x,recv_loc[0]-dx_s], [Ps_low_y,layer_thickness], c='k')
	# PpPs
	PpPs_low_x    = ((1/plane_right)/((1/plane_right) + plane_right))*(recv_loc[0]-dx_s-2*dx_p)
	PpPs_low_y    = 0.5-plane_right*PpPs_low_x
	PpPs_low,     = axs[2].plot([PpPs_low_x,recv_loc[0]-dx_s-2*dx_p], [PpPs_low_y,layer_thickness], c='k')
	# PsPs
	PsPs_low_x    = ((1/plane_right)/((1/plane_right) + plane_right))*(recv_loc[0]-2*dx_s-dx_p)
	PsPs_low_y    = 0.5-plane_right*PsPs_low_x
	PsPs_low,     = axs[2].plot([PsPs_low_x,recv_loc[0]-2*dx_s-dx_p], [PsPs_low_y,layer_thickness], c='k')
	# PpSs
	PpSs_low_x    = ((1/plane_right)/((1/plane_right) + plane_right))*(recv_loc[0]-2*dx_s-dx_p)
	PpSs_low_y    = 0.5-plane_right*PpSs_low_x
	PpSs_low,     = axs[2].plot([PpSs_low_x,recv_loc[0]-2*dx_s-dx_p], [PpSs_low_y,layer_thickness], c='k')
	# PsSs
	PsSs_low_x    = ((1/plane_right)/((1/plane_right) + plane_right))*(recv_loc[0]-3*dx_s)
	PsSs_low_y    = 0.5-plane_right*PsSs_low_x
	PsSs_low,     = axs[2].plot([PsSs_low_x,recv_loc[0]-3*dx_s], [PsSs_low_y,layer_thickness], c='k')

	# calculate time-amplitude scaling
	delta_t = np.max(t_axis[t_inds])-np.min(t_axis[t_inds])
	delta_a = np.max(ref_rf)-np.min(ref_rf)
	lamb    = delta_t/delta_a

	# calculate time distance matrix
	ntimes = len(t_axis[t_inds])
	M_t = distance_matrix_1d(t_axis[t_inds,np.newaxis], t_axis[t_inds,np.newaxis])

	# Initialize distance plot
	f_dist, = axs[1].plot(t_axis[t_inds], np.zeros(ntimes), linewidth=2.5, alpha=1.0, c=cmap(0.0))
	current_plots_dist.append(f_dist)
	axs[1].set_xlim(tlim[0], tlim[1])
	axs[1].set_xlabel("Time [s]")

	# cache length for lines plotted
	n_lines = 25

	# define update function that changes the data when sliders are updated
	def update(val):
		# update receiver function lines
		pert_model = copy.deepcopy(ref_model)
		pert_model.vp[0]*=(1+(s_vp.val/100))
		pert_model.vs[0]*=(1+(s_vs.val/100))
		pert_model.update_tensor()
		# simulate RF through perturbed model
		pert_rf = simulate_RF(pert_model, s_p.val, baz, npts, dt, freq=flim, vels=None).data
		
		# -----------------
		# PLOT PERTURBED RF
		# -----------------
		f_cur, = axs[0].plot(t_axis, pert_rf, linewidth=2.5)
		current_plots_rf.append(f_cur)

		# Update colors and transparencies of cached RFs
		# Remove RFs once they exceed the cache length
		if len(current_plots_rf) > n_lines:
			current_plots_rf[0].remove()
			current_plots_rf.remove(current_plots_rf[0])

			for i in range(len(current_plots_rf)):
				current_plots_rf[i].set_alpha(i/len(current_plots_rf))
				current_plots_rf[i].set_color(cmap(1-i/len(current_plots_rf)))
		else:
			interval_places = np.linspace(1.0 - (len(current_plots_rf)/n_lines),1.0,len(current_plots_rf))

			for i in range(len(current_plots_rf)):
				current_plots_rf[i].set_alpha(interval_places[i])
				current_plots_rf[i].set_color(cmap(1-interval_places[i]))

		# ---------------------
		# PLOT PLANAR WAVEFRONT
		# ---------------------
		plane_right = np.tan(np.arcsin(s_p.val * pert_model.vp[1]))
		plane.set_data([0,1], [layer_thickness, layer_thickness-plane_right])

		# UPDATE RAYS
		# ------------------------
		# PLOT RAYS IN UPPER LAYER
		# ------------------------
		dx_p = layer_thickness*np.tan(np.arcsin(s_p.val * pert_model.vp[0]))
		dx_s = layer_thickness*np.tan(np.arcsin(s_p.val * pert_model.vs[0]))
		# Pp
		Pp_up.set_data([recv_loc[0]-dx_p,recv_loc[0]], [layer_thickness,1])
		# Ps
		Ps_up.set_data([recv_loc[0]-dx_s,recv_loc[0]], [layer_thickness,1])
		# PpPs
		PpPs_up_up1.set_data([recv_loc[0]-dx_s-2*dx_p,recv_loc[0]-dx_s-dx_p], [layer_thickness,1])
		PpPs_up_down.set_data([recv_loc[0]-dx_s-dx_p,recv_loc[0]-dx_s], [1,layer_thickness])
		PpPs_up_up2.set_data([recv_loc[0]-dx_s,recv_loc[0]], [layer_thickness,1])
		# PsPs
		PsPs_up_up1.set_data([recv_loc[0]-2*dx_s-dx_p,recv_loc[0]-dx_s-dx_p], [layer_thickness,1])
		PsPs_up_down.set_data([recv_loc[0]-dx_s-dx_p,recv_loc[0]-dx_s], [1,layer_thickness])
		PsPs_up_up2.set_data([recv_loc[0]-dx_s,recv_loc[0]], [layer_thickness,1])
		# PpSs
		PpSs_up_up1.set_data([recv_loc[0]-2*dx_s-dx_p,recv_loc[0]-2*dx_s], [layer_thickness,1])
		PpSs_up_down.set_data([recv_loc[0]-2*dx_s,recv_loc[0]-dx_s], [1,layer_thickness])
		PpSs_up_up2.set_data([recv_loc[0]-dx_s,recv_loc[0]], [layer_thickness,1])
		# PsSs
		PsSs_up_up1.set_data([recv_loc[0]-3*dx_s,recv_loc[0]-2*dx_s], [layer_thickness,1])
		PsSs_up_down.set_data([recv_loc[0]-2*dx_s,recv_loc[0]-dx_s], [1,layer_thickness])
		PsSs_up_up2.set_data([recv_loc[0]-dx_s,recv_loc[0]], [layer_thickness,1])

		# ------------------------
		# PLOT RAYS IN LOWER LAYER
		# ------------------------
		# Pp
		Pp_low_x=((1/plane_right)/((1/plane_right) + plane_right))*(recv_loc[0]-dx_p)
		Pp_low_y = 0.5-plane_right*Pp_low_x
		Pp_low.set_data([Pp_low_x,recv_loc[0]-dx_p], [Pp_low_y,layer_thickness])
		# Ps
		Ps_low_x=((1/plane_right)/((1/plane_right) + plane_right))*(recv_loc[0]-dx_s)
		Ps_low_y = 0.5-plane_right*Ps_low_x
		Ps_low.set_data([Ps_low_x,recv_loc[0]-dx_s], [Ps_low_y,layer_thickness])
		# PpPs
		PpPs_low_x=((1/plane_right)/((1/plane_right) + plane_right))*(recv_loc[0]-dx_s-2*dx_p)
		PpPs_low_y = 0.5-plane_right*PpPs_low_x
		PpPs_low.set_data([PpPs_low_x,recv_loc[0]-dx_s-2*dx_p], [PpPs_low_y,layer_thickness])
		# PsPs
		PsPs_low_x=((1/plane_right)/((1/plane_right) + plane_right))*(recv_loc[0]-2*dx_s-dx_p)
		PsPs_low_y = 0.5-plane_right*PsPs_low_x
		PsPs_low.set_data([PsPs_low_x,recv_loc[0]-2*dx_s-dx_p], [PsPs_low_y,layer_thickness])
		# PpSs
		PpSs_low_x=((1/plane_right)/((1/plane_right) + plane_right))*(recv_loc[0]-2*dx_s-dx_p)
		PpSs_low_y = 0.5-plane_right*PpSs_low_x
		PpSs_low.set_data([PpSs_low_x,recv_loc[0]-2*dx_s-dx_p], [PpSs_low_y,layer_thickness])
		# PsSs
		PsSs_low_x=((1/plane_right)/((1/plane_right) + plane_right))*(recv_loc[0]-3*dx_s)
		PsSs_low_y = 0.5-plane_right*PsSs_low_x
		PsSs_low.set_data([PsSs_low_x,recv_loc[0]-3*dx_s], [PsSs_low_y,layer_thickness])

		# Calculate the amplitude distance matrix
		M_a = distance_matrix_1d(pert_rf[t_inds,np.newaxis], ref_rf[t_inds,np.newaxis])
		# Weighted combination of the time and amplitude distances
		M_tlp = M_t + s_lamb.val*lamb*M_a

		# ----- Calculate the OT plan -----
		a=np.ones((ntimes,))/float(ntimes) # uniform distribution over reference points
		b=np.ones((ntimes,))/float(ntimes) # uniform distribution over current points
		p=ot.partial.partial_wasserstein(a,b,M_tlp,m=s_m.val)

		# distances unaffected by partial OT
		valid_inds = np.sum(p,axis=0)!=0

		# Integrate the OT dist over amplitude to give a time series of distances
		d_t=np.sum(p*M_t,axis=0)
		d_a=np.sum(p*M_a,axis=0)
		d=np.sum(p*M_tlp,axis=0)

		# ----------------
		# PLOT OT DISTANCE
		# ----------------
		f_dist, = axs[1].plot(t_axis[t_inds], d_t, linewidth=2.5, alpha=1.0, c=cmap(0.0))

		current_plots_dist.append(f_dist)
		if len(current_plots_dist) > n_lines:
			current_plots_dist[0].remove()
			current_plots_dist.remove(current_plots_dist[0])

			for i in range(len(current_plots_dist)):
				current_plots_dist[i].set_alpha(i/len(current_plots_dist))
				current_plots_dist[i].set_color(cmap(1-i/len(current_plots_dist)))
		else:
			interval_places = np.linspace(1.0 - (len(current_plots_dist)/n_lines),1.0,len(current_plots_dist))

			for i in range(len(current_plots_dist)):
				current_plots_dist[i].set_alpha(interval_places[i])
				current_plots_dist[i].set_color(cmap(1-interval_places[i]))

		fig.canvas.draw_idle()

	# update the plot when the sliders change
	s_p.on_changed(update)
	s_vp.on_changed(update)
	s_vs.on_changed(update)
	s_lamb.on_changed(update)
	s_m.on_changed(update)

	plt.show()

if __name__=="__main__":
	main()