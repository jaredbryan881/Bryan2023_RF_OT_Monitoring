import numpy as np
from telewavesim import utils as ut

def simulate_RF(model, slow, baz, npts, dt, wvtype='P', obs=False, freq=None, vels=None, len_factor=1, noise=None, comp='R'):
	"""
	Simulate the propagation of a plane wave through a stack of layers and calculate the resulting (radial) receiver function.

	This is essentially a wrapper for telewavesim's run_plane and tf_from_xyz, 
	followed by filtering to mimic the bandwidth of teleseismic body waves.

	Arguments
	---------
	:param model: tws.utils.Model
		Model object from telewavesim
	:param slow: float
		Slowness of the incident plane wave
	:param baz: float
		Backazimuth of the incident plane wave
	:param npts: int
		Number of points in the receiver function
	:param dt: float
		Sampling frequency of the receiver function
	:param wvtype: char
		Type of incoming wave
	:param obs: bool
		Ocean Bottom Seismometer or not
	:param freq: float or np.array(float)
		Lowpass frequency if float, else bandpass frequencies if np.array(float)
	:param len_factor: int
		Intermediate time expansion factor used to prevent artifacts in the acausal portion of the RF

	Returns
	-------
	:return rf: obspy.Trace 
		Synthetic receiver function from the given velocity model and source parameters
	"""
	# Optionally expand number of points before generating the RF to prevent artifacts
	npts*=len_factor
	if npts%2==0:
		npts+=1

	# generate displacement time series
	trxyz = ut.run_plane(model, slow, npts, dt, baz=baz, wvtype=wvtype, obs=obs)

	# get transfer functions from displacement time series
	if vels is not None:
		# rotate into P-SV-SH
		tfs = ut.tf_from_xyz(trxyz, pvh=True, vp=vels[0], vs=vels[1])
	else:
		# don't rotate
		tfs = ut.tf_from_xyz(trxyz, pvh=False)

	if comp=='R':
		ind=0
	elif comp=="T":
		ind=1

	if noise is not None:
		tfs[ind].data+=noise

	# get receiver function by lowpass filtering transfer function
	if freq is not None:
		if type(freq)==list:
			rf = tfs[ind].filter('bandpass', freqmin=freq[0], freqmax=freq[1], corners=2, zerophase=True)
		else:
			rf = tfs[ind].filter('lowpass', freq=freq, corners=2, zerophase=True)
	else:
		rf = tfs[ind]

	# trim RF to undo time expansion
	if len_factor>1:
		rf.data = rf.data[npts//(2*len_factor):-npts//(2*len_factor)-1]

	return rf