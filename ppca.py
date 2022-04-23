import numpy as np
import scipy

class PPCA:
	"""Probabilistic Principal Component Analysis (PPCA)

	Arguments
	---------
	:param ncomp: int
		Number of principal components
	:param tol: float
		Convergence tolerence
	:param max_iter: int
		Maximum number of EM iterations

	Notes
	-----
	Method implementation based on
	"Tipping, M.E. and Bishop, C.M., 1999. 
	Probabilistic principal component analysis. 
	Journal of the Royal Statistical Society: 
	Series B (Statistical Methodology), 61(3), 
	pp. 611-622."

	Code implementation based on sklearn, e.g.
	https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/decomposition/_fastica.py

	Inspiration from
	https://github.com/allentran/pca-magic
	https://github.com/shergreen/pyppca
	http://lear.inrialpes.fr/~verbeek

	"""
    def __init__(self, n_components, tol=1e-5, max_iter=500):
    	if max_iter < 1:
    		raise ValueError("max_iter should be greater than 1, got max_iter={}".format(max_iter))
    	if tol < 0:
    		raise ValueErro("tol should be positive, got tol={}".format(tol))

    	self.q = n_components
    	self.tol = tol
    	self.max_iter = max_iter

    def fit(self, X):
    	"""Fit the model
    	"""
    	print("here")

    def fit_transform(self, X):
    	"""Fit the model and apply the dimensionality reduction.
    	"""
    	print("here")

	def ppca(self, data):
		"""Apply probabilistic principal component analysis (PPCA) to incomplete data.

		Arguments
		---------
		:param data: np.array
			Input (potentially incomplete) data, dimensions [M, N] are M realizations of N-dimensional observations.
		"""
		M,N=data.shape

		# create a mask for missing data
		missing=np.isnan(data)
		n_missing=np.sum(missing)

		# demean data
		mu=np.nanmean(data, axis=0)
		data-=mu

		# fill missing values with 0 for the moment
		data[missing]=0

		# initialize W [N,q]
		# C=W W^T + \sigma^2 I [NxN]
		# M=W^T W + \sigma^2 I [qxq]
		W = np.random.randn(N,self.q)
		X = data @ W @ np.linalg.inv(W.T @ W)
		recon = X @ W.T
		recon[missing] = 0
		var0 = np.sum((recon-data)**2)/(N*D - n_missing)

		n_iter=0
		v0=np.inf
		while (tol) and (n_iter<max_iter):
			Sx = np.linalg.inv(np.eye(self.q) + (W.T @ W)/ss)

			# E-step
			ss0=ss
			proj = X @ W.T
			data[missing] = proj[missing]
			X=data @ W @ Sx/ss

			# M-step
			XX = X.T @ X
			W = data.T @ X @ np.linalg.pinv(XX + M*Sx)
			recon = X @ W.T
			recon[missing]=0
			ss = (np.sum((recon-data)**2) + M*np.sum((W.T @ W)*Sx) + n_missing*ss0)/(M*self.q)

			# Calculate convergence criterion
			det = np.log(np.linalg.det(Sx))
			v1 = M*(self.q*np.log(ss) + np.trace(Sx) - det) + np.trace(XX) - n_missing*np.log(ss0)
			diff = np.abs(v1/v0 - 1)

			n_iter+=1
			v0=v1

def ppca(Y, d, dia, tol=1e-4):
    """
    Implements probabilistic PCA for data with missing values,
    using a factorizing distribution over hidden states and hidden observations.

    Args:
        Y:   (N by D ) input numpy ndarray of data vectors
        d:   (  int  ) dimension of latent space
        dia: (boolean) if True: print objective each step

    Returns:
        C:  (D by d ) C*C' + I*ss is covariance model, C has scaled principal directions as cols
        ss: ( float ) isotropic variance outside subspace
        M:  (D by 1 ) data mean
        X:  (N by d ) expected states
        Ye: (N by D ) expected complete observations (differs from Y if data is missing)

        Based on MATLAB code from J.J. VerBeek, 2006. http://lear.inrialpes.fr/~verbeek
    """
    N, D = shape(Y)  # N observations in D dimensions (i.e. D is number of features, N is samples)
    hidden = isnan(Y)
    missing = hidden.sum()
    M = nanmean(Y, axis=0)
    Ye = Y - repmat(M, N, 1)
    Ye[hidden] = 0

    # initialize
    C = normal(loc=0.0, scale=1.0, size=(D, d))
    CtC = C.T @ C
    X = Ye @ C @ inv(CtC)
    recon = X @ C.T
    recon[hidden] = 0
    ss = np.sum((recon - Ye) ** 2) / (N * D - missing)

    count = 1
    old = np.inf

    # EM Iterations
    while (count):
        Sx = inv(eye(d) + CtC / ss)  # E-step, covariances
        ss_old = ss
        if missing > 0:
            proj = X @ C.T
            Ye[hidden] = proj[hidden]

        X = Ye @ C @ Sx / ss  # E-step: expected values

        SumXtX = X.T @ X  # M-step
        C = Ye.T @ X @ (SumXtX + N * Sx).T @ inv(((SumXtX + N * Sx) @ (SumXtX + N * Sx).T))
        CtC = C.T @ C
        ss = (np.sum((X @ C.T - Ye) ** 2) + N * np.sum(CtC * Sx) + missing * ss_old) / (N * D)
        # transform Sx determinant into numpy longdouble in order to deal with high dimensionality
        Sx_det = np.min(Sx).astype(np.longdouble) ** shape(Sx)[0] * det(Sx / np.min(Sx))
        objective = N * D + N * (D * log(ss) + tr(Sx) - log(Sx_det)) + tr(SumXtX) - missing * log(ss_old)

        rel_ch = np.abs(1 - objective / old)
        old = objective

        count = count + 1
        if rel_ch < tol and count > 5:
            count = 0
        if dia:
            print(f"Objective: {objective:.2f}, Relative Change {rel_ch:.5f}")

    C = orth(C)
    covM = cov((Ye @ C).T)
    vals, vecs = eig(covM)
    ordr = np.argsort(vals)[::-1]
    vecs = vecs[:, ordr]

    C = C @ vecs
    X = Ye @ C

    # add data mean to expected complete data
    Ye = Ye + repmat(M, N, 1)

    return C, ss, M, X, Ye