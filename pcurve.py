import sklearn
import numpy as np
from sklearn.decomposition import PCA
from scipy.interpolate import UnivariateSpline


class PrincipalCurve:
    def __init__(self, k = 3):
        """
        Constructs a Principal Curve with degree k.
        Attributes:
          order: argsort of pseudotimes
          points: curve
          points_interp: data projected onto curve
          pseudotimes: pseudotimes
          pseudotimes_interp: pseudotimes of data projected onto curve in data order
        :param k: polynomial spline degree
        """
        self.k = k
        self.order = None
        self.points = None
        self.pseudotimes = None
        self.points_interp = None
        self.pseudotimes_interp = None

    @staticmethod
    def from_params(s, p, order=None):
        """
        Constructs a PrincipalCurve. If no order given, an ordered input is assumed.
        """
        curve = PrincipalCurve()
        curve.update(s, p, order=order)
        return curve

    def update(self, s_interp, p_interp, order=None):
        self.pseudotimes_interp = s_interp
        self.points_interp = p_interp
        if order is None:
            self.order = np.arange(s_interp.shape[0])
        else:
            self.order = order

    def project(self, X):
        s_interp, p_interp, d_sq = self.project_on(X, self.points, self.pseudotimes)
        self.pseudotimes_interp = s_interp
        self.points_interp = p_interp
        return s_interp, p_interp, d_sq

    def project_on(self, X, points, pseudotimes):
        """
        Get interpolating s values for projection of X onto the curve defined by (p, s)
        @param X: data
        @param points: curve points
        @param pseudotimes: curve parameterisation
        @returns: interpolating parameter values, projected points on curve, sum of square distances
        """
        s_interp = np.zeros(X.shape[0])
        p_interp = np.zeros(X.shape)
        d_sq = list()

        for i in range(0, X.shape[0]):
            z = X[i, :]
            numerator = (points[1:] - points[0:-1]).T * np.einsum('ij,ij->i', z - points[0:-1], points[1:] - points[0:-1])
            denominator = np.power(np.linalg.norm(points[1:] - points[0:-1], axis=1), 2)
            seg_proj = (numerator / denominator).T  # compute parallel component
            proj_dist = (z - points[0:-1]) - seg_proj  # compute perpendicular component
            dist_endpts = np.minimum(np.linalg.norm(z - points[0:-1], axis=1), np.linalg.norm(z - points[1:], axis = 1))
            dist_seg = np.maximum(np.linalg.norm(proj_dist, axis = 1), dist_endpts)

            idx_min = np.argmin(dist_seg)
            q = seg_proj[idx_min] 
            s_interp[i] = (np.linalg.norm(q) / np.linalg.norm(points[idx_min + 1, :] - points[idx_min, :])) * (pseudotimes[idx_min + 1] - pseudotimes[idx_min]) + pseudotimes[idx_min]
            p_interp[i] = (s_interp[i] - pseudotimes[idx_min]) * (points[idx_min + 1, :] - points[idx_min, :]) + points[idx_min, :]
            d_sq.append(np.linalg.norm(proj_dist[idx_min])**2)

        d_sq = np.array(d_sq)
        self.order = s_interp.argsort()
        return s_interp, p_interp, d_sq

    def project_to_curve(self, X, p):
        s = self.renorm_parameterisation(p)
        s_interp, p_interp, d_sq = self.project_on(X, p, s)
        return s_interp, p_interp, d_sq

    def unpack_params(self):
        return self.pseudotimes_interp, self.points_interp, self.order

    def project_and_spline(self, X, p, s):
        s = self.renorm_parameterisation(p)
        s_interp, p_interp, d_sq = self.project_on(X, p, s)
        order = np.argsort(s_interp)

        spline = [UnivariateSpline(s_interp[order], X[order, j], k=self.k, w=None) for j in range(0, X.shape[1])]

        # p is the set of J functions producing a smooth curve in R^J
        p = np.zeros((len(s_interp), X.shape[1]))
        for j in range(0, X.shape[1]):
            p[:, j] = spline[j](s_interp[order])

        idx = [i for i in range(0, p.shape[0] - 1) if (p[i] != p[i + 1]).any()]
        p = p[idx, :]
        s = self.renorm_parameterisation(p)  # normalise to unit speed
        return s, p, d_sq

    def renorm_parameterisation(self, p):
        '''
        Renormalise curve to unit speed 
        @param p: curve points
        @returns: new parameterisation
        '''
        seg_lens = np.linalg.norm(p[1:] - p[0:-1], axis = 1)
        s = np.zeros(p.shape[0])
        s[1:] = np.cumsum(seg_lens)
        s = s/sum(seg_lens)
        return s
    
    def fit(self, X, p = None, w = None, max_iter = 10, tol = 1e-3):
        '''
        Fit principal curve to data
        @param X: data
        @param p: starting curve (optional) if None, then first principal components is used
        @param w: data weights (optional)
        @param max_iter: maximum number of iterations 
        @param tol: tolerance for stopping condition
        @returns: None
        '''
        if p is None:
            pca = sklearn.decomposition.PCA(n_components=X.shape[1])
            pca.fit(X)
            pc1 = pca.components_[:, 0]
            p = np.kron(np.dot(X, pc1)/np.dot(pc1, pc1), pc1).reshape(X.shape) # starting point for iteration
            order = np.argsort([np.linalg.norm(p[0, :] - p[i, :]) for i in range(0, p.shape[0])])
            p = p[order]
        s = self.renorm_parameterisation(p)
        
        p_interp = np.zeros(X.shape)
        s_interp = np.zeros(X.shape[0])
        d_sq_old = np.Inf
        
        for i in range(0, max_iter):
            # 1. Project data onto curve and set the pseudotime s_interp to be the arc length of the projections
            s_interp, p_interp, d_sq = self.project_on(X, p, s)
            d_sq = d_sq.sum()
            if np.abs(d_sq - d_sq_old) < tol:
                break
            d_sq_old = d_sq

            # 2. Use pseudotimes (s_interp) to order the data and apply a spline interpolation in each data dimension j
            order = np.argsort(s_interp)

            spline = [UnivariateSpline(s_interp[order], X[order, j], k=self.k, w=w) for j in range(0, X.shape[1])]

            # p is the set of J functions producing a smooth curve in R^J
            p = np.zeros((len(s_interp), X.shape[1]))
            for j in range(0, X.shape[1]):
                p[:, j] = spline[j](s_interp[order])

            idx = [i for i in range(0, p.shape[0]-1) if (p[i] != p[i+1]).any()]
            p = p[idx, :]
            s = self.renorm_parameterisation(p)  # normalise to unit speed
            
        self.pseudotimes = s
        self.points = p
        self.points_interp = p_interp
        self.pseudotimes_interp = s_interp
