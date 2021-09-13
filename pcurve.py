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

    def project_to_curve(self, X, points=None, stretch=2):
        """Python translation of R/C++ package `princurve`"""
        if points is None:
            points = self.points
        nseg = points.shape[0] - 1
        npts = X.shape[0]
        ncols = X.shape[1]
        print('nseg', nseg, 'ncols', ncols)
        print('x', X.shape, 's', points.shape)

        # argument checks
        if points.shape[1] != ncols:
            raise "'x' and 's' must have an equal number of columns"

        if points.shape[0] < 2:
            raise "'s' must contain at least two rows."

        if X.shape[0] == 0:
            raise "'x' must contain at least one row."

        if stretch < 0:
            raise "Argument 'stretch' should be larger than or equal to 0"

        # perform stretch on end points of s
        # only perform stretch if s contains at least two rows
        if stretch > 0 and points.shape[0] >= 2:
            points = points.copy()
            n = points.shape[0]
            diff1 = points[0, :] - points[1, :]
            diff2 = points[n - 1, :] - points[n - 2, :]
            points[0, :] = points[0, :] + stretch * diff1
            points[n - 1, :] = points[n - 1, :] + stretch * diff2

        # precompute distances between successive points in the curve
        # and the length of each segment
        # diff = np.zeros((nseg, ncols))
        # length = np.zeros(nseg)
        diff = points[1:] - points[:-1]
        length = np.square(diff).sum(axis=1)
        # for i in range(nseg):
            # OPTIMISATION: compute length manually
            #   diff(i, _) = s(i + 1, _) - s(i, _)
            #   length[i] = sum(pow(diff(i, _), 2))
            # w = 0
            # for k in range(ncols):
                # v = points[i + 1, k] - points[i, k]
                ## diff[k * nseg + i] = v
                # diff[i, k] = v
                # w += v * v
            # length[i] = w
            # END OPTIMISATION

        # allocate output data structures
        new_points = np.zeros((npts, ncols))  # projections of x onto s
        pseudotime = np.zeros(npts)  # distance from start of the curve
        dist_ind = np.zeros(npts)  # distances between x and new_s

        # pre-allocate intermediate vectors
        n = np.zeros(ncols)

        # iterate over points in x
        for i in range(X.shape[0]):
            # store information on the closest segment
            p = X[i, :]  # p is vector of dimensions

            numerator = diff.T * np.einsum('ij,ij->i', p - points[:-1], diff)  # multiply and sum along second axis
            seg_proj = (numerator / length).T  # compute parallel component

            n_test = points[:-1] + seg_proj
            # project p orthogonally onto the segment
            v = (diff * (p - points[:-1])).sum(axis=1) / length
            v[v < 0] = 0.
            v[v > 1.] = 1.
            w = np.square(n_test - p).sum(axis=1)
            j = w.argmin()
            # calculate position of projection and the distance

            dist_ind[i] = w[j]
            pseudotime[i] = j + .1 + .9 * v[j]
            new_points[i] = n_test[j]
            # save the best projection to the output data structures
            # for k in range(ncols):
                # new_s[k * npts + i] = n[k]

        # get ordering from old pseudotime
        new_ord = pseudotime.argsort()

        # calculate total dist
        dist = dist_ind.sum()

        # recalculate pseudotime for new_s
        pseudotime[new_ord[0]] = 0

        for i in range(1, new_ord.shape[0]):
            l = new_ord[i - 1]
            m = new_ord[i]

            # OPTIMISATION: compute pseudotime[o1] manually
            #   NumericVector p1 = new_s(o1, _)
            #   NumericVector p0 = new_s(o0, _)
            #   pseudotime[o1] = pseudotime[o0] + sqrt(sum(pow(p1 - p0, 2.0)))
            # for k in range(ncols):
            v = new_points[m, :] - new_points[l, :]
            w = (v * v).sum()
            pseudotime[m] = pseudotime[l] + np.sqrt(w)
        # END OPTIMISATION
        pseudotime_min = pseudotime.min()
        pseudotime = (pseudotime - pseudotime_min) / (pseudotime.max() - pseudotime_min)
        self.pseudotimes_interp = pseudotime
        self.points_interp = new_points
        self.order = new_ord
        return self, dist_ind, dist

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
        diff = points[1:] - points[:-1]  # first difference
        # denominator = np.power(np.linalg.norm(diff, axis=1), 2)
        length = np.square(diff).sum(axis=1)
        for i in range(X.shape[0]): # for each point
            z = X[i, :]  # z is the vector of the dimensions for the point
            numerator = diff.T * np.einsum('ij,ij->i', z - points[:-1], diff)
            seg_proj = (numerator / length).T  # compute parallel component
            proj_dist = z - points[:-1] - seg_proj  # compute perpendicular component
            dist_endpts = np.minimum(np.linalg.norm(z - points[:-1], axis=1), np.linalg.norm(z - points[1:], axis=1))
            dist_seg = np.maximum(np.linalg.norm(proj_dist, axis=1), dist_endpts)

            idx_min = np.argmin(dist_seg)
            q = seg_proj[idx_min]
            s_interp[i] = (np.linalg.norm(q) / np.linalg.norm(points[idx_min + 1, :] - points[idx_min, :])) * (pseudotimes[idx_min + 1] - pseudotimes[idx_min]) + pseudotimes[idx_min]
            p_interp[i] = (s_interp[i] - pseudotimes[idx_min]) * (points[idx_min + 1, :] - points[idx_min, :]) + points[idx_min, :]

            #####
            n_test = points[:-1] + seg_proj
            w = np.square(n_test - z).sum(axis=1)
            # print('w', w.argmax())
            p_interp[i] = points[w.argmax()]
            # p_interp[i] =

            #####
            d_sq.append(np.linalg.norm(proj_dist[idx_min])**2)

        d_sq = np.array(d_sq)
        self.order = s_interp.argsort()
        return s_interp, p_interp, d_sq

    def _project_to_curve(self, X, points=None):
        if points is None:
            points = self.points_interp
        s = self.renorm_parameterisation(points)
        s_interp, p_interp, d_sq = self.project_on(X, points, s)
        self.pseudotimes_interp = s_interp
        self.points_interp = p_interp
        self.order = s_interp.argsort()
        return self, d_sq, d_sq.sum()

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
        seg_lens = np.linalg.norm(p[1:] - p[:-1], axis=1)
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
            print('i', i)
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

            idx = [i for i in range(0, p.shape[0]-1) if (p[i] != p[i+1]).any()] # remove duplicate consecutive points?
            p = p[idx, :]
            s = self.renorm_parameterisation(p)  # normalise to unit speed
            
        self.pseudotimes = s
        self.points = p
        self.pseudotimes_interp = s_interp
        self.points_interp = p_interp
        self.order = order