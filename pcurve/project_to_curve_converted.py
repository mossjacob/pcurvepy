import numpy as np


def project_to_curve(x, s, stretch = 2):
    nseg = s.shape[0] - 1
    npts = x.shape[0]
    ncols = x.shape[1]

    # argument checks
    if s.shape[1] != ncols:
        raise "'x' and 's' must have an equal number of columns"

    if s.shape[0] < 2:
        raise "'s' must contain at least two rows."

    if x.shape[0] == 0:
        raise "'x' must contain at least one row."

    if stretch < 0:
        raise "Argument 'stretch' should be larger than or equal to 0"


    # perform stretch on end points of s
    # only perform stretch if s contains at least two rows
    if stretch > 0 and s.shape[0] >= 2:
        s = s.copy()
        n = s.shape[0]
        diff1 = s[0, :] - s[1, :]
        diff2 = s[n - 1, :] - s[n - 2, :]
        s[0, :] = s[0, :] + stretch * diff1
        s[n - 1, :] = s[n - 1, :] + stretch * diff2


    # precompute distances between successive points in the curve
    # and the length of each segment
    diff = np.zeros((nseg, ncols))
    length = np.zeros(nseg)

    # preallocate variables
    i, j, k, l, m = 0, 0, 0, 0, 0
    u, v, w = 0., 0., 0.

    for i in range(nseg):
        # OPTIMISATION: compute length manually
        #   diff(i, _) = s(i + 1, _) - s(i, _)
        #   length[i] = sum(pow(diff(i, _), 2))
        w = 0
        for k in range(ncols):
            v = s[i + 1, k] - s[i, k]
            # diff[k * nseg + i] = v
            diff[i, k] = v
            w += v * v
        length[i] = w
        # END OPTIMISATION


    # allocate output data structures
    new_points = np.zeros((npts, ncols))    # projections of x onto s
    pseudotime = np.zeros(npts)           # distance from start of the curve
    dist_ind = np.zeros(npts)         # distances between x and new_points

    # pre-allocate intermediate vectors
    n_test = np.zeros(ncols)
    n = np.zeros(ncols)
    p = np.zeros(ncols)

    # iterate over points in x
    for i in range(npts):
        # store information on the closest segment
        bestlam = -1
        bestdi = np.iinfo(np.int64).max

        # copy current point to p
        for k in range(ncols):
            p[k] = x[i, k]

        # iterate over the segments
        for j in range(nseg):
            # project p orthogonally onto the segment
            # OPTIMISATION: do not allocate diff1 and diff2 compute t manually
            #   NumericVector diff1 = s(j + 1, _) - s(j, _)
            #   NumericVector diff2 = p - s(j, _)
            #   double t = sum(diff1 * diff2) / length(j)
            v = 0
            for k in range(ncols):
               v += diff[j, k] * (p[k] - s[j, k])

            v /= length[j]
            # END OPTIMISATION
            if v < 0:
                v = 0.0
            if v > 1:
                v = 1.0

            # calculate position of projection and the distance
            # OPTIMISATION: compute di and n_test manually
            #   NumericVector n_test = s(j, _) + t * diff(j, _)
            #   double di = sum(pow(n_test - p, 2.0))
            w = 0
            for k in range(ncols):
                u = s[j, k] + v * diff[j, k]
                n_test[k] = u
                w += (u - p[k]) * (u - p[k])

            # END OPTIMISATION

            # if this is better than what was found earlier, store it
            if w < bestdi:
                bestdi = w
                bestlam = j + .1 + .9 * v
                for k in range(ncols):
                   n[k] = n_test[k]

        # save the best projection to the output data structures
        pseudotime[i] = bestlam
        dist_ind[i] = bestdi
        for k in range(ncols):
            # new_points[k * npts + i] = n[k]
            new_points[i, k] = n[k]


    # get ordering from old pseudotime
    new_ord = pseudotime.argsort()

    # calculate total dist
    dist = dist_ind.sum()

    # recalculate pseudotime for new_points
    pseudotime[new_ord[0]] = 0

    for i in range(1, new_ord.shape[0]):
        l = new_ord[i - 1]
        m = new_ord[i]

        # OPTIMISATION: compute pseudotime[o1] manually
        #   NumericVector p1 = new_points(o1, _)
        #   NumericVector p0 = new_points(o0, _)
        #   pseudotime[o1] = pseudotime[o0] + sqrt(sum(pow(p1 - p0, 2.0)))
        w = 0
        for k in range(ncols):
            v = new_points[m, k] - new_points[l, k]
            w += v * v
        pseudotime[m] = pseudotime[l] + np.sqrt(w)
    # END OPTIMISATION

    return pseudotime, new_points, new_ord, dist_ind, dist
