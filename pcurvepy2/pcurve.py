from __future__ import annotations

import numpy as np
from scipy.interpolate import UnivariateSpline
from sklearn.decomposition import PCA


class PrincipalCurve:
    def __init__(self, k: int = 3) -> None:
        """
        Constructs a Principal Curve of degree k.
        Attributes:
          order: argsort of pseudotimes
          points: curve
          points_interp: data projected onto curve
          pseudotimes: pseudotimes
          pseudotimes_interp: pseudotimes of data projected onto curve in data order
        :param k: polynomial spline degree
        """
        self.k: int = k
        self.order: np.ndarray | None = None
        self.points: np.ndarray | None = None
        self.pseudotimes: np.ndarray | None = None
        self.points_interp: np.ndarray | None = None
        self.pseudotimes_interp: np.ndarray | None = None

    @staticmethod
    def from_params(pseudotime: np.ndarray, points: np.ndarray, order: np.ndarray | None = None) -> PrincipalCurve:
        """
        Constructs a PrincipalCurve. If no order given, an ordered input is assumed.
        """
        curve = PrincipalCurve()
        curve.update(pseudotime, points, order=order)
        return curve

    def update(self, pseudotime_interp: np.ndarray, points_interp: np.ndarray, order: np.ndarray | None = None) -> None:
        self.pseudotimes_interp = pseudotime_interp
        self.points_interp = points_interp
        if order is None:
            self.order = np.arange(pseudotime_interp.shape[0])
        else:
            self.order = order

    def project_to_curve(
        self, data: np.ndarray, points: np.ndarray | None = None, stretch: float = 0.0
    ) -> tuple[np.ndarray, float]:
        """
        Originally a Python translation of R/C++ package `princurve`
        Projects set of points `X` to the closest point on a curve made up
        of points `points`. Finds the projection index for a matrix of points `X`.
        The curve need not be of the same
        length as the number of points.
        Parameters:
            data: a matrix of data points.
            points: a parametrized curve, represented by a polygon.
            stretch: A stretch factor for the endpoints of the curve,
                     allowing the curve to grow to avoid bunching at the end.
                     Must be a numeric value between 0 and 2.

        """
        if points is None:
            points = self.points

        assert points is not None, "points must be provided or fitted"

        # Num segments = points.shape[0] - 1
        n_pts = data.shape[0]
        n_features = data.shape[1]

        # argument checks
        if points.shape[1] != n_features:
            raise ValueError("'x' and 's' must have an equal number of columns")

        if points.shape[0] < 2:
            raise ValueError("'s' must contain at least two rows.")

        if data.shape[0] == 0:
            raise ValueError("'x' must contain at least one row.")

        if stretch < 0:
            raise ValueError("Argument 'stretch' should be larger than or equal to 0")

        # perform stretch on end points of s
        # only perform stretch if s contains at least two rows
        if stretch > 0 and points.shape[0] >= 2:
            points = points.copy()
            num_points = points.shape[0]
            diff_start = points[0, :] - points[1, :]
            diff_end = points[num_points - 1, :] - points[num_points - 2, :]
            points[0, :] = points[0, :] + stretch * diff_start
            points[num_points - 1, :] = points[num_points - 1, :] + stretch * diff_end

        # precompute distances between successive points in the curve
        # and the length of each segment
        segment_diffs = points[1:] - points[:-1]
        segment_lengths = np.square(segment_diffs).sum(axis=1)
        # segment_lengths = np.power(np.linalg.norm(segment_diffs, axis=1), 2)
        segment_lengths += 1e-7
        # allocate output data structures
        new_points = np.zeros((n_pts, n_features))  # projections of x onto s
        new_pseudotimes = np.zeros(n_pts)  # distance from start of the curve
        dist_ind = np.zeros(n_pts)  # distances between x and new_s

        # iterate over points in x
        for point_idx in range(data.shape[0]):
            current_point = data[point_idx, :]

            # project current_point orthogonally onto the segment --  compute parallel component
            seg_proj = (segment_diffs * (current_point - points[:-1])).sum(axis=1)
            seg_proj /= segment_lengths
            seg_proj[seg_proj < 0] = 0.0
            seg_proj[seg_proj > 1.0] = 1.0

            projection = (seg_proj * segment_diffs.T).T
            proj_dist = current_point - points[:-1] - projection
            proj_sq_dist = np.square(proj_dist).sum(axis=1)

            # calculate position of projection and the distance
            min_segment_idx = proj_sq_dist.argmin()
            dist_ind[point_idx] = proj_sq_dist[min_segment_idx]
            new_pseudotimes[point_idx] = min_segment_idx + 0.1 + 0.9 * seg_proj[min_segment_idx]
            new_points[point_idx] = current_point - proj_dist[min_segment_idx]

        # get ordering from old pseudotime
        new_order = new_pseudotimes.argsort()

        # calculate total dist
        total_distance = dist_ind.sum()

        # recalculate pseudotime for new_s
        new_pseudotimes[new_order[0]] = 0

        for idx in range(1, new_order.shape[0]):
            prev_point_idx = new_order[idx - 1]
            curr_point_idx = new_order[idx]

            # OPTIMISATION: compute pseudotime[o1] manually
            #   NumericVector p1 = new_s(o1, _)
            #   NumericVector p0 = new_s(o0, _)
            #   pseudotime[o1] = pseudotime[o0] + sqrt(sum(pow(p1 - p0, 2.0)))
            point_diff = new_points[curr_point_idx, :] - new_points[prev_point_idx, :]
            arc_length = np.linalg.norm(point_diff)
            new_pseudotimes[curr_point_idx] = new_pseudotimes[prev_point_idx] + arc_length

        self.pseudotimes_interp = new_pseudotimes
        self.points_interp = new_points
        self.order = new_order
        return dist_ind, total_distance

    def unpack_params(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert self.pseudotimes_interp is not None
        assert self.points_interp is not None
        assert self.order is not None
        return self.pseudotimes_interp, self.points_interp, self.order

    def renorm_parameterisation(self, curve_points: np.ndarray) -> np.ndarray:
        """
        Renormalise curve to unit speed
        Args:
            curve_points: principal curve points
        Returns:
            normalised parameterisation
        """
        segment_lengths = np.linalg.norm(curve_points[1:] - curve_points[:-1], axis=1)
        normalized_params = np.zeros(curve_points.shape[0])
        normalized_params[1:] = np.cumsum(segment_lengths)
        total_length: float = float(np.sum(segment_lengths))
        return normalized_params / total_length  # type: ignore

    def fit(
        self,
        data: np.ndarray,
        initial_points: np.ndarray | None = None,
        weights: np.ndarray | None = None,
        param_s: float | None = None,
        max_iter: int = 10,
        tol: float = 1e-3,
    ) -> None:
        """
        Fit principal curve to data
        Args:
            data: data matrix
            initial_points: starting curve (optional) if None, then first principal components is used
            weights: data weights (optional)
            param_s: positive smoothing factor used to choose the number of knots. Number of knots will be increased until the smoothing condition is satisfied.
            max_iter: maximum number of iterations
            tol: tolerance for stopping condition

        """
        if initial_points is None and self.points is None:
            pca = PCA(n_components=data.shape[1])
            pca.fit(data)
            first_component = pca.components_[:, 0]

            projected_data = np.kron(
                np.dot(data, first_component) / np.dot(first_component, first_component), first_component
            ).reshape(data.shape)  # starting point for iteration
            sorted_order = np.argsort(
                [np.linalg.norm(projected_data[0, :] - projected_data[i, :]) for i in range(0, projected_data.shape[0])]
            )
            initial_points = projected_data[sorted_order]

        if self.pseudotimes_interp is None:
            self.project_to_curve(data, points=initial_points)

        distance_sq_old = np.inf

        for _ in range(0, max_iter):
            # 1. Use pseudotimes (s_interp) to order the data and
            # apply a spline interpolation in each data dimension
            order = self.order
            pseudotimes_interp = self.pseudotimes_interp
            assert order is not None and pseudotimes_interp is not None
            pseudotimes_unique, unique_indices = np.unique(pseudotimes_interp[order], return_index=True)

            splines = [
                UnivariateSpline(
                    pseudotimes_unique,
                    data[order, dim_idx][unique_indices],
                    k=self.k,
                    s=param_s,
                    w=weights[order][unique_indices] if weights is not None else None,
                )
                for dim_idx in range(0, data.shape[1])
            ]
            # curve_points is the set of functions producing a smooth curve
            curve_points = np.zeros((len(pseudotimes_interp), data.shape[1]))
            for dim_idx in range(0, data.shape[1]):
                curve_points[:, dim_idx] = splines[dim_idx](pseudotimes_interp[order])

            non_duplicate_indices = [
                i for i in range(0, curve_points.shape[0] - 1) if (curve_points[i] != curve_points[i + 1]).any()
            ]  # remove duplicate consecutive points?
            curve_points = curve_points[non_duplicate_indices, :]
            normalized_pseudotimes = self.renorm_parameterisation(curve_points)  # normalise to unit speed

            # 2. Project data onto curve and set the pseudotime to be the arc length of the projections
            dist_ind, distance_sq = self.project_to_curve(
                data,
                points=curve_points,
            )

            total_distance_sq: float = float(dist_ind.sum())
            if np.abs(total_distance_sq - distance_sq_old) < tol:
                break
            distance_sq_old = total_distance_sq

        self.pseudotimes = normalized_pseudotimes
        self.points = curve_points
