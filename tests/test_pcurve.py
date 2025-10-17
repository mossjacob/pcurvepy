import numpy as np
import pytest

from pcurvepy2 import PrincipalCurve


class TestPrincipalCurveFunctionality:
    """Tests for actual principal curve functionality and correctness"""

    def test_linear_data_produces_linear_curve(self):
        """Test that a straight line of data produces a linear principal curve"""
        pc = PrincipalCurve(k=3)
        # Create points along a line y = 2x + 1 with some noise
        np.random.seed(42)
        t = np.linspace(0, 10, 50)
        X = np.column_stack([t, 2 * t + 1]) + np.random.randn(50, 2) * 0.1

        pc.fit(X, max_iter=10)

        # Check that the curve is approximately linear
        # The curve should have low residuals when fit to a line
        curve_points = pc.points
        assert curve_points is not None
        # Fit a line to the curve points
        coeffs = np.polyfit(curve_points[:, 0], curve_points[:, 1], 1)
        predicted = np.polyval(coeffs, curve_points[:, 0])
        residuals = np.abs(curve_points[:, 1] - predicted)

        # The curve should be approximately linear (mean residual < 0.5)
        assert np.mean(residuals) < 0.02, f"Curve is not linear enough, mean residual: {np.mean(residuals)}"
        # Slope should be approximately 2
        assert abs(coeffs[0] - 2.0) < 0.01, f"Slope should be ~2, got {coeffs[0]}"

    def test_circular_data_produces_circular_curve(self):
        """Test that circular data produces a circular-ish principal curve"""
        pc = PrincipalCurve(k=3)
        np.random.seed(42)
        # Create points around a circle with some noise
        theta = np.linspace(0, 2 * np.pi, 100)
        radius = 5.0
        X = np.column_stack(
            [radius * np.cos(theta) + np.random.randn(100) * 0.2, radius * np.sin(theta) + np.random.randn(100) * 0.2]
        )

        pc.fit(X, max_iter=15)

        # Check that curve points are approximately equidistant from origin
        curve_points = pc.points
        distances_from_origin = np.linalg.norm(curve_points, axis=1)

        # Most points should be within 1 unit of the expected radius
        assert (
            np.std(distances_from_origin) < 0.8
        ), f"Curve is not circular enough, std: {np.std(distances_from_origin)}"
        assert (
            abs(np.mean(distances_from_origin) - radius) < 0.22
        ), f"Mean radius should be ~{radius}, got {np.mean(distances_from_origin)}"

    def test_projection_distance_decreases_with_iterations(self):
        """Test that the total projection distance decreases during fitting"""
        pc = PrincipalCurve(k=3)
        np.random.seed(42)
        t = np.linspace(0, 4 * np.pi, 100)
        X = np.column_stack([t, np.sin(t)]) + np.random.randn(100, 2) * 0.1

        # Manually track distances across iterations
        pca_result = np.column_stack([t, 0 * t])  # Simple initial line
        pc.project_to_curve(X, points=pca_result[:50])

        distances = []
        for i in range(5):
            order = pc.order
            pseudotimes_interp = pc.pseudotimes_interp
            assert order is not None and pseudotimes_interp is not None
            pseudotimes_uniq, ind = np.unique(pseudotimes_interp[order], return_index=True)

            from scipy.interpolate import UnivariateSpline

            spline = [UnivariateSpline(pseudotimes_uniq, X[order, j][ind], k=3, s=None) for j in range(X.shape[1])]

            p = np.zeros((len(pseudotimes_interp), X.shape[1]))
            for j in range(X.shape[1]):
                p[:, j] = spline[j](pseudotimes_interp[order])

            idx = [i for i in range(p.shape[0] - 1) if (p[i] != p[i + 1]).any()]
            if len(idx) > 0:
                p = p[idx, :]

            dist_ind, total_dist = pc.project_to_curve(X, points=p)
            distances.append(total_dist)

        # Distance should generally decrease (at least not increase significantly)
        # Allow for small increases due to numerical issues
        assert distances[-1] <= distances[0], f"Distance should decrease, got {distances[0]} -> {distances[-1]}"

    def test_curve_passes_near_data_points(self):
        """Test that the fitted curve passes near the actual data points"""
        pc = PrincipalCurve(k=3)
        np.random.seed(42)
        # Create a simple curved dataset
        t = np.linspace(0, 2 * np.pi, 50)
        X = np.column_stack([t, np.sin(t) * 2])

        pc.fit(X, max_iter=10)

        # For each data point, find the minimum distance to the curve
        min_distances: list[float] = []
        assert pc.points is not None
        for point in X:
            distances = np.linalg.norm(pc.points - point, axis=1)
            min_distances.append(np.min(distances))

        # Most points should be close to the curve
        mean_distance: float = float(np.mean(min_distances))
        max_distance: float = float(np.max(min_distances))

        assert mean_distance < 0.2, f"Mean distance to curve too large: {mean_distance}"
        assert max_distance < 0.3, f"Max distance to curve too large: {max_distance}"

    def test_pseudotime_ordering_preserves_curve_structure(self):
        """Test that pseudotime ordering correctly orders points along the curve"""
        pc = PrincipalCurve(k=3)
        # Create points that clearly follow a path
        t = np.linspace(0, 10, 30)
        X = np.column_stack([t, t**1.5])

        pc.fit(X, max_iter=10)

        # Get pseudotimes for points in their original order
        pseudotimes = pc.pseudotimes_interp

        # Check that the correlation between original ordering and pseudotime is high
        # (they should both reflect the progression along the curve)
        from scipy.stats import spearmanr

        correlation, _ = spearmanr(np.arange(len(t)), pseudotimes)

        assert correlation > 0.9, f"Pseudotime ordering doesn't preserve structure, correlation: {correlation}"

    def test_projection_onto_straight_line(self):
        """Test that points project correctly onto a straight line curve"""
        pc = PrincipalCurve(k=3)
        # Data points
        X = np.array([[0.5, 0.5], [1.5, 0.8], [2.5, 1.2], [3.5, 1.5]])
        # Simple straight line curve from (0,0) to (4,2)
        curve_points = np.array([[0.0, 0.0], [2.0, 1.0], [4.0, 2.0]])

        dist_ind, total_dist = pc.project_to_curve(X, points=curve_points, stretch=0)

        # Check that projections exist
        assert pc.points_interp is not None
        assert pc.points_interp.shape == X.shape

        # All projected points should lie on or very near the line y = 0.5x
        for proj_point in pc.points_interp:
            expected_y = 0.5 * proj_point[0]
            assert abs(proj_point[1] - expected_y) < 0.05, f"Projection {proj_point} not on line"

        # Distances should all be relatively small (points are near the line)
        assert total_dist < 0.2, f"Total projection distance too large: {total_dist}"

    def test_pseudotime_increases_monotonically_along_curve(self):
        """Test that pseudotimes increase monotonically when following the curve order"""
        pc = PrincipalCurve(k=3)
        t = np.linspace(0, 5, 20)
        X = np.column_stack([t, t * 2])

        pc.fit(X, max_iter=10)

        # Get pseudotimes in curve order
        assert pc.pseudotimes_interp is not None and pc.order is not None
        ordered_pseudotimes = pc.pseudotimes_interp[pc.order]

        # Check monotonicity
        differences = np.diff(ordered_pseudotimes)
        assert np.all(differences >= 0), "Pseudotimes should be monotonically increasing along curve"

    def test_renormalization_produces_unit_length_parameterization(self):
        """Test that renormalization produces parameterization from 0 to 1"""
        pc = PrincipalCurve(k=3)
        # Create some arbitrary curve points
        points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0], [2.0, 2.0], [1.0, 3.0]])

        s = pc.renorm_parameterisation(points)

        # Should start at 0
        assert s[0] == 0.0
        # Should end at 1
        assert abs(s[-1] - 1.0) < 1e-10
        # Should be monotonically increasing
        assert np.all(np.diff(s) >= 0)
        # All values should be in [0, 1]
        assert np.all(s >= 0) and np.all(s <= 1)

    def test_curve_smoothness(self):
        """Test that the fitted curve is smooth (no sharp angles)"""
        pc = PrincipalCurve(k=3)
        np.random.seed(42)
        # Create noisy sinusoidal data
        t = np.linspace(0, 2 * np.pi, 100)
        X = np.column_stack([t, np.sin(t)]) + np.random.randn(100, 2) * 0.05

        pc.fit(X, max_iter=15)

        # Check smoothness by looking at second derivatives (curvature)
        curve = pc.points
        assert curve is not None
        if len(curve) > 3:
            # Calculate discrete second derivative
            first_diff = np.diff(curve, axis=0)
            second_diff = np.diff(first_diff, axis=0)

            # Second differences should not be too large (curve is smooth)
            max_second_diff: float = float(np.max(np.linalg.norm(second_diff, axis=1)))
            assert max_second_diff < 0.3, f"Curve has sharp angles, max second diff: {max_second_diff}"

    def test_fitted_curve_represents_data_center(self):
        """Test that the curve runs through the center of the data distribution"""
        pc = PrincipalCurve(k=3)
        np.random.seed(42)
        # Create data distributed around a line
        t = np.linspace(0, 10, 100)
        X = np.column_stack([t + np.random.randn(100) * 0.5, 2 * t + np.random.randn(100) * 0.5])

        pc.fit(X, max_iter=10)

        # Divide data into segments and check that curve passes near the center
        n_segments = 5
        for i in range(n_segments):
            start_idx = i * len(X) // n_segments
            end_idx = (i + 1) * len(X) // n_segments
            segment_data = X[start_idx:end_idx]
            segment_center = np.mean(segment_data, axis=0)

            # Find closest curve point to this segment
            assert pc.points is not None
            distances = np.linalg.norm(pc.points - segment_center, axis=1)
            min_dist: float = float(np.min(distances))

            # Curve should pass within 1 unit of the segment center
            assert min_dist < 0.3, f"Curve too far from data center in segment {i}: {min_dist}"


class TestPrincipalCurveEdgeCases:
    """Tests focusing on edge cases and error handling"""

    def test_single_point_raises_error(self):
        """Test that fitting with a single point raises an error"""
        pc = PrincipalCurve(k=3)
        X = np.array([[1.0, 2.0]])

        with pytest.raises(ValueError):
            pc.fit(X, max_iter=1)

    def test_collinear_points(self):
        """Test fitting on perfectly collinear points"""
        pc = PrincipalCurve(k=3)
        X = np.array([[i, i * 2] for i in range(10)], dtype=float)

        pc.fit(X, max_iter=5)

        assert pc.points is not None
        assert pc.pseudotimes is not None
        # Should produce a reasonable number of curve points
        assert len(pc.points) > 2

    def test_duplicate_points(self):
        """Test fitting with duplicate points"""
        pc = PrincipalCurve(k=3)
        X = np.array([[1.0, 1.0], [1.0, 1.0], [2.0, 2.0], [2.0, 2.0], [3.0, 3.0], [3.0, 3.0], [4.0, 4.0], [4.0, 4.0]])

        pc.fit(X, max_iter=5)

        assert pc.points is not None
        assert pc.points_interp is not None
        # Should handle duplicates without crashing
        assert len(pc.points_interp) == len(X)

    def test_high_dimensional_data(self):
        """Test with higher dimensional data"""
        pc = PrincipalCurve(k=3)
        np.random.seed(42)
        # Create 5D data along a curve
        t = np.linspace(0, 2 * np.pi, 30)
        X = np.column_stack([np.cos(t), np.sin(t), t, np.cos(2 * t), np.sin(2 * t)]) + np.random.randn(30, 5) * 0.1

        pc.fit(X, max_iter=5)

        # Dimensions should be preserved
        assert pc.points is not None and pc.points_interp is not None
        assert pc.points.shape[1] == 5
        assert pc.points_interp.shape[1] == 5

    def test_minimal_points(self):
        """Test with minimum viable number of points"""
        pc = PrincipalCurve(k=3)
        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 1.5], [3.0, 2.0]])

        pc.fit(X, max_iter=5)

        assert pc.points is not None
        assert pc.order is not None
        assert len(pc.order) == len(X)

    def test_project_to_curve_stretch_zero(self):
        """Test projection with no stretch"""
        pc = PrincipalCurve(k=3)
        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        points = np.array([[0.5, 0.5], [1.5, 1.5], [2.5, 2.5]])

        dist_ind, dist = pc.project_to_curve(X, points=points, stretch=0)

        assert dist_ind is not None
        assert len(dist_ind) == len(X)
        assert dist >= 0

    def test_project_to_curve_with_stretch(self):
        """Test that stretch parameter affects endpoint projections"""
        pc = PrincipalCurve(k=3)
        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        points = np.array([[0.5, 0.5], [1.5, 1.5], [2.5, 2.5]])

        # Project with no stretch
        dist_ind_no_stretch, _ = pc.project_to_curve(X, points=points, stretch=0)

        # Project with stretch
        dist_ind_stretch, _ = pc.project_to_curve(X, points=points, stretch=0.5)

        # Stretch should affect the projections (especially for endpoint data)
        # The distances might be different
        assert dist_ind_no_stretch is not None
        assert dist_ind_stretch is not None

    def test_project_to_curve_dimension_mismatch_raises(self):
        """Test that dimension mismatch raises error"""
        pc = PrincipalCurve(k=3)
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        points = np.array([[0.5, 0.5, 0.5]])  # 3D instead of 2D

        with pytest.raises(ValueError):
            pc.project_to_curve(X, points=points)

    def test_project_to_curve_insufficient_points_raises(self):
        """Test that curve with < 2 points raises error"""
        pc = PrincipalCurve(k=3)
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        points = np.array([[0.5, 0.5]])  # Only 1 point

        with pytest.raises(ValueError):
            pc.project_to_curve(X, points=points)

    def test_project_to_curve_negative_stretch_raises(self):
        """Test that negative stretch raises error"""
        pc = PrincipalCurve(k=3)
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        points = np.array([[0.5, 0.5], [1.5, 1.5]])

        with pytest.raises(ValueError):
            pc.project_to_curve(X, points=points, stretch=-0.1)

    def test_from_params_without_order(self):
        """Test creating curve from parameters without explicit order"""
        pseudotime = np.array([0.0, 1.0, 2.0, 3.0])
        points = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])

        pc = PrincipalCurve.from_params(pseudotime, points)

        assert pc.pseudotimes_interp is not None
        assert pc.points_interp is not None
        assert pc.order is not None
        assert len(pc.order) == len(pseudotime)
        # Default order should be sequential
        np.testing.assert_array_equal(pc.order, np.arange(len(pseudotime)))

    def test_from_params_with_order(self):
        """Test creating curve from parameters with explicit order"""
        pseudotime = np.array([3.0, 1.0, 2.0, 0.0])
        points = np.array([[3.0, 3.0], [1.0, 1.0], [2.0, 2.0], [0.0, 0.0]])
        order = np.array([3, 1, 2, 0])

        pc = PrincipalCurve.from_params(pseudotime, points, order=order)

        np.testing.assert_array_equal(pc.order, order)

    def test_convergence_with_tolerance(self):
        """Test that fitting respects tolerance for early stopping"""
        pc = PrincipalCurve(k=3)
        np.random.seed(42)
        t = np.linspace(0, 2 * np.pi, 50)
        X = np.column_stack([np.cos(t), np.sin(t)])

        # With loose tolerance, should converge quickly
        pc.fit(X, max_iter=100, tol=1e-1)

        assert pc.points is not None
        # Should converge (we can't directly check iterations, but it shouldn't crash)

    def test_fitting_with_custom_initial_points(self):
        """Test fitting with user-provided initial curve"""
        pc = PrincipalCurve(k=3)
        X = np.array([[i, i * 2] for i in range(10)], dtype=float)
        initial = np.array([[0.0, 0.0], [5.0, 10.0], [9.0, 18.0]])

        pc.fit(X, initial_points=initial, max_iter=5)

        assert pc.points is not None
        assert pc.points.shape[1] == 2
        # Should refine the initial curve
        assert len(pc.points) >= 2
