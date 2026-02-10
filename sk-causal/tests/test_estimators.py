"""Tests for causal inference estimators."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression, LinearRegression

from sk_causal import (
    PropensityScoreMatching,
    InversePropensityWeighting,
    DoublyRobust,
)


def generate_synthetic_data(n_samples=500, true_ate=2.0, seed=42):
    """Generate synthetic data for testing causal estimators.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    true_ate : float
        True average treatment effect.
    seed : int
        Random seed.

    Returns
    -------
    X : ndarray
        Covariates.
    treatment : ndarray
        Treatment indicator.
    y : ndarray
        Outcomes.
    """
    rng = np.random.RandomState(seed)

    # Generate covariates
    X = rng.randn(n_samples, 3)

    # Treatment assignment depends on X
    propensity = 1 / (1 + np.exp(-(X[:, 0] + 0.5 * X[:, 1])))
    treatment = (rng.rand(n_samples) < propensity).astype(int)

    # Outcome depends on X and treatment
    y0 = X[:, 0] + X[:, 1] + rng.randn(n_samples) * 0.5
    y1 = y0 + true_ate
    y = np.where(treatment == 1, y1, y0)

    return X, treatment, y


class TestPropensityScoreMatching:
    """Tests for PropensityScoreMatching estimator."""

    def test_fit_and_estimate(self):
        """Test that PSM can fit and estimate ATE."""
        X, treatment, y = generate_synthetic_data(n_samples=500, true_ate=2.0)
        psm = PropensityScoreMatching(n_neighbors=3, random_state=42)
        psm.fit(X, treatment, y)

        ate = psm.estimate_ate()
        assert isinstance(ate, float)
        # Check that estimate is reasonably close to true ATE
        assert abs(ate - 2.0) < 1.0

    def test_estimate_att(self):
        """Test ATT estimation."""
        X, treatment, y = generate_synthetic_data()
        psm = PropensityScoreMatching(random_state=42)
        psm.fit(X, treatment, y)

        att = psm.estimate_att()
        assert isinstance(att, float)

    def test_propensity_scores(self):
        """Test that propensity scores are computed."""
        X, treatment, y = generate_synthetic_data()
        psm = PropensityScoreMatching(random_state=42)
        psm.fit(X, treatment, y)

        assert hasattr(psm, "propensity_scores_")
        assert len(psm.propensity_scores_) == len(X)
        assert np.all(psm.propensity_scores_ >= 0)
        assert np.all(psm.propensity_scores_ <= 1)

    def test_caliper(self):
        """Test matching with caliper."""
        X, treatment, y = generate_synthetic_data()
        psm = PropensityScoreMatching(n_neighbors=1, caliper=0.1, random_state=42)
        psm.fit(X, treatment, y)

        assert hasattr(psm, "valid_matches_")
        # Some matches should be invalid with tight caliper
        assert psm.valid_matches_.sum() < len(psm.valid_matches_)

    def test_custom_propensity_model(self):
        """Test with custom propensity model."""
        X, treatment, y = generate_synthetic_data()
        custom_model = LogisticRegression(C=0.5, random_state=42)
        psm = PropensityScoreMatching(propensity_model=custom_model, random_state=42)
        psm.fit(X, treatment, y)

        ate = psm.estimate_ate()
        assert isinstance(ate, float)

    def test_not_fitted_error(self):
        """Test error when estimating without fitting."""
        psm = PropensityScoreMatching()
        with pytest.raises(ValueError, match="not been fitted"):
            psm.estimate_ate()


class TestInversePropensityWeighting:
    """Tests for InversePropensityWeighting estimator."""

    def test_fit_and_estimate(self):
        """Test that IPW can fit and estimate ATE."""
        X, treatment, y = generate_synthetic_data(n_samples=500, true_ate=2.0)
        ipw = InversePropensityWeighting(random_state=42)
        ipw.fit(X, treatment, y)

        ate = ipw.estimate_ate()
        assert isinstance(ate, float)
        # Check that estimate is reasonably close to true ATE
        assert abs(ate - 2.0) < 1.0

    def test_propensity_scores(self):
        """Test that propensity scores are computed."""
        X, treatment, y = generate_synthetic_data()
        ipw = InversePropensityWeighting(random_state=42)
        ipw.fit(X, treatment, y)

        assert hasattr(ipw, "propensity_scores_")
        assert len(ipw.propensity_scores_) == len(X)

    def test_weights(self):
        """Test that IPW weights are computed."""
        X, treatment, y = generate_synthetic_data()
        ipw = InversePropensityWeighting(random_state=42)
        ipw.fit(X, treatment, y)

        assert hasattr(ipw, "weights_")
        assert len(ipw.weights_) == len(X)
        assert np.all(ipw.weights_ > 0)

    def test_clipping(self):
        """Test propensity score clipping."""
        X, treatment, y = generate_synthetic_data()
        ipw = InversePropensityWeighting(clip=(0.1, 0.9), random_state=42)
        ipw.fit(X, treatment, y)

        # Weights should be bounded due to clipping
        max_weight = 1 / 0.1
        assert np.all(ipw.weights_ <= max_weight + 1e-10)

    def test_no_normalization(self):
        """Test without weight normalization."""
        X, treatment, y = generate_synthetic_data()
        ipw = InversePropensityWeighting(normalize_weights=False, random_state=42)
        ipw.fit(X, treatment, y)

        ate = ipw.estimate_ate()
        assert isinstance(ate, float)

    def test_not_fitted_error(self):
        """Test error when estimating without fitting."""
        ipw = InversePropensityWeighting()
        with pytest.raises(ValueError, match="not been fitted"):
            ipw.estimate_ate()


class TestDoublyRobust:
    """Tests for DoublyRobust estimator."""

    def test_fit_and_estimate(self):
        """Test that DR can fit and estimate ATE."""
        X, treatment, y = generate_synthetic_data(n_samples=500, true_ate=2.0)
        dr = DoublyRobust(random_state=42)
        dr.fit(X, treatment, y)

        ate = dr.estimate_ate()
        assert isinstance(ate, float)
        # DR should give good estimates
        assert abs(ate - 2.0) < 0.5

    def test_propensity_scores(self):
        """Test that propensity scores are computed."""
        X, treatment, y = generate_synthetic_data()
        dr = DoublyRobust(random_state=42)
        dr.fit(X, treatment, y)

        assert hasattr(dr, "propensity_scores_")
        assert len(dr.propensity_scores_) == len(X)

    def test_potential_outcomes(self):
        """Test that potential outcomes are predicted."""
        X, treatment, y = generate_synthetic_data()
        dr = DoublyRobust(random_state=42)
        dr.fit(X, treatment, y)

        assert hasattr(dr, "mu0_")
        assert hasattr(dr, "mu1_")
        assert len(dr.mu0_) == len(X)
        assert len(dr.mu1_) == len(X)

    def test_cate_estimation(self):
        """Test CATE estimation."""
        X, treatment, y = generate_synthetic_data()
        dr = DoublyRobust(random_state=42)
        dr.fit(X, treatment, y)

        X_test = np.random.randn(10, 3)
        cate = dr.estimate_cate(X_test)

        assert len(cate) == 10
        assert isinstance(cate, np.ndarray)

    def test_custom_models(self):
        """Test with custom propensity and outcome models."""
        X, treatment, y = generate_synthetic_data()
        custom_ps = LogisticRegression(C=0.5, random_state=42)
        custom_outcome = LinearRegression()

        dr = DoublyRobust(
            propensity_model=custom_ps, outcome_model=custom_outcome, random_state=42
        )
        dr.fit(X, treatment, y)

        ate = dr.estimate_ate()
        assert isinstance(ate, float)

    def test_not_fitted_error(self):
        """Test error when estimating without fitting."""
        dr = DoublyRobust()
        with pytest.raises(ValueError, match="not been fitted"):
            dr.estimate_ate()


class TestInputValidation:
    """Tests for input validation across estimators."""

    @pytest.mark.parametrize(
        "EstimatorClass",
        [PropensityScoreMatching, InversePropensityWeighting, DoublyRobust],
    )
    def test_mismatched_lengths(self, EstimatorClass):
        """Test error on mismatched input lengths."""
        X = np.random.randn(100, 3)
        treatment = np.random.randint(0, 2, 50)  # Wrong length
        y = np.random.randn(100)

        estimator = EstimatorClass(random_state=42)
        with pytest.raises(ValueError, match="samples"):
            estimator.fit(X, treatment, y)

    @pytest.mark.parametrize(
        "EstimatorClass",
        [PropensityScoreMatching, InversePropensityWeighting, DoublyRobust],
    )
    def test_non_binary_treatment(self, EstimatorClass):
        """Test error on non-binary treatment."""
        X = np.random.randn(100, 3)
        treatment = np.random.randint(0, 3, 100)  # Non-binary
        y = np.random.randn(100)

        estimator = EstimatorClass(random_state=42)
        with pytest.raises(ValueError, match="binary"):
            estimator.fit(X, treatment, y)

    @pytest.mark.parametrize(
        "EstimatorClass",
        [PropensityScoreMatching, InversePropensityWeighting, DoublyRobust],
    )
    def test_1d_covariates(self, EstimatorClass):
        """Test that 1D covariates are reshaped."""
        X = np.random.randn(100)
        treatment = np.random.randint(0, 2, 100)
        y = np.random.randn(100)

        estimator = EstimatorClass(random_state=42)
        estimator.fit(X, treatment, y)

        ate = estimator.estimate_ate()
        assert isinstance(ate, float)


class TestSklearnCompatibility:
    """Tests for scikit-learn API compatibility."""

    @pytest.mark.parametrize(
        "EstimatorClass",
        [PropensityScoreMatching, InversePropensityWeighting, DoublyRobust],
    )
    def test_get_params(self, EstimatorClass):
        """Test get_params method."""
        estimator = EstimatorClass(random_state=42)
        params = estimator.get_params()

        assert isinstance(params, dict)
        assert "random_state" in params
        assert params["random_state"] == 42

    @pytest.mark.parametrize(
        "EstimatorClass",
        [PropensityScoreMatching, InversePropensityWeighting, DoublyRobust],
    )
    def test_set_params(self, EstimatorClass):
        """Test set_params method."""
        estimator = EstimatorClass(random_state=42)
        estimator.set_params(random_state=123)

        assert estimator.random_state == 123

    @pytest.mark.parametrize(
        "EstimatorClass",
        [PropensityScoreMatching, InversePropensityWeighting, DoublyRobust],
    )
    def test_repr(self, EstimatorClass):
        """Test string representation."""
        estimator = EstimatorClass(random_state=42)
        repr_str = repr(estimator)

        assert EstimatorClass.__name__ in repr_str
