"""Causal inference estimators."""

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import NearestNeighbors

from .base import BaseCausalEstimator


class PropensityScoreMatching(BaseCausalEstimator):
    """Propensity Score Matching estimator for causal inference.

    Estimates treatment effects by matching treated units to control units
    with similar propensity scores.

    Parameters
    ----------
    n_neighbors : int, default=1
        Number of neighbors to use for matching.
    caliper : float or None, default=None
        Maximum distance for a match. If None, no caliper is used.
    propensity_model : estimator or None, default=None
        Model to estimate propensity scores. Must have fit and predict_proba
        methods. If None, uses LogisticRegression.
    random_state : int or None, default=None
        Random state for reproducibility.

    Attributes
    ----------
    ate_ : float
        Estimated average treatment effect after fitting.
    att_ : float
        Estimated average treatment effect on the treated after fitting.
    propensity_scores_ : ndarray of shape (n_samples,)
        Estimated propensity scores.
    matched_indices_ : ndarray
        Indices of matched control units for each treated unit.

    Examples
    --------
    >>> from sk_causal import PropensityScoreMatching
    >>> import numpy as np
    >>> X = np.random.randn(100, 3)
    >>> treatment = (X[:, 0] > 0).astype(int)
    >>> y = treatment * 2 + X[:, 0] + np.random.randn(100) * 0.5
    >>> psm = PropensityScoreMatching(n_neighbors=1)
    >>> psm.fit(X, treatment, y)
    >>> print(f"ATE: {psm.estimate_ate():.2f}")
    """

    def __init__(
        self, n_neighbors=1, caliper=None, propensity_model=None, random_state=None
    ):
        super().__init__(random_state=random_state)
        self.n_neighbors = n_neighbors
        self.caliper = caliper
        self.propensity_model = propensity_model

    def fit(self, X, treatment, y):
        """Fit the propensity score matching estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariate matrix.
        treatment : array-like of shape (n_samples,)
            Binary treatment indicator (0 or 1).
        y : array-like of shape (n_samples,)
            Observed outcomes.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, treatment, y = self._validate_inputs(X, treatment, y)

        self.X_ = X
        self.treatment_ = treatment
        self.y_ = y

        # Fit propensity score model
        if self.propensity_model is None:
            self._propensity_model = LogisticRegression(
                random_state=self.random_state, max_iter=1000
            )
        else:
            self._propensity_model = self.propensity_model

        self._propensity_model.fit(X, treatment)
        self.propensity_scores_ = self._propensity_model.predict_proba(X)[:, 1]

        # Perform matching
        treated_mask = treatment == 1
        control_mask = treatment == 0

        treated_ps = self.propensity_scores_[treated_mask].reshape(-1, 1)
        control_ps = self.propensity_scores_[control_mask].reshape(-1, 1)
        control_indices = np.where(control_mask)[0]

        nn = NearestNeighbors(n_neighbors=self.n_neighbors, metric="euclidean")
        nn.fit(control_ps)
        distances, indices = nn.kneighbors(treated_ps)

        self.matched_indices_ = control_indices[indices]
        self.matched_distances_ = distances

        # Apply caliper if specified
        if self.caliper is not None:
            self.valid_matches_ = distances.max(axis=1) <= self.caliper
        else:
            self.valid_matches_ = np.ones(len(treated_ps), dtype=bool)

        # Compute treatment effects
        treated_outcomes = y[treated_mask][self.valid_matches_]
        matched_control_outcomes = y[self.matched_indices_[self.valid_matches_]].mean(
            axis=1
        )

        self.att_ = np.mean(treated_outcomes - matched_control_outcomes)
        self.ate_ = self.att_  # PSM primarily estimates ATT

        return self

    def estimate_ate(self):
        """Estimate the Average Treatment Effect (ATE).

        Note: PSM primarily estimates ATT. This returns the ATT estimate.

        Returns
        -------
        ate : float
            Estimated average treatment effect.
        """
        if not hasattr(self, "ate_"):
            raise ValueError("Estimator has not been fitted. Call fit() first.")
        return self.ate_

    def estimate_att(self):
        """Estimate the Average Treatment Effect on the Treated (ATT).

        Returns
        -------
        att : float
            Estimated average treatment effect on the treated.
        """
        if not hasattr(self, "att_"):
            raise ValueError("Estimator has not been fitted. Call fit() first.")
        return self.att_


class InversePropensityWeighting(BaseCausalEstimator):
    """Inverse Propensity Weighting (IPW) estimator for causal inference.

    Estimates treatment effects using inverse probability of treatment
    weighting to create a pseudo-population where treatment is independent
    of covariates.

    Parameters
    ----------
    propensity_model : estimator or None, default=None
        Model to estimate propensity scores. Must have fit and predict_proba
        methods. If None, uses LogisticRegression.
    clip : tuple of (float, float) or None, default=(0.01, 0.99)
        Clip propensity scores to avoid extreme weights. If None, no clipping.
    normalize_weights : bool, default=True
        Whether to normalize weights to sum to 1 within each treatment group.
    random_state : int or None, default=None
        Random state for reproducibility.

    Attributes
    ----------
    ate_ : float
        Estimated average treatment effect after fitting.
    propensity_scores_ : ndarray of shape (n_samples,)
        Estimated propensity scores.
    weights_ : ndarray of shape (n_samples,)
        IPW weights for each observation.

    Examples
    --------
    >>> from sk_causal import InversePropensityWeighting
    >>> import numpy as np
    >>> X = np.random.randn(100, 3)
    >>> treatment = (X[:, 0] > 0).astype(int)
    >>> y = treatment * 2 + X[:, 0] + np.random.randn(100) * 0.5
    >>> ipw = InversePropensityWeighting()
    >>> ipw.fit(X, treatment, y)
    >>> print(f"ATE: {ipw.estimate_ate():.2f}")
    """

    def __init__(
        self,
        propensity_model=None,
        clip=(0.01, 0.99),
        normalize_weights=True,
        random_state=None,
    ):
        super().__init__(random_state=random_state)
        self.propensity_model = propensity_model
        self.clip = clip
        self.normalize_weights = normalize_weights

    def fit(self, X, treatment, y):
        """Fit the IPW estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariate matrix.
        treatment : array-like of shape (n_samples,)
            Binary treatment indicator (0 or 1).
        y : array-like of shape (n_samples,)
            Observed outcomes.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, treatment, y = self._validate_inputs(X, treatment, y)

        self.X_ = X
        self.treatment_ = treatment
        self.y_ = y

        # Fit propensity score model
        if self.propensity_model is None:
            self._propensity_model = LogisticRegression(
                random_state=self.random_state, max_iter=1000
            )
        else:
            self._propensity_model = self.propensity_model

        self._propensity_model.fit(X, treatment)
        self.propensity_scores_ = self._propensity_model.predict_proba(X)[:, 1]

        # Clip propensity scores
        ps = self.propensity_scores_.copy()
        if self.clip is not None:
            ps = np.clip(ps, self.clip[0], self.clip[1])

        # Compute IPW weights
        self.weights_ = np.where(treatment == 1, 1 / ps, 1 / (1 - ps))

        # Compute ATE using IPW
        treated_mask = treatment == 1
        control_mask = treatment == 0

        if self.normalize_weights:
            # Normalized (Hajek) estimator
            w_treated = self.weights_[treated_mask]
            w_control = self.weights_[control_mask]

            mean_treated = np.sum(w_treated * y[treated_mask]) / np.sum(w_treated)
            mean_control = np.sum(w_control * y[control_mask]) / np.sum(w_control)
        else:
            # Horvitz-Thompson estimator
            n = len(y)
            mean_treated = np.sum(self.weights_[treated_mask] * y[treated_mask]) / n
            mean_control = np.sum(self.weights_[control_mask] * y[control_mask]) / n

        self.ate_ = mean_treated - mean_control

        return self

    def estimate_ate(self):
        """Estimate the Average Treatment Effect (ATE).

        Returns
        -------
        ate : float
            Estimated average treatment effect.
        """
        if not hasattr(self, "ate_"):
            raise ValueError("Estimator has not been fitted. Call fit() first.")
        return self.ate_


class DoublyRobust(BaseCausalEstimator):
    """Doubly Robust (AIPW) estimator for causal inference.

    Combines inverse propensity weighting with outcome regression to create
    an estimator that is consistent if either the propensity score model
    or the outcome model is correctly specified.

    Parameters
    ----------
    propensity_model : estimator or None, default=None
        Model to estimate propensity scores. Must have fit and predict_proba
        methods. If None, uses LogisticRegression.
    outcome_model : estimator or None, default=None
        Model to predict outcomes. Must have fit and predict methods.
        If None, uses Ridge regression.
    clip : tuple of (float, float) or None, default=(0.01, 0.99)
        Clip propensity scores to avoid extreme weights. If None, no clipping.
    random_state : int or None, default=None
        Random state for reproducibility.

    Attributes
    ----------
    ate_ : float
        Estimated average treatment effect after fitting.
    propensity_scores_ : ndarray of shape (n_samples,)
        Estimated propensity scores.
    mu0_ : ndarray of shape (n_samples,)
        Predicted outcomes under control.
    mu1_ : ndarray of shape (n_samples,)
        Predicted outcomes under treatment.

    Examples
    --------
    >>> from sk_causal import DoublyRobust
    >>> import numpy as np
    >>> X = np.random.randn(100, 3)
    >>> treatment = (X[:, 0] > 0).astype(int)
    >>> y = treatment * 2 + X[:, 0] + np.random.randn(100) * 0.5
    >>> dr = DoublyRobust()
    >>> dr.fit(X, treatment, y)
    >>> print(f"ATE: {dr.estimate_ate():.2f}")
    """

    def __init__(
        self,
        propensity_model=None,
        outcome_model=None,
        clip=(0.01, 0.99),
        random_state=None,
    ):
        super().__init__(random_state=random_state)
        self.propensity_model = propensity_model
        self.outcome_model = outcome_model
        self.clip = clip

    def fit(self, X, treatment, y):
        """Fit the doubly robust estimator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariate matrix.
        treatment : array-like of shape (n_samples,)
            Binary treatment indicator (0 or 1).
        y : array-like of shape (n_samples,)
            Observed outcomes.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, treatment, y = self._validate_inputs(X, treatment, y)

        self.X_ = X
        self.treatment_ = treatment
        self.y_ = y

        treated_mask = treatment == 1
        control_mask = treatment == 0

        # Fit propensity score model
        if self.propensity_model is None:
            self._propensity_model = LogisticRegression(
                random_state=self.random_state, max_iter=1000
            )
        else:
            self._propensity_model = self.propensity_model

        self._propensity_model.fit(X, treatment)
        self.propensity_scores_ = self._propensity_model.predict_proba(X)[:, 1]

        # Clip propensity scores
        ps = self.propensity_scores_.copy()
        if self.clip is not None:
            ps = np.clip(ps, self.clip[0], self.clip[1])

        # Fit outcome models
        if self.outcome_model is None:
            self._outcome_model_0 = Ridge(random_state=self.random_state)
            self._outcome_model_1 = Ridge(random_state=self.random_state)
        else:
            from sklearn.base import clone

            self._outcome_model_0 = clone(self.outcome_model)
            self._outcome_model_1 = clone(self.outcome_model)

        self._outcome_model_0.fit(X[control_mask], y[control_mask])
        self._outcome_model_1.fit(X[treated_mask], y[treated_mask])

        # Predict potential outcomes
        self.mu0_ = self._outcome_model_0.predict(X)
        self.mu1_ = self._outcome_model_1.predict(X)

        # Compute AIPW estimator
        # E[Y(1)] = E[mu1(X) + T(Y - mu1(X))/e(X)]
        # E[Y(0)] = E[mu0(X) + (1-T)(Y - mu0(X))/(1-e(X))]
        n = len(y)

        aipw_treated = self.mu1_ + treatment * (y - self.mu1_) / ps
        aipw_control = self.mu0_ + (1 - treatment) * (y - self.mu0_) / (1 - ps)

        self.ate_ = np.mean(aipw_treated) - np.mean(aipw_control)

        return self

    def estimate_ate(self):
        """Estimate the Average Treatment Effect (ATE).

        Returns
        -------
        ate : float
            Estimated average treatment effect.
        """
        if not hasattr(self, "ate_"):
            raise ValueError("Estimator has not been fitted. Call fit() first.")
        return self.ate_

    def estimate_cate(self, X):
        """Estimate Conditional Average Treatment Effects (CATE).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariate matrix for which to estimate CATE.

        Returns
        -------
        cate : ndarray of shape (n_samples,)
            Estimated conditional average treatment effects.
        """
        if not hasattr(self, "mu0_"):
            raise ValueError("Estimator has not been fitted. Call fit() first.")

        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        mu0 = self._outcome_model_0.predict(X)
        mu1 = self._outcome_model_1.predict(X)

        return mu1 - mu0
