"""Base classes for causal inference estimators."""

from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator


class BaseCausalEstimator(BaseEstimator, ABC):
    """Base class for causal inference estimators.

    All causal estimators should inherit from this class and implement
    the `fit` and `estimate_ate` methods.

    Parameters
    ----------
    random_state : int or None, default=None
        Random state for reproducibility.
    """

    def __init__(self, random_state=None):
        self.random_state = random_state

    @abstractmethod
    def fit(self, X, treatment, y):
        """Fit the causal estimator.

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
        pass

    @abstractmethod
    def estimate_ate(self):
        """Estimate the Average Treatment Effect (ATE).

        Returns
        -------
        ate : float
            Estimated average treatment effect.
        """
        pass

    def estimate_att(self):
        """Estimate the Average Treatment Effect on the Treated (ATT).

        Returns
        -------
        att : float
            Estimated average treatment effect on the treated.

        Raises
        ------
        NotImplementedError
            If the estimator does not support ATT estimation.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support ATT estimation."
        )

    def _validate_inputs(self, X, treatment, y):
        """Validate input arrays.

        Parameters
        ----------
        X : array-like
            Covariate matrix.
        treatment : array-like
            Treatment indicator.
        y : array-like
            Outcomes.

        Returns
        -------
        X : ndarray
            Validated covariate matrix.
        treatment : ndarray
            Validated treatment array.
        y : ndarray
            Validated outcome array.
        """
        X = np.asarray(X)
        treatment = np.asarray(treatment).ravel()
        y = np.asarray(y).ravel()

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n_samples = X.shape[0]
        if treatment.shape[0] != n_samples:
            raise ValueError(
                f"treatment has {treatment.shape[0]} samples, "
                f"but X has {n_samples} samples."
            )
        if y.shape[0] != n_samples:
            raise ValueError(
                f"y has {y.shape[0]} samples, but X has {n_samples} samples."
            )

        unique_treatments = np.unique(treatment)
        if not np.array_equal(unique_treatments, [0, 1]):
            if set(unique_treatments).issubset({0, 1}):
                pass  # OK if only one treatment level present (edge case)
            else:
                raise ValueError(
                    f"treatment must be binary (0 or 1), got values: {unique_treatments}"
                )

        return X, treatment, y
