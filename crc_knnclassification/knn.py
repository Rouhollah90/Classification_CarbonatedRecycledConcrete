"""Weighted K-nearest neighbours classifiers and distance utilities."""

from dataclasses import dataclass

import numpy as np


def lp_distance_matrix(XA, XB=None, p=1.5):
    XA = np.asarray(XA, dtype=float)
    XB = XA if XB is None else np.asarray(XB, dtype=float)
    diff = np.abs(XA[:, None, :] - XB[None, :, :]) ** p
    return np.sum(diff, axis=2) ** (1.0 / p)


def silverman_bandwidth(X):
    X = np.asarray(X, dtype=float)
    n = X.shape[0]

    if n < 2:
        raise ValueError("At least two samples are required to compute bandwidth.")

    s_hat = np.mean(np.std(X, axis=0, ddof=1))
    return max(1.6 * s_hat * (n ** (-1.0 / 5.0)), 1e-12)


def compute_knn_weights(distances, weighting="gaussian", phi=None, epsilon=1e-12):
    distances = np.asarray(distances, dtype=float)

    if weighting == "uniform":
        return np.ones_like(distances)
    if weighting == "inverse-distance":
        return 1.0 / (distances + epsilon)
    if weighting == "gaussian":
        if phi is None:
            raise ValueError("phi is required for Gaussian weighting.")
        return np.exp(-(distances ** 2) / (2.0 * (phi ** 2)))

    raise ValueError("weighting must be 'uniform', 'inverse-distance', or 'gaussian'.")


def gaussian_weights(distances, phi):
    return compute_knn_weights(distances, weighting="gaussian", phi=phi)


@dataclass
class TunableWeightedKNN:
    k: int = 5
    p: float = 1.5
    weighting: str = "gaussian"
    phi: float = None
    epsilon: float = 1e-12

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        self._validate_training_data()
        self.classes_ = np.unique(self.y_train)
        self.phi_ = silverman_bandwidth(self.X_train) if self.phi is None else float(self.phi)
        return self

    def _validate_training_data(self):
        if self.X_train.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if len(self.X_train) != len(self.y_train):
            raise ValueError("X and y must have the same number of samples.")
        if self.k < 1:
            raise ValueError("k must be at least 1.")
        if self.k > len(self.X_train):
            raise ValueError("k cannot be larger than the number of training samples.")

    def _predict_one(self, x):
        x = np.asarray(x, dtype=float).reshape(1, -1)
        distances = lp_distance_matrix(x, self.X_train, p=self.p).ravel()
        nn_idx = np.argsort(distances)[: self.k]
        nn_distances = distances[nn_idx]
        nn_labels = self.y_train[nn_idx]
        nn_weights = compute_knn_weights(
            nn_distances,
            weighting=self.weighting,
            phi=self.phi_,
            epsilon=self.epsilon,
        )
        class_scores = {c: np.sum(nn_weights[nn_labels == c]) for c in self.classes_}
        return max(class_scores, key=class_scores.get)

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.asarray([self._predict_one(x) for x in X])


@dataclass
class WeightedKNN:
    k: int = 5
    p: float = 1.5
    phi: float = None

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        self.y_train = np.asarray(y)
        self._validate_training_data()
        self.classes_ = np.unique(self.y_train)
        self.phi_ = silverman_bandwidth(self.X_train) if self.phi is None else float(self.phi)
        return self

    def _validate_training_data(self):
        if self.X_train.ndim != 2:
            raise ValueError("X must be a 2D array.")
        if len(self.X_train) != len(self.y_train):
            raise ValueError("X and y must have the same number of samples.")
        if self.k < 1:
            raise ValueError("k must be at least 1.")
        if self.k > len(self.X_train):
            raise ValueError("k cannot be larger than the number of training samples.")

    def _predict_one_with_details(self, x):
        x = np.asarray(x, dtype=float).reshape(1, -1)
        distances = lp_distance_matrix(x, self.X_train, p=self.p).ravel()
        nn_idx = np.argsort(distances)[: self.k]
        nn_distances = distances[nn_idx]
        nn_labels = self.y_train[nn_idx]
        nn_weights = gaussian_weights(nn_distances, self.phi_)
        class_scores = {c: np.sum(nn_weights[nn_labels == c]) for c in self.classes_}

        total_score = np.sum(list(class_scores.values()))
        if total_score <= 0:
            class_probs = {c: 0.0 for c in self.classes_}
        else:
            class_probs = {c: class_scores[c] / total_score for c in self.classes_}

        return (
            max(class_scores, key=class_scores.get),
            class_probs,
            float(np.mean(nn_distances)),
            nn_idx,
        )

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.asarray([self._predict_one_with_details(x)[0] for x in X])

    def predict_confidence(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.asarray(
            [1.0 / (1.0 + self._predict_one_with_details(x)[2]) for x in X]
        )

    def predict_posterior_confidence(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return np.asarray(
            [max(self._predict_one_with_details(x)[1].values()) for x in X]
        )
