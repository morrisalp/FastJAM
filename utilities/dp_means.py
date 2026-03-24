"""
Minimal pure-numpy DP-Means implementation.
Drop-in replacement for pdc_dp_means.DPMeans, which is broken on
modern Python/scikit-learn due to Cython ABI incompatibilities.

Algorithm: Lloyd-style DP-Means (Kulis & Jordan, 2012), matching the
behaviour of the original pdc-dp-means Cython implementation:
- delta is compared against **squared** Euclidean distance (not L2).
- At most one new cluster is spawned per iteration (the farthest point).
- Best run is selected by penalized inertia: sum_sq_dist + delta * n_clusters.
"""
import numpy as np


class DPMeans:
    def __init__(self, n_clusters=1, delta=1.0, n_init=10, max_iter=300, tol=1e-4, random_state=None):
        self.delta = delta
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        best_inertia = np.inf
        rng = np.random.RandomState(self.random_state)

        for _ in range(self.n_init):
            centers, labels = self._run(X, rng)
            inertia = np.sum((X - centers[labels]) ** 2) + self.delta * len(centers)
            if inertia < best_inertia:
                best_inertia = inertia
                self.cluster_centers_ = centers
                self.labels_ = labels

        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def _run(self, X, rng):
        n = len(X)
        centers = [X[rng.randint(n)].copy()]

        prev_labels = None
        for _ in range(self.max_iter):
            centers_arr = np.array(centers)

            # Squared distances from each point to each center: (n, k)
            diff = X[:, np.newaxis, :] - centers_arr[np.newaxis, :, :]  # (n, k, d)
            sq_dists = np.sum(diff ** 2, axis=2)  # (n, k)

            nearest = np.argmin(sq_dists, axis=1)          # (n,)
            min_sq_dists = sq_dists[np.arange(n), nearest]  # (n,)

            # Spawn one new cluster: the point farthest from its nearest center
            max_idx = np.argmax(min_sq_dists)
            if min_sq_dists[max_idx] > self.delta:
                centers.append(X[max_idx].copy())
                centers_arr = np.array(centers)
                # Re-assign all points with the new center included
                diff = X[:, np.newaxis, :] - centers_arr[np.newaxis, :, :]
                sq_dists = np.sum(diff ** 2, axis=2)
                nearest = np.argmin(sq_dists, axis=1)
                spawned = True
            else:
                spawned = False

            labels = nearest.astype(np.int32)

            # Update step: recompute centers as cluster means
            new_centers = np.array([
                X[labels == k].mean(axis=0) if np.any(labels == k) else centers_arr[k]
                for k in range(len(centers_arr))
            ])

            shift = np.max(np.sqrt(np.sum((new_centers - centers_arr) ** 2, axis=1)))
            centers = list(new_centers)

            if not spawned and shift < self.tol:
                break
            if not spawned and prev_labels is not None and np.array_equal(labels, prev_labels):
                break
            prev_labels = labels

        return np.array(centers), labels
