import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph

class SpectralClustering:
    def __init__(self, n_clusters, affinity='kneighbors', n_neighbors=10):
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.labels_ = None

    def _laplacian(self, A):
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.sum(A, axis=1)))
        return np.eye(A.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt

    def _k_means(self, X):
        kmeans = KMeans(n_clusters=self.n_clusters, n_init=10)
        kmeans.fit(X)
        return kmeans.labels_

    def _euclidean_distances(self, X):
        XX = np.sum(X**2, axis=1)[:, np.newaxis]
        distances = XX - 2 * np.dot(X, X.T) + XX.T
        np.maximum(distances, 0, out=distances)
        np.sqrt(distances, out=distances)
        return distances

    def _kneighbors_affinity_graph(self, X):
        n_samples = X.shape[0]
        distances = self._euclidean_distances(X)
        A = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            nearest_neighbors = np.argsort(distances[i])[:self.n_neighbors+1]
            A[i, nearest_neighbors] = 1
            A[nearest_neighbors, i] = 1
        return A

    def fit(self, X):
        if self.affinity == 'nearest_neighbors':
            A = kneighbors_graph(X, n_neighbors=self.n_neighbors, mode='distance', include_self=True).toarray()
        elif self.affinity == 'kneighbors':
            A = self._kneighbors_affinity_graph(X)
        else:
            raise ValueError("Unknown affinity type")

        L = self._laplacian(A)
        eig_val, eig_vec = np.linalg.eig(L)
        idx = np.argsort(eig_val)
        eig_vec = eig_vec[:, idx]
        X = eig_vec[:, :self.n_clusters].real
        rows_norm = np.linalg.norm(X, axis=1, ord=2)
        Y = (X.T / rows_norm).T
        self.labels_ = self._k_means(Y)
        return self.labels_