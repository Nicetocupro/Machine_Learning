import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans

DEBUG = True


def debug(*args, **kwargs):
    global DEBUG
    if DEBUG:
        print(*args, **kwargs)


class GMM:
    def __init__(self, K, max_iter=100, tol=1e-6):
        self.K = K
        self.max_iter = max_iter
        self.tol = tol

    def _phi(self, Y, mu_k, cov_k):
        norm = multivariate_normal(mean=mu_k, cov=cov_k)
        return norm.pdf(Y)

    def _get_expectation(self, Y, mu, cov, alpha):
        N = Y.shape[0]
        K = alpha.shape[0]

        assert N > 1, "There must be more than one sample!"
        assert K > 1, "There must be more than one gaussian model!"

        gamma = np.zeros((N, K))
        prob = np.zeros((N, K))

        for k in range(K):
            prob[:, k] = self._phi(Y, mu[k], cov[k])

        for k in range(K):
            gamma[:, k] = alpha[k] * prob[:, k]

        gamma /= gamma.sum(axis=1, keepdims=True)
        return gamma

    def _maximize(self, Y, gamma):
        N, D = Y.shape
        K = gamma.shape[1]

        mu = np.zeros((K, D))
        cov = []
        alpha = np.zeros(K)

        for k in range(K):
            Nk = np.sum(gamma[:, k])
            mu[k, :] = np.sum(gamma[:, k, np.newaxis] * Y, axis=0) / Nk
            cov_k = ((Y - mu[k]).T @ (gamma[:, k, np.newaxis] * (Y - mu[k]))) / Nk
            cov.append(cov_k)
            alpha[k] = Nk / N

        cov = np.array(cov)
        return mu, cov, alpha

    def _scale_data(self, Y):
        for i in range(Y.shape[1]):
            max_ = Y[:, i].max()
            min_ = Y[:, i].min()
            Y[:, i] = (Y[:, i] - min_) / (max_ - min_)
        debug("Data scaled.")
        return Y

    def _init_params(self, Y):
        N, D = Y.shape
        kmeans = KMeans(n_clusters=self.K, init='k-means++', n_init=10, random_state=42)
        kmeans.fit(Y)
        mu = kmeans.cluster_centers_
        cov = np.array([np.eye(D) * 0.1 for _ in range(self.K)])
        alpha = np.array([1.0 / self.K] * self.K)
        debug("Parameters initialized.")
        debug("mu:", mu, "cov:", cov, "alpha:", alpha, sep="\n")
        return mu, cov, alpha

    def fit(self, Y):
        Y = self._scale_data(Y)
        mu, cov, alpha = self._init_params(Y)
        prev_log_likelihood = -np.inf

        for i in range(self.max_iter):
            gamma = self._get_expectation(Y, mu, cov, alpha)
            mu_new, cov_new, alpha_new = self._maximize(Y, gamma)

            # Calculate log likelihood
            log_likelihood = 0
            for k in range(self.K):
                log_likelihood += np.sum(gamma[:, k] * (np.log(alpha_new[k]) + self._phi(Y, mu_new[k], cov_new[k])))

            if np.abs(log_likelihood - prev_log_likelihood) < self.tol:
                break

            mu, cov, alpha = mu_new, cov_new, alpha_new
            prev_log_likelihood = log_likelihood

        debug("{sep} Result {sep}".format(sep="-" * 20))
        debug("mu:", mu, "cov:", cov, "alpha:", alpha, sep="\n")
        return mu, cov, alpha
