import numpy as np


def compute_pca(data):
    m = np.mean(data, axis=0)
    data_c = np.array([obs -m for obs in data])
    T = np.dot(data_c, data_c.T)
    [u, s, v] = np.linalg.svd(T)

    pcs = [np.dot(data_c.T, item) for item in u.T]

    pcs = np.array([d / np.linalg.norm(d) for d in pcs])
    return pcs, m, s, T, u


def compute_projections(I, pcs, m):
    projections = []
    for i in I:
        w = []
        for p in pcs:
            w.append(np.dot(i - m, p))
        projections.append(w)
    return projections


def reconstruct(w, X, m, dim=5):
    return np.dot(w[: dim], X[:dim, :]) + m


def normalize(samples, maxs=None):
    while not maxs:
        maxs = np.max(samples)
        print(maxs)
    return np.array([np.ravel(s) / maxs for s in samples])


if __name__ == '__main__':
    x = np.random.randn(10, 100)
    print(x)
    print(x[0, 0])
    print('++++++++++')
    print(normalize(x))
















