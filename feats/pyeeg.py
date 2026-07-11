import numpy as np

def embed_seq(X, Tau, D):
    X = np.asarray(X)
    shape = (X.size - Tau * (D - 1), D)
    strides = (X.itemsize, Tau * X.itemsize)
    return np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)

def pfd(X, D=None):
    X = np.asarray(X, dtype=np.float64)
    D = np.diff(X) if D is None else np.asarray(D)
    N_delta = np.sum(D[1:] * D[:-1] < 0)
    n = len(X)
    return np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * N_delta)))

def ap_entropy(X, M, R):
    X = np.asarray(X, dtype=np.float64)
    N = len(X)
    if N <= M + 1:
        return np.nan
    Em = embed_seq(X, 1, M)
    A = np.tile(Em, (len(Em), 1, 1))
    B = np.transpose(A, [1, 0, 2])
    D = np.abs(A - B)
    InRange = np.max(D, axis=2) <= R
    Cm = InRange.mean(axis=0)
    Dp = np.abs(np.tile(X[M:], (N - M, 1)) - np.tile(X[M:], (N - M, 1)).T)
    Cmp = np.logical_and(Dp <= R, InRange[:-1, :-1]).mean(axis=0)
    with np.errstate(divide="ignore"):
        Phi_m, Phi_mp = np.sum(np.log(Cm)), np.sum(np.log(Cmp))
    return (Phi_m - Phi_mp) / (N - M)
