import numpy as np
from scipy.linalg import toeplitz

def chebdif(N,M):

    L = np.eye(N, dtype=bool)

    n1 = N // 2
    n2 = (N+1) // 2

    k = np.arange(N)
    th = k*np.pi/(N-1)

    x = np.sin(((np.pi*np.arange(N-1,-N,-2)) / (2*(N-1))))

    T = np.tile(th[:, None]/2, (1, N))

    DX = 2*np.sin(np.transpose(T) + T) * np.sin(np.transpose(T) - T)
    DX = np.vstack((DX[:n1, :], -np.flipud(np.fliplr(DX[:n2, :]))))

    DX[L] = 1

    C = toeplitz((-1.0)**k)

    C[0,:] *= 2
    C[-1,:] *= 2
    C[:,0] /= 2
    C[:,-1] /= 2

    Z = 1/DX
    Z[L] = 0

    D = np.eye(N)
    DM = np.zeros((N, N, M))

    for ell in range(1, M + 1):
        D = ell*Z*(C*np.tile(np.diag(D)[:, None], (1, N)) - D)
        D[L] = -np.sum(D, axis=1)
        DM[:,:,ell - 1] = D

    return x, DM

