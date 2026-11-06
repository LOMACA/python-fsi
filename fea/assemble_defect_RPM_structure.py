"""Grounded monoatomic PnC with edge defect, finite N-mass chain.

Lin et al. (2026), Eq. (1): mass 0 is the defect (mDef, kgDef, distinct from
the periodic masses 1..N-1), nearest-neighbour coupling k throughout, uniform
damping c on every mass. All quantities nondimensional (outer units):
m = m_dim/(rho*h^3), c = c_dim/(rho*Ub*h^2), k = k_dim/(rho*Ub^2*h).
"""

import numpy as np


def assemble_defect_chain(m, k, kg, c, mDef, kgDef, N):

    mVec = m*np.ones(N)
    mVec[0] = mDef
    M = np.diag(mVec)

    C = c*np.eye(N)

    K = np.zeros((N, N))
    K[0, 0] = k + kgDef

    for i in range(1, N-1):
        K[i, i] = 2*k + kg

    K[N-1, N-1] = k + kg

    for i in range(N-1):
        K[i, i+1] -= k
        K[i+1, i] -= k

    forcedDoF = 0   # defect mass = fluid interface

    return M, C, K, forcedDoF