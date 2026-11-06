"""Application of coiling cycles to the MM rod, following procedure by Harris et al. (2026)"""

import numpy as np


def coilRod(M, C, K, freeDoF, junctNodes, omega, nCycles):

    nActive = nCycles + 1

    activeIdx = np.arange(0, 2*nCycles + 1,2)

    assert np.max(activeIdx) < len(junctNodes), "Not enough unit cells/junctions for number of coiling cycles."

    activeJunctions = junctNodes[activeIdx]

    nDoF = np.size(M, axis=0)
    F = np.zeros(nDoF)
    F[activeJunctions] = 1/nActive

    Z = K - omega**2*M + 1j*omega*C

    u = np.zeros(nDoF, dtype=complex)
    u[freeDoF] = np.linalg.solve(Z[np.ix_(freeDoF, freeDoF)], F[freeDoF])

    etaJ = u[activeJunctions]
    eta = np.sum(etaJ)

    return eta, etaJ, activeJunctions


