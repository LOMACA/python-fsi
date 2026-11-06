"""Structural assembly of the Metamaterial (Harris et al. (2026)) and application
of coiling cycles"""

import numpy as np
from rod_elements import rodElement

def assemble_structure(EA, rhoA, q1, q2, LUC, nElem, Njunct, kres, mres):

    nRes = len(kres)
    assert len(mres) == nRes, "kres and mres must have the same length."
    assert nElem >= nRes + 1, "need nElem >= nRes+1 elements per cell."

    nCells = Njunct
    Le = LUC / nElem
    totalElem = nCells * nElem
    nBase = totalElem + 1
    nDoF = nBase + nCells*nRes

    junctNodes = np.arange(Njunct)*nElem
    hostNode = junctNodes[:, None] + np.arange(1, nRes+1)

    K = np.zeros((nDoF, nDoF))
    M = np.zeros((nDoF, nDoF))
    K_base = np.zeros((nBase, nBase))
    M_base = np.zeros((nBase, nBase))

    ke, me = rodElement(EA, rhoA, Le)

    for e in range(totalElem):

        nodes = np.array([e, e+1])
        K_base[np.ix_(nodes, nodes)] += ke
        M_base[np.ix_(nodes, nodes)] += me

    K[:nBase, :nBase] = K_base
    M[:nBase, :nBase] = M_base

    resDoF = nBase - 1

    # local resonance effects

    for c in range(nCells):
        for r in range(nRes):

            resDoF += 1
            n = hostNode[c, r]
            M[resDoF, resDoF] = M[resDoF, resDoF] + mres[r]
            K[n, n] = K[n, n] + kres[r]
            K[resDoF, resDoF] = K[resDoF, resDoF] + kres[r]
            K[n, resDoF] = K[n, resDoF] - kres[r]
            K[resDoF, n] = K[resDoF, n] - kres[r]


    C = np.zeros((nDoF, nDoF))
    C[:nBase, :nBase] = q1*M_base + q2*K_base

    fixedDoF = nBase - 1
    forceDoF = 0
    freeDoF = np.setdiff1d(np.arange(nDoF), [fixedDoF])

    return M, C, K, freeDoF, fixedDoF, forceDoF, junctNodes
