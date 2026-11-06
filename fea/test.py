import numpy as np

from assemble_structure import assemble_structure
from coil_rod import coilRod

EA, rhoA, q1, q2, LUC = 2.0, 3.0, 0.1, 1e-4, 0.02
nElem, Njunct = 4, 3
kres = np.array([5.0, 7.0])
mres = np.array([0.1, 0.2])

M, C, K, freeDoF, fixedDoF, forceDoF, junctNodes = assemble_structure(
    EA, rhoA, q1, q2, LUC, nElem, Njunct, kres, mres)

print(np.max(np.abs(K @ np.ones(K.shape[0]))))
print(np.max(np.abs(K - K.T)))
print(np.sum(M) - (rhoA*LUC*Njunct + Njunct*np.sum(mres)))
print(fixedDoF, K.shape)

eta, etaJ, activeJunctions = coilRod(M, C, K, freeDoF, junctNodes, 50, 1)
print(f"{eta.real:.10e} {eta.imag:.10e}")