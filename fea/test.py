import numpy as np

from assemble_coild_MM_structure import assemble_structure
from coil_rod import coilRod
from assemble_defect_RPM_structure import assemble_defect_chain
from modal_analysis import modalFRF

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

M, C, K, forcedDoF = assemble_defect_chain(1.0, 2.0, 3.0, 0.1, 4.0, 5.0, 6)
print(K @ np.ones(6))

M, C, K, freeDoF, fixedDoF, forceDoF, junctNodes = assemble_structure(
    2.0, 3.0, 0.0, 0.0, 0.02, 4, 3, np.array([5.0, 7.0]), np.array([0.1, 0.2]))

eta_direct, _, _ = coilRod(M, C, K, freeDoF, junctNodes, 50, 1)
eta_modal, modes = modalFRF(M, C, K, freeDoF, junctNodes[[0, 2]], 50)
print(np.abs(eta_modal[0] - eta_direct))

M, C, K, freeDoF, fixedDoF, forceDoF, junctNodes = assemble_structure(
    2.0, 3.0, 0.1, 1e-4, 0.02, 4, 3, np.array([5.0, 7.0]), np.array([0.1, 0.2]))

eta_modal, modes = modalFRF(M, C, K, freeDoF, junctNodes[[0, 2]], 50)
print(f"{eta_modal[0].real:.10e} {eta_modal[0].imag:.10e} {modes.coupling:.10e}")