from struct_admittance import structAdmittance
from fea.assemble_coild_MM_structure import assemble_structure
from fea.coil_rod import coilRod

import numpy as np

M, C, K, freeDoF, fixedDoF, forceDoF, junctNodes = assemble_structure(
    2.0, 3.0, 0.1, 1e-4, 0.02, 4, 3, np.array([5.0, 7.0]), np.array([0.1, 0.2]))

eta, _, _ = coilRod(M, C, K, freeDoF, junctNodes, 50, 1)
Y = structAdmittance(M, C, K, freeDoF, junctNodes[[0, 2]], np.array([50.0]), 1)
print(np.abs(Y[0] - 1j*50*np.conj(eta)))