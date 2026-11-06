"""2-node linear axial rod element matrices.
[ke, me] = rodElement(EA, rhoA, L) returns the 2x2 element stiffness
and mass matrices for a 2-node rod element of length L, and axial
rigidity EA"""

import numpy as np

def rodElement(EA, rhoA, L):

    ke = (EA / L) * np.array([[1, -1], [-1,1]])
    me = (rhoA * L/6) * np.array([[2, 1], [1, 2]])

    return ke, me
