"""resolvent wall admittance computation

eta = interface displacement per unit force e^{i*w*t).
The resolvent uses e^{-i*w*t}, so the FRF is conjugated,
and the wall pressure pushes into the structure, f = -p*A. Together:

Y+ = v+/p+ = i*w+ * A+ * conj(eta+(w+))

A+ = interface area in wall units: panel area l1+*L3+ for a structure whose
masses/forces describe a whole panel (Lin), 1 for a structure already
normalised per unit interface area (Harris).
Inputs M, C, K must already be in wall units; w+ is a vector."""

import numpy as np

def structAdmittance(M, C, K, freeDoF, activeDoF, wP, Ap):

    Mf = M[np.ix_(freeDoF, freeDoF)]
    Cf = C[np.ix_(freeDoF, freeDoF)]
    Kf = K[np.ix_(freeDoF, freeDoF)]

    activeDoF = np.atleast_1d(activeDoF)
    wP = np.atleast_1d(wP)

    loc = np.searchsorted(freeDoF, activeDoF)
    nActive = len(loc)
    F = np.zeros(len(freeDoF))
    F[loc] = 1 / nActive

    eta = np.zeros(len(wP), dtype=complex)

    for i in range(len(wP)):

        u = np.linalg.solve(Kf - wP[i]**2*Mf + 1j*wP[i]*Cf, F)
        eta[i] = np.sum(u[loc])

    Yp = 1j * wP * Ap * np.conjugate(eta)

    return Yp
