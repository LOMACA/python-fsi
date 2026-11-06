"""Modal interface FRF
Undamped modes of the free system, mass-normalized (Phi' M Phi = I):
eta(omega) = sum_j r_j / (Omega_j^2 - omega^2 + i*omega*c_j)
with interface residue r_j = (sum_{active} phi_ij)^2 / nActive, which is
exactly coilRod's input/output (force 1/nActive on each active junction,
response summed), and modal damping c_j = phi_j' C phi_j = 2*zeta_j*Omega_j"""

from scipy.linalg import eigh
import numpy as np

from dataclasses import dataclass

@dataclass
class modalFRFResult:

    Omega: np.ndarray
    r: np.ndarray
    zeta: np.ndarray
    c: np.ndarray
    coupling: float


def modalFRF(M, C, K, freeDoF, activeDoF, omega, nModes=None, residual=False):

    Mf = M[np.ix_(freeDoF, freeDoF)]
    Cf = C[np.ix_(freeDoF, freeDoF)]
    Kf = K[np.ix_(freeDoF, freeDoF)]

    activeDoF = np.atleast_1d(activeDoF)
    omega = np.atleast_1d(omega)

    W2, Phi = eigh(Kf, Mf)

    Omega = np.sqrt(W2)

    loc = np.searchsorted(freeDoF, activeDoF)
    nActive = len(loc)
    r = (np.sum(Phi[loc, :], axis=0))**2 / nActive
    Cm = Phi.T@Cf@Phi
    c = np.diag(Cm)
    zeta = c / (2*Omega)

    offDiag = Cm - np.diag(c)
    coupling = np.linalg.norm(offDiag, 'fro') / np.linalg.norm(np.diag(c), 'fro')

    if nModes is None:
        nModes = len(Omega)

    rj = r[:nModes, None]
    Oj = Omega[:nModes, None]
    cj = c[:nModes, None]
    eta = np.sum(rj / (Oj ** 2 - omega ** 2 + 1j * omega * cj), axis=0)

    if residual and nModes < len(Omega):

        eta = eta + np.sum(r[nModes:] / Omega[nModes:] ** 2)

    modes = modalFRFResult(Omega=Omega, r=r, zeta=zeta,c=c, coupling=coupling)

    return eta, modes