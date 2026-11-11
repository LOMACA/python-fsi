''' Primitive-variable resolvent code for channel flow
written by Mitul Luhar 01/06/2015
Modified by Ryan McMullen 03/22/2017
Modified by Lorenzo Cappello 05/11/2026

For details on the primitive-variable resolvent analysis see:
Luhar, Sharma, McKeon (2014), Opposition Control within the Resolvent
Analysis Framework, J. Fluid Mech., 749:597-626
This particular code performs phase-shifted opposition control on both
walls '''

import numpy as np
import scipy.linalg

from resolvent.mean_vel import channelMeanVel
from resolvent.compliant_bc import full_channel_compliant_bc
from dataclasses import dataclass


@dataclass
class ResolventResult:
    u: np.ndarray    # response modes [u;v;w;p], 4N x nsvd
    s: np.ndarray    # singular values
    v: np.ndarray    # forcing modes
    fro: float       # Frobenius norm of the weighted resolvent
    H: np.ndarray    # full resolvent, 4N x 4N

def robust_svd(A):
    """Dense SVD; falls back to LAPACK gesvd if the fast gesdd fails to converge."""
    try:
        return np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        return scipy.linalg.svd(A, full_matrices=False, lapack_driver='gesvd')

def channelResolventCompliant(Re, kx, kz, cP, N, nsvd, admittance):

    omega = cP * kx

    y, D1, D2, dy, U0, nuT = channelMeanVel(Re, N)
    y = y + 1

    I = np.eye(N)
    Z = np.zeros((N,N))

    dU0 = D1@U0
    U0 = np.diag(U0)
    dU0 = np.diag(dU0)

    ikU0 = 1j*kx*U0
    LAP = -(kx**2)*I - (kz**2)*I + D2

    L1 = [-ikU0+LAP/Re, -dU0, Z, -1j*kx*I]
    L2 = [Z, -ikU0+LAP/Re, Z, -D1]
    L3 = [Z, Z, -ikU0+LAP/Re, -1j*kz*I]
    L4 = [-1j*kx*I, -D1, -1j*kz*I, Z]
    L = np.block([L1, L2, L3, L4])

    M = np.block([[I, Z, Z, Z],
                  [Z, I, Z, Z],
                  [Z, Z, I, Z],
                  [Z, Z, Z, Z]])

    LHS = -1j*omega*M-L
    RHS = M

    H0, HC = full_channel_compliant_bc(LHS, RHS, omega, N, admittance, dU0)

    IW = np.diag(np.sqrt(dy))
    iIW = np.diag(1/np.sqrt(dy))

    sqW = np.block([[IW, Z, Z, Z],
                    [Z, IW, Z, Z],
                    [Z, Z, IW, Z],
                    [Z, Z, Z, Z]])

    isqW = np.block([[iIW, Z, Z, Z],
                     [Z, iIW, Z, Z],
                     [Z, Z, iIW, Z],
                     [Z, Z, Z, Z]])

    H0W = sqW@H0@isqW
    HCW = sqW@HC@isqW

    H0Wfro = np.linalg.norm(H0W, 'fro')

    u0W, s0W, v0W = robust_svd(H0W)
    u0W = u0W[:, :nsvd]
    s0W = s0W[:nsvd]
    v0W = v0W[:nsvd, :].conj().T
    u0 = isqW @ u0W
    v0 = isqW @ v0W
    s0 = s0W

    uCW, sCW, vCW = robust_svd(HCW)
    uCW = uCW[:, :nsvd]
    sCW = sCW[:nsvd]
    vCW = vCW[:nsvd, :].conj().T
    uC = isqW @ uCW
    vC = isqW @ vCW
    sC = sCW

    # Fix the arbitrary SVD phase using the point nearest the critical layer
    U_vec = np.diag(U0)
    if cP < U_vec.max():
        ind = np.argmax(U_vec > cP)     # first index where U > cP
    else:
        ind = N - 1

    phase_shift = -1j*np.angle(u0[ind, :])
    v0 = v0 @ np.diag(np.exp(phase_shift))
    phase_shift = -1j*np.angle(uC[ind, :])
    vC = vC @ np.diag(np.exp(phase_shift))

    # The weights zero the pressure, so recover the full modes from the unweighted resolvent
    u0 = H0 @ v0
    u0 = u0 @ np.diag(1/s0)

    uC = HC @ vC
    uC = uC @ np.diag(1/sC)

    RIG = ResolventResult(u=u0, s=s0, v=v0, fro=H0Wfro, H=H0)
    CON = ResolventResult(u=uC, s=sC, v=vC, fro=np.linalg.norm(HCW, 'fro'), H=HC)

    return RIG, CON, y, D1, D2, dy, U_vec





