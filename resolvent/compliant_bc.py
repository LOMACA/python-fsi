import numpy as np


def full_channel_compliant_bc(LHS, RHS, omega, N, admittance, dU0):

    """Impose BCs on (-i*omega*M - L)[u;v;w;p] = M[fx;fy;fz;0] and compute the resolvent.

    Returns H0 (no-slip) and HC (compliant wall):
        i*omega*u = v*dU0   [1]
        Y = v/p             [2]
    """
    # 0-based wall rows: top and bottom of u, v and w
    wall_rows = [0, N-1, N, 2*N-1, 2*N, 3*N-1]

    # Rigid wall: no slip
    LHS0 = LHS.copy()
    RHS0 = RHS.copy()
    for ni in wall_rows:
        LHS0[ni, :] = 0
        RHS0[ni, :] = 0
        LHS0[ni, ni] = 1
    H0 = np.linalg.solve(LHS0, RHS0)

    # Compliant wall
    LHSC = LHS.copy()
    RHSC = RHS.copy()
    for ni in wall_rows:
        LHSC[ni, :] = 0
        RHSC[ni, :] = 0

    # Eq. [1] for u at the top and bottom walls
    LHSC[0, 0] = -1j*omega
    LHSC[N-1, N-1] = -1j*omega
    LHSC[0, N] = dU0[0, 0]
    LHSC[N-1, 2*N-1] = dU0[N-1, N-1]

    # Eq. [2] for v
    LHSC[N, N] = -1
    LHSC[2*N-1, 2*N-1] = 1
    LHSC[N, 3*N] = -admittance
    LHSC[2*N-1, 4*N-1] = -admittance

    # No slip for w
    LHSC[2*N, 2*N] = 1
    LHSC[3*N-1, 3*N-1] = 1

    # Compute resolvent
    HC = np.linalg.solve(LHSC, RHSC)

    return H0, HC