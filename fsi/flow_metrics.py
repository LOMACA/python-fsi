''' Computation of mode gains and channel integrated Reynolds stresses '''


import numpy as np
from dataclasses import dataclass


@dataclass
class ModeMetrics:
    s: np.ndarray        # singular values of the pair
    RS: float            # channel-integrated RS of ranks 1+2 (Luhar eq. 2.17)
    eta: float           # max wall displacement |eta+| (top/bottom), unit forcing
    u_max: float         # max |u|, pair energy sum
    v_max: float
    p_wall: float        # max wall |p|


def flow_metrics(R, y, dy, N, kx, cP, Re, nPair=2):

    j = np.arange(nPair)
    s = R.s[j]
    u = R.u[:N, j] * s                   # sigma_j * psi_j, columns = ranks
    v = R.u[N:2*N, j] * s
    p = R.u[3*N:4*N, j] * s

    RS = np.sum(np.real(np.conj(u) * v) * ((y - 1) * dy)[:, None])

    amp = lambda Q: np.sqrt(np.sum(np.abs(Q)**2, axis=1))   # pair-invariant profile
    wP = cP * kx / Re
    walls = [0, N - 1]                   # top (y = 2), bottom (y = 0)

    return ModeMetrics(s=s, RS=RS,
                       eta=np.max(amp(v)[walls]) / wP,
                       u_max=np.max(amp(u)), v_max=np.max(amp(v)),
                       p_wall=np.max(amp(p)[walls]))