''' Build the Bunya input: grids, settings, and every wall's admittance at every (kx, c) '''

import sys
import numpy as np
from fsi.walls import Flow, Wall, rigid, sr, modal

TEST = '--test' in sys.argv

RE, H = 2000, 0.05
N, NSVD = 201, 10
Y_SKIP = 1e-6
OMEGA_T = 0.0053
M_EFF = [1e2, 1e3, 1e4]
ZETA = [0.1, 0.3, 1]
REFINE = np.exp([-0.35, -0.12, -0.03, -0.01, 0, 0.01, 0.03, 0.12, 0.35])
EPS = 1e-3

if TEST:
    KX, KZ, CP = np.array([1.0, 12.0]), np.array([10.0, 120.0]), np.array([10.0, 16.0])
    OUT = r'C:\Users\uqlcapp1\Simulations\Bunya\input\sweep_test_in.npz'
else:
    KX = np.union1d(0.25 * np.arange(1, 9),
                    np.union1d(np.logspace(np.log10(2.4), np.log10(150), 25), [12]))
    KZ = np.union1d((2 / 3) * np.arange(1, 7),
                    np.union1d(np.logspace(np.log10(4.8), np.log10(800), 30), [10, 120]))
    CP = np.union1d(np.round(np.linspace(2, 26, 37), 10), [10, 16])
    OUT = r'C:\Users\uqlcapp1\Simulations\Bunya\input\sweep_v4_in.npz'

flow = Flow(Re_tau=RE, h=H)
walls = [rigid(), sr(flow, 0), sr(flow, 3)]
walls += [modal(OMEGA_T, m, z) for m in M_EFF for z in ZETA]
walls += [Wall('dY re', lambda wP: EPS * np.ones_like(np.atleast_1d(wP), dtype=complex)),
          Wall('dY im', lambda wP: 1j * EPS * np.ones_like(np.atleast_1d(wP), dtype=complex))]
assert walls[0].name == 'Rigid'

CPall = np.stack([np.sort(np.concatenate([CP, OMEGA_T * RE / kx * REFINE])) for kx in KX])   # (nkx, nck)
Yall = np.stack([np.array([w.Y(c * kx / RE) for w in walls], dtype=complex).T
                 for kx, c in zip(KX, CPall)])                                                # (nkx, nck, nW)

np.savez(OUT, kx=KX, kz=KZ, cP_base=CP, cP=CPall, Y=Yall, walls=[w.name for w in walls],
         Re=RE, h=H, N=N, nsvd=NSVD, Y_skip=Y_SKIP, omega_t=OMEGA_T, m_eff=M_EFF, zeta=ZETA, refine=REFINE)
print(f'{OUT}: {len(KX)} kx x {len(KZ)} kz x {CPall.shape[1]} c = {Yall.shape[0]*len(KZ)*CPall.shape[1]} points, '
      f'{len(walls)} walls, skipped {np.mean(np.abs(Yall) < Y_SKIP):.2f}')