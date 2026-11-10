''' resolvent - fea fsi for single turbulent modes - rigid vs compliant '''

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from fsi.walls import Flow, rigid, rpm, sr
from fsi.flow_metrics import flow_metrics
from resolvent.resolvent_compliant import channelResolventCompliant
from fsi.plots import plot_mode

plt.rcParams.update({'font.size': 14,
                     'axes.titlesize': 18,
                     'axes.labelsize': 15,
                     'legend.fontsize': 15})

# --------------------------------- USER INPUTS ------------------------

flow = Flow(Re_tau = 2000, h = 0.05)   # friction Reynolds number and channel height [m]

modes = {'NW':   dict(kx=12, kz=120, cP=10, ylim=80),
         'VLSM': dict(kx=1,  kz=10,  cP=16, ylim=400)}

N = 401                     # number of resolvent modes
nsvd = 10                   # SVD ranks in the resolvent

R = rpm()
target = dict(omega_tar=R.w_res, m_tar=1/(R.A * R.modes.r[0]))
walls = [rigid(), R, sr(flow, 0), sr(flow, 3), sr(flow, 0, **target), sr(flow, 3, **target)]
assert walls[0].name == 'Rigid', 'walls[0] must be rigid() (reference for all ratios)'

# -------------------------------- RESULTS INITIATION --------------------

out = Path('results') / 'modes.npz'

Re = flow.Re_tau

nM = len(modes)
nW = len(walls)

fields = ['s1', 'RS', 'eta', 'u_max', 'v_max','p_wall']

res = {f: np.zeros((nM, nW)) for f in fields}

Ytab = np.zeros((nM, nW), dtype = complex)

# -------------------------------- FSI COMPUTATION ---------------------------

Q = np.zeros((nM, nW, 4*N), dtype=complex)
Umode = np.zeros((nM, N))

for i, (name, mode) in enumerate(modes.items()):

    kx, kz, cP = mode['kx'], mode['kz'], mode['cP']
    wP = cP * kx / Re

    for k, w in enumerate(walls):

        Y = w.Y(wP)[0]

        _, CON, y, _, _, dy, U = channelResolventCompliant(Re, kx, kz, cP, N, nsvd, Y)

        Q[i, k] = CON.u[:, 0]
        Umode[i] = U

        m = flow_metrics(CON, y, dy, N, kx, cP, Re)

        Ytab[i, k] = Y
        res['s1'][i, k] = m.s[0]

        for f in fields[1:]:

            res[f][i, k] = getattr(m, f)

# ------------------------------- OUTPUT  --------------------------------------

names = [w.name for w in walls]

for i, name in enumerate(modes):

    print(f'\n{name}  (w+ = {modes[name]["cP"] * modes[name]["kx"] / Re:.4f})')
    print(f'{"wall":10s} {"|Y+|":>10s} {"s1/s1_rig":>10s} {"RS/RS_rig":>10s} {"eta+":>10s}')
    for j, n in enumerate(names):
        print(f'{n:10s} {abs(Ytab[i, j]):10.3e} '
              f'{res["s1"][i, j] / res["s1"][i, 0]:10.4f} '
              f'{res["RS"][i, j] / res["RS"][i, 0]:10.4f} '
              f'{res["eta"][i, j]:10.3e}')


out.parent.mkdir(exist_ok=True)
np.savez(out, modes=list(modes), walls=names, Y=Ytab,
         kx=[m['kx'] for m in modes.values()], kz=[m['kz'] for m in modes.values()],
         cP=[m['cP'] for m in modes.values()], h=flow.h, Re=Re, Q=Q, Umode=Umode, **res)

# ----------------------------------- PLOTS ----------------------------------------

idxHalf = np.arange((N - 1)//2, N)
U_half = U[idxHalf]

idxHalf = np.arange((N - 1)//2, N)
icrit = idxHalf[np.argmin(np.abs(U[idxHalf] - cP))]
ycP = y[icrit] * Re

for i, (name, mode) in enumerate(modes.items()):
    plot_mode(i, name, Q, Umode, y, walls, mode['cP'], N, Re, ylim=mode['ylim'])
plt.show()















