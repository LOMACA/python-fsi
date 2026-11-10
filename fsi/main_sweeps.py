''' Sweep over (kx, kz, c+) for all walls: metrics per unit forcing + rigid profiles for NNLS '''

import os
for v in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(v, '1')            # one BLAS thread per worker (before numpy import)

import sys
import time
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from resolvent.resolvent_compliant import channelResolventCompliant
from fsi.flow_metrics import flow_metrics

# --------------------------------- USER INPUTS ------------------------
TEST = '--test' in sys.argv                  # python -m fsi.sweep_modes --test

RE, H = 2000, 0.05                           # Re_tau, channel half-height [m]
N, NSVD = 201, 10                            # grid points, SVD ranks
Y_SKIP = 1e-4                                # |Y+| below this -> copy rigid result
N_WORKERS = max(1, (os.cpu_count() or 2) - 1)

if TEST:
    KX = np.array([1.0, 12.0])
    KZ = np.array([10.0, 120.0])
    CP = np.array([10.0, 16.0])
    OUT = Path('results') / 'sweep_test'
else:
    KX = np.union1d(np.logspace(np.log10(0.5), np.log10(150), 30), [1, 12])
    KZ = np.union1d(np.logspace(0, np.log10(800), 36), [10, 120])
    CP = np.union1d(np.linspace(2, 26, 25), [10, 16])
    OUT = Path('results') / 'sweep'

SCALARS = ['s1', 's2', 'RS', 'RS_all', 'eta', 'eta_all', 'p_wall', 'u_max', 'v_max']
PROFILES = ['uu', 'vv', 'ww', 'uv']          # rigid, sum over all ranks of sigma^2 * ...
# -----------------------------------------------------------------------


def build_walls():
    """Walls are built in the main process only (closures cannot go to workers)."""
    from fsi.walls import Flow, rigid, rpm, sr
    flow = Flow(Re_tau=RE, h=H)
    R = rpm()
    target = dict(omega_tar=R.w_res, m_tar=1/(R.A * R.modes.r[0]))
    walls = [rigid(), R, sr(flow, 0), sr(flow, 3),
             sr(flow, 0, **target), sr(flow, 3, **target)]
    assert walls[0].name == 'Rigid'
    return flow, walls


def point(kx, kz, cP, Y):
    """One resolvent solve -> scalar metrics and all-rank profiles."""
    _, CON, y, _, _, dy, _ = channelResolventCompliant(RE, kx, kz, cP, N, NSVD, Y)
    m = flow_metrics(CON, y, dy, N, kx, cP, RE)

    q = CON.u * CON.s                        # sigma_j * psi_j, all ranks
    u, v, w = q[:N], q[N:2*N], q[2*N:3*N]
    uv = np.sum(np.real(np.conj(u) * v), axis=1)
    v_wall = np.sqrt(np.sum(np.abs(v[[0, N-1]])**2, axis=1))

    sc = dict(s1=CON.s[0], s2=CON.s[1], RS=m.RS, RS_all=np.sum(uv * (y - 1) * dy),
              eta=m.eta, eta_all=v_wall.max() / (cP*kx/RE),
              p_wall=m.p_wall, u_max=m.u_max, v_max=m.v_max)
    pr = dict(uu=np.sum(np.abs(u)**2, axis=1), vv=np.sum(np.abs(v)**2, axis=1),
              ww=np.sum(np.abs(w)**2, axis=1), uv=uv)
    return sc, pr


def solve_task(args):
    """All walls at one (kx, kz, c). Ys[0] is the rigid wall."""
    kx, kz, cP, Ys = args
    sc0, pr0 = point(kx, kz, cP, 0.0)
    rows = [sc0]
    for Y in Ys[1:]:
        rows.append(sc0 if abs(Y) < Y_SKIP else point(kx, kz, cP, Y)[0])
    S = np.array([[r[f] for f in SCALARS] for r in rows])                 # (nW, nS)
    P = np.array([pr0[f] for f in PROFILES], dtype=np.float32)          # (4, N)
    return S, P


def main():
    flow, walls = build_walls()
    names = [w.name for w in walls]
    nW, nS, nkz, nc = len(walls), len(SCALARS), len(KZ), len(CP)
    OUT.mkdir(parents=True, exist_ok=True)

    # grid + metadata
    _, _, y, _, _, dy, U = channelResolventCompliant(RE, 12.0, 120.0, 10.0, N, NSVD, 0.0)
    np.savez(OUT / 'meta.npz', kx=KX, kz=KZ, cP=CP, walls=names, scalars=SCALARS,
             profiles=PROFILES, Re=RE, h=H, N=N, nsvd=NSVD, Y_skip=Y_SKIP, y=y, dy=dy, U=U)

    # timing estimate on the NW point
    wP = 10*12/RE
    t0 = time.time()
    solve_task((12.0, 120.0, 10.0, [w.Y(wP)[0] for w in walls]))
    t1 = time.time() - t0
    n_pts = len(KX) * nkz * nc
    print(f'{n_pts} points, {nW} walls, {N_WORKERS} workers')
    print(f'one point: {t1:.2f} s  ->  estimate {t1*n_pts/N_WORKERS/3600:.1f} h', flush=True)

    t_start = time.time()
    with ProcessPoolExecutor(N_WORKERS) as ex:
        for i, kx in enumerate(KX):
            f = OUT / f'kx_{i:03d}.npz'
            if f.exists():
                print(f'kx = {kx:.3f} done already, skipped', flush=True)
                continue

            wP = CP * kx / RE
            Yc = np.array([w.Y(wP) for w in walls]).T                       # (nc, nW)
            tasks = [(kx, kz, c, Yc[l]) for kz in KZ for l, c in enumerate(CP)]
            out = list(ex.map(solve_task, tasks, chunksize=4))

            S = np.array([o[0] for o in out]).reshape(nkz, nc, nW, nS)
            P = np.array([o[1] for o in out]).reshape(nkz, nc, len(PROFILES), N)
            np.savez(f, S=S, P=P, Y=Yc, skipped=np.abs(Yc) < Y_SKIP)

            el = time.time() - t_start
            print(f'[{i+1}/{len(KX)}] kx = {kx:8.3f}   elapsed {el/3600:.2f} h', flush=True)

    # assemble one file
    S = np.stack([np.load(OUT / f'kx_{i:03d}.npz')['S'] for i in range(len(KX))])   # (nkx,nkz,nc,nW,nS)
    P = np.stack([np.load(OUT / f'kx_{i:03d}.npz')['P'] for i in range(len(KX))])
    Y = np.stack([np.load(OUT / f'kx_{i:03d}.npz')['Y'] for i in range(len(KX))])   # (nkx,nc,nW)
    np.savez(OUT / 'sweep_all.npz', S=S, P=P, Y=Y)

    # check against main_modes (N there may differ)
    iRS = SCALARS.index('RS')
    for name, (kx, kz, c) in {'NW': (12, 120, 10), 'VLSM': (1, 10, 16)}.items():
        a, b, l = np.argmin(abs(KX-kx)), np.argmin(abs(KZ-kz)), np.argmin(abs(CP-c))
        r = S[a, b, l, :, iRS] / S[a, b, l, 0, iRS]
        print(f'{name}: RS/RS_rig = ' + ', '.join(f'{n} {x:.4f}' for n, x in zip(names, r)))


if __name__ == '__main__':
    main()