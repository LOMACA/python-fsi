''' DNS weighting: NNLS forcing X(k,c) from HJ spectra -> weighted DR and eta+_rms per wall '''

import os
for v in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ.setdefault(v, '2')

import numpy as np
from pathlib import Path
from scipy.optimize import nnls

# --------------------------------- USER INPUTS ------------------------
SWEEP = Path('results') / 'sweep'
DNS = Path('results') / 'dns' / 'HJ2000_on_sweep.npz'
OUT = Path('results') / 'dns' / 'weighting.npz'
W_UV = 3.0                                   # extra weight of the uv block in the fit
# -----------------------------------------------------------------------

meta = np.load(SWEEP / 'meta.npz')
KX, KZ, CP = meta['kx'], meta['kz'], meta['cP']
y, N = meta['y'], int(meta['N'])
names, SC = list(meta['walls']), list(meta['scalars'])
iRS, iEta = SC.index('RS_all'), SC.index('eta_all')
nW, nc = len(names), len(CP)

D = np.load(DNS)
E, w, T_cell = D['E'], D['w'], D['T_cell']          # E: (planes, var, nkx, nkz)
eta_d = D['yp'] / float(D['Re'])                    # DNS heights, outer units

bot = np.arange((N - 1)//2, N)[::-1]                # y = 0 -> 1
top = np.arange(0, (N - 1)//2 + 1)                  # 2 - y = 0 -> 1
SIGN = [1, 1, 1, -1]                                # uv is antisymmetric


def to_planes(P):
    """Profiles (nkz, nc, 4, N) on y in [0,2] -> (nkz, nc, 4, n_planes), both halves averaged."""
    out = np.empty(P.shape[:3] + (len(eta_d),))
    for j, l, v in np.ndindex(P.shape[:3]):
        p = P[j, l, v]
        out[j, l, v] = 0.5*(np.interp(eta_d, y[bot], p[bot])
                            + SIGN[v]*np.interp(eta_d, 2 - y[top], p[top]))
    return out


X = np.full((len(KX), len(KZ), nc), np.nan)
J = np.full((len(KX), len(KZ), nW), np.nan)
T_res = np.full((len(KX), len(KZ)), np.nan)
res = np.full((len(KX), len(KZ), 4), np.nan)
eta2 = np.zeros(nW)
done = np.zeros(len(KX), bool)
ratio = np.full((len(KX), len(KZ), 4), np.nan)

FIT = [3]          # fitted components: 0 uu, 1 vv, 2 ww, 3 uv   (put in USER INPUTS)
ALPHA = 0.2
T_fit = np.full((len(KX), len(KZ)), np.nan)

for f in sorted(SWEEP.glob('kx_*.npz')):
    i = int(f.stem[3:])
    C = np.load(f)
    Pp, S = to_planes(C['P'].astype(float)), C['S']       # (nkz,nc,4,np), (nkz,nc,nW,nS)
    done[i] = True

    for j in range(len(KZ)):
        b = E[:, :, i, j].T                                # (4, np)
        if not np.any(b):
            X[i, j] = 0.0
            continue
        ok = np.all(np.isfinite(Pp[j]), axis=(1, 2)) & np.all(np.isfinite(S[j, :, :, iRS]), axis=1)
        A = Pp[j][ok].transpose(1, 2, 0)  # (4, np, n_ok)
        n_ok = ok.sum()
        scale = np.linalg.norm(b, axis=1)
        scale[scale == 0] = 1.0
        Af = (A[FIT] / scale[FIT, None, None]).reshape(-1, n_ok)
        bf = (b[FIT] / scale[FIT, None]).ravel()
        L = np.diff(np.eye(n_ok), 2, axis=0)  # second difference over c
        lam = ALPHA * np.linalg.norm(Af) / np.sqrt(n_ok)
        x_ok, _ = nnls(np.vstack([Af, lam * L]), np.r_[bf, np.zeros(n_ok - 2)])
        pred = A @ x_ok
        yw = np.r_[0.0, eta_d]
        T_fit[i, j] = 3 * np.trapezoid((1 - yw) * (-np.r_[0.0, pred[3]]), yw)
        # (4, np)
        ratio[i, j] = pred.sum(axis=1) / np.where(b.sum(axis=1) == 0, np.nan, b.sum(axis=1))
        x = np.zeros(nc)
        x[ok] = x_ok              # failed speeds (not ok) keep X = 0
        X[i, j] = x
        X[i, j] = x
        res[i, j] = np.linalg.norm(A @ x_ok - b, axis=1) / scale

        RS = S[j, ok][:, :, iRS]                            # (n_ok, nW)
        RSr = x_ok @ RS[:, 0]
        T_res[i, j] = 1.5 * RSr
        J[i, j] = (x_ok @ RS) / RSr if RSr != 0 else 1.0
        eta2 += x_ok @ S[j, ok][:, :, iEta]**2

# ---------------------------------- OUTPUT --------------------------------
m = done[:, None] & (w > 0)
print(f'finished kx chunks: {done.sum()}/{len(KX)},  w covered: {w[m].sum():.4f} of {w.sum():.4f} (sweep grid)')
print('median fit residual  ' + '  '.join(f'{v} {np.nanmedian(res[..., c][m]):.3f}'
                                          for c, v in enumerate(['uu', 'vv', 'ww', 'uv'])))
print(f'T reconstructed / DNS (sum over fitted cells): {np.nansum(T_res[m]) / T_cell[m].sum():.4f}')

print(f'\n{"wall":20s} {"DR (partial)":>13s} {"eta+_rms":>10s}')
for k, n in enumerate(names):
    DRk = np.nansum(w[m] * (1 - J[..., k][m]))
    print(f'{n:20s} {DRk:13.5f} {np.sqrt(eta2[k]):10.3e}')
print('median pred/DNS  ' + '  '.join(f'{v} {np.nanmedian(ratio[..., c][m]):.2f}'
                                      for c, v in enumerate(['uu', 'vv', 'ww', 'uv'])))
print(f'T fit on DNS planes / DNS: {np.nansum(T_fit[m]) / T_cell[m].sum():.4f}')

np.savez(OUT, X=X, J=J, T_res=T_res, res=res, done=done, w=w, names=names, W_UV=W_UV)