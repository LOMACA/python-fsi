''' DNS weighting of resolvent modes: HJ 2D spectra (Re_tau = 2000) -> sweep cells -> weights,
    friction change per wall, first-order sensitivity, linear screening and paper figures '''

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d
sys.path.insert(0, r'C:\Users\uqlcapp1\Simulations\DNS\code')        # folder with dns_tools.py
from dns_tools import HJ_SPECS, load_spectrum_2d, read_fortran_records
from scipy.optimize import nnls
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter
import matplotlib.patches as mpatches

# --------------------------------- SETTINGS ---------------------------------
Re = 2000                                                  # DNS case
ROOT = Path(__file__).resolve().parent.parent              # fsi -> Python (project root)
SWEEP = ROOT / 'results' / 'sweep_v3'                      # resolvent sweep folder
WALL_CHECK = 'modal m=1e+02 z=3e-02'                       # wall for the ratio check plot (v2: 'SR 3 cyc matched')
FIG_WALLS = ['modal m=1e+02 z=1e-02', 'modal m=1e+02 z=1e-01']   # walls for paper figure 1
VARS = ['uu', 'vv', 'ww', 'uv']                            # spectra to read
SHOW_CHECKS = False                                        # True: file exploration + all check plots
EPS = 1e-3                                                 # probe |Y| of the dY walls in main_sweeps
PASSIVE_RE = -1                                            # passive wall: PASSIVE_RE * Re Y >= 0 (check convention!)
BAND = (4e-3, 0.4)                                         # omega+ band that carries the friction
# ----------------------------------------------------------------------------

files = sorted(HJ_SPECS[Re].glob('*.spe.j*'))
print(len(files), 'files:', files[0].name, '...', files[-1].name)     # check for correct files


# ============================ 0. FILE EXPLORATION (optional) ============================
if SHOW_CHECKS:

    G0 = load_spectrum_2d(files[0])
    print(type(G0))                                        # type of what the reader returns
    for key, val in G0.items():                            # contents of the dict for one file (one plane)
        if isinstance(val, np.ndarray):
            print(f'{key:8s} array, shape {val.shape}, dtype {val.dtype}')
        elif isinstance(val, dict):
            print(f'{key:8s} dict with keys {list(val.keys())}')
            for k2, v2 in val.items():
                print(f'   {k2:6s} array, shape {v2.shape}')
        else:
            print(f'{key:8s} {type(val).__name__} = {val}')

    records = read_fortran_records(files[0])               # the raw records of the file
    print(len(records), 'records')
    for i, r in enumerate(records):
        print(f'record {i}: {len(r)} bytes = {len(r)//4} numbers')

    print(np.round(G0['y_all'], 5))                        # plane heights y/h
    print(np.round(G0['y_all'] * G0['utau'] * G0['re'], 1))  # same heights in y+

    P = load_spectrum_2d(files[2])
    Re_tau = P['utau'] * P['re']
    uu = P['sp']['uu'] / P['utau']**2                      # 2D energy spectrum of u, energy per bin, plus units
    print('y+ = ', P['y'] * Re_tau)
    print('uu[0,0] = ', uu[0, 0], 'sqrt = ', np.sqrt(uu[0, 0]))   # U+^2 and U+ at that y+
    uu[0, 0] = 0.0                                         # remove the mean flow (kx = kz = 0)
    print('uu binned sum', uu.sum())                       # u'^2 at that y+ (Parseval)
    print('u_rms', np.sqrt(uu.sum()))                      # rms of u' at that y+

    # premultiplied spectrum
    kx, kz = P['kx'][1:], P['kz'][1:]                      # wavenumbers excluding 0
    dkx, dkz = P['alp'], P['bet']                          # bin widths
    phi = uu[1:, 1:] / (dkx * dkz)                         # energy per bin -> energy density
    pm = kx[:, None] * kz[None, :] * phi                   # energy per unit d(ln kx) d(ln kz)
    lx, lz = 2*np.pi*Re_tau/kx, 2*np.pi*Re_tau/kz          # wavelengths in plus units
    fig, ax = plt.subplots()
    cs = ax.contourf(lx, lz, pm.T, levels=20, cmap='viridis')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$\lambda_x^+$'); ax.set_ylabel(r'$\lambda_z^+$')
    ax.set_title(f'$k_x k_z \\Phi_{{uu}}$ at $y^+$ = {P["y"]*Re_tau:.1f}')
    fig.colorbar(cs)

    # log-polar spectrum (Lee & Moser 2019, eq. 2.27)
    KREF = 1/50000                                         # k_ref+
    dg = 0.01                                              # grid spacing in kx#, kz#
    g = np.arange(0, 4.9 + dg/2, dg)
    GX, GZ = np.meshgrid(g, g, indexing='ij')
    XI = np.hypot(GX, GZ)                                  # radius xi
    TH = np.arctan2(GZ, GX)                                # angle: 0 = kx axis, pi/2 = kz axis
    K = KREF * 10**XI * Re_tau                             # |k| in outer units
    KXq, KZq = K*np.cos(TH), K*np.sin(TH)                  # the real (kx, kz) of every grid point
    Phi = uu / (P['alp'] * P['bet'])                       # density on the full DNS grid
    fi = RegularGridInterpolator((P['kx'], P['kz']), Phi, bounds_error=False, fill_value=0.0)
    Phi_q = fi(np.stack([KXq, KZq], axis=-1))
    with np.errstate(divide='ignore', invalid='ignore'):
        D = np.log(10) * K**2 * Phi_q / XI
    D[XI == 0] = 0.0
    print('integral of D =', D.sum() * dg**2, '  vs  sum of uu =', uu.sum())
    fig, ax = plt.subplots(figsize=(6, 6))
    Ds = gaussian_filter(D, 6)
    levels = np.linspace(0.05, 1, 20) * Ds.max()
    cs = ax.contourf(g, g, Ds.T, levels=levels, cmap='magma_r')
    th = np.linspace(0, np.pi/2, 100)
    for lp in [1e1, 1e2, 1e3, 1e4]:                        # circles of constant |lambda+|
        r = np.log10(2*np.pi/lp / KREF)
        ax.plot(r*np.cos(th), r*np.sin(th), 'k:', lw=0.8)
        ax.text(r/np.sqrt(2), r/np.sqrt(2), rf'$\lambda^+=10^{{{int(np.log10(lp))}}}$', fontsize=8)
    ax.plot([0, 4.9], [0, 4.9], 'k--', lw=0.5)             # kz = kx
    ax.set_aspect('equal')
    ax.set_xlabel(r'$k_x^\#$'); ax.set_ylabel(r'$k_z^\#$')
    ax.set_title(f'log-polar $uu$ at $y^+$ = {P["y"]*Re_tau:.1f}')
    fig.colorbar(cs)


# ============================ A. SWEEP GRID ============================
M = np.load(SWEEP / 'meta.npz')                            # sweep metadata (grid, walls, y, U)
KX, KZ, CP = M['kx'], M['kz'], M['cP']                     # resolvent wavenumbers and base wave speeds
names = list(M['walls'])
print(f'sweep kx: {KX.min():.2f} ... {KX.max():.2f} ({len(KX)} values)')
print(f'sweep kz: {KZ.min():.2f} ... {KZ.max():.1f} ({len(KZ)} values)')
print('walls:', names)


# ============================ B. DNS GRID AND CROP ============================
G = load_spectrum_2d(files[0], keep=())                    # grid only (same for all planes)
Re_tau = G['utau'] * G['re']                               # friction Re
kx_top = KX[-1] * np.sqrt(KX[-1] / KX[-2])                 # upper edge of the last kx cell
kz_top = KZ[-1] * np.sqrt(KZ[-1] / KZ[-2])                 # upper edge of the last kz cell
ix = G['kx'] <= kx_top                                     # DNS kx bins we keep
iz = G['kz'] <= kz_top                                     # DNS kz bins we keep
print(f'keep {ix.sum()} of {len(ix)} kx bins (kx <= {kx_top:.1f}), '
      f'{iz.sum()} of {len(iz)} kz bins (kz <= {kz_top:.1f})')


# ============================ C. READ THE 20 PLANES ============================
nP = len(files)
yp = np.empty(nP)                                          # height of each plane in y+
tot = np.empty((nP, len(VARS)))                            # full variance per plane: sum over all bins
E = np.empty((nP, len(VARS), ix.sum(), iz.sum()), np.float32)   # cropped spectra, energy per bin

for p, f in enumerate(files):
    P = load_spectrum_2d(f, keep=VARS)
    assert P['jind'] == p + 1, f'{f.name} is plane {P["jind"]}, expected {p+1}'
    yp[p] = P['y'] * Re_tau
    for v, name in enumerate(VARS):
        s = P['sp'][name] / G['utau']**2                   # plus units
        s[0, 0] = 0.0                                      # remove the mean
        tot[p, v] = s.sum()                                # variance at this height (Parseval)
        E[p, v] = s[np.ix_(ix, iz)]                        # cropped, energy per bin
    print(f'{f.name}: y+ = {yp[p]:7.1f}   ' + '  '.join(f'{n} {t:7.3f}' for n, t in zip(VARS, tot[p])))

coverage = np.divide(E.sum(axis=(2, 3)), tot, out=np.full_like(tot, np.nan), where=tot != 0)
print(f'\nCoverage: fraction of each variance kept by the crop (kx <= {kx_top:.1f}, kz <= {kz_top:.1f})')
print('   y+     ' + '   '.join(f'{n:>5s}' for n in VARS))
for p in range(nP):
    print(f'{yp[p]:7.1f}  ' + '  '.join(f'{c:6.3f}' for c in coverage[p]))


# ============================ D. CELL EDGES AND BIN MATRICES ============================
def cell_edges(K):

    mid = np.sqrt(K[1:] * K[:-1])
    lo = K[0] / np.sqrt(K[1] / K[0])
    hi = K[-1] * np.sqrt(K[-1] / K[-2])
    return np.concatenate([[lo], mid, [hi]])

def bin_matrix(k, e):

    return ((k[None, :] >= e[:-1, None]) & (k[None, :] < e[1:, None])).astype(np.float32)

ex, ez = cell_edges(KX), cell_edges(KZ)
kx_d, kz_d = G['kx'][ix], G['kz'][iz]                      # kept DNS wavenumbers
Ax, Az = bin_matrix(kx_d, ex), bin_matrix(kz_d, ez)
print(f'kx cells from {ex[0]:.3f} to {ex[-1]:.1f},  kz cells from {ez[0]:.3f} to {ez[-1]:.1f}')
print('Ax', Ax.shape, '  Az', Az.shape)
print('DNS bins per kx cell:', Ax.sum(axis=1).astype(int))
print('DNS bins per kz cell:', Az.sum(axis=1).astype(int))


# ============================ E. SUM THE DNS BINS INTO THE CELLS ============================
Ec = Ax @ E @ Az.T                                         # (20, 4, nkx, nkz): energy in each sweep cell at each plane
print('Ec', Ec.shape)
in_cells = Ec.sum(axis=(2, 3)) / np.where(tot != 0, tot, np.nan)
print('fraction of uv inside the cells, per plane:', np.round(in_cells[:, VARS.index('uv')], 3))


# ============================ F. FIK INTEGRAL PER CELL AND WEIGHTS ============================
iuv = VARS.index('uv')
yw = np.concatenate([[0.0], yp])                           # heights incl. the wall, y+ = 0
fik = 1 - yw / Re_tau                                      # FIK weight (1 - y/h)

uv_cell = Ec[:, iuv]                                       # (20, nkx, nkz)
uv_cell = np.concatenate([np.zeros((1,) + uv_cell.shape[1:]), uv_cell])   # + uv = 0 at the wall
T_cell = 3 / Re_tau * np.trapezoid(fik[:, None, None] * -uv_cell, yw, axis=0)   # FIK friction of each cell

uv_tot = np.concatenate([[0.0], tot[:, iuv]])              # full uv profile (all scales)
T_tot = 3 / Re_tau * np.trapezoid(fik * -uv_tot, yw)

w = T_cell / T_tot
print(f'T_tot = {T_tot:.4f},   sum of w over the sweep cells = {w.sum():.4f}')


# ============================ G. RESOLVENT GRID -> DNS HEIGHTS ============================
y, N = M['y'], int(M['N'])
print(f'resolvent y: index 0 -> {y[0]:.3f}, index {(N-1)//2} -> {y[(N-1)//2]:.3f}, index {N-1} -> {y[-1]:.3f}')

mid = (N - 1) // 2                                         # centreline index
eta_top = 2 - y[:mid+1]                                    # top half: distance from the top wall, 0 -> 1
eta_bot = y[mid:][::-1]                                    # bottom half, flipped so it runs 0 -> 1
SIGN = np.array([1, 1, 1, -1])[:, None]                    # uv changes sign in the top half
eta_d = np.clip(yp / Re_tau, 0, 1)                         # DNS plane heights in y/h
U_bot = M['U'][mid:][::-1]                                 # mean velocity U+, wall -> centreline
y_bot = eta_bot * Re_tau                                   # the same heights in y+

def to_planes(Pr):

    top = interp1d(eta_top, Pr[..., :mid+1], axis=-1)(eta_d)
    bot = interp1d(eta_bot, Pr[..., mid:][..., ::-1], axis=-1)(eta_d)
    return 0.5 * (bot + SIGN * top)


# ============================ H. RIGID PROFILES OF ALL CHUNKS ============================
Rp = np.full((len(KX), len(KZ), len(CP), 4, len(yp)), np.nan)   # rigid profiles at the DNS planes, base c grid
for i in range(len(KX)):
    f = SWEEP / f'kx_{i:03d}.npz'
    if f.exists():
        d = np.load(f)
        CPk = d['cP'] if 'cP' in d.files else CP             # v3: per-kx grid, v1/v2: base grid
        ib = np.array([np.flatnonzero(np.isclose(CPk, c))[0] for c in CP])   # base speeds inside CPk
        Pc = d['P']
        if Pc.ndim == 5:                                     # v2/v3: (nkz, nck, nW, 5, N) -> rigid wall
            Pc = Pc[:, :, 0]
        Rp[i] = to_planes(Pc[:, ib, :4, :])                  # uu, vv, ww, uv -> (nkz, nc, 4, 20)
done = ~np.isnan(Rp[:, 0, 0, 0, 0])
print(f'{done.sum()} of {len(KX)} kx chunks loaded,  Rp shape {Rp.shape}')

if SHOW_CHECKS:                                              # DNS cell vs resolvent profiles
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (cn, kx0, kz0, cs) in zip(axes, [('NW', 12, 120, [4, 8, 10, 14, 20]),
                                             ('VLSM', 1, 10, [8, 12, 16, 20, 24])]):
        a, b = np.argmin(abs(KX - kx0)), np.argmin(abs(KZ - kz0))
        dns = -Ec[:, iuv, a, b]                              # DNS -uv of this cell at the 20 planes
        ax.plot(yp, dns / dns.max(), 'ko-', lw=2, label='DNS cell')
        for c in cs:
            l = np.argmin(abs(CP - c))
            r = -Rp[a, b, l, iuv]                            # resolvent -uv per unit forcing
            line, = ax.plot(yp, r / r.max(), label=f'$c^+$ = {CP[l]:.1f}')
            yc = np.interp(CP[l], U_bot, y_bot)              # critical layer: U(yc) = c
            ax.axvline(yc, color=line.get_color(), ls=':', lw=1)
        ax.set_xscale('log')
        ax.set_xlabel('$y^+$'); ax.set_ylabel('$-uv$ / max')
        ax.set_title(f'{cn}: $k_x$ = {KX[a]:.2f}, $k_z$ = {KZ[b]:.1f},  w = {w[a, b]:.2e}')
        ax.legend(fontsize=8)
    fig.tight_layout()


# ============================ I. FIT CP VALUES OF RESOLVENT CELLS TO DNS ============================
ALPHA = 0.05

def fit_cell(a, b):

    d = Ec[:, iuv, a, b]                                 # target: DNS uv of this cell at the 20 planes
    A = Rp[a, b, :, iuv, :].T                            # column i = resolvent uv profile at speed CP[i]
    ok = ~np.isnan(A).any(axis=0)
    Xc = np.zeros(len(CP))
    if not d.any() or ok.sum() < 3:
        return Xc, np.nan, np.zeros_like(d)

    Aok = A[:, ok]
    n = ok.sum()
    L = np.diff(np.eye(n), 2, axis=0)
    lam = ALPHA * np.linalg.norm(Aok / np.sqrt(n))      # how strongly smoothness counts against fitting the data
    xok, _ = nnls(np.vstack([Aok, lam * L]), np.concatenate([d, np.zeros(n - 2)]))
    Xc[ok] = xok
    fit = Aok @ xok
    rc = np.linalg.norm(fit - d) / np.linalg.norm(d)    # relative misfit
    return Xc, rc, fit

if SHOW_CHECKS:                                          # fit check at NW and VLSM
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for row, (cn, kx0, kz0) in enumerate([('NW', 12, 120), ('VLSM', 1, 10)]):
        a, b = np.argmin(abs(KX - kx0)), np.argmin(abs(KZ - kz0))
        Xc, rc, fc = fit_cell(a, b)
        ax = axes[row, 0]
        ax.plot(yp, -Ec[:, iuv, a, b], 'ko-', label='DNS cell')
        ax.plot(yp, -fc, 'r-', lw=2, label=f'resolvent fit, residual {rc:.3f}')
        ax.set_xscale('log'); ax.set_xlabel('$y^+$'); ax.set_ylabel('$-uv$')
        ax.set_title(f'{cn}: $k_x$ = {KX[a]:.2f}, $k_z$ = {KZ[b]:.1f}'); ax.legend()
        ax = axes[row, 1]
        A = Rp[a, b, :, iuv, :].T
        Az0 = np.vstack([np.zeros((1, A.shape[1])), np.nan_to_num(A)])   # add uv = 0 at the wall
        Tc = 3 / Re_tau * np.trapezoid(fik[:, None] * -Az0, yw, axis=0)  # FIK integral of each speed's profile
        ax.bar(CP, Xc * Tc, width=0.5)
        ax.set_xlabel('$c^+$'); ax.set_ylabel(r'$X_c\,T_c$'); ax.set_title(f'{cn}: stress over speeds')
        print(f'{cn}: sum X_c T_c = {np.sum(Xc * Tc):.3e}   vs   T_cell = {T_cell[a, b]:.3e}')
    fig.tight_layout()


# ============================ J. FIT ALL CELLS ============================
nkx, nkz, nc = len(KX), len(KZ), len(CP)
X = np.zeros((nkx, nkz, nc))                               # forcing per cell and speed
res = np.full((nkx, nkz), np.nan)                          # relative misfit per cell
for a in range(nkx):
    for b in range(nkz):
        X[a, b], res[a, b], _ = fit_cell(a, b)

uvR = np.nan_to_num(Rp[..., iuv, :])                       # rigid uv profiles
uvR = np.concatenate([np.zeros(uvR.shape[:-1] + (1,)), uvR], axis=-1)   # uv = 0 at the wall
T_c = 3 / Re_tau * np.trapezoid(fik * -uvR, yw, axis=-1)  # FIK friction per unit forcing, per speed

ok = ~np.isnan(res)
print(f'fitted {ok.sum()} of {res.size} cells')
print(f'w-weighted mean residual: {np.sum(w[ok] * res[ok]) / np.sum(w[ok]):.3f}')
print(f'sum X_c T_c / sum T_cell: {np.sum(X * T_c) / T_cell.sum():.4f}')


# ============================ K. SUM OVER ALL CELLS ============================
def cell_widths(c):                                        # midpoint cells: any spacing
    e = np.concatenate([[c[0] - (c[1] - c[0]) / 2], (c[1:] + c[:-1]) / 2,
                        [c[-1] + (c[-1] - c[-2]) / 2]])
    return np.diff(e)

nW, nS = len(names), len(M['scalars'])
S_all = CPall = None
for a in range(nkx):
    f = SWEEP / f'kx_{a:03d}.npz'
    if f.exists():
        d = np.load(f)
        CPk = d['cP'] if 'cP' in d.files else CP
        if S_all is None:                                  # allocate once the chunk size is known
            S_all = np.full((nkx, nkz, len(CPk), nW, nS), np.nan)
            CPall = np.full((nkx, len(CPk)), np.nan)
        S_all[a], CPall[a] = d['S'], CPk
print('S_all', S_all.shape, '  c points per kx:', CPall.shape[1])


# forcing on each chunk's own c grid
dc0 = cell_widths(CP)
Xk = np.zeros((nkx, nkz, CPall.shape[1]))
for a in range(nkx):
    if np.isnan(CPall[a]).any():
        continue
    dck = cell_widths(CPall[a])
    for b in range(nkz):
        xd = X[a, b] / dc0                                 # forcing per unit c
        Xk[a, b] = np.interp(CPall[a], CP, xd, left=0, right=0) * dck
print(f'forcing kept after regridding: {Xk.sum() / X.sum():.4f}')   # 1 up to interpolation

# rigid-wall pressure^2 at the wall per unit forcing, all ranks (from the saved 'pp' profile)
ipp = list(M['profiles']).index('pp') if 'profiles' in M.files else 4    # uu, vv, ww, uv, pp
PPW = np.full((nkx, nkz, CPall.shape[1]), np.nan)
for a in range(nkx):
    f = SWEEP / f'kx_{a:03d}.npz'
    if f.exists():
        Pc = np.load(f)['P']                                  # (nkz, nck, nW, 5, N)
        if Pc.ndim == 5:
            PPW[a] = 0.5 * (Pc[:, :, 0, ipp, 0] + Pc[:, :, 0, ipp, -1])   # mean of both walls
pp_model = np.nansum(Xk * PPW)
print(f'wall pressure from the DNS-weighted model: p_rms+ = {np.sqrt(pp_model):.2f}  '
      f'(DNS at Re_tau ~ 2000: about 2.8-3.0)')


RS = S_all[..., list(M['scalars']).index('RS_all')]
num = np.sum(Xk[..., None] * RS, axis=2)
den = num[..., 0]
with np.errstate(invalid='ignore', divide='ignore'):
    J = num / den[..., None]
bad = np.isnan(J[..., 0])
J = np.nan_to_num(J, nan=1.0)
print(f'{bad.sum()} cells without forcing (J set to 1),  max deviation of rigid J from 1: {np.abs(J[..., 0] - 1).max()}')

a, b = np.argmin(abs(KX - 12)), np.argmin(abs(KZ - 120))   # NW cell
print(f'NW cell (kx = {KX[a]:.2f}, kz = {KZ[b]:.1f}), w = {w[a, b]:.2e}')
for name, j in zip(names, J[a, b]):
    print(f'   {name:24s} J = {j:.4f}')

k = names.index(WALL_CHECK)
RS_lead = S_all[..., list(M['scalars']).index('RS')]
ca = CPall[a]                                              # this kx's speeds
ratio_all = RS[a, b, :, k] / RS[a, b, :, 0]
ratio_lead = RS_lead[a, b, :, k] / RS_lead[a, b, :, 0]
ic = np.argmin(abs(ca - 10))
print(f'c = {ca[ic]:.2f}: RS_all ratio = {ratio_all[ic]:.4f}, '
      f'RS leading ratio = {ratio_lead[ic]:.4f}, cell J = {J[a, b, k]:.4f}')

if SHOW_CHECKS:                                            # ratio vs c on top of where the stress sits
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(CP, X[a, b] * T_c[a, b], width=0.5, color='0.8', label=r'$X_c T_c$')
    ax.set_xlabel(r'$c^+$'); ax.set_ylabel(r'$X_c T_c$')
    ax2 = ax.twinx()
    ax2.plot(ca, ratio_all, 'C0o-', ms=3, label='RS rank 10 ratio')
    ax2.plot(ca, ratio_lead, 'C1s--', ms=3, label='RS (leading = rank 2) ratio')
    ax2.axhline(1, color='k', lw=0.8)
    if 'omega_t' in M.files:
        ax2.axvline(M['omega_t'] * Re_tau / KX[a], color='r', ls=':', label=r'$c_{res}$')
    ax2.set_ylabel(r'$RS_{wall}/RS_{rigid}$')
    ax.set_title(f'{names[k]}, kx = {KX[a]:.2f}, kz = {KZ[b]:.1f}')
    fig.legend(loc='upper right'); fig.tight_layout()

# ---------- friction change per wall (negative = reduction) ----------
dCf = np.sum(T_cell[..., None] * (J - 1), axis=(0, 1))     # T_tot is already inside T_cell
for name, dv in zip(names, dCf):
    print(f'   {name:24s} dCf = {100*dv:+.3f} %')
np.savez(ROOT / 'results' / f'dns_weights_{SWEEP.name}.npz', X=X, Xk=Xk, CPall=CPall, w=w,
         T_cell=T_cell, T_c=T_c, T_tot=T_tot, KX=KX, KZ=KZ, CP=CP)

# ---------- dCf table over (m_eff, zeta) ----------
im = [i for i, n in enumerate(names) if n.startswith('modal')]   # modal walls, m outer, zeta inner
if im:
    m_eff, zeta = M['m_eff'], M['zeta']
    dCf_mz = 100 * dCf[im].reshape(len(m_eff), len(zeta))
    print('\ndCf [%]  (negative = reduction)  rows m_eff, columns zeta')
    print('m_eff \\ zeta ' + ''.join(f'{z:>10.2f}' for z in zeta))
    for m, row in zip(m_eff, dCf_mz):
        print(f'{m:12.0e} ' + ''.join(f'{dv:+10.3f}' for dv in row))

# ---------- dCf per kx ----------
dCf_kx = 100 * np.sum(T_cell[..., None] * (J - 1), axis=1)  # sum over kz only -> (nkx, nW) in %
print(f'check: sum over kx = {dCf_kx[:, k].sum():+.3f} %  vs  dCf {100*dCf[k]:+.3f} %  ({names[k]})')

if SHOW_CHECKS:
    fig, ax = plt.subplots(figsize=(8, 4))
    for wn in ['modal m=1e+02 z=3e-02', 'modal m=1e+02 z=1e-02', 'dY im', 'dY re']:
        if wn in names:
            ax.semilogx(KX, dCf_kx[:, names.index(wn)], 'o-', ms=3, label=wn)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel(r'$k_x$'); ax.set_ylabel(r'$\Delta C_f$ per $k_x$ column [%]')
    ax.set_title(r'Where in $k_x$ the friction change comes from (negative = reduction)')
    ax.legend(fontsize=8); fig.tight_layout()


# ============================ L. FREQUENCY SENSITIVITY g(omega) ============================
ire, iim = names.index('dY re'), names.index('dY im')

Wp = np.nan_to_num(T_cell[..., None] * Xk / np.where(den != 0, den, np.nan)[..., None])   # (nkx, nkz, nck)
RS0 = np.nan_to_num(RS[..., 0])
gr = np.nan_to_num((RS[..., ire] - RS[..., 0]) / EPS)    # d RS / d Re Y per point
gi = np.nan_to_num((RS[..., iim] - RS[..., 0]) / EPS)    # d RS / d Im Y per point
om = np.broadcast_to(CPall[:, None, :] * KX[:, None, None] / Re_tau, Wp.shape)   # omega+ of each point

print(f'\ncheck: sum W RS0 = {np.sum(Wp * RS0):.4f}  vs  sum T_cell = {T_cell.sum():.4f}')
print(f'check: EPS * sum W gi = {100*EPS*np.sum(Wp * gi):+.4f} %  vs  dY im dCf = {100*dCf[iim]:+.4f} %')

edges = np.logspace(-4, 0.5, 46)                           # log bins in omega+
oc = np.sqrt(edges[1:] * edges[:-1])
hist = lambda q: np.histogram(om.ravel(), bins=edges, weights=q.ravel())[0]
T_om, Gr, Gi = hist(Wp * RS0), hist(Wp * gr), hist(Wp * gi)

G = np.hypot(Gr, Gi)                                       # first-order optimum for |Y| = EPS at every omega
opt_free = -EPS * G.sum()                                  # best phase per bin, no passivity
passive_ok = lambda g: -PASSIVE_RE * g > 0                 # PASSIVE_RE = sign of passive Re Y (-1); optimum Re Y ~ -g
opt_pass = -EPS * np.where(passive_ok(Gr), G, np.abs(Gi)).sum()   # Re Y restricted to the passive side
print(f'first order, |Y| = {EPS:g} at all omega:  unconstrained {100*opt_free:+.4f} %,  '
      f'passive {100*opt_pass:+.4f} %,  dY im {100*EPS*Gi.sum():+.4f} %')

if SHOW_CHECKS:
    fig, axs = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    axs[0].semilogx(oc, T_om / T_om.sum(), 'k.-')
    axs[0].set_ylabel(r'$T(\omega)$ / total'); axs[0].set_title('friction carried per frequency bin')
    axs[1].semilogx(oc, 100 * Gr, 'C3.-', label=r'$G_r$: per unit Re $Y$')
    axs[1].semilogx(oc, 100 * Gi, 'C2.-', label=r'$G_i$: per unit Im $Y$')
    axs[1].axhline(0, color='k', lw=0.8)
    axs[1].set_ylabel(r'$\partial \Delta C_f / \partial Y$ [% per unit Y]'); axs[1].set_xlabel(r'$\omega^+$')
    axs[1].set_title('first-order sensitivity (negative = that sign of Y reduces friction)')
    for axx in axs:
        if 'omega_t' in M.files:
            axx.axvline(M['omega_t'], color='r', ls=':', lw=1)
    axs[1].legend(); fig.tight_layout()


# ============================ M. LINEAR SCREENING ============================
om_kc = CPall * KX[:, None] / Re_tau                       # (nkx, nck): omega depends on kx, c only

def Y_modal(wv, Omega, m_eff, zeta):                       # single-mode wall, same convention as walls.py
    return 1j * wv * np.conj((1 / m_eff) / (Omega**2 - wv**2 + 2j * zeta * Omega * wv))

def screen(Ykc):
    """first-order friction change (fraction, negative = reduction) for Y given on (nkx, nck)"""
    Y3 = Ykc[:, None, :]                                   # broadcast over kz
    return np.sum(Wp * (gr * Y3.real + gi * Y3.imag))

def mz(name):                                              # m_eff, zeta from a modal wall name
    return float(name.split('m=')[1].split()[0]), float(name.split('z=')[1])

if 'omega_t' in M.files:                                   # validate against the full modal walls
    print('\nlinear screening vs full sweep  (dCf %, negative = reduction)')
    for i in im:
        m_, z_ = mz(names[i])
        lin = screen(Y_modal(om_kc, M['omega_t'], m_, z_))
        print(f'   {names[i]:24s} full {100*dCf[i]:+.4f}   linear {100*lin:+.4f}')

# bounds at |Y| = EPS: what freedom in Y is worth
g_abs = np.hypot(gr, gi)
b_point = -EPS * np.sum(Wp * g_abs)                        # Y free per (kx, kz, omega)
b_point_pass = -EPS * np.sum(Wp * np.where(passive_ok(gr), g_abs, np.abs(gi)))
print(f'\nbounds at |Y| = {EPS:g}:  Y(omega) only: free {100*opt_free:+.4f} %, passive {100*opt_pass:+.4f} %')
print(f'                       Y(kx,kz,omega):  free {100*b_point:+.4f} %, passive {100*b_point_pass:+.4f} %')

if SHOW_CHECKS:                                            # design map per unit m_eff
    OM = np.logspace(-2, 0.5, 51)
    ZE = np.array([0.003, 0.01, 0.03, 0.1, 0.3, 1.0])
    F = np.array([[screen(Y_modal(om_kc, O, 1.0, z)) for O in OM] for z in ZE])   # m_eff = 1
    fig, ax = plt.subplots(figsize=(8, 4))
    for z, fz in zip(ZE, F):
        ax.semilogx(OM, 100 * fz, label=f'$\\zeta$ = {z:g}')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel(r'$\Omega^+$'); ax.set_ylabel(r'$m_{eff}\,\Delta C_f$ [%]')
    ax.set_title('linear design map: single-mode wall (negative = reduction; divide by $m_{eff}$)')
    ax.legend(fontsize=8); fig.tight_layout()


# ============================ N. DESIGN MAP PER UNIT PEAK ADMITTANCE ============================
wmin, wmax = om_kc[om_kc > 0].min(), om_kc.max()           # all forced frequencies
wg = np.logspace(np.log10(wmin), np.log10(wmax), 4000)     # dense grid to find the peak |Y|

def Y_peak(Omega, zeta):                                   # max |Y| over all forced omegas, m_eff = 1
    wv = np.append(wg, Omega) if wmin <= Omega <= wmax else wg
    return np.abs(Y_modal(wv, Omega, 1.0, zeta)).max()

OM2 = np.logspace(np.log10(0.002), np.log10(3), 61)
ZE2 = np.array([0.03, 0.1, 0.3, 1.0])                      # zeta < 0.03 is under-sampled in omega
S = np.array([[screen(Y_modal(om_kc, O, 1.0, z)) / Y_peak(O, z) for O in OM2] for z in ZE2])

S_spring = screen(1j * om_kc) / wmax                       # massless spring Y = i omega / k (peak at top omega)
S_const = Gi.sum()                                         # constant spring Y = i |Y|, per unit |Y|

iz_, io_ = np.unravel_index(np.argmin(S), S.shape)
print(f'\nper unit |Y|_max:  best single mode Omega = {OM2[io_]:.4f}, zeta = {ZE2[iz_]:g}: {100*S[iz_, io_]:+.2f} %')
print(f'                   massless spring {100*S_spring:+.2f} %,  constant spring {100*S_const:+.2f} %')
print(f'                   bounds: passive {100*opt_pass/EPS:+.2f} %,  free {100*opt_free/EPS:+.2f} %')
LX, LZ = np.meshgrid(2*np.pi*Re_tau/KX, 2*np.pi*Re_tau/KZ, indexing='ij')   # wavelengths in wall units
nw = (LX > 300) & (LX < 3000) & (LZ > 50) & (LZ < 200)
vl = LX > 3*Re_tau
print(f'NW region   ({nw.sum()} cells): {100*T_cell[nw].sum():.1f} %')
print(f'VLSM region ({vl.sum()} cells): {100*T_cell[vl].sum():.1f} %')
print(f'all cells: {100*T_cell.sum():.1f} %')

# ============================ Z. PAPER FIGURES ============================
FIG_DIR = ROOT / 'results' / 'figures' / SWEEP.name
FIG_DIR.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 10, 'axes.titlesize': 10, 'legend.fontsize': 8,
                     'axes.spines.top': False, 'axes.spines.right': False})
W1, H1 = 6.5, 3.6                                          # single-panel size [in]
ZCOL = {0.01: 'C0', 0.03: 'C1', 0.1: 'C2', 0.3: 'C4', 1.0: 'C3'}   # one colour per zeta, all figures
CELLS = {'NW': (12, 120), 'VLSM': (1, 10)}

def save(fig, fname):
    fig.tight_layout()
    fig.savefig(FIG_DIR / f'{fname}.png', dpi=200)
    print(f'saved {FIG_DIR / fname}.png')

def cell(kx0, kz0):
    return np.argmin(abs(KX - kx0)), np.argmin(abs(KZ - kz0))

# ---------- 1. single mode vs DNS-weighted cell ----------
fig, axs = plt.subplots(1, 2, figsize=(2 * W1, H1))
for ax, (cn, (kx0, kz0)) in zip(axs, CELLS.items()):
    a, b = cell(kx0, kz0)
    st = X[a, b] * T_c[a, b]
    ax.bar(CP, st / st.max(), width=0.5, color='0.85', label=r'stress carried, $X_c T_c$ (norm.)')
    ax.set_xlabel(r'$c^+$')
    ax.set_ylabel(r'$X_c T_c$ / max')
    ax2 = ax.twinx(); ax2.spines['right'].set_visible(True)
    for wn in FIG_WALLS:
        kw = names.index(wn)
        rr = RS[a, b, :, kw] / RS[a, b, :, 0]
        st_k = np.interp(CPall[a], CP, st, left=0, right=0)  # stress at the chunk's speeds
        rr = np.where(st_k > 0.02 * st.max(), rr, np.nan)  # hide where the cell carries no stress
        ax2.plot(CPall[a], rr, '-', color=ZCOL[mz(wn)[1]], lw=1.5,
                 label=f'{wn}:  cell $J$ = {J[a, b, kw]:.4f}')
    ax2.axhline(1, color='k', lw=0.6)
    if 'omega_t' in M.files:
        ax2.axvline(M['omega_t'] * Re_tau / KX[a], color='r', ls=':', lw=1, label=r'$c_{res}$')
    ax2.set_ylabel(r'$RS_{wall}/RS_{rigid}$ at each $c^+$')
    ax.set_xlim(CP[0] - 0.5, CP[-1] + 0.5)
    name = 'NW' if KX[a] > 5 else 'VLSM'
    ax.set_title(f'{name} cell ($k_x$ = {KX[a]:.2f}, $k_z$ = {KZ[b]:.1f}),  '
                 f'$T_{{cell}}$ = {100 * T_cell[a, b]:.2f} % of $C_f$')
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc='upper right')
save(fig, 'fig1_single_vs_weighted')

# ---------- 2. which speeds can the wall reach ----------
sc = list(M['scalars'])
ip, i1 = sc.index('p_wall'), sc.index('s1')
fig, axs = plt.subplots(1, 2, figsize=(2 * W1, H1), sharey=True)
for ax, (cn, (kx0, kz0)) in zip(axs, CELLS.items()):
    a, b = cell(kx0, kz0)
    kp = np.hypot(KX[a], KZ[b]) / Re_tau                   # |k|+ of the mode
    yc = np.interp(CP, U_bot, y_bot)                       # critical layer U(yc) = c
    st = X[a, b] * T_c[a, b]
    ax.bar(CP, st / st.max(), width=0.5, color='0.85', label='stress carried $X_c T_c$ (norm.)')
    ax.plot(CP, np.exp(-kp * yc), 'k-', lw=1.5, label=r'estimate $e^{-k^+ y_c^+}$')

    ck = CPall[a]
    inb = (ck >= CP[0]) & (ck <= CP[-1])
    pw = PPW[a, b]  # |p_wall|^2 per unit forcing, all ranks
    xpd = Xk[a, b] / cell_widths(ck) * pw  # DNS-weighted, per unit c
    ax.plot(ck[inb], pw[inb] / np.nanmax(pw[inb]), 'C3-', lw=1.5,
            label=r'$|p_{wall}|^2$ (norm.)')
    ax.plot(ck[inb], xpd[inb] / np.nanmax(xpd[inb]), 'C0-', lw=1.5,
            label=r'$X_c |p_{wall}|^2$ (DNS-weighted, norm.)')
    ax.set_xlabel(r'$c^+$'); ax.set_ylim(0, 1.05); ax.set_xlim(CP[0] - 0.5, CP[-1] + 0.5)
    ax.set_title(f'{cn}: wall-flow-response estimate in each cell: $1/k^+$ = {1/kp:.0f}')
    ax.legend(loc='upper right')
#axs[0].set_ylabel('normalised')
save(fig, 'fig2_wall_coupling')

# ---------- 3. friction change per kx ----------
# ---------- 3. friction change per kx ----------
dlnk = np.diff(np.log(cell_edges(KX)))                    # log width of each kx column
fig, ax = plt.subplots(figsize=(W1, H1))
for wn in [n for n in names if n.startswith('modal m=1e+02')]:
    ax.semilogx(KX, dCf_kx[:, names.index(wn)] / dlnk, '.-', color=ZCOL[mz(wn)[1]], label=wn)
ax.axhline(0, color='k', lw=0.6)
if 'omega_t' in M.files:
    OMT = float(M['omega_t'])
    ax.axvspan(OMT * Re_tau / 15, OMT * Re_tau / 10, color='0.92', zorder=0,
               label=r'$c_{res}$ on stress-carrying speeds (10-15)')
    sec = ax.secondary_xaxis('top', functions=(lambda k: OMT * Re_tau / np.maximum(k, 1e-9),
                                               lambda c: OMT * Re_tau / np.maximum(c, 1e-9)))
    sec.set_xlabel(r'$c^+_{res} = \Omega^+ Re_\tau / k_x$')
ax.set_xlabel(r'$k_x$'); ax.set_ylabel(r'$d\Delta C_f / d\ln k_x$ [%]')
ax.set_title('friction change by streamwise scale (negative = reduction)', pad=28)
ax.legend()
save(fig, 'fig3_dCf_per_kx')


# ---------- 4. friction and sensitivity per frequency (per decade, smoothed, split by kx) ----------
lg = np.log10(np.where(om > 0, om, np.nan))               # log10 omega+ of every point
xg = np.linspace(-3.7, 0.5, 300)                          # plotting grid in log10 omega
disc = np.broadcast_to((KX <= 2)[:, None, None], Wp.shape)   # one DNS mode per cell (linear kx grid)
SIG = 0.1                                                 # smoothing width [decades]

def per_decade(q, mask):                                  # Gaussian-smoothed density per decade
    ok = mask & np.isfinite(lg)
    x, wq = lg[ok], q[ok]
    k = np.exp(-0.5 * ((xg[:, None] - x[None, :]) / SIG)**2) / (SIG * np.sqrt(2 * np.pi))
    return k @ wq

T0 = np.sum(Wp * RS0)
og = 10**xg
fig, axs = plt.subplots(2, 1, figsize=(W1, 1.6 * H1), sharex=True)
dxg = xg[1] - xg[0]
dxg = xg[1] - xg[0]
for mask, ls, lab in [(~disc, '-', r'$k_x > 2$'), (disc, '--', r'$k_x \leq 2$ (single DNS modes)')]:
    f0 = per_decade(Wp * RS0, mask) / T0
    axs[0].semilogx(og, f0, 'k', ls=ls, label=f'{lab}: {100*f0.sum()*dxg:.0f} % of friction')
    axs[1].semilogx(og, 100 * per_decade(Wp * gr, mask), color='C3', ls=ls)
    axs[1].semilogx(og, 100 * per_decade(Wp * gi, mask), color='C2', ls=ls)
axs[0].set_ylabel('friction per decade (per factor of 10 in omega) \n(fraction of total)'); axs[0].legend()
axs[0].set_title(r'where the friction sits in $\omega^+$, and how a wall changes it')
axs[1].plot([], [], 'C3', label=r'$G_r$: per unit Re $Y$')
axs[1].plot([], [], 'C2', label=r'$G_i$: per unit Im $Y$')
axs[1].axhline(0, color='k', lw=0.6)
axs[1].set_xlabel(r'$\omega^+$')
axs[1].set_ylabel(r'$\partial\Delta C_f/\partial Y$' + '\n[% per unit Y per decade]')
axs[1].text(0.01, 0.04, r'$G_i<0$: spring (Im $Y>0$) helps;   $G_r>0$: damper (Re $Y<0$, passive) helps',
            transform=axs[1].transAxes, fontsize=8)
for axx in axs:
    axx.axvspan(*BAND, color='0.94', zorder=0)
    if 'omega_t' in M.files:
        axx.axvline(float(M['omega_t']), color='r', ls=':', lw=1)
axs[1].legend()
save(fig, 'fig4_sensitivity_omega')


# ---------- 5. design map per unit peak admittance ----------
fig, ax = plt.subplots(figsize=(1.5*W1, H1))
for z, s in zip(ZE2, S):
    ax.semilogx(OM2, 100 * s, color=ZCOL[z], label=f'single mode, $\\zeta$ = {z:g}')
ax.axhline(100 * S_const, color='k', ls='--', lw=1,
           label=f'constant spring ({100 * S_const:.1f} %)')
ax.axhline(100 * opt_pass / EPS, color='k', ls=':', lw=1,
           label=f'best passive $Y(\\omega)$ ({100 * opt_pass / EPS:.1f} %)')
ax.axhline(100 * opt_free / EPS, color='k', ls='-.', lw=1,
           label=f'best $Y(\\omega)$ ({100 * opt_free / EPS:.1f} %)')
ax.plot(OM2[io_], 100 * S[iz_, io_], 'k*', ms=10, zorder=5,
        label=f'best single mode: $\\Omega^+$ = {OM2[io_]:.4f}, $\\zeta$ = {ZE2[iz_]:g} '
              f'({100 * S[iz_, io_]:.1f} %)')
# arrow showing the gap between the best single mode and the passive bound
ax.annotate('', xy=(OM2[io_], 100 * opt_pass / EPS), xytext=(OM2[io_], 100 * S[iz_, io_]),
            arrowprops=dict(arrowstyle='<->', color='0.4', lw=1))
ax.text(OM2[io_] * 1.15, 50 * (S[iz_, io_] + opt_pass / EPS),
        f'{100 * S[iz_, io_] / (opt_pass / EPS):.0f} % of\npassive bound',
        fontsize=8, color='0.3', va='center')
ax.axhline(0, color='k', lw=0.6)
ax.axvspan(*BAND, color='0.94', zorder=0)
ax.set_xlabel(r'resonance $\Omega^+$');
ax.set_ylabel(r'$\Delta C_f$ per unit $|Y|_{max}$ [%]')
ax.set_title('single-mode walls vs the best possible admittance')
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8, frameon=False)
save(fig, 'fig5_design_map')

# ---------- 6. bounds: what each level of freedom is worth ----------
labs = ['massless\nspring', 'best single\nmode', 'constant\nspring', 'best passive\n$Y(\\omega)$',
        'best\n$Y(\\omega)$', 'best passive\n$Y(k_x,k_z,\\omega)$', 'best\n$Y(k_x,k_z,\\omega)$']
vals = 100 * np.array([S_spring, S.min(), S_const, opt_pass / EPS, opt_free / EPS,
                       b_point_pass / EPS, b_point / EPS])
cols = ['0.6', '0.6', '0.6', 'C0', 'C0', 'C1', 'C1']
active = [False, False, False, False, True, False, True]        # Re Y > 0 allowed

fig, ax = plt.subplots(figsize=(W1, H1))
bars = ax.bar(range(len(vals)), vals, color=cols, edgecolor='k', lw=0.6,
              hatch=['//' if a else None for a in active])
ax.bar_label(bars, labels=[f'{v:.1f}' if abs(v) >= 0.05 else '≈ 0' for v in vals],
             fontsize=12, padding=2)
ax.plot(0, vals[0], 'k_', ms=20, mew=2)                           # makes the ~0 bar visible
ax.set_xticks(range(len(vals))); ax.set_xticklabels(labs, fontsize=7)
ax.axhline(0, color='k', lw=0.6)
ax.set_ylabel(r'$\Delta C_f$ per unit $|Y|_{max}$ [%]')
ax.set_title('friction change per unit admittance (negative = reduction)')
ax.legend(handles=[mpatches.Patch(color='0.6', label='realisable wall shapes'),
                   mpatches.Patch(color='C0', label=r'bound: one $Y(\omega)$ for all scales (local wall)'),
                   mpatches.Patch(color='C1', label=r'bound: $Y$ chosen per $(k_x,k_z,\omega)$ (scale-selective wall)'),
                   mpatches.Patch(facecolor='w', edgecolor='k', hatch='//', label=r'active (Re $Y$ > 0 allowed)')],
          loc='lower left', fontsize=8, frameon=False)
save(fig, 'fig6_bounds')

# ---------- 7. linear vs full: where nonlinearity sets in ----------
if 'omega_t' in M.files:
    from matplotlib.lines import Line2D
    Y_SKIP_SWEEP = 1e-4                                    # the sweep copied rigid where |Y+| < this
    OM7 = float(M['omega_t'])
    yl, full, lin, zc = [], [], [], []
    for i in im:
        m_, z_ = mz(names[i])
        Yv = Y_modal(om_kc, OM7, m_, z_)
        yl.append(Y_peak(OM7, z_) / m_)
        full.append(100 * dCf[i])
        lin.append(100 * screen(np.where(np.abs(Yv) < Y_SKIP_SWEEP, 0, Yv)))   # same skip as the sweep
        zc.append(ZCOL[z_])
    yl, full, lin = map(np.array, (yl, full, lin))

    fig, axs = plt.subplots(1, 2, figsize=(2 * W1, H1))
    for ax in axs:
        ax.axvspan(yl.min() / 2, 0.05, color='0.93', zorder=0)
        ax.set_xscale('log'); ax.set_xlabel(r'$|Y|_{max}$')
    axs[0].scatter(yl, full, c=zc, marker='o')
    axs[0].scatter(yl, lin, c=zc, marker='x')
    axs[0].set_yscale('log')                               # all these walls increase friction
    axs[0].set_ylabel(r'$\Delta C_f$ [%]')
    axs[0].set_title(f'modal walls at $\\Omega^+$ = {OM7:g}')
    axs[0].text(0.03, 0.95, 'linear region', transform=axs[0].transAxes, fontsize=8, va='top', color='0.4')

    axs[1].scatter(yl, full / lin, c=zc)
    axs[1].axhline(1, color='k', lw=0.6)
    axs[1].set_ylabel('full / linear')
    axs[1].set_title('full / linear')

    hz = [Line2D([], [], ls='', marker='o', color=c, label=f'$\\zeta$ = {z:g}')
          for z, c in ZCOL.items() if z in M['zeta']]
    hm = [Line2D([], [], ls='', marker='o', color='k', label='full sweep'),
          Line2D([], [], ls='', marker='x', color='k', label='linear screening')]
    axs[0].legend(handles=hz + hm, fontsize=8, loc='lower right')
    axs[1].legend(handles=hz, fontsize=8, loc='upper left')
    save(fig, 'fig7_linear_vs_full')

print('\n[C7c] linear screening with the same skip as the sweep  (Y_SKIP = 1e-4)')
OMt = float(M['omega_t']) if 'omega_t' in M.files else 0.12
Y_SKIP = 1e-4
for z_ in sorted({mz(names[i])[1] for i in im}):
    for i in sorted([i for i in im if mz(names[i])[1] == z_], key=lambda i: -mz(names[i])[0]):
        m_ = mz(names[i])[0]
        Y_ = Y_modal(om_kc, OMt, m_, z_)
        lin = screen(Y_)
        lin_s = screen(np.where(np.abs(Y_) < Y_SKIP, 0, Y_))
        print(f'     zeta {z_:5g}  m {m_:8.0e}  |Y|max {1/(2*z_*OMt*m_):8.1e}   full/lin {dCf[i]/lin:5.2f}'
              f'   full/lin_skip {dCf[i]/lin_s:5.2f}   (skip changes lin by {100*(lin_s-lin):+.1e} %)')

for n in ['SR 0 cyc', 'SR 3 cyc']:
    i = names.index(n)
    same = (np.nan_to_num(RS[..., i]) == RS0) & (Wp > 0)
    print(f'{n}: {same.sum()/(Wp > 0).sum():.3f} of points left rigid (|Y| < Y_SKIP everywhere if 1.000)')


plt.show()

# ============================ O. CHECKS ============================
print('\n======================== CHECKS ========================')
WCHK = [n for n in ['modal m=1e+02 z=1e-02', 'modal m=1e+02 z=3e-02', 'dY re', 'dY im'] if n in names]
iW = [names.index(n) for n in WCHK]
cc = np.broadcast_to(CPall[:, None, :], Xk.shape)          # c+ of every (kx, kz, c) point
KXg, KZg = np.meshgrid(KX, KZ, indexing='ij')
dc0 = cell_widths(CP)
F = Wp * RS0                                               # friction carried by every point (sums to sum T_cell)
passive_ok = lambda g: -PASSIVE_RE * g > 0                 # optimum Re Y ~ -g lies on the passive side

def wq(v, wt, q):                                          # weighted quantile
    v, wt = np.ravel(v), np.ravel(wt)
    ok = np.isfinite(v) & (wt > 0)
    o = np.argsort(v[ok])
    cw = np.cumsum(wt[ok][o]) / wt[ok].sum()
    return np.interp(q, cw, v[ok][o])

def regrid(Xf):                                            # base c grid -> per-kx c grid (same as K)
    out = np.zeros(Xk.shape)
    for a in range(nkx):
        if np.isnan(CPall[a]).any():
            continue
        dck = cell_widths(CPall[a])
        for b in range(nkz):
            out[a, b] = np.interp(CPall[a], CP, Xf[a, b] / dc0, left=0, right=0) * dck
    return out

def dcf_from_X(Xf):                                        # forcing -> dCf of every wall (same as K)
    num_ = np.sum(regrid(Xf)[..., None] * RS, axis=2)
    with np.errstate(invalid='ignore', divide='ignore'):
        J_ = np.nan_to_num(num_ / num_[..., :1], nan=1.0)
    return np.sum(T_cell[..., None] * (J_ - 1), axis=(0, 1))

def wp_from_Xk(Xk_):                                       # point weights W_p for a given forcing (same as L)
    den_ = np.sum(Xk_ * RS[..., 0], axis=2)
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.nan_to_num(T_cell[..., None] * Xk_ / den_[..., None])

def refit():                                               # refit all cells with the current ALPHA and Rp
    X_, r_ = np.zeros_like(X), np.full_like(res, np.nan)
    for a in range(nkx):
        for b in range(nkz):
            X_[a, b], r_[a, b], _ = fit_cell(a, b)
    return X_, r_

e45 = np.logspace(-4, 0.5, 46); oc45 = np.sqrt(e45[1:] * e45[:-1]); dl45 = np.diff(np.log10(e45))
h45 = lambda q: np.histogram(om.ravel(), bins=e45, weights=np.nan_to_num(q).ravel())[0] / dl45   # per decade

hdr = ''.join(f'{n[-16:]:>18s}' for n in WCHK)
row = lambda d_: ''.join(f'{100*d_[i]:+18.4f}' for i in iW)

# ---------- C1. bookkeeping ----------
Ub = abs(np.trapezoid(M['U'], y)) / 2
print(f'\n[C1] FIK closure: 3Ub+/Re_tau + T_tot = {3*Ub/Re_tau:.3f} + {T_tot:.3f} = {3*Ub/Re_tau + T_tot:.3f}  (expect ~1)')
print(f'     friction inside the sweep box: sum T_cell / T_tot = {T_cell.sum()/T_tot:.3f}')
print(f'     sum Wp*RS0 = {F.sum():.4f}  vs  sum T_cell = {T_cell.sum():.4f}')
print(f'     rebuilt pipeline reproduces dCf: max |diff| = {np.abs(dcf_from_X(X) - dCf).max():.1e}')

# ---------- C2. fit quality ----------
okc = np.isfinite(res) & (T_cell > 0)
with np.errstate(invalid='ignore', divide='ignore'):
    clos = np.sum(X * T_c, axis=2) / T_cell                # per cell: sum X_c T_c / T_cell
print('\n[C2] fit misfit, friction-weighted:  '
      f'median {wq(res[okc], T_cell[okc], 0.5):.3f},  90 % {wq(res[okc], T_cell[okc], 0.9):.3f}')
for thr in [0.1, 0.2, 0.5]:
    print(f'     friction share in cells with misfit > {thr}: {T_cell[okc & (res > thr)].sum()/T_cell.sum():.3f}')
print(f'     stress closure sum X_c T_c / T_cell, friction-weighted: 10 % {wq(clos[okc], T_cell[okc], 0.1):.3f}, '
      f'median {wq(clos[okc], T_cell[okc], 0.5):.3f}, 90 % {wq(clos[okc], T_cell[okc], 0.9):.3f}')

LXg, LZg = 2*np.pi*Re_tau/KXg, 2*np.pi*Re_tau/KZg
fig, axs = plt.subplots(1, 2, figsize=(2*W1, H1))
for ax, q, t, vr in [(axs[0], res, 'fit misfit', (0, 0.5)), (axs[1], clos, r'$\sum X_c T_c / T_{cell}$', (0.5, 1.5))]:
    pc = ax.pcolormesh(LXg, LZg, np.where(T_cell > 1e-3*T_cell.max(), q, np.nan), shading='nearest',
                       vmin=vr[0], vmax=vr[1], cmap='viridis')
    ax.contour(LXg, LZg, T_cell / T_cell.max(), levels=[0.1, 0.5], colors='w', linewidths=0.8)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel(r'$\lambda_x^+$'); ax.set_ylabel(r'$\lambda_z^+$')
    ax.set_title(t + ' (white: 10 %, 50 % of max $T_{cell}$)'); fig.colorbar(pc, ax=ax)
fig.tight_layout()

# ---------- C3. smoothing (ALPHA) sensitivity: net dCf and the shape of G_r, G_i ----------
print(f'\n[C3] ALPHA sensitivity (dCf in %)\n{"ALPHA":>7s}{"misfit":>8s}{"XT/T":>8s}{"c<5":>7s}' + hdr)
fig, axs = plt.subplots(2, 1, figsize=(W1, 1.6*H1), sharex=True)
ALPHA0 = ALPHA
for ALPHA, ls in [(0.01, ':'), (0.05, '-'), (0.2, '--'), (1.0, '-.'), (0.0, None)]:   # fit_cell reads the global ALPHA
    X_, r_ = refit()
    ok_ = np.isfinite(r_)
    XT_ = X_ * T_c
    print(f'{ALPHA:7.2f}{np.sum(w[ok_]*r_[ok_])/np.sum(w[ok_]):8.3f}{XT_.sum()/T_cell.sum():8.3f}'
          f'{XT_[..., CP < 5].sum()/XT_.sum():7.3f}' + row(dcf_from_X(X_)))
    if ls is None:                                         # alpha = 0 is unstable, not plotted
        continue
    Wp_ = wp_from_Xk(regrid(X_))
    axs[0].semilogx(oc45, 100*h45(Wp_ * gr), color='C3', ls=ls, label=rf'$\alpha$ = {ALPHA:g}')
    axs[1].semilogx(oc45, 100*h45(Wp_ * gi), color='C2', ls=ls, label=rf'$\alpha$ = {ALPHA:g}')
ALPHA = ALPHA0
for ax, t in zip(axs, [r'$G_r$ [% per unit Re $Y$ per decade]', r'$G_i$ [% per unit Im $Y$ per decade]']):
    ax.axhline(0, color='k', lw=0.6); ax.set_ylabel(t); ax.legend(fontsize=8)
axs[0].set_title(r'does the shape of $G_r(\omega)$, $G_i(\omega)$ survive the smoothing choice?')
axs[1].set_xlabel(r'$\omega^+$'); fig.tight_layout()

# ---------- C4. low-c forcing ----------
lo = cc < 5
pw = np.nan_to_num(Xk * PPW)
print(f'\n[C4] share of friction from c < 5: {F[lo].sum()/F.sum():.3f}')
print(f'     model wall pressure: mean speed {np.sum(pw*cc)/pw.sum():.1f},  share from c < 5: {pw[lo].sum()/pw.sum():.3f}'
      f'   (DNS wall-pressure convection speed ~ 10-13)')
print(f'     p_rms+ model = {np.sqrt(np.nansum(Xk*PPW)):.2f}  (DNS ~2.8-3.0)')
for i in iW:
    dp = Wp * (np.nan_to_num(RS[..., i]) - RS0)
    print(f'     {names[i]:24s} dCf from c < 5: {100*dp[lo].sum():+.4f} %  of total {100*dp.sum():+.4f} %')
Rp_bak = Rp
Rp = Rp_bak.copy(); Rp[:, :, CP < 5] = np.nan              # refit without the c < 5 profiles
X5, r5 = refit()
Rp = Rp_bak
ok5 = np.isfinite(r5)
print(f'     refit without c < 5: misfit {np.sum(w[ok5]*r5[ok5])/np.sum(w[ok5]):.3f} '
      f'(was {np.sum(w[okc]*res[okc])/np.sum(w[okc]):.3f})\n{"":13s}' + hdr)
print(f'{"   with c<5":13s}' + row(dCf) + f'\n{"   without":13s}' + row(dcf_from_X(X5)))

# ---------- C5. resolution of the Y(omega) bounds ----------
inr = (om >= 1e-4) & (om <= 10**0.5)
g_abs_ = np.hypot(gr, gi)
bp_free = -100 * np.sum(Wp * g_abs_)                       # per-point (scale-selective) limits, % per unit |Y|
bp_pass = -100 * np.sum(Wp * np.where(passive_ok(gr), g_abs_, np.abs(gi)))
print(f'\n[C5] friction weight outside the omega histogram range: {F[~inr].sum()/F.sum():.4f}')
print(f'{"bins":>6s}{"dlog10w":>9s}{"zeta_eq":>9s}{"free":>9s}{"passive":>9s}   (% per unit |Y|; section L uses 45)')
NB = [6, 11, 23, 45, 90, 180, 360, 720]
bnd = []
for nb in NB:
    e = np.logspace(-4, 0.5, nb + 1)
    h = lambda q: np.histogram(om.ravel(), bins=e, weights=q.ravel())[0]
    Gr_, Gi_ = h(Wp * gr), h(Wp * gi)
    G_ = np.hypot(Gr_, Gi_)
    d_ = 4.5 / nb
    bnd.append((d_, -100*G_.sum(), -100*np.where(passive_ok(Gr_), G_, np.abs(Gi_)).sum()))
    print(f'{nb:6d}{d_:9.3f}{(10**d_ - 1)/2:9.3f}{bnd[-1][1]:9.2f}{bnd[-1][2]:9.2f}')
print(f'{"point":>6s}{"":18s}{bp_free:9.2f}{bp_pass:9.2f}   (Y chosen per (kx, kz, c))')
bnd = np.array(bnd)
fig, ax = plt.subplots(figsize=(W1, H1))
ax.semilogx(bnd[:, 0], bnd[:, 1], 'ko-', label=r'best $Y(\omega)$')
ax.semilogx(bnd[:, 0], bnd[:, 2], 'ks--', mfc='w', label=r'best passive $Y(\omega)$')
ax.axhline(bp_free, color='C1', lw=1, label=r'per point $Y(k_x,k_z,\omega)$')
ax.axhline(bp_pass, color='C1', ls='--', lw=1, label=r'per point, passive')
ax.axvline(0.1, color='0.6', ls=':', lw=1)
ax.set_xlabel(r'bin width $\Delta\log_{10}\omega$ (how sharply $Y$ may vary)')
ax.set_ylabel(r'$\Delta C_f$ per unit $|Y|$ [%]')
sec = ax.secondary_xaxis('top', functions=(lambda d: (10**d - 1)/2, lambda z: np.log10(1 + 2*np.maximum(z, 1e-9))))
sec.set_xlabel(r'equivalent damping ratio $\zeta \approx \Delta\omega/2\omega$')
ax.set_title('local-wall bound depends on how sharp Y may be', pad=28); ax.legend(fontsize=8)

plt.show()

# ---------- C6. sign conventions ----------
OMt = float(M['omega_t']) if 'omega_t' in M.files else 0.12
wt = np.logspace(-4, 1, 4000)
print('\n[C6] modal wall convention (m_eff = 1):')
for z in [0.01, 0.1, 1.0]:
    Yt = Y_modal(wt, OMt, 1.0, z)
    print(f'     zeta {z:4g}: max Re Y = {Yt.real.max():+.2e} (passive -> <= 0),  '
          f'Im Y at 0.1*Omega {Y_modal(np.array([0.1*OMt]), OMt, 1.0, z).imag[0]:+.2e} (spring -> > 0),  '
          f'Im Y at 10*Omega {Y_modal(np.array([10*OMt]), OMt, 1.0, z).imag[0]:+.2e} (mass -> < 0)')
psign = np.sign(Y_modal(np.array([OMt]), OMt, 1.0, 0.1).real[0])
print(f'     passive side is Re Y {"<" if psign < 0 else ">"} 0,  PASSIVE_RE = {PASSIVE_RE:+}  ->  '
      f'{"OK" if PASSIVE_RE == psign else "MISMATCH: PASSIVE_RE must be the sign of passive Re Y"}')
print(f'     passive_ok picks Gr > 0 (damper helps):  {"OK" if passive_ok(1.0) and not passive_ok(-1.0) else "WRONG"}')
print('     linear screening with Y and with conj(Y) vs full  (dCf %):')
for i in im:
    m_, z_ = mz(names[i])
    Y_ = Y_modal(om_kc, OMt, m_, z_)
    print(f'     {names[i]:24s} full {100*dCf[i]:+.4f}   Y {100*screen(Y_):+.4f}   conj(Y) {100*screen(np.conj(Y_)):+.4f}')

# ---------- C7. linearity and precision floor of the full sweep ----------
print('\n[C7] full dCf * m_eff should be constant in m at small |Y| (the linear one is, by construction)')
for z_ in sorted({mz(names[i])[1] for i in im}):
    for i in sorted([i for i in im if mz(names[i])[1] == z_], key=lambda i: -mz(names[i])[0]):
        m_ = mz(names[i])[0]
        lin = screen(Y_modal(om_kc, OMt, m_, z_))
        print(f'     zeta {z_:5g}  m {m_:8.0e}  |Y|max {1/(2*z_*OMt*m_):8.1e}   full*m {100*dCf[i]*m_:+10.4f}'
              f'   lin*m {100*lin*m_:+10.4f}   ratio {dCf[i]/lin:6.2f}')

i0 = max(im, key=lambda i: np.prod(mz(names[i])))          # smallest |Y|max = largest m*zeta
m0, z0 = mz(names[i0])
Y0 = np.nan_to_num(Y_modal(om_kc, OMt, m0, z0))[:, None, :]
full_pt = Wp * (np.nan_to_num(RS[..., i0]) - RS0)
lin_pt = np.nan_to_num(Wp * (gr * Y0.real + gi * Y0.imag))
with np.errstate(invalid='ignore', divide='ignore'):
    near = np.abs(np.log(om / OMt)) < np.log(1.5)          # within a factor 1.5 of the resonance
print(f'     per point, {names[i0]}: full {100*full_pt.sum():+.2e} %, linear {100*lin_pt.sum():+.2e} %;  '
      f'near resonance: full {100*full_pt[near].sum():+.2e}, linear {100*lin_pt[near].sum():+.2e}')

f0 = next(SWEEP / f'kx_{a:03d}.npz' for a in range(nkx) if (SWEEP / f'kx_{a:03d}.npz').exists())
d_ = np.load(f0)
print(f'     stored dtypes ({f0.name}): S {d_["S"].dtype}, P {d_["P"].dtype}  '
      f'(float32 resolves relative changes down to ~1e-7)')
dRS = np.abs(np.nan_to_num(RS[..., i0]) - RS0)
far = ~near & (Wp > 0)
Yabs = np.broadcast_to(np.abs(Y0), Wp.shape)
with np.errstate(invalid='ignore', divide='ignore'):
    rel = dRS / np.abs(RS0)
    rel_lin = np.abs(gr * Y0.real + gi * Y0.imag) / np.abs(RS0)   # relative change the linear model expects
print(f'     off-resonance points: {far.sum()},  exactly unchanged in the full sweep: {np.mean(dRS[far] == 0):.2f}')
print(f'     off resonance, median |Y| {np.median(Yabs[far]):.1e},  median relative change: '
      f'full {np.nanmedian(rel[far]):.1e}, linear expects {np.nanmedian(rel_lin[far]):.1e}')
print(f'     off-resonance share of the linear dCf: {lin_pt[far].sum()/lin_pt.sum():+.2f},  of the full dCf: '
      f'{full_pt[far].sum()/full_pt.sum():+.2f}')

fig, ax = plt.subplots(figsize=(W1, H1))
ax.semilogx(oc45, 100*h45(full_pt), 'C0.-', label='full sweep')
ax.semilogx(oc45, 100*h45(lin_pt), 'C3x--', label='linear screening')
ax.axvspan(OMt/1.5, OMt*1.5, color='0.92', zorder=0, label='near resonance (x1.5)')
ax.axvline(OMt, color='r', ls=':', lw=1); ax.axhline(0, color='k', lw=0.6)
ax.set_xlabel(r'$\omega^+$'); ax.set_ylabel(r'$\Delta C_f$ per decade [%]')
ax.set_title(f'where full and linear differ: {names[i0]}'); ax.legend(fontsize=8)

# ---------- C8. box-scale column ----------
print(f'\n[C8] first kx column (kx = {KX[0]:.2f}, lambda_x+ = {2*np.pi*Re_tau/KX[0]:.0f}): '
      f'{T_cell[0].sum()/T_cell.sum():.3f} of in-box friction')
for i in iW:
    print(f'     {names[i]:24s} dCf from that column {dCf_kx[0, i]:+.4f} %  of {100*dCf[i]:+.4f} %')

# ---------- C9. where stress and wall pressure sit in c (attached-eddy scaling) ----------
c_st = CP[np.argmax(X * T_c, axis=2)]                      # stress peak speed per cell
inb = (CPall >= CP[0]) & (CPall <= CP[-1])
PPm = np.where(np.isfinite(PPW) & inb[:, None, :], PPW, -np.inf)
c_pr = CPall[np.arange(nkx)[:, None], np.argmax(PPm, axis=2)]   # wall-pressure peak speed per unit forcing
has = np.isfinite(PPm).any(axis=2) & (T_cell > 0)
ky = np.hypot(KXg, KZg) / Re_tau * np.interp(c_st, U_bot, y_bot)
print(f'\n[C9] k+ y_c+ at the stress peak (friction-weighted): 10 % {wq(ky[has], T_cell[has], 0.1):.2f}, '
      f'median {wq(ky[has], T_cell[has], 0.5):.2f}, 90 % {wq(ky[has], T_cell[has], 0.9):.2f}  (attached eddies: O(1))')
dcp = (c_pr - c_st)[has]
print(f'     pressure peak - stress peak speed: 10 % {wq(dcp, T_cell[has], 0.1):+.1f}, '
      f'median {wq(dcp, T_cell[has], 0.5):+.1f}, 90 % {wq(dcp, T_cell[has], 0.9):+.1f}  (figs 2: -2 to -3)')
fig, ax = plt.subplots(figsize=(W1, H1))
ax.hist(np.log10(ky[has]), bins=40, weights=T_cell[has] / T_cell[has].sum(), color='0.6')
ax.set_xlabel(r'$\log_{10}(k^+ y_c^+)$ at the stress peak'); ax.set_ylabel('friction share')
ax.set_title('wall-normal position of the stress vs scale')

k0 = kx_d == 0
uv0 = np.concatenate([[0.0], E[:, iuv][:, k0].sum(axis=(1, 2))])
T0 = 3 / Re_tau * np.trapezoid(fik * -uv0, yw)
print(f'kx = 0 modes: {T0:.4f} of Cf  ({T0/T_tot:.3f} of turbulent friction)')

# ============================ P. CHECK SUMMARY FIGURE ============================
fig, axs = plt.subplots(2, 3, figsize=(3 * W1 * 0.8, 2 * H1))
axs = axs.ravel()

# (a) friction budget: does everything add up to 1?
k0 = kx_d == 0
T0 = 3 / Re_tau * np.trapezoid(fik * -np.concatenate([[0.0], E[:, iuv][:, k0].sum(axis=(1, 2))]), yw)
parts = [('laminar $3U_b^+/Re_\\tau$', 3 * Ub / Re_tau), ('in sweep cells', T_cell.sum()),
         ('$k_x$ = 0 modes', T0), ('rest (net)', T_tot - T_cell.sum() - T0)]
labs_a = [p[0] for p in parts] + ['total']
vals_a = [p[1] for p in parts] + [sum(p[1] for p in parts)]
axs[0].barh(labs_a[::-1], vals_a[::-1], color=['k', '0.6', '0.6', 'C0', '0.6'])
for i, v in enumerate(vals_a[::-1]):
    axs[0].text(max(v, 0) + 0.01, i, f'{v:.3f}', va='center', fontsize=8)
axs[0].axvline(1, color='r', ls=':', lw=1)
axs[0].set_xlim(-0.05, 1.2); axs[0].set_xlabel('fraction of $C_f$')
axs[0].set_title('(a) friction budget (FIK)')

# (b) conservation over bins: fine DNS bins vs sweep cells
kxin, kzin = Ax.sum(axis=0) > 0, Az.sum(axis=0) > 0        # DNS bins that belong to a cell
uvd = E[:, iuv][:, kxin][:, :, kzin]                       # (20, kx in cells, kz in cells)
Tint = lambda q: 3 / Re_tau * np.trapezoid(fik[:, None] * -np.vstack([np.zeros((1, q.shape[1])), q]), yw, axis=0)
last = lambda A, kd: np.array([kd[A[i] > 0].max() if A[i].any() else np.nan for i in range(A.shape[0])])
for kd, Td, A_, Tc_, col, lab in [(kx_d[kxin], Tint(uvd.sum(axis=2)), Ax[:, kxin], T_cell.sum(axis=1), 'C0', '$k_x$'),
                                  (kz_d[kzin], Tint(uvd.sum(axis=1)), Az[:, kzin], T_cell.sum(axis=0), 'C3', '$k_z$')]:
    axs[1].semilogx(kd, np.cumsum(Td), color=col, lw=1.5, label=f'DNS bins, {lab}')
    axs[1].semilogx(last(A_, kd), np.cumsum(Tc_), 'o', color=col, mfc='none', ms=5, label=f'sweep cells, {lab}')
axs[1].set_xlabel('wavenumber'); axs[1].set_ylabel('cumulative friction')
axs[1].set_title('(b) binning conserves friction'); axs[1].legend(fontsize=7)

# (c) stress profile: DNS total, inside cells, fitted model
uv_fit = np.einsum('abcp,abc->p', np.nan_to_num(Rp[:, :, :, iuv, :]), X)
axs[2].semilogx(yp, -tot[:, iuv], 'k-', lw=2, label='DNS, all scales')
axs[2].semilogx(yp, -Ec[:, iuv].sum(axis=(1, 2)), 'C0--', lw=1.5, label='DNS, inside cells')
axs[2].semilogx(yp, -uv_fit, 'C3:', lw=2, label=r' weighted resolvent $\sum X_c\,uv^R_c$')
axs[2].set_xlabel('$y^+$'); axs[2].set_ylabel(r'$-\overline{uv}^+$')
axs[2].set_title('(c) the fit reproduces the stress'); axs[2].legend(fontsize=7)

# (d) per-cell closure
XTc = np.sum(X * T_c, axis=2)
mk = T_cell > 1e-6
#axs[3].scatter(T_cell[mk], XTc[mk], c='k', s=6)
sc = axs[3].scatter(T_cell[mk], XTc[mk], c=res[mk], s=6, vmin=0, vmax=0.5, cmap='viridis')
lim = [T_cell[mk].min(), T_cell[mk].max()]
axs[3].plot(lim, lim, 'k-', lw=0.8)
axs[3].set_xscale('log'); axs[3].set_yscale('log')
axs[3].set_xlabel('$T_{cell}$ (DNS)'); axs[3].set_ylabel(r'$\sum_c X_c T_c$ (resolvent)')
axs[3].set_title('(d) friction per cell: weighted resolvent vs DNS');
fig.colorbar(sc, ax=axs[3], label='(fitted uv - DNS uv) / DNS uv')

# (e) smoothing robustness
AL = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
iA = [i for i, n in enumerate(names) if n.startswith('modal m=1e+02')] + [names.index('dY im')]
ALPHA0, D = ALPHA, []
for ALPHA in AL:                                           # fit_cell reads the global ALPHA
    D.append(dcf_from_X(refit()[0])[iA])
ALPHA = ALPHA0
D = np.array(D) / np.array(D)[AL.index(0.05)]
for k_, i in enumerate(iA):
    axs[4].semilogx(AL, D[:, k_], 'o-', ms=3, label=names[i])
axs[4].axhspan(0.85, 1.15, color='0.93', zorder=0)
axs[4].axvline(0.05, color='k', ls=':', lw=1)
axs[4].set_xticks(AL); axs[4].set_xticklabels([f'{a:g}' for a in AL], fontsize=7)
axs[4].set_xlabel(r'smoothing $\alpha$'); axs[4].set_ylabel(r'$\Delta C_f / \Delta C_f$')
axs[4].set_ylabel(r'$\Delta C_f / \Delta C_f(\alpha = 0.05)$')
axs[4].set_title('(e) results vs smoothing (band: ±15 %)'); axs[4].legend(fontsize=7)

# (f) where the model wall pressure travels
pw = np.nan_to_num(Xk * PPW)
axs[5].hist(cc.ravel(), bins=np.arange(0, 30, 1), weights=pw.ravel() / pw.sum(), color='0.6')
axs[5].axvspan(10, 13, color='C2', alpha=0.25, label='probable DNS convection speed range (maybe from literature)')
axs[5].axvline(np.sum(pw * cc) / pw.sum(), color='k', lw=1.5, label=f'weighted resolvent mean {np.sum(pw*cc)/pw.sum():.1f}')
axs[5].set_xlabel('$c^+$'); axs[5].set_ylabel('share of wall pressure')
axs[5].set_title('(f) wall-pressure variance by phase speed'); axs[5].legend(fontsize=7)

fig.tight_layout()
save(fig, 'DNS weighting checks')
plt.show()

Tq = lambda q: 3 / Re_tau * np.trapezoid(fik * -np.concatenate([[0.0], q]), yw)
T_kept = Tq(E[:, iuv].sum(axis=(1, 2)))                           # all kept DNS bins
T_kz0 = Tq(E[:, iuv][:, kx_d > 0][:, :, kz_d == 0].sum(axis=(1, 2)))
print(f'kz = 0 modes (kx > 0): {T_kz0:+.4f}   cropped small scales: {T_tot - T_kept:+.4f}   '
      f'check: {T_cell.sum() + T0 + T_kz0 + (T_tot - T_kept):.4f} vs T_tot {T_tot:.4f}')