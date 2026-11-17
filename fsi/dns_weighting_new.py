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
opt_pass = -EPS * np.where(PASSIVE_RE * Gr > 0, G, np.abs(Gi)).sum()   # Re Y restricted to the passive side
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
b_point_pass = -EPS * np.sum(Wp * np.where(PASSIVE_RE * gr > 0, g_abs, np.abs(gi)))
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
    ax.set_xlabel(r'$c^+$'); ax.set_ylabel(r'$X_c T_c$ / max')
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
    ax2.set_ylabel(r'$RS_{wall}/RS_{rigid}$ per speed')
    ax.set_xlim(CP[0] - 0.5, CP[-1] + 0.5)
    ax.set_title(f'{cn} cell ($k_x$ = {KX[a]:.2f}, $k_z$ = {KZ[b]:.1f})')
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc='upper right')
save(fig, 'fig1_single_vs_weighted')

# ---------- 2. which speeds can the wall reach ----------
fig, axs = plt.subplots(1, 2, figsize=(2 * W1, H1), sharey=True)
for ax, (cn, (kx0, kz0)) in zip(axs, CELLS.items()):
    a, b = cell(kx0, kz0)
    kp = np.hypot(KX[a], KZ[b]) / Re_tau                   # |k|+ of the mode
    yc = np.interp(CP, U_bot, y_bot)                       # critical layer U(yc) = c
    st = X[a, b] * T_c[a, b]
    ax.bar(CP, st / st.max(), width=0.5, color='0.85', label='stress carried (norm.)')
    ax.plot(CP, np.exp(-kp * yc), 'k-', lw=1.5, label=r'wall coupling $e^{-k^+ y_c^+}$')
    ax.set_xlabel(r'$c^+$'); ax.set_ylim(0, 1.05)
    ax.set_title(f'{cn}: $1/k^+$ = {1/kp:.0f}')
    ax.legend(loc='upper right')
axs[0].set_ylabel('normalised')
save(fig, 'fig2_wall_coupling')

# ---------- 3. friction change per kx ----------
fig, ax = plt.subplots(figsize=(W1, H1))
for wn in [n for n in names if n.startswith('modal m=1e+02')]:
    ax.semilogx(KX, dCf_kx[:, names.index(wn)], '.-', color=ZCOL[mz(wn)[1]], label=wn)
ax.axhline(0, color='k', lw=0.6)
if 'omega_t' in M.files:
    for c in (15, 10):
        ax.axvline(M['omega_t'] * Re_tau / c, color='r', ls=':', lw=1)
    ax.text(M['omega_t'] * Re_tau / 15, ax.get_ylim()[1], r'$c_{res}$ = 15, 10  ', color='r',
            ha='right', va='top', fontsize=8)
ax.set_xlabel(r'$k_x$'); ax.set_ylabel(r'$\Delta C_f$ per $k_x$ column [%]')
ax.set_title('friction change by streamwise scale (negative = reduction)')
ax.legend()
save(fig, 'fig3_dCf_per_kx')

# ---------- 4. friction and sensitivity per frequency (per decade, bin-independent) ----------
e4 = np.logspace(-3.7, 0.5, 29)
o4 = np.sqrt(e4[1:] * e4[:-1])
dl = np.diff(np.log10(e4))
h4 = lambda q: np.histogram(om.ravel(), bins=e4, weights=q.ravel())[0] / dl
fig, axs = plt.subplots(2, 1, figsize=(W1, 1.6 * H1), sharex=True)
axs[0].semilogx(o4, h4(Wp * RS0) / np.sum(Wp * RS0), 'k.-')
axs[0].set_ylabel('friction per decade\n(fraction of total)')
axs[0].set_title(r'where the friction sits in $\omega^+$, and how a wall changes it')
axs[1].semilogx(o4, 100 * h4(Wp * gr), '.-', color='C3', label=r'$G_r$: per unit Re $Y$')
axs[1].semilogx(o4, 100 * h4(Wp * gi), '.-', color='C2', label=r'$G_i$: per unit Im $Y$')
axs[1].axhline(0, color='k', lw=0.6)
for axx in axs:
    axx.axvspan(*BAND, color='0.94', zorder=0)
axs[1].set_xlabel(r'$\omega^+$')
axs[1].set_ylabel(r'$\partial\Delta C_f/\partial Y$' + '\n[% per unit Y per decade]')
axs[1].text(0.02, 0.05, 'passive: Re $Y$ < 0,  spring: Im $Y$ > 0', transform=axs[1].transAxes, fontsize=8)
axs[1].legend()
save(fig, 'fig4_sensitivity_omega')

# ---------- 5. design map per unit peak admittance ----------
fig, ax = plt.subplots(figsize=(W1, H1))
for z, s in zip(ZE2, S):
    ax.semilogx(OM2, 100 * s, color=ZCOL[z], label=f'single mode, $\\zeta$ = {z:g}')
ax.axhline(100 * S_const, color='C2', ls='--', lw=1, label='constant spring')
ax.axhline(100 * opt_pass / EPS, color='k', ls=':', lw=1, label=r'best passive $Y(\omega)$')
ax.axhline(100 * opt_free / EPS, color='k', ls='-.', lw=1, label=r'best $Y(\omega)$')
ax.axhline(0, color='k', lw=0.6)
ax.axvspan(*BAND, color='0.94', zorder=0)
ax.set_xlabel(r'resonance $\Omega^+$'); ax.set_ylabel(r'$\Delta C_f$ per unit $|Y|_{max}$ [%]')
ax.set_title('single-mode walls vs the best possible admittance (linear)')
ax.legend(loc='lower right', ncol=2)
save(fig, 'fig5_design_map')

# ---------- 6. bounds: what each level of freedom is worth ----------
labs = ['massless\nspring', 'best single\nmode', 'constant\nspring', 'best passive\n$Y(\\omega)$',
        'best\n$Y(\\omega)$', 'best passive\n$Y(k_x,k_z,\\omega)$', 'best\n$Y(k_x,k_z,\\omega)$']
vals = 100 * np.array([S_spring, S.min(), S_const, opt_pass / EPS, opt_free / EPS,
                       b_point_pass / EPS, b_point / EPS])
cols = ['0.6', '0.6', '0.6', 'C0', 'C0', 'C1', 'C1']
fig, ax = plt.subplots(figsize=(W1, H1))
bars = ax.bar(range(len(vals)), vals, color=cols)
ax.bar_label(bars, fmt='%.1f', fontsize=8, padding=2)
ax.set_xticks(range(len(vals))); ax.set_xticklabels(labs, fontsize=7)
ax.axhline(0, color='k', lw=0.6)
ax.set_ylabel(r'$\Delta C_f$ per unit $|Y|$ [%]')
ax.set_title('first-order friction change per unit admittance (negative = reduction)')
save(fig, 'fig6_bounds')

# ---------- 7. linear vs full: where nonlinearity sets in ----------
if 'omega_t' in M.files:
    yl, full, lin, zc = [], [], [], []
    for i in im:
        m_, z_ = mz(names[i])
        yl.append(Y_peak(M['omega_t'], z_) / m_)
        full.append(100 * dCf[i])
        lin.append(100 * screen(Y_modal(om_kc, M['omega_t'], m_, z_)))
        zc.append(ZCOL[z_])
    yl, full, lin = map(np.array, (yl, full, lin))
    fig, axs = plt.subplots(1, 2, figsize=(2 * W1, H1))
    axs[0].scatter(yl, full, c=zc, marker='o', label='full sweep')
    axs[0].scatter(yl, lin, c=zc, marker='x', label='linear screening')
    axs[0].set_xscale('log'); axs[0].set_yscale('symlog', linthresh=1e-4)
    axs[0].set_xlabel(r'$|Y|_{max}$'); axs[0].set_ylabel(r'$\Delta C_f$ [%]')
    axs[0].set_title(f'modal walls at $\\Omega^+$ = {float(M["omega_t"]):g}'); axs[0].legend()
    axs[1].scatter(yl, full / lin, c=zc)
    axs[1].axhline(1, color='k', lw=0.6)
    axs[1].set_xscale('log')
    axs[1].set_xlabel(r'$|Y|_{max}$'); axs[1].set_ylabel('full / linear')
    axs[1].set_title('linear screening is valid where the ratio is 1')
    for z, c in ZCOL.items():
        if z in M['zeta']:
            axs[1].scatter([], [], c=c, label=f'$\\zeta$ = {z:g}')
    axs[1].legend()
    save(fig, 'fig7_linear_vs_full')

plt.show()