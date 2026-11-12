''' DNS weighting of resolvent modes: HJ 2D spectra (Re_tau = 2000) -> sweep cells -> weights '''

import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.interpolate import interp1d
sys.path.insert(0, r'C:\Users\uqlcapp1\Simulations\DNS\code')        # folder with dns_tools.py
from dns_tools import HJ_SPECS, load_spectrum_2d, read_fortran_records
from scipy.optimize import nnls

# --------------------------------- SETTINGS ---------------------------------
Re = 2000                                                  # DNS case
ROOT = Path(__file__).resolve().parent.parent              # fsi -> Python (project root)
SWEEP = ROOT / 'results' / 'sweep'                         # resolvent sweep folder
VARS = ['uu', 'vv', 'ww', 'uv']                            # spectra to read
SHOW_CHECKS = False                                        # True: run the file exploration and spectrum plots
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
    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import gaussian_filter
    KREF = 1/50000                                         # k_ref+
    d = 0.01                                               # grid spacing in kx#, kz#
    g = np.arange(0, 4.9 + d/2, d)
    GX, GZ = np.meshgrid(g, g, indexing='ij')
    XI = np.hypot(GX, GZ)                                  # radius xi
    TH = np.arctan2(GZ, GX)                                # angle: 0 = kx axis, pi/2 = kz axis
    K = KREF * 10**XI * Re_tau                             # |k| in outer units
    KXq, KZq = K*np.cos(TH), K*np.sin(TH)                  # the real (kx, kz) of every grid point
    Phi = uu / (P['alp'] * P['bet'])                       # density on the full DNS grid
    f = RegularGridInterpolator((P['kx'], P['kz']), Phi, bounds_error=False, fill_value=0.0)
    Phi_q = f(np.stack([KXq, KZq], axis=-1))
    with np.errstate(divide='ignore', invalid='ignore'):
        D = np.log(10) * K**2 * Phi_q / XI
    D[XI == 0] = 0.0
    print('integral of D =', D.sum() * d**2, '  vs  sum of uu =', uu.sum())
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
    plt.show()


# ============================ A. SWEEP GRID ============================
M = np.load(SWEEP / 'meta.npz')                            # sweep metadata (grid, walls, y, U)
KX, KZ, CP = M['kx'], M['kz'], M['cP']                     # resolvent wavenumbers and wave speeds
print(f'sweep kx: {KX.min():.2f} ... {KX.max():.2f} ({len(KX)} values)')
print(f'sweep kz: {KZ.min():.2f} ... {KZ.max():.1f} ({len(KZ)} values)')


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
Ec = Ax @ E @ Az.T                                         # (20, 4, 37, 42): energy of each cell at each plane (energy in each sweep cell)
print('Ec', Ec.shape)
in_cells = Ec.sum(axis=(2, 3)) / np.where(tot != 0, tot, np.nan)
print('fraction of uv inside the cells, per plane:', np.round(in_cells[:, VARS.index('uv')], 3))


# ============================ F. FIK INTEGRAL PER CELL AND WEIGHTS ============================
iuv = VARS.index('uv')
yw = np.concatenate([[0.0], yp])                           # heights incl. the wall, y+ = 0
fik = 1 - yw / Re_tau                                      # FIK weight (1 - y/h)

uv_cell = Ec[:, iuv]                                       # variable 3 from Ec (20, 37, 42)
uv_cell = np.concatenate([np.zeros((1,) + uv_cell.shape[1:]), uv_cell])   # (21, 37, 42): + uv = 0 at the wall
T_cell = 3 / Re_tau * np.trapezoid(fik[:, None, None] * -uv_cell, yw, axis=0)   # integral for all scales at once integrated along y+ (37, 42)

uv_tot = np.concatenate([[0.0], tot[:, iuv]])              # full uv profile (all scales)
T_tot = 3 / Re_tau * np.trapezoid(fik * -uv_tot, yw)

w = T_cell / T_tot
print(f'T_tot = {T_tot:.4f},   sum of w over the sweep cells = {w.sum():.4f}')

uvE = E[:, iuv]                                            # (20, 653, 1308): cropped DNS uv per bin
parts = {'kx = 0':                uvE[:, 0, :].sum(axis=1),          # DNS kx index 0
         'kx = 0.25':             uvE[:, 1, :].sum(axis=1),          # DNS kx index 1
         'kz = 0, 0.667 (kx>=0.5)': uvE[:, 2:, :2].sum(axis=(1, 2))} # kz indices 0, 1, rest of kx
for name, prof in parts.items():
    Tp = 3 / Re_tau * np.trapezoid(fik * -np.concatenate([[0.0], prof]), yw)
    print(f'{name:24s}: {Tp / T_tot:.3f} of T_tot')


# ============================ G. RESOLVENT GRID -> DNS HEIGHTS ============================
y, N = M['y'], int(M['N'])
print(f'resolvent y: index 0 -> {y[0]:.3f}, index {(N-1)//2} -> {y[(N-1)//2]:.3f}, index {N-1} -> {y[-1]:.3f}')

mid = (N - 1) // 2                                         # centreline index
eta_top = 2 - y[:mid+1]                                    # top half: distance from the top wall, 0 -> 1
eta_bot = y[mid:][::-1]                                    # bottom half, flipped so it runs 0 -> 1
SIGN = np.array([1, 1, 1, -1])[:, None]                    # uv changes sign in the top half
eta_d = np.clip(yp / Re_tau, 0, 1)                         # DNS plane heights in y/h

def to_planes(Pr):

    top = interp1d(eta_top, Pr[..., :mid+1], axis=-1)(eta_d)
    bot = interp1d(eta_bot, Pr[..., mid:][..., ::-1], axis=-1)(eta_d)
    return 0.5 * (bot + SIGN * top)


# ============================ H. RIGID PROFILES OF ALL CHUNKS ============================
Rp = np.full((len(KX), len(KZ), len(CP), 4, len(yp)), np.nan)   # rigid profiles at the DNS planes
for i in range(len(KX)):
    f = SWEEP / f'kx_{i:03d}.npz'
    if f.exists():
        Pc = np.load(f)['P']
        if Pc.ndim == 5:                                   # v2: (nkz, nc, nW, 5, N) -> rigid wall only
            Pc = Pc[:, :, 0]
        Rp[i] = to_planes(Pc[..., :4, :])                  # uu, vv, ww, uv -> (nkz, nc, 4, 20)                 # (nkz, nc, 4, N) -> (nkz, nc, 4, 20) -> resolvent to DNS planes
done = ~np.isnan(Rp[:, 0, 0, 0, 0])
print(f'{done.sum()} of {len(KX)} kx chunks loaded,  Rp shape {Rp.shape}')


# ============================ CHECK: DNS CELL vs RESOLVENT PROFILES ============================
U_bot = M['U'][mid:][::-1]                                 # mean velocity U+, wall -> centreline
y_bot = eta_bot * Re_tau                                   # the same heights in y+

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (name, kx0, kz0, cs) in zip(axes, [('NW', 12, 120, [4, 8, 10, 14, 20]),
                                           ('VLSM', 1, 10, [8, 12, 16, 20, 24])]):
    a, b = np.argmin(abs(KX - kx0)), np.argmin(abs(KZ - kz0))
    dns = -Ec[:, iuv, a, b]                                # DNS -uv of this cell at the 20 planes
    ax.plot(yp, dns / dns.max(), 'ko-', lw=2, label='DNS cell')
    for c in cs:
        l = np.argmin(abs(CP - c))
        r = -Rp[a, b, l, iuv]                              # resolvent -uv per unit forcing
        line, = ax.plot(yp, r / r.max(), label=f'$c^+$ = {CP[l]:.1f}')
        yc = np.interp(CP[l], U_bot, y_bot)                # critical layer: U(yc) = c
        ax.axvline(yc, color=line.get_color(), ls=':', lw=1)
    ax.set_xscale('log')
    ax.set_xlabel('$y^+$'); ax.set_ylabel('$-uv$ / max')
    ax.set_title(f'{name}: $k_x$ = {KX[a]:.2f}, $k_z$ = {KZ[b]:.1f},  w = {w[a, b]:.2e}')
    ax.legend(fontsize=8)
fig.tight_layout()

# ============================ I. FIT CP SPECTRUM OF RESOLVENT CELLS TO DNS ============================

ALPHA = 0.05

def fit_cell(a, b):

    d = Ec[:, iuv, a, b]                                 # target: DNS uv of this cell at the 20 planes
    A = Rp[a, b, :, iuv, :].T                            # column i = resolvent uv profile at speed CP[i]
    ok = ~np.isnan(A).any(axis=0)
    X = np.zeros(len(CP))
    if not d.any() or ok.sum() < 3:

        return X, np.nan, np.zeros_like(d)

    Aok = A[:, ok]
    n = ok.sum()
    L = np.diff(np.eye(n), 2, axis = 0)

    lam = ALPHA * np.linalg.norm(Aok / np.sqrt(n))      # how strongly smoothness counts against fitting the data
    xok, _ = nnls(np.vstack([Aok, lam * L]), np.concatenate([d, np.zeros(n - 2)]))
    X[ok] = xok
    fit = Aok @ xok
    res = np.linalg.norm(fit - d) / np.linalg.norm(d)   # relative misfit

    return X, res, fit

# FIT CHECK

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for row, (name, kx0, kz0) in enumerate([('NW', 12, 120), ('VLSM', 1, 10)]):
    a, b = np.argmin(abs(KX - kx0)), np.argmin(abs(KZ - kz0))
    X, res, fit = fit_cell(a, b)
    ax = axes[row, 0]
    ax.plot(yp, -Ec[:, iuv, a, b], 'ko-', label='DNS cell')
    ax.plot(yp, -fit, 'r-', lw=2, label=f'resolvent fit, residual {res:.3f}')
    ax.set_xscale('log'); ax.set_xlabel('$y^+$'); ax.set_ylabel('$-uv$')
    ax.set_title(f'{name}: $k_x$ = {KX[a]:.2f}, $k_z$ = {KZ[b]:.1f}'); ax.legend()
    ax = axes[row, 1]
    #ax.bar(CP, X, width=0.5)
    ax.set_xlabel('$c^+$'); ax.set_ylabel('$X_c$'); ax.set_title(f'{name}: forcing over speeds')
    A = Rp[a, b, :, iuv, :].T                                        # (20, nc) profiles, as in fit_cell
    Az0 = np.vstack([np.zeros((1, A.shape[1])), np.nan_to_num(A)])   # add uv = 0 at the wall
    T_c = 3 / Re_tau * np.trapezoid(fik[:, None] * -Az0, yw, axis=0) # FIK integral of each speed's profile
    ax.bar(CP, X * T_c, width=0.5)
    ax.set_ylabel(r'$X_c\,T_c$')
    print(f'{name}: sum X_c T_c = {np.sum(X * T_c):.3e}   vs   T_cell = {T_cell[a, b]:.3e}')
fig.tight_layout()
plt.show()

