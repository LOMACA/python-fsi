import sys
import numpy as np
sys.path.insert(0, r'C:\Users\uqlcapp1\Simulations\DNS\code')  # folder for postprocessing functions
from dns_tools import HJ_SPECS, load_spectrum_2d, read_fortran_records
import matplotlib.pyplot as plt
from pathlib import Path


# SOME POST-PROCESSING TESTS ABOUT FILE STRUCTURE 

Re = 2000
files = sorted(HJ_SPECS[Re].glob('*.spe.j*'))
print(len(files), 'files:', files[0].name, '...', files[-1].name)  # check for correct files

G = load_spectrum_2d(files[0])
print(type(G))   # type of what the reader returns

for key, val in G.items():        # what keys and values are stored in the files at different y+

    if isinstance(val, np.ndarray):

         print(f'{key:8s} array, shape {val.shape}, dtype {val.dtype}')

    elif isinstance(val, dict):

        print(f'{key:8s} dict with keys {list(val.keys())}')

        for k2, v2 in val.items():

            print(f'   {k2:6s} array, shape {v2.shape}')

    else:

        print(f'{key:8s} {type(val).__name__} = {val}')

records = read_fortran_records(files[0])
print(len(records), 'records')               # what is the actual data in the files?

SHOW_CHECKS = False
if SHOW_CHECKS:

    for i, r in enumerate(records):
        print(f'record {i}: {len(r)} bytes = {len(r)//4} numbers')

    print(np.round(G['y_all'], 5))                           # y locations in the channel in outer units
    print(np.round(G['y_all'] * G['utau'] * G['re'], 1))     # y+ locations in the channel

    P = load_spectrum_2d(files[2])
    Re_tau = P['utau'] * P['re']                                 # friction Re

    uu = P['sp']['uu'] / P['utau']**2                            # 2D energy spectrum of u, energy per bin, in plus units (FT of streamwise velocity autocorrelation)
    print('y+ = ', P['y'] * Re_tau)
    print('uu[0,0] = ', uu[0, 0], 'sqrt = ', np.sqrt(uu[0, 0]))    # U_mean^2 and U_mean at that y+
    uu[0,0] = 0.0                                                  # remove the mean flow U+^2 (kx = kz = 0)
    print('uu binned sum', uu.sum())                               # variance of streamwise velocity at that y+
    print('u_prime', np.sqrt(uu.sum()))                            # rms of streamwise velocity fluctuation at that y+

    kx, kz = P['kx'][1:], P['kz'][1:]                              # array of streamwise and spanwise wavenumbers excluding 0
    dkx, dkz = P['alp'], P['bet']                                  # bin widths
    phi = uu[1:, 1:] / (dkx * dkz)                                 # energy per bin -> energy density
    pm = kx[:, None] * kz[None, :] * phi                           # premultiplied spectrum -> energy per unit d(ln kx) d(ln kz)
    lx, lz = 2*np.pi*Re_tau/kx, 2*np.pi*Re_tau/kz                  # wavelengths

    fig, ax = plt.subplots()
    cs = ax.contourf(lx, lz, pm.T, levels = 20, cmap = 'viridis')
    ax.set_xscale('log'), ax.set_yscale('log')
    ax.set_xlabel(r'$\lambda_x^+$')
    ax.set_ylabel(r'$\lambda_z^+$')
    ax.set_title(f'$k_x k_z \\Phi_{{uu}}$ at $y^+$ = {P["y"]*Re_tau:.1f}')
    fig.colorbar(cs)

    # LOG-POLAR MAPPING OF SPECTRA

    from scipy.interpolate import RegularGridInterpolator
    from scipy.ndimage import gaussian_filter

    KREF = 1/50000                          # k_ref+, as in the paper (independent of Re)
    d = 0.01                                # grid spacing in kx#, kz# (paper: 0.01)
    g = np.arange(0, 4.9 + d/2, d)          # kx#, kz# from 0 to 4.9
    GX, GZ = np.meshgrid(g, g, indexing='ij')   # GX[i, j] = g[i], GZ[i, j] = g[j]
    XI = np.hypot(GX, GZ)                   # radius:  xi = sqrt(kx#^2 + kz#^2)
    TH = np.arctan2(GZ, GX)                 # angle: 0 = kx axis, pi/2 = kz axis
    K = KREF * 10**XI * Re_tau              # |k| in outer units, the same units as P2['kx']
    KXq, KZq = K*np.cos(TH), K*np.sin(TH)   # the real (kx, kz) of every grid point

    Phi = uu / (P['alp'] * P['bet'])      # density on the full DNS grid ((0,0) already set to 0)
    f = RegularGridInterpolator((P['kx'], P['kz']), Phi, bounds_error=False, fill_value=0.0)
    Phi_q = f(np.stack([KXq, KZq], axis=-1))   # Phi at every log-polar grid point

    with np.errstate(divide='ignore', invalid='ignore'):
        D = np.log(10) * K**2 * Phi_q / XI   # LM19 eq. 2.27
    D[XI == 0] = 0.0                         # the origin: xi = 0, division by zero
    print('integral of D =', D.sum() * d**2, '  vs  sum of uu =', uu.sum())

    fig, ax = plt.subplots(figsize=(6, 6))
    Ds = gaussian_filter(D, 6)
    levels = np.linspace(0.05, 1, 20) * Ds.max()        # start at 5% of the peak: below that stays white
    cs = ax.contourf(g, g, Ds.T, levels=levels, cmap='magma_r', extend='neither')
    th = np.linspace(0, np.pi/2, 100)
    for lp in [1e1, 1e2, 1e3, 1e4]:                    # circles of constant |lambda+|
        r = np.log10(2*np.pi/lp / KREF)                # their radius xi
        ax.plot(r*np.cos(th), r*np.sin(th), 'k:', lw=0.8)
        ax.text(r/np.sqrt(2), r/np.sqrt(2), rf'$\lambda^+=10^{{{int(np.log10(lp))}}}$', fontsize=8)
    ax.plot([0, 4.9], [0, 4.9], 'k--', lw=0.5)         # kz = kx
    ax.set_aspect('equal')
    ax.set_xlabel(r'$k_x^ #$'); ax.set_ylabel(r'$k_z^ #$')
    ax.set_title(f'log-polar $u u$ at $y^+$ = {P["y"]*Re_tau:.1f}')
    fig.colorbar(cs)
    plt.show()


''' DNS WEIGHTING OF RESOLVENT MODES '''

# resolvent sweep data import

ROOT = Path(__file__).resolve().parent.parent           # fsi -> Python (project root)
M = np.load(ROOT / 'results' / 'sweep' / 'meta.npz')    # resolvent results load
KX, KZ = M['kx'], M['kz']                               # resolvent kx and kz
print(f'sweep kx: {KX.min():.2f} ... {KX.max():.2f} ({len(KX)} values)')
print(f'sweep kz: {KZ.min():.2f} ... {KZ.max():.1f} ({len(KZ)} values)')

G = load_spectrum_2d(files[0], keep = ())               # load grid only
Re_tau = G['utau'] * G['re']                            # friction Re
kx_top = KX[-1] * np.sqrt(KX[-1] / KX[-2])              # cap DNS wavenumbers at upper edge of last cell
kz_top = KZ[-1] * np.sqrt(KZ[-1] / KZ[-2])
ix = G['kx'] <= kx_top
iz = G['kz'] <= kz_top
print(f'keep {ix.sum()} of {len(ix)} kx bins (kx <= {kx_top:.1f}), '
      f'{iz.sum()} of {len(iz)} kz bins (kz <= {kz_top:.1f})')

VARS = ['uu', 'vv', 'ww', 'uv']
nP = len(files)
yp = np.empty(nP)                                       # height of each plane in y+
tot = np.empty((nP, len(VARS)))                         # full variance per plane: sum over all bins
E = np.empty((nP, len(VARS),  ix.sum(), iz.sum()), np.float32)   # cropped spectra, energy per bin

# 2D spectra import

for p, f in enumerate(files):

    P = load_spectrum_2d(f, keep=VARS)
    assert P['jind'] == p + 1, f'{f.name} is plane {P["jind"]}, expected {p+1}'
    yp[p] = P['y'] * Re_tau

    for v, name in enumerate(VARS):
        
        s = P['sp'][name] / G['utau']**2
        s[0, 0] = 0.0                                   # remove the mean
        tot[p, v] = s.sum()                             # variance at this height (Parseval)
        E[p, v] = s[np.ix_(ix, iz)]                     # energy per bin

    print(f'{f.name}: y+ = {yp[p]:7.1f}   ' + '  '.join(f'{n} {t:7.3f}' for n, t in zip(VARS, tot[p])))

coverage = np.divide(E.sum(axis=(2, 3)), tot, out=np.full_like(tot, np.nan), where=tot != 0)   # fraction of the variance inside the crop
print(f'\nCoverage: fraction of each variance kept by the crop (kx <= {kx_top:.1f}, kz <= {kz_top:.1f})')
print('\n   y+     ' + '   '.join(f'{n:>5s}' for n in VARS))
for p in range(nP):
    print(f'{yp[p]:7.1f}  ' + '  '.join(f'{c:6.3f}' for c in coverage[p]))

# SUMMATION OF DNS BINS INTO THE RESOLVENT SWEEP CELLS

def cell_edges(K):

    mid = np.sqrt(K[1:]*K[:-1])
    lo = K[0] / np.sqrt(K[1] / K[0])            # half a step below the first point
    hi = K[-1] * np.sqrt(K[-1] / K[-2])  # half a step above the last point (= kx_top, kz_top)
    return np.concatenate([[lo], mid, [hi]])

def bin_matrix(k, e):

    return ((k[None, :] >= e[:-1, None]) & (k[None, :] < e[1:, None])).astype(np.float32)

ex, ez = cell_edges(KX), cell_edges(KZ)
kx_d, kz_d = G['kx'][ix], G['kz'][iz]               # kept DNS wavenumbers
Ax, Az = bin_matrix(kx_d, ex), bin_matrix(kz_d, ez)

print(f'kx cells from {ex[0]:.3f} to {ex[-1]:.1f},  kz cells from {ez[0]:.3f} to {ez[-1]:.1f}')
print('Ax', Ax.shape, '  Az', Az.shape)
print('DNS bins per kx cell:', Ax.sum(axis=1).astype(int))
print('DNS bins per kz cell:', Az.sum(axis=1).astype(int))
