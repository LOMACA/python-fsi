import numpy as np

from fsi.walls import Flow, rigid, sr, default_walls, modal
from fea.assemble_coiled_MM_structure import assemble_structure
from fea.coil_rod import coilRod

flow = Flow()                                   # air, Re_tau = 2000, h = 0.05 m
kx, cP = 12, 10
wP = cP*kx/flow.Re_tau                          # NW: omega+ = 0.06
print(f'u_tau = {flow.u_tau:.4f} m/s,  t* = {flow.t_star:.4e} s,  m* = {flow.m_star:.4e} kg/m^2')

# 1. table: first resonance, effective modal mass per area, Y+ at the NW mode
for w in default_walls(flow):
    Y = w.Y(wP)[0]
    m_eff = 1/(w.A*w.modes.r[0]) if w.modes is not None else np.nan
    print(f'{w.name:10s}  w_res+ = {w.w_res:.6e}   m_eff+ = {m_eff:.4e}'
          f'   Y+ = {Y.real:.10e} {Y.imag:+.10e}j')

# 2. rigid is exactly zero, also for vectors
print('rigid max |Y|:', np.max(np.abs(rigid().Y(np.linspace(0.01, 0.1, 5)))))

# 3. SR unit conversion: first resonance back in Hz (expect ~278.56)
s0 = sr(flow, 0)
print('SR f1 [Hz]:', flow.f_hz(s0.w_res))

# 4. SR scaling against the unscaled SI structure via coilRod
#    w = w+/t*,  eta+ = eta_SI * rho_f u_tau^3/nu,  Y+ = i w+ conj(eta+)
E, rho, LUC = 0.2e9, 450, 0.02
mres = np.array([3, 5]) * rho * LUC
kres = mres * (2*np.pi*np.array([5000.0, 20000.0]))**2
M, C, K, freeDoF, _, _, jN = assemble_structure(E, rho, 0, 6e-8, LUC, 3, 10, kres, mres)

for n in [0, 3]:
    eta_SI, _, _ = coilRod(M, C, K, freeDoF, jN, wP / flow.t_star, n)
    Y_ref = 1j * wP * np.conj(eta_SI * flow.rho_f * flow.u_tau**3 / flow.nu)
    Y_wall = sr(flow, n).Y(wP)[0]
    print(f'SR {n} cyc rel. error:', abs(Y_wall - Y_ref) / abs(Y_ref))

import numpy as np, matplotlib.pyplot as plt
from fsi.walls import Flow, sr, modal

flow = Flow(Re_tau=2000, h=0.05)                      # H as in main_sweeps
S3 = sr(flow, 3); m = S3.modes
O, r, z = m.Omega[0], m.r[0], m.zeta[0]
W = modal(O, 1/r, z)

w = O * np.sort(np.concatenate([np.linspace(0.5, 1.5, 2001), np.linspace(0.999, 1.001, 2001)]))
Y1 = 1j*w*np.conj(r / (O**2 - w**2 + 2j*z*O*w))
print('modal vs closed form, max rel err:', np.max(np.abs(W.Y(w) - Y1) / np.abs(Y1)))

plt.semilogy(w/O, abs(S3.Y(w)), label='SR 3 cyc (all modes)')
plt.semilogy(w/O, abs(W.Y(w)), '--', label='modal, mode 1 only')
plt.xlabel(r'$\omega/\Omega$'); plt.ylabel('|Y|'); plt.legend(); plt.show()