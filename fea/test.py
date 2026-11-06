import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from fea.assemble_coild_MM_structure import assemble_structure
from fea.coil_rod import coilRod
from assemble_defect_RPM_structure import assemble_defect_chain
from fea.modal_analysis import modalFRF
from scipy.linalg import eigh

"""
EA, rhoA, q1, q2, LUC = 2.0, 3.0, 0.1, 1e-4, 0.02
nElem, Njunct = 4, 3
kres = np.array([5.0, 7.0])
mres = np.array([0.1, 0.2])

M, C, K, freeDoF, fixedDoF, forceDoF, junctNodes = assemble_structure(
    EA, rhoA, q1, q2, LUC, nElem, Njunct, kres, mres)

print(np.max(np.abs(K @ np.ones(K.shape[0]))))
print(np.max(np.abs(K - K.T)))
print(np.sum(M) - (rhoA*LUC*Njunct + Njunct*np.sum(mres)))
print(fixedDoF, K.shape)

eta, etaJ, activeJunctions = coilRod(M, C, K, freeDoF, junctNodes, 50, 1)
print(f"{eta.real:.10e} {eta.imag:.10e}")

M, C, K, forcedDoF = assemble_defect_chain(1.0, 2.0, 3.0, 0.1, 4.0, 5.0, 6)
print(K @ np.ones(6))

M, C, K, freeDoF, fixedDoF, forceDoF, junctNodes = assemble_structure(
    2.0, 3.0, 0.0, 0.0, 0.02, 4, 3, np.array([5.0, 7.0]), np.array([0.1, 0.2]))

eta_direct, _, _ = coilRod(M, C, K, freeDoF, junctNodes, 50, 1)
eta_modal, modes = modalFRF(M, C, K, freeDoF, junctNodes[[0, 2]], 50)
print(np.abs(eta_modal[0] - eta_direct))

M, C, K, freeDoF, fixedDoF, forceDoF, junctNodes = assemble_structure(
    2.0, 3.0, 0.1, 1e-4, 0.02, 4, 3, np.array([5.0, 7.0]), np.array([0.1, 0.2]))

eta_modal, modes = modalFRF(M, C, K, freeDoF, junctNodes[[0, 2]], 50)
print(f"{eta_modal[0].real:.10e} {eta_modal[0].imag:.10e} {modes.coupling:.10e}")
"""
E_eff, rho_eff, LUC = 0.2e9, 450, 0.02
nElem, Njunct = 3, 10
q1, q2 = 0, 6e-8

# Resonators
fres = np.array([5000.0, 20000.0])
mult = np.array([3, 5])
mres = mult*rho_eff*LUC
kres = mres*(2*np.pi*fres)**2

M, C, K, freeDoF, fixedDoF, forceDoF, junctNodes = assemble_structure(
    E_eff, rho_eff, q1, q2, LUC, nElem, Njunct, kres, mres)

# Direct frequency sweep
f = np.linspace(1, 3500, 2000)
omega = 2*np.pi*f
Cycles = [0, 1, 2, 3]

eta = np.zeros((len(Cycles), len(omega)), dtype=complex)
for ci, nCycles in enumerate(Cycles):
    for wi, w in enumerate(omega):
        eta[ci, wi], _, _ = coilRod(M, C, K, freeDoF, junctNodes, w, nCycles)

amp = np.abs(eta)
phase = np.angle(1j*eta)
P = amp*phase

# Check 1: first resonance and antiresonance, uncoiled
pk, _ = find_peaks(amp[0])
tr, _ = find_peaks(-amp[0])
print(f"First resonance {f[pk[0]]:.1f} Hz (target 278)")
print(f"First antiresonance {f[tr[0]]:.1f} Hz (target 555)")

# Check 2: modal FRF against the direct solve, 3 coiling cycles
eta_modal, modes = modalFRF(M, C, K, freeDoF, junctNodes[[0, 2, 4, 6]], omega)
print("Max relative difference:", np.max(np.abs(eta_modal - eta[3])/np.abs(eta[3])))
print("Modal coupling:", modes.coupling)

# Fig. 2(b), (c): uncoiled and 3 cycles
fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7, 6))
ax[0].semilogy(f, amp[0], 'k--', label='Uncoiled')
ax[0].semilogy(f, amp[3], 'k-', label='3 cycles')
ax[0].set_ylabel('|eta| (m/N)')
ax[0].legend()
ax[1].plot(f, phase[0], 'k--')
ax[1].plot(f, phase[3], 'k-')
ax[1].set_ylabel('phase (rad)')
ax[1].set_xlabel('f (Hz)')
plt.tight_layout()

# Fig. 2(d): performance metric
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(9, 6))
for ci, a in enumerate(ax.flat):
    a.plot(f, P[ci], 'k-')
    a.axhline(0, color='k', linewidth=0.5)
    a.fill_between(f, P[ci], 0, where=P[ci] > 0, color='red', alpha=0.3)
    a.fill_between(f, P[ci], 0, where=P[ci] < 0, color='green', alpha=0.3)
    a.set_title(f"{Cycles[ci]} coiling cycles")
    a.set_xlim(0, 2000)
    a.set_ylim(-5e-9, 5e-9)

plt.show()

Kf = K[np.ix_(freeDoF, freeDoF)]
Mf = M[np.ix_(freeDoF, freeDoF)]

W2_res = eigh(Kf, Mf, eigvals_only=True)
W2_anti = eigh(Kf[1:, 1:], Mf[1:, 1:], eigvals_only=True)

print("Resonance:", np.sqrt(W2_res[0])/(2*np.pi))
print("Antiresonance:", np.sqrt(W2_anti[0])/(2*np.pi))

rel = np.abs(eta_modal - eta[3])/np.abs(eta[3])
print("Worst at", f[np.argmax(rel)], "Hz")
