import numpy as np
from chebdif import chebdif
from clencurt import clencurt
from mean_vel import channelMeanVel
import matplotlib.pyplot as plt
from resolvent.resolvent_compliant import channelResolventCompliant

Re =2000

x, DM = chebdif(21, 2)
print(np.max(np.abs(DM[:,:,0] @ x**2 - 2*x)))
print(np.max(np.abs(DM[:,:,1] @ x**3 - 6*x)))

x, w = clencurt(21)
print(np.sum(w) - 2)
print(np.sum(w*x**2) - 2/3)

RIG, CON, y, D1, D2, dy, U0 = channelResolventCompliant(Re, 12, 120, 10, 601, 10, 0)
print(np.max(np.abs(CON.H - RIG.H)) / np.max(np.abs(RIG.H)))
print(RIG.s[:4])

# Y nonzero: compare with MATLAB
RIG, CON, y, D1, D2, dy, U0 = channelResolventCompliant(Re, 12, 120, 10, 601, 10, 0.02-0.01j)
print(CON.s[:4])

Re, kx, kz, cP, N = 2000, 1, 10, 16, 601
RIG, CON, y, D1, D2, dy, U0 = channelResolventCompliant(Re, kx, kz, cP, N, 10, 0)

# First singular mode
u = RIG.u[:N, 0]
v = RIG.u[N:2*N, 0]
w = RIG.u[2*N:3*N, 0]
p = RIG.u[3*N:4*N, 0]

yp = Re*y
low = yp <= 400     # bottom-wall region shown in the paper

# Critical layer in the bottom half
bottom = np.where(y < 1)[0]
i_c = bottom[np.argmin(np.abs(U0[bottom] - cP))]

# Energy checks
E = dy*(np.abs(u)**2 + np.abs(v)**2 + np.abs(w)**2)
print("singular values:", RIG.s[:4])
print("total energy:", np.sum(E))
print("bottom-half energy fraction:", np.sum(E[y < 1])/np.sum(E))
print("peaks |u|, |v|, |w|:", np.abs(u).max(), np.abs(v).max(), np.abs(w).max())
print("critical layer y+:", yp[i_c])


def phase(q):
    # Phase/2pi, with the overall offset set so u = 0.55 at the critical layer
    shifted = q*np.exp(-1j*np.angle(u[i_c]))
    return (np.angle(shifted)/(2*np.pi) + 0.55) % 1

RS = -0.5*np.real(np.conj(u)*v)

fig, ax = plt.subplots(1, 3, sharey=True, figsize=(9, 4))

ax[0].plot(np.abs(u[low]), yp[low], 'k-', label='|u|')
ax[0].plot(100*np.abs(v[low]), yp[low], 'k:', label='100|v|')
ax[0].plot(100*np.abs(p[low]), yp[low], 'b--', label='100|p|')
ax[0].set_ylabel('y+')
ax[0].legend()

ax[1].plot(phase(u)[low], yp[low], 'k-')
ax[1].plot(phase(v)[low], yp[low], 'k--')
ax[1].plot(phase(p)[low], yp[low], 'b:')
ax[1].set_xlim(0, 1)

ax[2].plot(RS[low], yp[low], 'k-')

for a in ax:
    a.axhline(yp[i_c], color='k', linestyle=':', linewidth=0.8)
ax[0].set_ylim(0, 400)

plt.tight_layout()
plt.show()
