import numpy as np
import matplotlib.pyplot as plt


def plot_mode(i, mname, Q, U, y, walls, cP, N, Re, ylim=80):
    fig, ax = plt.subplots(1, 3, figsize=(14, 7), sharey=True)
    for k, w in enumerate(walls):
        yp, u, v, w_, p, ycP = bottom_profiles(Q[i, k], U[i], y, cP, N, Re)
        ls = '-' if k == 0 else '--'
        ax[0].plot(abs(u), yp, ls, label=f'u {w.name}')
        ax[0].plot(10*abs(v), yp, ls, label=f'10v {w.name}')
        ax[0].plot(10*abs(p), yp, ls, label=f'10p {w.name}')
        ax[1].plot(np.mod(np.angle(u), 2*np.pi)/(2*np.pi), yp, ls, label=f'u {w.name}')
        ax[1].plot(np.mod(np.angle(v), 2*np.pi)/(2*np.pi), yp, ls, label=f'v {w.name}')
        ax[2].plot(-0.5*np.real(np.conj(u)*v), yp, ls, label=w.name)
    for a, t, xl in zip(ax, ['amplitudes', 'phases', 'Reynolds stress'],
                        [r'$|u|, 10|v|, 10|p|$', r'phase $/2\pi$', r'$-\frac{1}{2}\mathrm{Re}(u^*v)$']):
        a.axhline(ycP, color='0.5', ls='--', lw=0.8)
        a.set_title(f'{mname}: {t}'); a.set_xlabel(xl); a.set_ylim(0, ylim); a.legend(fontsize=7)
    ax[1].set_xlim(0, 1); ax[2].axvline(0, color='k', ls=':', lw=0.8)
    ax[0].set_ylabel(r'$y^+$')
    plt.tight_layout()


def bottom_profiles(q, U, y, cP, N, Re):

    idxHalf = np.arange((N - 1)//2, N)
    icrit = idxHalf[np.argmin(np.abs(U[idxHalf] - cP))]
    q = q * np.exp(1j*(np.pi - np.angle(q[icrit])))
    u, v, w, p = (q[n*N:(n+1)*N][idxHalf] for n in range(4))
    return y[idxHalf]*Re, u, v, w, p, y[icrit]*Re