import numpy as np
from dataclasses import dataclass
from typing import Callable

from fea.assemble_coiled_MM_structure import assemble_structure
from fea.assemble_defect_RPM_structure import assemble_defect_chain
from fea.modal_analysis import modalFRF
from fsi.struct_admittance import structAdmittance


@dataclass
class Flow:
    Re_tau: float = 2000
    h: float = 0.05          # channel half-height [m]
    rho_f: float = 1.2       # density [kg/m^3]
    nu: float = 1.5e-5       # kinematic viscosity [m^2/s]

    @property
    def u_tau(self):         # friction velocity [m/s]
        return self.Re_tau * self.nu / self.h

    @property
    def t_star(self):        # viscous time [s]:  w+ = w * t_star
        return self.nu / self.u_tau**2

    @property
    def m_star(self):        # mass per area [kg/m^2]:  m+ = m / m_star
        return self.rho_f * self.nu / self.u_tau

    def omega_plus(self, f):
        """Dimensional frequency [Hz] -> omega+."""
        return 2*np.pi*f * self.t_star

    def f_hz(self, wP):
        """omega+ -> dimensional frequency [Hz]."""
        return wP / self.t_star / (2*np.pi)

@dataclass
class Wall:
    name: str
    Y: Callable              # Y(wP) -> complex admittance Y+ in wall units
    w_res: float = np.nan    # first resonance frequency, omega+
    A: float = 1.0           # interface area A+
    modes: object = None     # modalFRFResult of the structure in wall units


def rigid():
    return Wall('Rigid', lambda wP: np.zeros_like(np.atleast_1d(wP), dtype=complex))


def rpm(m=6.4e6, k=5.6e4, kg=1.4e6, c=3.0e3, mDef=8.687e6, kgDef=2.981e4,
        N=5, L=206, Re=186):

    """defect RPM case 10, A+ = panel area."""

    M, C, K, forcedDoF = assemble_defect_chain(m, k, kg, c, mDef, kgDef, N)
    freeDoF = np.arange(N)
    S = L * 2*np.pi*Re
    _, modes = modalFRF(M, C, K, freeDoF, forcedDoF, 1.0)

    def Y(wP):
        return structAdmittance(M, C, K, freeDoF, forcedDoF, wP, S)

    return Wall('RPM', Y, modes.Omega[0], S, modes)


def sr(flow, nCycles=0, omega_tar=None, m_tar=None,
       E=0.2e9, rho=450, LUC=0.02, nElem=3, Njunct=10,
       q1=0, q2=6e-8, fres=(5000, 20000), mult=(3, 5)):

    if (omega_tar is None) != (m_tar is None):
        raise ValueError('give both omega_tar and m_tar, or neither')

    fres, mult = np.asarray(fres, float), np.asarray(mult, float)
    mres = mult * rho * LUC
    kres = mres * (2*np.pi*fres)**2
    M, C, K, freeDoF, _, _, junctNodes = assemble_structure(
        E, rho, q1, q2, LUC, nElem, Njunct, kres, mres)

    alpha, beta = 1/flow.m_star, flow.t_star                  # physical scaling
    if omega_tar is not None:                                 # matched (MATLAB): from uncoiled mode 1
        _, m0 = modalFRF(alpha*M, alpha*beta*C, alpha*beta**2*K, freeDoF, junctNodes[:1], 1.0)
        a, b = m_tar * m0.r[0], omega_tar / m0.Omega[0]       # extra factors on top of physical
        print(f'SR matched: equivalent rho = {rho*a:.3e} kg/m^3, E = {E*a*b**2:.3e} Pa')
        alpha, beta = alpha*a, beta*b

    Ms, Cs, Ks = alpha*M, alpha*beta*C, alpha*beta**2*K
    act = junctNodes[0:2*nCycles+1:2]
    _, modes = modalFRF(Ms, Cs, Ks, freeDoF, act, 1.0)

    def Y(wP):
        return structAdmittance(Ms, Cs, Ks, freeDoF, act, wP, 1.0)

    tag = ' matched' if omega_tar is not None else ''
    return Wall(f'SR {nCycles} cyc{tag}', Y, modes.Omega[0], 1.0, modes)


def default_walls(flow):

    return [rigid(), rpm(), sr(flow, 0), sr(flow, 3)]