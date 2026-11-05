## Mean velocity calculation based on R-T eddy viscosity model
import numpy as np
from chebdif import chebdif
from clencurt import clencurt

def channelMeanVel(Re, N):

    kappa = 0.426
    Amp = 25.4

    y, DM = chebdif(N, 2)
    D1 = DM[:,:,0]
    D2 = DM[:,:,1]

    _,dy = clencurt(N-1)

    nuT = 0.5*((1 + ((1/3)*kappa*Re*(1 - y**2)*(1 + 2*y**2)*(1 - np.exp(-(1 - np.abs(y))*Re/Amp)) )**2 )**(1/2) - 1)

    DnuT = D1 @ nuT

    LHS = np.diag(1+nuT) @ D2 + np.diag(DnuT) @ D1
    RHS = -Re*np.ones(N)

    LHS[0,:] = 0
    LHS[-1,:] = 0
    LHS[0,0] = 1
    LHS[-1,-1] = 1
    RHS[0] = 0
    RHS[-1] = 0

    U0 = np.linalg.solve(LHS, RHS)

    return y, D1, D2, dy, U0, nuT