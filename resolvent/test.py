import numpy as np
from chebdif import chebdif
from clencurt import clencurt
from mean_vel import channelMeanVel
import matplotlib.pyplot as plt
from resolvent_compliant import channelResolventCompliant

x, DM = chebdif(21, 2)
print(np.max(np.abs(DM[:,:,0] @ x**2 - 2*x)))
print(np.max(np.abs(DM[:,:,1] @ x**3 - 6*x)))

x, w = clencurt(21)
print(np.sum(w) - 2)
print(np.sum(w*x**2) - 2/3)

y, D1, D2, dy, U0, nuT = channelMeanVel(2000, 201)

plt.plot(y,U0)
plt.show()

print(f"{U0.max():.10f}")

RIG, CON, y, D1, D2, dy, U0 = channelResolventCompliant(2000, 12, 120, 10, 201, 10, 0)
print(np.max(np.abs(CON.H - RIG.H)) / np.max(np.abs(RIG.H)))
print(RIG.s[:4])

# Y nonzero: compare with MATLAB
RIG, CON, y, D1, D2, dy, U0 = channelResolventCompliant(2000, 12, 120, 10, 201, 10, 0.02-0.01j)
print(CON.s[:4])

