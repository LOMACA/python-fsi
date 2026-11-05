import numpy as np
from chebdif import chebdif
from clencurt import clencurt
from mean_vel import channelMeanVel
import matplotlib.pyplot as plt

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

