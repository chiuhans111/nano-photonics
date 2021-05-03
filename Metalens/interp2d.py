import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

f = interpolate.interp2d([0, 0, 1, 1], [0, 1, 0, 1], [0, 0, 1, 0])

x = np.linspace(0, 1, 10)
y = np.linspace(0, 1, 10)
plt.imshow(f(x, y)[::-1], extent=[0, 1, 0, 1])
plt.show()
