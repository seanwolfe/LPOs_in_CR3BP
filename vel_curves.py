import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fsolve

# This file will visualize zero velocity curves in the CR3BP

def equation(var, c):

    x, y = var

    mu_SE = 3.04042e-6
    mu_em = 0.01215
    mu = mu_SE

    rp1 = np.sqrt((x + mu) ** 2 + y ** 2)
    rp2 = np.sqrt((x - 1 + mu) ** 2 + y ** 2)
    Om3 = 0.5 * (x ** 2 + y ** 2) + (1 - mu) / rp1 + mu / rp2 + 0.5 * mu * (1 - mu)

    return c - 2*Om3

delta = 0.0001
x, y = np.meshgrid(np.arange(0.985, 1.015, delta), np.arange(-0.02, 0.02, delta))

mu_SE = 3.04042e-6
mu_em = 0.01215
mu = mu_SE

rp1 = np.sqrt((x + mu) ** 2 + y ** 2)
rp2 = np.sqrt((x - 1 + mu) ** 2 + y ** 2)
Om3 = 0.5 * (x ** 2 + y ** 2) + (1 - mu) / rp1 + mu / rp2 + 0.5 * mu * (1 - mu)

for i in range(0,1000):
    print(i)
    f = 3.0005 + i/20000. - 2*Om3
    plt.contour(x, y, f)

plt.show()

