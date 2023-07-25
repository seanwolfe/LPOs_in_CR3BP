import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm

# Generates a nice figure for lyapunov orbits, based on the Jacobi constant

# Specify the directory path
directory_l1 = os.path.join(os.getcwd(), 'Horizontal Lyapunov Orbits - L1 - Sun and EMS - Integrations Results')
directory_l2 = os.path.join(os.getcwd(), 'Horizontal Lyapunov Orbits - L2 - Sun and EMS - Integrations Results')

mu_SE = 3.04042e-6
mu_em = 0.0121505856
mu = mu_SE

sun = [-mu, 0., 0.]
earth = [1 - mu, 0., 0.]

quintic_poly_l1 = [1., -(3-mu), 3-2*mu, -mu, 2*mu, -mu]
roots = np.roots(quintic_poly_l1)
l_1 = [earth[0] - roots.real[abs(roots.imag) < 1e-5][0], 0. , 0.]

quintic_poly_l2 = [1., (3-mu), 3-2*mu, -mu, -2*mu, -mu]
roots_l2 = np.roots(quintic_poly_l2)
l_2 = [earth[0] + roots_l2.real[abs(roots_l2.imag)<1e-5][0], 0., 0.]
print(l_2)

xs_l1 = []
ys_l1 = []
cjs_l1 = []
i=0
for root, dirs, files in os.walk(directory_l1):
    for file in files:
        # Process each file
        file_path = os.path.join(root, file)
        print(file_path)
        # Perform desired operations with the file
        df = pd.read_csv(file_path, sep=" ", header=0)
        xs_l1 = np.hstack([xs_l1, df['x']])
        ys_l1 = np.hstack([ys_l1, df['y']])
        cjs_l1 = np.hstack([cjs_l1, df['cj']])

        i = i + 1
i_l1 = i

xs_l2 = []
ys_l2 = []
cjs_l2 = []
i=0
fig, ax = plt.subplots()
for root, dirs, files in os.walk(directory_l2):
    for file in files:
        # Process each file
        file_path = os.path.join(root, file)
        print(file_path)
        # Perform desired operations with the file
        df = pd.read_csv(file_path, sep=" ", header=0)
        xs_l2 = np.hstack([xs_l2, df['x']])
        ys_l2 = np.hstack([ys_l2, df['y']])
        cjs_l2 = np.hstack([cjs_l2, df['cj']])

        i = i + 1
i_l2 = i

i = max([i_l1, i_l2])
chopped_xs = []
chopped_ys = []
chopped_cjs = []
for j in range(0, i-80):
    j = j + 0
    if j % 2 == 0:
        chopped_xs = np.hstack([chopped_xs, xs_l1[-(1200*j + 1200):-1200*j], xs_l2[-(1200*(j+2) + 1200):-1200*(j+2)]])
        chopped_cjs = np.hstack([chopped_cjs, cjs_l1[-(1200*j + 1200):-1200*j], cjs_l2[-(1200*(j+2) + 1200):-1200*(j+2)]])
        chopped_ys = np.hstack([chopped_ys, ys_l1[-(1200*j + 1200):-1200*j], ys_l2[-(1200*(j+2) + 1200):-1200*(j+2)]])

r = 579734.2 / 149.e6
plt.scatter(earth[0], earth[1], color='b', label='Earth')
plt.scatter(l_1[0], l_1[1], color='black', label='L1')
plt.scatter(l_2[0], l_2[1], color='black', label='L2')
c2 = plt.Circle((earth[0], earth[1]), r, linestyle='--', edgecolor='black', alpha=0.1)
ax.add_patch(c2)
sc = plt.scatter(chopped_xs, chopped_ys, c=chopped_cjs, cmap='coolwarm', s=5)
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
ax.set_aspect('equal')
cbar = fig.colorbar(sc, label='Jacobi Constant')
plt.show()

