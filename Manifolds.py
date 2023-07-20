import os
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd

def model(state, time, mu=0.01215):
    # Define the dynamics of the system
    # state: current state vector
    # time: current time
    # return: derivative of the state vector

    x, y, z, vx, vy, vz = state  # position and velocity

    dUdx = -(mu * (mu + x - 1))/np.power(((mu + x - 1)**2 + y**2 + z**2), (3/2)) - \
           ((1 - mu) * (mu + x))/np.power(((mu + x)**2 + y**2 + z**2),(3/2)) + x
    dUdy = - (mu * y)/np.power(((mu + x - 1)**2 + y**2 + z**2), (3/2)) - \
           ((1 - mu) * y)/np.power(((mu + x)**2 + y**2 + z**2), (3/2)) + y
    dUdz = - (mu * z)/np.power(((mu + x - 1)**2 + y**2 + z**2), (3/2)) - \
           ((1 - mu) * z)/np.power(((mu + x)**2 + y**2 + z**2), (3/2))

    dxdt = vx  # derivative of position is velocity
    dydt = vy
    dzdt = vz
    dvxdt = dUdx + 2*dydt  # derivative of velocity is acceleration
    dvydt = dUdy - 2*dxdt
    dvzdt = dUdz

    dXdt = np.array([dxdt, dydt, dzdt, dvxdt, dvydt, dvzdt])

    return dXdt

start = 0
end = 15
space = 5000
t = np.linspace(start, end, space)  # time is non-dimensionalized
mu = 3.04042e-6
sun = [-mu, 0., 0.]
earth = [1 - mu, 0., 0.]
r = 579734.2 / 149.e6

quintic_poly_l1 = [1., -(3-mu), 3-2*mu, -mu, 2*mu, -mu]
roots = np.roots(quintic_poly_l1)
l_1 = [earth[0] - roots.real[abs(roots.imag) < 1e-5][0], 0. , 0.]

quintic_poly_l2 = [1., (3-mu), 3-2*mu, -mu, -2*mu, -mu]
roots_l2 = np.roots(quintic_poly_l2)
l_2 = [earth[0] + roots_l2.real[abs(roots_l2.imag)<1e-5][0], 0., 0.]

dir_path = os.path.join(os.getcwd(), 'Horizontal Lyapunov Orbits - L1 - Sun and EMS - Integrations Results')

file_list = os.listdir(dir_path)

# Iterate over the files
for idx, file_name in enumerate(file_list):
    file_path = os.path.join(dir_path, file_name)

    # Check if the current item is a file (not a subdirectory)
    if os.path.isfile(file_path) and idx > 95:

        dir_path_l2 = os.path.join(os.getcwd(), 'Horizontal Lyapunov Orbits - L2 - Sun and EMS - Integrations Results')
        file_list_l2 = os.listdir(dir_path_l2)

        for idx_l2, file_name_l2 in enumerate(file_list_l2):

            if os.path.isfile(file_list_l2) and idx > 95:

                fig, ax = plt.subplots()
                plt.scatter(earth[0], earth[1], color='b', label='Earth')
                plt.scatter(l_1[0], l_1[1], color='black', label='L1')
                plt.scatter(l_2[0], l_2[1], color='black', label='L2')
                c2 = plt.Circle((earth[0], earth[1]), r, linestyle='--', edgecolor='black', alpha=0.1)
                ax.add_patch(c2)
                ax.set_aspect('equal')

                orbit = pd.read_csv(file_path, sep=' ', header=0, names=['x', 'y', 'z', 'vx', 'vy', 'vz', 'cj'])
                orbit_l2 = pd.read_csv(file_list_l2, sep=' ', header=0, names=['x', 'y', 'z', 'vx', 'vy', 'vz', 'cj'])

                for i in range(len(orbit['x'])):
                    print(len(orbit['x']))
                    print(i)
                    X_0 = [orbit['x'].iloc[i], orbit['y'].iloc[i], orbit['z'].iloc[i], orbit['vx'].iloc[i], orbit['vy'].iloc[i],
                           orbit['vz'].iloc[i]]
                    state = np.array(X_0)

                    # solve ODE
                    res = odeint(model, state, t, args=(mu,))

                    # identify when inside the sphere of influence of the EMS
                    dist = [np.linalg.norm([res[i, 0] - earth[0], res[i, 1] - earth[1], res[i, 2] - earth[2]]) for i in range(len(res))]

                    in_ems = np.NAN * np.zeros((len(res),))
                    in_ems_idxs = [index for index, value in enumerate(dist) if value <= r]

                    if in_ems_idxs:
                        final_res = res[:in_ems_idxs[0], :]

                        plt.plot(final_res[:, 0], final_res[:, 1], 'blue')

                plt.show()

