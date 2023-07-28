import os
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd

def jacobi(res, mu):
    """

    :param res: the resultant state vector from a periodic orbit
    :param mu: the gravitational parameter
    :return: the jacobi constant
    """
    x, y, z, vx, vy, vz = res
    r1 = np.sqrt((x + mu) ** 2 + y ** 2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y ** 2)
    U = 0.5 * (x ** 2 + y ** 2) + (1 - mu) / r1 + mu / r2

    return 2*U - vx**2 - vy**2 - vz**2

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
end = 30
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

# L1
dir_path = os.path.join(os.getcwd(), 'Horizontal Lyapunov Orbits - L1 - Sun and EMS - Integrations Results')
file_list = os.listdir(dir_path)

# L2
dir_path_l2 = os.path.join(os.getcwd(), 'Horizontal Lyapunov Orbits - L2 - Sun and EMS - Integrations Results')
file_list_l2 = os.listdir(dir_path_l2)

l2_cjs = []
# make a list of possible cjs from L2
for idx_l2, file_name_l2 in enumerate(file_list_l2):

    file_path_l2 = os.path.join(dir_path_l2, file_name_l2)
    if os.path.isfile(file_path_l2):
        orbit_l2 = pd.read_csv(file_path_l2, sep=' ', header=0, names=['x', 'y', 'z', 'vx', 'vy', 'vz', 'cj'])
        l2_cjs.append(orbit_l2['cj'].iloc[0])


# Iterate over the files for L1
for idx, file_name in enumerate(file_list):

    # final results
    ems_state = []

    file_path = os.path.join(dir_path, file_name)
    # Check if the current item is a file (not a subdirectory)
    if os.path.isfile(file_path):
        print(idx)
        print("Reading file: " + str(file_name))
        orbit = pd.read_csv(file_path, sep=' ', header=0, names=['x', 'y', 'z', 'vx', 'vy', 'vz', 'cj'])
        cj = orbit['cj'].iloc[0]

        print("Finding closest L2 file...")
        # find the closest value in L2 jacobi constant list to this current
        closest_value = None
        min_difference = float('inf')
        closest_idx = None
        for idx, value in enumerate(l2_cjs):
            difference = abs(cj - value)
            if difference < min_difference:
                min_difference = difference
                closest_value = value
                closest_idx = idx

        print("Found file: " + str(l2_cjs[closest_idx]))
        # get the data for the file with the closest cj
        file_path_close = dir_path_l2 + '/' + str(l2_cjs[closest_idx])
        orbit_l2_close = pd.read_csv(file_path_close, sep=' ', header=0, names=['x', 'y', 'z', 'vx', 'vy', 'vz', 'cj'])

        print("Propogating orbit to SOI of EMS for L1....")
        # integrate the L1 orbit for each point of data until it hits the SOI of the EMS
        for i in range(len(orbit['x'])):

            X_0 = [orbit['x'].iloc[i], orbit['y'].iloc[i], orbit['z'].iloc[i], orbit['vx'].iloc[i], orbit['vy'].iloc[i],
                   orbit['vz'].iloc[i]]
            state = np.array(X_0)

            # solve ODE
            res = odeint(model, state, t, args=(mu,))

            # identify when inside the sphere of influence of the EMS
            dist = [np.linalg.norm([res[i, 0] - earth[0], res[i, 1] - earth[1], res[i, 2] - earth[2]]) for i in range(len(res))]

            # get soi of ems entries
            in_ems_idx = []
            hit = 0
            if dist[0] < r:
                inside = 1
            else:
                inside = 0
            for idx_ems, value in enumerate(dist):

                # if we are inside SOI of EMS
                if value < r:
                    if inside == 1:  # we were previously inside as well
                        pass
                    else:  # we were previously outside
                        inside = 1
                        if orbit['cj'].iloc[i] >= 3.00033 and hit == 0:
                            in_ems_idx.append(idx_ems)
                            hit = 1
                        elif orbit['cj'].iloc[i] >= 3.00033 and hit == 1:
                            pass
                        else:
                            in_ems_idx.append(idx_ems)
                # we are outside the EMS
                else:
                    # we were just inside the SOI of the EMS, we are exiting
                    inside = 0

            # fig, ax = plt.subplots()
            # plt.scatter(earth[0], earth[1], color='b', label='Earth', zorder=15)
            # plt.scatter(l_1[0], l_1[1], color='black', label='L1', zorder=15)
            # plt.scatter(l_2[0], l_2[1], color='black', label='L2', zorder=15)
            # c2 = plt.Circle((earth[0], earth[1]), r, linestyle='--', edgecolor='black', alpha=0.1, zorder=15)
            # plt.plot(res[:, 0], res[:, 1])
            # if in_ems_idx:
            #     plt.scatter(res[in_ems_idx, 0], res[in_ems_idx, 1])
            # plt.scatter(res2[:, 0], res2[:, 1])
            # plt.xlim([0.96, 1.04])
            # plt.ylim([-0.04, 0.04])
            # ax.add_patch(c2)
            # ax.set_aspect('equal')
            # plt.xlabel('Synodic x (non-dimensional)')
            # plt.ylabel('Synodic y (non-dimensional)')
            # plt.legend(loc='upper right')
            # plt.show()

            if in_ems_idx:
                # plt.plot(res[passed_r_idx:in_ems_idx, 0], res[passed_r_idx:in_ems_idx, 1], 'blue')
                # plt.scatter(res[in_ems_idx, 0], res[in_ems_idx, 1])
                for idxs in range(len(in_ems_idx)):
                    temp = np.append(res[in_ems_idx[idxs], :], 1)
                    ems_state.append(temp)


        print("Propogating orbit to SOI of EMS for L2...")
        # do the same for the L2 orbit
        for j in range(len(orbit_l2_close['x'])):

            X_0 = [orbit_l2_close['x'].iloc[j], orbit_l2_close['y'].iloc[j], orbit_l2_close['z'].iloc[j],
                   orbit_l2_close['vx'].iloc[j], orbit_l2_close['vy'].iloc[j], orbit_l2_close['vz'].iloc[j]]
            state = np.array(X_0)

            # solve ODE
            res2 = odeint(model, state, t, args=(mu,))

            # identify when inside the sphere of influence of the EMS
            dist = [np.linalg.norm([res2[i, 0] - earth[0], res2[i, 1] - earth[1], res2[i, 2] - earth[2]]) for i in range(len(res2))]

            # get soi of ems entries
            in_ems_idx = []
            hit = 0
            if dist[0] < r:
                inside = 1
            else:
                inside = 0
            for idx_ems, value in enumerate(dist):

                # if we are inside SOI of EMS
                if value < r:
                    if inside == 1:  # we were previously inside as well
                        pass
                    else:  # we were previously outside
                        inside = 1
                        if orbit_l2_close['cj'].iloc[j] >= 3.00033 and hit == 0:
                            in_ems_idx.append(idx_ems)
                            hit = 1
                        elif orbit['cj'].iloc[i] >= 3.00033 and hit == 1:
                            pass
                        else:
                            in_ems_idx.append(idx_ems)
                            # we are outside the EMS
                else:
                    # we were just inside the SOI of the EMS, we are exiting
                    inside = 0

            if in_ems_idx:
                # plt.plot(res[passed_r_idx:in_ems_idx, 0], res[passed_r_idx:in_ems_idx, 1], 'red')
                # plt.scatter(res[in_ems_idx, 0], res[in_ems_idx, 1])
                for idxs in range(len(in_ems_idx)):
                    temp = np.append(res2[in_ems_idx[idxs], :], 1)
                    ems_state.append(temp)

        l12_pd = pd.DataFrame(ems_state, columns=['x', 'y', 'z', 'vx', 'vy', 'vz', 'L'])
        cj_ems = jacobi(l12_pd.iloc[0, :6], mu)
        ems_file_path = os.path.join(os.getcwd(), 'States_at_EMS_new')
        l12_pd.to_csv(ems_file_path + '/' + format(cj_ems, ".15f") + '.csv', sep=' ', header=True, index=False)

        # plt.show()
        # plt.savefig('Figures/' + format(cj_ems, ".15f") + '.svg', format='svg')
        # plt.close()
        print("Saved file: " + str(cj_ems))

