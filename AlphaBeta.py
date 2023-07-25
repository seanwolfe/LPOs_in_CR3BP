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


def alpha_beta_jacobi_from_manifolds():

    # go through all the files of test particles
    main_dir = os.path.join(os.getcwd(), 'States_at_EMS')
    file_list = os.listdir(main_dir)

    mu = 3.04042e-6

    all_alphas = []
    all_betas = []
    all_jacobis = []

    for idx, file_name in enumerate(file_list):

        file_path = os.path.join(main_dir, file_name)
        if os.path.isfile(file_path) and idx > -1:

            print("Reading file: "  + str(file_name))
            data = pd.read_csv(file_path, sep=' ', header=0, names=['x', 'y', 'z', 'vx', 'vy', 'vz', 'L'])

            jacobis = []
            alphas = []
            betas = []

            for i in range(len(data['x'])):

                jacobi_i = jacobi(data.iloc[i, :6], mu)
                jacobis.append(jacobi_i)
                alpha_i = np.rad2deg(np.arctan2((data['y'].iloc[i]), (data['x'].iloc[i] - 1.0))) # arctan2 from -180 to 180
                if alpha_i < 0:
                    alpha_i += 360
                alphas.append(alpha_i)
                psi = np.rad2deg(np.arctan2(data['vy'].iloc[i], data['vx'].iloc[i]))
                if psi < 0:
                    psi += 360
                beta_i = psi - alpha_i - 90
                if beta_i < 0:
                    beta_i += 360
                betas.append(-beta_i)  # negative to match with qi

                all_alphas.append(alpha_i)
                all_betas.append(-beta_i)
                all_jacobis.append(jacobi_i)

            data['alpha_I'] = alphas
            data['beta_I'] = betas
            data['Jacobi'] = jacobis


            data.to_csv(os.getcwd() + '/States_at_EMS_with_alpha_beta/' + format(jacobis[0], ".15f") + '.csv', sep=' ', header=True, index=False)

            # fig, ax = plt.subplots()
            # plt.scatter(1.0, 0., color='b', label='Earth', zorder=15)
            # c2 = plt.Circle((1.0, 0.),  579734.2 / 149.e6, linestyle='--', edgecolor='black', alpha=0.1, zorder=15)
            # plt.scatter(data['x'], data['y'])
            # ax.add_patch(c2)
            # ax.set_aspect('equal')

            fig = plt.figure()
            sc = plt.scatter(alphas, betas, c=jacobis, cmap='coolwarm', s=5)
            cbar = fig.colorbar(sc, label='Jacobi Constant')
            plt.savefig(os.getcwd() + '/States_at_EMS_with_alpha_beta/' + format(jacobis[0], ".15f") + '_pc.svg', format='svg')
            # plt.show()
            plt.close()

    fig = plt.figure()
    sc = plt.scatter(all_alphas, all_betas, c=all_jacobis, cmap='gist_rainbow', s=5)
    cbar = fig.colorbar(sc, label='Jacobi Constant')
    plt.savefig('Figures/final_alpha_beta_pc.svg', format='svg')
    plt.show()
    plt.close()

    return


alpha_beta_jacobi_from_manifolds()
