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
    plt.xlabel(r'$\alpha_I$ (degrees)')
    plt.ylabel(r'$\beta_I$ (degrees)')
    plt.xlim([0, 360])
    plt.ylim([-180, 0])
    plt.show()
    plt.close()

    return


# def reverse_alpha_beta_jacobi():
#
#     r = 579734.2 / 149.e6
#     mu = 3.04042e-6
#     points = 2
#     # x_temp = np.linspace(1-r, 1+r, points)
#     # y_pos = np.sqrt(abs(r ** 2 - (x_temp - 1) ** 2))
#     # y_neg = -y_pos
#     # x = np.hstack((x_temp, x_temp))
#     # y = np.hstack((y_pos, y_neg))
#
#     # fig = plt.figure()
#     # plt.scatter(x, y)
#     # plt.show()
#
#     # x_dot = np.linspace(-.1, .1, 2*points)
#     # y_dot = np.linspace(-.1, .1, 2*points)
#     #
#     # alphas = []
#     # for i, value in enumerate(x):
#     #     alpha_i = np.rad2deg(np.arctan2(y[i], x[i] - 1))  # arctan2 from -180 to 180
#     #     if alpha_i < 0:
#     #         alpha_i += 360
#     #     alphas.append(alpha_i)
#     #
#     # psis = []
#     # for j, value in enumerate(x_dot):
#     #     psi = np.rad2deg(np.arctan2(y_dot[j], x_dot[j]))
#     #     if psi < 0:
#     #         psi += 360
#     #     psis.append(psi)
#     #
#     # betas = []
#     # for k, value in enumerate(psis):
#     #     beta_i = psis[k] - alphas[k] - 90
#     #     if beta_i < 0:
#     #         beta_i += 360
#     #     betas.append(-beta_i)
#
#     # Alpha_I, Beta_I = np.meshgrid(alphas, betas)
#     alphas = np.linspace(0, 360, 2*points)
#     betas = np.linspace(-180, 0, 2*points)
#     C_se = np.linspace(2.9, 3.0009, 2*points)
#     Alpha_I, Beta_I = np.meshgrid(alphas, betas)
#     for i, alpha in enumerate(Alpha_I[0]):
#         for j, beta in enumerate(Beta_I[0]):
#             x = r * np.cos(np.deg2rad(Alpha_I[i, j])) + 1 - mu
#             y = r * np.sin(np.deg2rad(Beta_I[i, j]))
#             r1 = np.sqrt((x + mu) ** 2 + y ** 2)
#             r2 = np.sqrt((x - 1 + mu) ** 2 + y ** 2)
#             U = 0.5 * (x ** 2 + y ** 2) + (1 - mu) / r1 + mu / r2 + 0.5 * mu * (1 - mu)
#             for k, cse in enumerate(C_se):
#                 v = np.sqrt(2 * U - cse)
#                 x_dot = -v * np.sin(np.deg2rad(Alpha_I[i, j] - Beta_I[i, j]))
#                 y_dot = v * np.cos(np.deg2rad(Alpha_I[i, j] - Beta_I[i, j]))
#                 Jacobi = 2 * U - x_dot ** 2 - y_dot ** 2
#                 print(cse)
#                 print(Jacobi)
#
#
#
#
#     # Create a 3D figure
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#
#     # Plot the 3D surface plot with color mapping
#     surface = ax.plot_surface(Alpha_I, Beta_I, V, cmap='viridis', facecolors=plt.cm.viridis(Jacobi))
#
#     # Add color bar for reference (optional)
#     plt.colorbar(surface)
#
#     # Show the plot
#     plt.show()


#
#
#
# alpha_beta_jacobi_from_manifolds()
# reverse_alpha_beta_jacobi()

