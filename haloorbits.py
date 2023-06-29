import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import pandas as pd


def correct_x_halo(state, mu):

    # deconstruct state
    x, y, z, vx, vy, vz = state[0:6]  # position and velocity
    phi = np.reshape(state[6:], (6, 6))

    # set corrections at time T/2 for vx and vz to the negative of what was present at that time
    delvx = -vx
    delvz = -vz

    # compute x and z acceleration at T/2
    dUdx = -(mu * (mu + x - 1)) / np.power(((mu + x - 1) ** 2 + y ** 2 + z ** 2), (3 / 2)) - \
           ((1 - mu) * (mu + x)) / np.power(((mu + x) ** 2 + y ** 2 + z ** 2), (3 / 2)) + x
    dUdz = -(mu * z) / np.power(((mu + x - 1) ** 2 + y ** 2 + z ** 2), (3 / 2)) - \
           ((1 - mu) * z) / np.power(((mu + x) ** 2 + y ** 2 + z ** 2), (3 / 2))

    acc_x = dUdx + 2 * vy  # derivative of velocity is acceleration
    acc_z = dUdz

    temp = 1 / vy * np.matmul(np.array([[acc_x], [acc_z]]), np.array([[phi[1, 0], phi[1, 4]]]))
    temp2 = np.array([[phi[3, 0], phi[3, 4]], [phi[5, 0], phi[5, 4]]]) - temp
    correction = np.matmul(np.linalg.inv(temp2), np.array([delvx, delvz]))

    return correction


def correct_z_halo(state, mu):
    # deconstruct state
    x, y, z, vx, vy, vz = state[:6]  # position and velocity
    phi = np.reshape(state[6:], (6, 6))

    # set corrections at time T/2 for vx and vz to the negative of what was present at that time
    delvx = -vx
    delvz = -vz

    # compute x and z acceleration at T/2
    dUdx = -(mu * (mu + x - 1)) / np.power(((mu + x - 1) ** 2 + y ** 2 + z ** 2), (3 / 2)) - \
           ((1 - mu) * (mu + x)) / np.power(((mu + x) ** 2 + y ** 2 + z ** 2), (3 / 2)) + x
    dUdz = - (mu * z) / np.power(((mu + x - 1) ** 2 + y ** 2 + z ** 2), (3 / 2)) - \
           ((1 - mu) * z) / np.power(((mu + x) ** 2 + y ** 2 + z ** 2), (3 / 2))

    acc_x = dUdx + 2 * vy  # derivative of velocity is acceleration
    acc_z = dUdz

    temp = 1 / vy * np.matmul(np.array([[acc_x], [acc_z]]), np.array([[phi[1, 2], phi[1, 4]]]))
    temp2 = np.array([[phi[3, 2], phi[3, 4]], [phi[5, 2], phi[5, 4]]]) - temp
    correction = np.matmul(np.linalg.inv(temp2), np.array([delvx, delvz]))

    return correction

def correct_x_vertical_lyapunov(state, mu):
    # deconstruct state
    x, y, z, vx, vy, vz = state[:6]  # position and velocity
    phi = np.reshape(state[6:], (6, 6))

    # set corrections at time T/2 for vx and vz to the negative of what was present at that time
    dely = -y
    delvx = -vx

    # compute x and z acceleration at T/2
    dUdx = -(mu * (mu + x - 1)) / np.power(((mu + x - 1) ** 2 + y ** 2 + z ** 2), (3 / 2)) - \
           ((1 - mu) * (mu + x)) / np.power(((mu + x) ** 2 + y ** 2 + z ** 2), (3 / 2)) + x

    acc_x = dUdx + 2 * vy  # derivative of velocity is acceleration

    temp = 1 / vz * np.matmul(np.array([[vy], [acc_x]]), np.array([[phi[2, 0], phi[2, 5]]]))
    temp2 = np.array([[phi[1, 0], phi[1, 5]], [phi[3, 0], phi[3, 5]]]) - temp
    correction = np.matmul(np.linalg.inv(temp2), np.array([dely, delvx]))

    return correction

def correct_vy_vertical_lyapunov(state, mu):
    # deconstruct state
    x, y, z, vx, vy, vz = state[:6]  # position and velocity
    phi = np.reshape(state[6:], (6, 6))

    # set corrections at time T/2 for vx and vz to the negative of what was present at that time
    dely = -y
    delvx = -vx

    # compute x and z acceleration at T/2
    dUdx = -(mu * (mu + x - 1)) / np.power(((mu + x - 1) ** 2 + y ** 2 + z ** 2), (3 / 2)) - \
           ((1 - mu) * (mu + x)) / np.power(((mu + x) ** 2 + y ** 2 + z ** 2), (3 / 2)) + x

    acc_x = dUdx + 2 * vy  # derivative of velocity is acceleration

    temp = 1 / vz * np.matmul(np.array([[vy], [acc_x]]), np.array([[phi[2, 0], phi[2, 5]]]))
    temp2 = np.array([[phi[1, 4], phi[1, 5]], [phi[3, 4], phi[3, 5]]]) - temp
    correction = np.matmul(np.linalg.inv(temp2), np.array([dely, delvx]))

    return correction


def correct_x_horizontal_lyapunov(state, mu):
    # deconstruct state
    x, y, z, vx, vy, vz = state[0:6]  # position and velocity
    phi = np.reshape(state[6:], (6, 6))

    # set corrections at time T/2 for vx and vz to the negative of what was present at that time
    delvx = -vx

    # compute x and z acceleration at T/2
    dUdx = -(mu * (mu + x - 1)) / np.power(((mu + x - 1) ** 2 + y ** 2 + z ** 2), (3 / 2)) - \
           ((1 - mu) * (mu + x)) / np.power(((mu + x) ** 2 + y ** 2 + z ** 2), (3 / 2)) + x

    acc_x = dUdx + 2 * vy  # derivative of velocity is acceleration

    correction = delvx / (phi[3, 4] - acc_x * phi[1, 4] / vy)

    return correction


def gen_F_matrix(x, y, z, mu):

    F = np.zeros((6, 6))
    F[0:3, 3:6] = np.eye(3)
    F[3:6, 3:6] = np.array([[0, 2, 0], [-2, 0, 0], [0, 0, 0]])

    # Second order partials
    U_xx = (mu - 1)/((mu + x)**2 + y**2 + z**2)**1.5000 - mu/((mu + x - 1)**2 + y**2 + z**2)**1.5000 +\
           (0.7500*mu*(2*x + 2*mu - 2)**2)/((mu + x - 1)**2 + y**2 + z**2)**2.5000 - \
           (0.7500*(2*x + 2*mu)**2*(mu - 1))/((mu + x)**2 + y**2 + z**2)**2.5000 + 1
    U_yy = (mu - 1)/((mu + x)**2 + y**2 + z**2)**1.5000 - mu/((mu + x - 1)**2 + y**2 + z**2)**1.5000 + \
           (3*mu*y**2)/((mu + x - 1)**2 + y**2 + z**2)**2.5000 - \
           (3*y**2*(mu - 1))/((mu + x)**2 + y**2 + z**2)**2.5000 + 1
    U_zz = (mu - 1)/((mu + x)**2 + y**2 + z**2)**1.5000 - mu/((mu + x - 1)**2 + y**2 + z**2)**1.5000 +\
           (3*mu*z**2)/((mu + x - 1)**2 + y**2 + z**2)**2.5000 - (3*z**2*(mu - 1))/((mu + x)**2 + y**2 + z**2)**2.5000
    U_xy = (1.5000*mu*y*(2*x + 2*mu - 2))/((mu + x - 1)**2 + y**2 + z**2)**2.5000 -\
           (1.5000*y*(2*x + 2*mu)*(mu - 1))/((mu + x)**2 + y**2 + z**2)**2.5000
    U_xz = (1.5000*mu*z*(2*x + 2*mu - 2))/((mu + x - 1)**2 + y**2 + z**2)**2.5000 -\
           (1.5000*z*(2*x + 2*mu)*(mu - 1))/((mu + x)**2 + y**2 + z**2)**2.5000
    U_yz = (3*mu*y*z)/((mu + x - 1)**2 + y**2 + z**2)**2.5000 -\
           (3*y*z*(mu - 1))/((mu + x)**2 + y**2 + z**2)**2.5000
    U_yx = U_xy
    U_zx = U_xz
    U_zy = U_yz

    F[3:6, 0:3] = np.array([[U_xx, U_xy, U_xz], [U_yx, U_yy, U_zy], [U_zx, U_zy, U_zz]])

    return F


def model(state, time, mu=0.01215):
    # Define the dynamics of the system
    # state: current state vector
    # time: current time
    # return: derivative of the state vector

    x, y, z, vx, vy, vz = state[:6]  # position and velocity
    phi = np.reshape(state[6:], (6, 6))

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

    F = gen_F_matrix(x, y, z, mu)

    dphidt = np.matmul(F, phi)

    return np.hstack((np.array(dXdt), dphidt.ravel()))


def halo_get_initial_X0(guess, mu):

    # good_X0 = [0.7232683951677348, 0.0, 0.04, 0.0, 0.19801932688532664, 0.0]
    tol_vx = 1
    tol_vz = 1
    # initial condition
    x_0, y_0, z_0, vx_0, vy_0, vz_0 = guess
    phi_0 = np.eye(6)
    start = 0
    time_mult = 1e13
    end = 4 * time_mult  # temporarily bigger
    space = 600
    ident = 0
    X_0 = [x_0, y_0, z_0, vx_0, vy_0, vz_0]  # [x y z vx vy vz] for halo: [x_0 0 z_0 0 vy_0 0]
    halfT = 0

    while (tol_vx > 10e-8) and (tol_vz > 10e-8):

        # time points
        t = np.linspace(start, end, space)  # time is non-dimensionalized
        t = t / time_mult

        state = np.hstack((np.array(X_0), phi_0.ravel()))

        # solve ODE
        res = odeint(model, state, t, args=(mu,))

        if ident == 0:  # find T/2
            for i in range(len(res)):
                if i > 1:
                    if abs(res[i-1, 1]) < 10e-11:  # identified T/2
                        ident = 1
                        start = 0
                        end = int(t[i] * time_mult)  # set T/2 as the end
                        X_0 = [x_0, y_0, z_0, vx_0, vy_0, vz_0]  # reset
                        phi_0 = np.eye(6)
                        halfT = int(t[i] * time_mult)
                        break
                    elif res[i, 1] * res[i - 1, 1] < 0:  # crossed T/2
                        start = int(t[i-1] * time_mult)
                        end = int(t[i] * time_mult)
                        X_0 = res[i-1, :6]
                        phi_0 = np.reshape(res[i-1, 6:], (6, 6))
                        break
        else:
            if tol_vx > 1e-8:  # if x dot norm larger that z dot norm, correct along x
                delx_0, delyd_0 = correct_x_halo(res[-1, :], mu)
                x_0 = x_0 + delx_0  # update new initial conditions
                vy_0 = vy_0 + delyd_0
            elif tol_vz > 1e-8:  # correct along z
                delz_0, delyd_0 = correct_z_halo(res[-1, :], mu)
                z_0 = z_0 + delz_0  # update new initial conditions
                vy_0 = vy_0 + delyd_0

            # Reinitialize
            ident = 0
            start = 0
            end = 5 * time_mult  # temporarily bigger

            X_0 = [x_0, y_0, z_0, vx_0, vy_0, vz_0]  # update

            tol_vx = abs(res[-1, 3])  # at T/2 because we set it as the end previously
            tol_vz = abs(res[-1, 5])

    return X_0, halfT, time_mult

def vertical_lyapunov_get_initial_X0(guess, mu):
    # good_X0 = [0.7232683951677348, 0.0, 0.04, 0.0, 0.19801932688532664, 0.0]
    tol_y = 1
    tol_vx = 1
    # initial condition
    x_0, y_0, z_0, vx_0, vy_0, vz_0 = guess
    phi_0 = np.eye(6)
    start = 0
    time_mult = 1e13
    end = 8 * time_mult  # temporarily bigger
    space = 600
    ident = 0
    X_0 = [x_0, y_0, z_0, vx_0, vy_0, vz_0]  # [x y z vx vy vz] for halo: [x_0 0 z_0 0 vy_0 0]
    halfT = 0
    crossed_once = 0
    iterations = 0
    while (tol_y > 10e-8) and (tol_vx > 10e-8):
        # print(iterations)
        # print(X_0)
        # if iterations > 20:
        #     X_0 = guess
        #     X_0[4] = X_0[4] + 0.00001
        #     iterations = 0
        #     continue

        # time points
        t = np.linspace(start, end, space)  # time is non-dimensionalized
        t = t / time_mult

        state = np.hstack((np.array(X_0), phi_0.ravel()))

        # solve ODE
        res = odeint(model, state, t, args=(mu,))

        if ident == 0:  # find T/2
            for i in range(len(res)):
                if abs(res[i - 1, 2]) < 10e-11 and i > 1:  # crossed zero second time
                    ident = 1
                    start = 0
                    end = int(t[i] * time_mult)  # set T/2 as the end
                    X_0 = [x_0, y_0, z_0, vx_0, vy_0, vz_0]  # reset
                    phi_0 = np.eye(6)
                    halfT = int((t[i]) * time_mult)
                    break
                elif res[i, 2] * res[i - 1, 2] < 0 and i > 1:  # crossed T/2
                    start = int(t[i - 1] * time_mult)
                    end = int(t[i] * time_mult)
                    X_0 = res[i - 1, :6]
                    phi_0 = np.reshape(res[i - 1, 6:], (6, 6))
                    break
        else:
            if tol_y > 1e-8:  # if x dot norm larger that z dot norm, correct along x
                delx_0, delvz_0 = correct_x_vertical_lyapunov(res[-1, :], mu)
                x_0 = x_0 + delx_0  # update new initial conditions
                vz_0 = vz_0 + delvz_0
            elif tol_vx > 1e-8:  # correct along z
                delvy_0, delvz_0 = correct_vy_vertical_lyapunov(res[-1, :], mu)
                vy_0 = vy_0 + delvy_0  # update new initial conditions
                vz_0 = vz_0 + delvz_0

            # Reinitialize
            ident = 0
            start = 0
            end = 5 * time_mult  # temporarily bigger

            X_0 = [x_0, y_0, z_0, vx_0, vy_0, vz_0]  # update

            tol_y = abs(res[-1, 1])  # at T/2 because we set it as the end previously
            tol_vx = abs(res[-1, 3])
        iterations = iterations + 1

    return X_0, halfT, time_mult

def horizontal_lyapunov_get_initial_X0(guess, mu):

    tol_vx = 1
    # initial condition
    x_0, y_0, z_0, vx_0, vy_0, vz_0 = guess
    phi_0 = np.eye(6)
    start = 0
    time_mult = 1e16
    end = 5 * time_mult  # temporarily bigger
    space = 600
    ident = 0
    X_0 = [x_0, y_0, z_0, vx_0, vy_0, vz_0]  # [x y z vx vy vz] for halo: [x_0 0 z_0 0 vy_0 0]
    halfT = 0
    iterations = 0
    while (tol_vx > 10e-8):

        t = np.linspace(start, end, space)  # time is non-dimensionalized
        t = t / time_mult

        state = np.hstack((np.array(X_0), phi_0.ravel()))

        # solve ODE
        res = odeint(model, state, t, args=(mu,))

        if ident == 0:  # find T/2
            for i in range(len(res)):
                if i > 1:
                    if abs(res[i-1, 1]) < 10e-11:  # identified T/2
                        ident = 1
                        start = 0
                        end = int(t[i] * time_mult)  # set T/2 as the end
                        X_0 = [x_0, y_0, z_0, vx_0, vy_0, vz_0]  # reset
                        phi_0 = np.eye(6)
                        halfT = int(t[i] * time_mult)
                        break
                    elif res[i, 1] * res[i - 1, 1] < 0:  # crossed T/2
                        start = int(t[i-1] * time_mult)
                        end = int(t[i] * time_mult)
                        X_0 = res[i-1, :6]
                        phi_0 = np.reshape(res[i-1, 6:], (6, 6))
                        break
        else:
            if tol_vx > 1e-8:  # if x dot norm larger that z dot norm, correct along x
                delyd_0 = correct_x_horizontal_lyapunov(res[-1, :], mu)
                vy_0 = vy_0 + delyd_0

            # Reinitialize
            ident = 0
            start = 0
            end = 5 * time_mult  # temporarily bigger

            X_0 = [x_0, y_0, z_0, vx_0, vy_0, vz_0]  # update

            tol_vx = abs(res[-1, 3])  # at T/2 because we set it as the end previously
        iterations = iterations + 1


    print(X_0)
    return X_0, halfT, time_mult

def jacobi(res, mu):

    x, y, z, vx, vy, vz = res
    r1 = np.sqrt((x + mu) ** 2 + y ** 2)
    r2 = np.sqrt((x - 1 + mu) ** 2 + y ** 2)
    U = 0.5 * (x ** 2 + y ** 2) + (1 - mu) / r1 + mu / r2

    return 2*U - vx**2 - vy**2 - vz**2


# good_X0 = [0.7232683951677348, 0.0, 0.04, 0.0, 0.19801932688532664, 0.0]  # mu = 0.04 from Howell for halo L1
# good_X0 = [0.8234, 0., 0.0224, 0., 0.1343, 0.]  # mu = em for Halo L1
# good_X0 = [0.8989995853538256, 0.0, -0.3906000000000003, 0.0, 0.10213119259847761, 0.0] # mu = em for Halo L1
# good_X0 = [0.7816, 0., 0., 0., 0.4432, 0.]  # mu = em for Horizontal Lyapunov L1
# good_X0 = [0.7816, 0., 0., 0., 0.4432, 0.0000]  # mu = em for axials L1 - you can use vertical Lyapunov code
# good_X0 = [0.25, 0., 0., 0., -1.83361401, -1.01998345] # mu = 0.499 for vertical lyapunov l1
# good_X0 = [ 9.91656301e-01,  6.73776287e-09,  0., -1.27840278e-08, 4.97184728e-03, -2.31393958e-02] # mu = se, l1 vertical lya
# good_X0 = [0.9903500018005715, 6.73776287e-09, 0.0, -1.27840278e-08, 0.0007718472800000006, -0.010417215114751176] # mu = se, l1 vertical
# good_X0 = [0.9870554733155437, 0., 0.0, 0., 0.0245251097803396, 0.]  # mu = se, l1 horizontal lyapunov
# good_X0 = [0.9970496786521372, 0., 0.0050185527811233, 0., -0.025691693196319, 0.]  # mu = se, l1 halo
# good_X0 = [1.0000599364305272, 0.0, 0.0021185527811233, 0.0, -0.05128769156354202, 0.0] # mu = se, l1 halo

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

vlyap_SE_L1_ini = [[0.9903500018005715, 6.73776287e-09, 0.0, -1.27840278e-08, 0.0007718472800000006, -0.010417215114751176],
                  np.array([0.000, 0., 0., 0., 0.0001, -0.0001]), 70, 'vlyap']  # max is 100
hlyap_SE_L1_ini = [[0.9870554733155437, 0., 0.0, 0., 0.0245251097803396, 0.], np.array([-0.0001, 0., 0., 0., 0.000, 0.000]),
                  70, 'hlyap']  # max 80
hlyap_SE_L1_ini = [[0.9791554733155445, 0.0, 0.0, 0.0, 0.041713004131169205, 0.0], np.array([0.0001, 0., 0., 0., 0.000, 0.000]),
                  70, 'hlyap']  # max 100
halo_SE_L1_ini = [[1.0000599364305272, 0.0, 0.0021185527811233, 0.0, -0.05128769156354202, 0.0],
                  np.array([0., 0., 0.00001, 0., 0.000, 0.000]), 70, 'halo']  # max around 290
# vlyap_SE_L2_ini = [[1.00900018005715, -6.73776287e-09, 0.0, 1.27840278e-08, -0.0007718472800000006, 0.010417215114751176],
#                   np.array([0.000, 0., 0., 0., 0.0001, -0.0001]), 2, 'vlyap']
hlyap_SE_L2_ini = [[1.0176999999999994, 0.0, 0.0, 0.0, -0.0357697193603642, 0.0], np.array([-0.0001, 0., 0., 0.000, 0.0001, 0.000]),
                 70, 'hlyap']  # max is 70
halo_SE_L2_ini = [[1.0000, 0.0, -0.002538552781123301, 0.0, 0.04651063631928344, 0.0],
                  np.array([0., 0., 0.00001, 0., 0.000, 0.000]), 70, 'halo']  # max around 210
ini_conditions = [vlyap_SE_L1_ini, hlyap_SE_L1_ini, halo_SE_L1_ini, hlyap_SE_L2_ini, halo_SE_L2_ini]
# ini_conditions = [hlyap_SE_L1_ini]

space = 1200
start = 0
phi_0 = np.eye(6)
results = [[], [], []]
plt.figure(4)
ax = plt.axes(projection='3d')

for ini_cond in ini_conditions:

    print(ini_cond[3])
    num_families = ini_cond[2]
    ini_guess = ini_cond[0]
    continuation = ini_cond[1]

    for i in range(num_families):

        print(i)

        if ini_cond[3] == 'vlyap':
            X_0, halfT, time_mult = vertical_lyapunov_get_initial_X0(ini_guess, mu)
            j = 0
        elif ini_cond[3] == 'hlyap':
            X_0, halfT, time_mult = horizontal_lyapunov_get_initial_X0(ini_guess, mu)
            j = 1
        else:
            X_0, halfT, time_mult = halo_get_initial_X0(ini_guess, mu) # for halo orbits
            j = 2

        # Final integration over complete period
        # time points
        t = np.linspace(start, 2 * halfT, space)  # over T instead of T/2
        t = t/time_mult

        state = np.hstack((np.array(X_0), phi_0.ravel()))

        # solve ODE
        res = odeint(model, state, t, args=(mu,))

        results[j].append(res[:, :6])

        ini_guess = X_0 + continuation  # continuation halo

# fix the plotting to grab correct values
for i in range(len(results)):

    reses = results[i]

    for j in range(len(reses)):

        resi = reses[j]
        cj = [jacobi(resij, mu) for resij in resi]
        df = pd.DataFrame(resi, columns=["x", "y", "z", "vx", "vy", "vz"])
        df["cj"] = cj
        # df.to_csv('Horizontal Lyapunov Orbits - L1 - Sun and EMS - Integrations Results/' + str(cj[0]),
        #           sep=" ", header=True, index=None)

        if j == len(reses) - 1:
            if i % 3 == 0:
                color = 'b'
                label = 'Vert. Lya.'
            elif i % 3 == 1:
                color = 'r'
                label = 'Hori. Lya.'
            else:
                color = 'g'
                label = 'Halo'
        else:
            if i % 3 == 0:
                color = 'b'
                label = None
            elif i % 3 == 1:
                color = 'r'
                label = None
            else:
                color = 'g'
                label = None

        if j % 20 == 0:
            plt.figure(1)
            plt.plot(resi[:, 0], resi[:, 1], color=color, label=label)
            plt.xlabel('x')
            plt.ylabel('y')
            if j == len(reses) - 1:
                if i == len(results) - 1:
                    plt.scatter(earth[0], earth[1], color='b', label='Earth')
                    plt.scatter(l_1[0], l_1[1], color='black', label='L1')
                    plt.scatter(l_2[0], l_2[1], color='black', label='L2')
                    plt.legend()

            plt.figure(2)
            plt.plot(resi[:, 0], resi[:, 2], color=color, label=label)
            plt.xlabel('x')
            plt.ylabel('z')
            if j == len(reses) - 1:
                if i == len(results) - 1:
                    plt.scatter(earth[0], earth[2], color='b', label='Earth')
                    plt.scatter(l_1[0], l_1[2], color='black', label='L1')
                    plt.scatter(l_2[0], l_2[2], color='black', label='L2')
                    plt.legend()

            plt.figure(3)
            plt.plot(resi[:, 1], resi[:, 2], color=color, label=label)
            plt.xlabel('y')
            plt.ylabel('z')
            if j == len(reses) - 1:
                if i == len(results) - 1:
                    plt.scatter(earth[1], earth[2], color='b', label='Earth')
                    plt.scatter(l_1[1], l_1[2], color='black', label='L1')
                    plt.scatter(l_2[1], l_2[2], color='black', label='L2')
                    plt.legend()

            plt.figure(4)
            ax.plot3D(resi[:, 0], resi[:, 1], resi[:, 2], color=color, label=label)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.set_zlabel('z')
            if j == len(reses) - 1:
                if i == len(results) - 1:
                    ax.scatter(earth[0], earth[1], earth[2], color='b', label='Earth')
                    ax.scatter(l_1[0], l_1[1], l_1[2], color='black', label='L1')
                    ax.scatter(l_2[0], l_2[1], l_2[2], color='black', label='L2')
                    ax.legend()

plt.show()