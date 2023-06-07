import numpy as np
import matplotlib.pyplot as plt

def dynamics(state, time):
    # Define the dynamics of the system
    # state: current state vector
    # time: current time
    # return: derivative of the state vector

    x, y, vx, vy = state  # position and velocity
    mu_SE = 3.04042e-6
    mu_em = 0.01215
    mu = mu_em

    # rp1 = np.sqrt((x + mu) ** 2 + y ** 2)
    # rp2 = np.sqrt((x - 1 + mu) ** 2 + y ** 2)
    # Om3 = 0.5 * (x ** 2 + y ** 2) + (1 - mu) / rp1 + mu / rp2 + 0.5 * mu * (1 - mu)

    dOm3dx = x - (mu * (2 * mu + 2 * x - 2))/(2 * np.power((mu + x - 1) ** 2 + y ** 2, 3/2)) + \
             ((2 * mu + 2 * x) * (mu - 1))/(2 * np.power((mu + x) ** 2 + y ** 2, 3/2))
    dOm3dy = y - (mu * y)/np.power((mu + x - 1) ** 2 + y ** 2, 3/2) + \
             (y * (mu - 1))/np.power((mu + x) ** 2 + y ** 2, 3/2)
    dxdt = vx  # derivative of position is velocity
    dydt = vy
    dvxdt = dOm3dx + 2*dydt  # derivative of velocity is acceleration
    dvydt = dOm3dy - 2*dxdt

    return np.array([dxdt, dydt, dvxdt, dvydt])

def propagate(initial_state, initial_time, final_time, time_step):
    # Propagate the dynamical system using RK4 method
    # initial_state: initial state vector
    # initial_time: initial time
    # final_time: final time
    # time_step: integration time step
    # return: time vector, state vector

    time = np.arange(initial_time, final_time + time_step, time_step)  # create time vector
    num_steps = len(time)
    state = np.zeros((num_steps, len(initial_state)))  # initialize state vector

    state[0] = initial_state  # set initial state

    for i in range(1, num_steps):
        current_state = state[i-1]
        current_time = time[i-1]

        k1 = dynamics(current_state, current_time)
        k2 = dynamics(current_state + 0.5 * time_step * k1, current_time + 0.5 * time_step)
        k3 = dynamics(current_state + 0.5 * time_step * k2, current_time + 0.5 * time_step)
        k4 = dynamics(current_state + time_step * k3, current_time + time_step)

        state[i] = current_state + (time_step / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)  # update state using RK4 method

    return time, state


if __name__ == '__main__':
    # Example usage
    initial_state = np.array([0.5, 0, 0.0000, 0.0000])  # initial position and velocity
    initial_time = 0.0
    final_time = 1 * 60 * 60
    time_step = 1.

    time, state = propagate(initial_state, initial_time, final_time, time_step)

    fig3 = plt.figure()
    plt.plot(state[:, 0], state[:, 1])

    plt.show()

    # Print the results
    for t, s in zip(time, state):
        print(f"Time: {t:.2f}, State: {s}")
