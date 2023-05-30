import numpy as np
from scipy.integrate import odeint


def mathieu(phi, t, delta, epsilon):
    # returns [\dot{u},\dot{p}] for Mathieu's equation
    # with parameters delta, epsilon
    return [phi[1], -(delta + epsilon * np.cos(t)) * phi[0]]


def monodromy(delta, epsilon):
    # returns the monodromy matrix Mon for Mathieu's equation
    # with parameters delta, epsilon and canonical initial conditions
    T = 2 * np.pi
    Mon = np.zeros((2, 2))
    i = 0
    for phi0 in np.eye(2):
        Mon[:, i] = odeint(mathieu, phi0, [0, T],
                    args=(delta, epsilon))[-1, :].T
        i += 1
    return Mon
