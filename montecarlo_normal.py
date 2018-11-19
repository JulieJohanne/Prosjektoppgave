import numpy as np


def generate_normal_rv(numSim):
    count = 0
    z = np.zeros((numSim, 1))
    while count <= numSim-1:
        u = np.random.random(1)
        y = -np.log(u)
        u = np.random.random(1)
        if u <= np.exp((-(y-1)**2)/2):
            z_abs = y
        else:
            continue
        u = np.random.random(1)
        if u <= 0.5:
            z[count] = z_abs
        else:
            z[count] = -1*z_abs

        count += 1
    return z
