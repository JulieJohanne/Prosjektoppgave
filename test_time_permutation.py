import numpy as np
from datetime import datetime
from skitools.numpytools import ndgrid

num_data = 10**4
size = 10
num_sim_per_data = 100
volatility = (np.sqrt(0.2)/252)*np.random.uniform(low=0, high=1, size=size)
initial_price = np.random.uniform(low=1, high=6, size=size)
t = np.arange(0, 59, 1)
time_to_maturity = t[-1]
strike_price = np.random.uniform(low=0.5, high=6.5, size=size)
risk_free_rate = np.random.uniform(low=0.015, high=0.025, size=size)
X = np.zeros((num_data, 5))


startTime = datetime.now()
x = np.arange(1,5,1)
y = np.arange(6,10,1)
z = np.arange(11,15,1)

[X, Y, Z] = n

print (sorted(list(set(itertools.permutations([1,1,1,2,2,2,3,3,3,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8,9], 3)))))