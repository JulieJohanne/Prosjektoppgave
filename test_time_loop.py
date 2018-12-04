import numpy as np
from datetime import datetime

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

for i in range(len(volatility)):
    for j in range(len(initial_price)):
        for k in range(len(strike_price)):
            for l in range(len(risk_free_rate)):
                X[l + k * len(risk_free_rate) + j * len(risk_free_rate) * len(strike_price) + i * len(risk_free_rate) *
                  len(strike_price) * len(initial_price), :] = [volatility[i], initial_price[j], strike_price[k],
                                                                risk_free_rate[l], time_to_maturity]

print(datetime.now() - startTime)