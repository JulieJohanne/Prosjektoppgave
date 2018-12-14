import numpy as np
import matplotlib.pyplot as plt
import simulate_stock_price as sp
import os


#directory = 'C:/Users/jju95/OneDrive/Documents/PYTHON/Prosjektoppgave/Prosjektoppgave/Prosjektoppgave'
data = np.loadtxt('stock_data.txt')

""""
# Look at historical volatility by splitting time periods
data10, data20 = np.array_split(data[0, :], 2)
data11, data21, data31, data41 = np.array_split(data[0, :], 4)
volatility_split = np.zeros(7)
_, volatility_split[0] = sp.parameter_estimation(data[0, :])
_, volatility_split[1] = sp.parameter_estimation(data10)
_, volatility_split[2] = sp.parameter_estimation(data20)
_, volatility_split[3] = sp.parameter_estimation(data11)
_, volatility_split[4] = sp.parameter_estimation(data21)
_, volatility_split[5] = sp.parameter_estimation(data31)
_, volatility_split[6] = sp.parameter_estimation(data41)

#plt.plot(data[0, :])

plt.plot(1, volatility[0], '*')
plt.plot(2*np.ones(2), volatility[1:3], '*')
plt.plot(3*np.ones(4), volatility[3:], '*')
plt.legend(['whole dataset', 'split into 2', 'split into 4'])
plt.show()

# Look at historical volatility by looking at different time periods
volatility_period = np.zeros(4)
# 3 mo, 6 mo, 1 y, entire dataset
_, volatility_period[0] = sp.parameter_estimation(data[0, -92:])
_, volatility_period[1] = sp.parameter_estimation(data[0, -183:])
_, volatility_period[2] = sp.parameter_estimation(data[0, -365:])
_, volatility_period[3] = sp.parameter_estimation(data[0, :])

plt.plot(np.arange(4), volatility_period, '*')
plt.show()
"""

#Genreate input and output data
size_exercise_ratio, size_risk_free_rate = 10, 100
exercise_ratio = np.random.uniform(low=0.5, high=1.5, size=size_exercise_ratio)
risk_free_rate = np.random.uniform(low=0.015, high=0.025, size=size_risk_free_rate)
time_to_expiry = 62  # 3 months
X = np.empty((0, 5), float)  # np.zeros((size_exercise_ratio*size_risk_free_rate*len(data[:,0])**3, 5))
Y = np.array([])  # np.zeros(size_exercise_ratio*size_risk_free_rate*len(data[:,0])**3)
for company in np.arange(len(data[:, 0])):
    exercise_price = exercise_ratio * data[company, -1]
    _, volatility = sp.parameter_estimation(data[company, (-time_to_expiry-365):(-time_to_expiry)])
    initial_price = data[company, -time_to_expiry]
    for i, r in enumerate(risk_free_rate):
        for j, E in enumerate(exercise_price):
            Y = np.append(Y, (sp.true_option_price(data[company, -time_to_expiry:], E, r)))
            X = np.vstack([X, [volatility, initial_price, E, r, time_to_expiry]])

np.savetxt('output_new.txt', Y)
np.savetxt('input_new.txt', X)
