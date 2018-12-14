import numpy as np
import simulate_stock_price as sp


num_data = 3**4
num_sim_per_data = 100
volatility = (np.sqrt(0.2)/252)*np.random.uniform(low=0, high=1, size=3)
initial_price = np.random.uniform(low=1, high=6, size=3)
exp_return = 0.1/252
t = np.arange(0, 59, 1)
time_to_maturity = t[-1]
strike_price = np.random.uniform(low=0.5, high=6.5, size=3)
risk_free_rate = np.random.uniform(low=0.015, high=0.025, size=3)
Y = np.zeros(num_data)
X = np.zeros((num_data, 5))
#Y[0, :] = 'Option price'
#X[0, :] = ['volatility', 'initial price', 'strike price', 'risk free_rate', 'time_to_maturity']

# Iterate over volatility
for i in range(len(volatility)):
    for j in range(len(initial_price)):
        for k in range(len(strike_price)):
            for l in range(len(risk_free_rate)):
                stock_price = sp.simulate_stock_price(num_sim_per_data, time_to_maturity, exp_return, volatility[i],
                                                      initial_price[j])
                Y[l + k * len(risk_free_rate) + j * len(risk_free_rate) * len(strike_price) + i * len(risk_free_rate) *
                  len(strike_price)*len(initial_price)] = sp.expected_option_price(stock_price, strike_price[k],
                                                                                   risk_free_rate[l])
                X[l + k * len(risk_free_rate) + j * len(risk_free_rate) * len(strike_price) + i * len(risk_free_rate) *
                  len(strike_price) * len(initial_price), :] = [volatility[i], initial_price[j], strike_price[k],
                                                                risk_free_rate[l], time_to_maturity]

np.savetxt('output.txt', Y)
np.savetxt('input.txt', X)
