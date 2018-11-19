import numpy as np
import matplotlib.pyplot as plt
# import montecarlo_normal as mc
import simulate_stock_price as sp
from scipy.stats import norm

# Simulate stick prices
num_sim = 1000
t = np.arange(0, 59, 1)
time_to_maturity = t[-1]
exp_return = 0.1/252
volatility = np.sqrt(0.2)/252
start_price = 4
stock_price = sp.simulate_stock_price(num_sim, time_to_maturity, exp_return, volatility,
                                      start_price*np.ones(num_sim))

for i in range(num_sim):
    plt.plot(t, stock_price[i, :])
plt.plot(t, np.mean(stock_price, axis=0), color='black', linewidth=2)
plt.figure(1)
plt.title('Stock price simulation')
plt.show()

# Calculate option price from stock price
risk_free_rate = 0.02  # risk free rate
strike_price = 4.15  # strike price

# C = np.maximum(np.zeros(N), S[:, -1]-K)
# for i in range(N):
#     plt.plot(t, C[i]*np.exp(-r*(t[-1]-t)), 'k')
# plt.figure(2)
# plt.show()
# C_exp = np.mean(C)*np.exp(-r*(t[-1]-t))

C = sp.expected_option_price(stock_price, strike_price, risk_free_rate)
print('Expected option price is ', C)

# Calculate with BS
# d1 = 1/(sigma*np.sqrt(t[-1])+1-t)*(np.log(S0/K)+(r+0.5*sigma**2)*(t[-1]+1-t))
# d2 = d1-sigma*np.sqrt(t[-1]+1-t)
#
# C_BS = norm.cdf(d1)*S0-norm.cdf(d2)*K*np.exp(-r*(t[-1]+1-t))
#
# plt.plot(t, C_BS)
# plt.figure(4)
# plt.show()
#plt.plot(t, sp.simulate_stock_price(N, t[-1], 0.1/252, np.sqrt(0.2)/252, 4*np.ones(N)))
#plt.show()
#Z = mc.generate_normal_rv(N)
#plt.plot(np.arange(1, N + 1), Z, '-*')
#plt.show()

# for i in range(2):
#     for j in range(3):
#         for k in range(4):
#             for l in range(5):
#                 #print(l+k*5+j*4*5+i*3*4*5)


num_data = 10*10*10*10
num_sim_per_data = 100
volatility = (np.sqrt(0.2)/252)*np.random.uniform(low=0, high=1, size=10)
initial_price = np.random.uniform(low=1, high=6, size=10)
exp_return = 0.1/252
t = np.arange(0, 59, 1)
time_to_maturity = t[-1]
strike_price = np.random.uniform(low=0.5, high=6.5, size=10)
risk_free_rate = np.random.uniform(low=0.015, high=0.025, size=10)
Y = np.zeros(num_data)
X = np.zeros((num_data, 5))

print(len(volatility), len(strike_price))

x = np.loadtxt('input.txt')
print(x)