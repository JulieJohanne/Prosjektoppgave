import numpy as np
import matplotlib.pyplot as plt
import simulate_stock_price as sp
from scipy.stats import norm

if __name__ == '__main__':

    filename = 'MSFT-daily.csv'
    data = sp.get_data_alpha_vantage(filename)
    no_time_steps, no_simulations, time_to_maturity = 100, 500, 60
    risk_free_rate = 0.02

    stock_price_simulated = sp.simulate_stock_price_historical(data, time_to_maturity, no_time_steps, no_simulations)

    for i in range(no_simulations):
        plt.plot(np.arange(0, time_to_maturity, time_to_maturity / (no_time_steps + 1)), stock_price_simulated[i, :])
    plt.plot(np.arange(100 - time_to_maturity, time_to_maturity, 1), data[-100:, 1], 'b')
    plt.show()



