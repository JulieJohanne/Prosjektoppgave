import numpy as np
import matplotlib.pyplot as plt
import simulate_stock_price as sp
from scipy.stats import norm

if __name__ == '__main__':
    """
    filename = 'MSFT-daily.csv'
    data = sp.get_data_alpha_vantage(filename)
    no_time_steps, no_simulations, time_to_maturity = 100, 500, 60
    risk_free_rate = 0.02

    stock_price_simulated = sp.simulate_stock_price_historical(data, time_to_maturity, no_time_steps, no_simulations)

    for i in range(no_simulations):
        plt.plot(np.arange(0, time_to_maturity, time_to_maturity / (no_time_steps + 1)), stock_price_simulated[i, :])
    plt.plot(np.arange(100 - time_to_maturity, time_to_maturity, 1), data[-100:, 1], 'b')
    plt.show()
    
    time_to_maturity = 90
    data = np.loadtxt('stock_data_150_5.txt')
    stock_price_simulated_0 = sp.simulate_stock_price_historical(data[0, :-time_to_maturity], time_to_maturity, 900,
                                                                 100)
    stock_price_simulated_1 = sp.simulate_stock_price_historical(data[1, :-time_to_maturity], time_to_maturity, 900,
                                                                 100)
    stock_price_simulated_2 = sp.simulate_stock_price_historical(data[2, :-time_to_maturity], time_to_maturity, 900,
                                                                 100)
    stock_price_simulated_3 = sp.simulate_stock_price_historical(data[3, :-time_to_maturity], time_to_maturity, 900,
                                                                 100)
    fig, ax = plt.subplots(nrows=4, ncols=1)
    for i in range(100):
        ax[0].plot(np.arange(0, time_to_maturity, time_to_maturity / (900 + 1)), stock_price_simulated_0[i, :])
    ax[0].plot(np.arange(-200+time_to_maturity, time_to_maturity, 1), data[0, -200:], 'b', label='Real stock price')
    ax[0].legend()
    #ax[0].xlabel('Time (Days)')
    #ax[0].ylabel('Stock price ($)')

    for i in range(100):
        ax[1].plot(np.arange(0, time_to_maturity, time_to_maturity / (900 + 1)), stock_price_simulated_1[i, :])
    ax[1].plot(np.arange(-200 + time_to_maturity, time_to_maturity, 1), data[0, -200:], 'b',
                  label='Real stock price')
    ax[1].legend()
    #ax[1, 0].xlabel('Time (Days)')
    #ax[1, 0].ylabel('Stock price ($)')

    for i in range(100):
        ax[2].plot(np.arange(0, time_to_maturity, time_to_maturity / (900 + 1)), stock_price_simulated_2[i, :])
    ax[2].plot(np.arange(-200 + time_to_maturity, time_to_maturity, 1), data[0, -200:], 'b',
                  label='Real stock price')
    ax[2].legend()
    #ax[2].xlabel('Time (Days)')
    #ax[2].ylabel('Stock price ($)')

    for i in range(100):
        ax[3].plot(np.arange(0, time_to_maturity, time_to_maturity / (900 + 1)), stock_price_simulated_3[i, :])
    ax[3].plot(np.arange(-200 + time_to_maturity, time_to_maturity, 1), data[0, -200:], 'b',
                  label='Real stock price')
    ax[3].legend()
    #ax[3].xlabel('Time (Days)')
    #ax[3].ylabel('Stock price ($)')


   # plt.savefig('MC_100_simulations.png')
   # plt.show()
    
    # Plot activation functions
    # Sigmoid, ELU, softsign
    x = np.arange(-10, 10, 0.1)
    a = 1
    fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True)
    axes[0, 0].plot(x, 1/(1+np.exp(-x)))
    axes[0, 0].set_title('Sigmoid')
   # axes[0, 1].plot(x, np.maximum(0, x))
    axes[0, 1].plot(x, np.log(1+np.exp(x)))
    axes[0, 1].set_title('Softplus')
    axes[1, 1].plot(x, a*(np.exp(x)-1)*(x < 0) + x*(x >= 0))
    axes[1, 1].set_title('ELU')
    axes[1, 0].plot(x, x/(1+np.abs(x)))
    axes[1, 0].set_title('Softsign')
    plt.savefig('Activation_functions.png')
    plt.show()
    """
    """
    time_to_maturity = 90
    data = np.loadtxt('stock_data_150_5.txt')
    S = np.arange(0, 50, 1)
    vol = np.arange(0.2, 0.6, 0.1)
    BS_call = np.zeros((len(vol), len(S)))
    for i in range(len(vol)):
        for j in range(len(S)):
            BS_call[i, j] = sp.simulate_BS(S[j], 30, time_to_maturity, 0.02, 0, volatility=vol[i])

    #S = np.arange(0, 50, 1)

    T = np.array([35, 30, 25])
    BS_put = np.zeros((len(T), len(S)))
    for i in range(len(T)):
        for j in range(len(S)):
            BS_put[i, j] = sp.simulate_put_BS(S[j], T[i], 90, 0.02, 0, volatility=0.4)

    plt.plot(S, BS_call[0, :], label='Black-Scholes solution, vol = 0.1')
    plt.plot(S, BS_call[1, :], label='Black-Scholes solution, vol = 0.2')
    plt.plot(S, BS_call[2, :], label='Black-Scholes solution, vol = 0.3')
    plt.plot(S, BS_call[3, :], label='Black-Scholes solution, vol = 0.4')
    plt.plot(S, np.maximum(S - 30, 0), label='Payoff function')
    plt.xlabel('Asset Price, S')
    plt.ylabel('Option Price, V(S,t)')
    plt.title('Call Option Price')
    plt.legend()
    plt.savefig('call_option_BS.png')
    plt.show()

    plt.plot(S, BS_put[0, :], label='Black-Scholes solution, E = 10')
    plt.plot(S, BS_put[1, :], label='Black-Scholes solution, E = 30')
    plt.plot(S, BS_put[2, :], label='Black-Scholes solution, E = 90')
    plt.plot(S, np.maximum(30 - S, 0), label='Payoff function')
    plt.xlabel('Asset Price, S')
    plt.ylabel('Option Price, V(S,t)')
    plt.title('Put Option Price')
    plt.legend()
    plt.savefig('put_option_BS.png')
    plt.show()
    """

    # Plot activation functions
    # Sigmoid, ELU, softsign
    x = np.arange(-10, 10, 0.1)
    a = 1
    #fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True)
    plt.plot(x, 1 / (1 + np.exp(-x)))
    plt.title('Sigmoid')
    plt.savefig('sigmoid.png')
    plt.show()
    plt.plot(x, np.log(1 + np.exp(x)))
    plt.title('Softplus')
    plt.savefig('softplus.png')
    plt.show()
    plt.plot(x, a * (np.exp(x) - 1) * (x < 0) + x * (x >= 0))
    plt.title('ELU')
    plt.savefig('elu.png')
    plt.show()
    plt.plot(x, np.maximum(0, x))
    plt.title('ReLU')
    plt.savefig('relu.png')
    plt.show()