import numpy as np
from scipy.stats import norm
import pandas as pd


def get_data_alpha_vantage(directory, filename):
    df = pd.read_csv(directory + '/' + filename, delimiter=',', header=0)
    df = df[['Timestamp', 'Close']]
    return df.get_values()


def parameter_estimation(data):
    # Input: 2D numpy array with first column as time and second column as stock price
    # Output: Annualised expected return and volatility
    stock_price = np.float64(data)
    log_returns = np.log(stock_price[1:]/stock_price[0:-1])
    m = np.mean(log_returns)
    volatility = np.std(log_returns)  # np.sqrt(1/(len(S)-2)*np.sum(np.power((S-m), 2)))
    mu = m - 0.5*volatility**2
    return mu*250, volatility*np.sqrt(250)


def simulate_random_walk(exp_return, volatility, current_price, time_to_maturity, no_time_steps, no_simulations):
    # Input: Annualised expected return, annualised volatility, current stock price, time to maturity in days,
    # number of time steps, number of simulations
    # Output: Simulated stock price
    delta_t = time_to_maturity/no_time_steps
    stock_price = np.zeros((no_simulations, no_time_steps + 1))
    stock_price[:, 0] = current_price
    for i in range(no_simulations):
        for j in range(no_time_steps):
            Z = np.random.normal(loc=0, scale=1, size=1)
            stock_price[i, j+1] = stock_price[i, j] * (1 + exp_return*delta_t + volatility * np.sqrt(delta_t) * Z)
    return stock_price


def simulate_stock_price_historical(data, time_to_maturity, no_of_time_steps, no_of_simulations):
    exp_return, volatility = parameter_estimation(data)
    time_to_maturity_annualised = time_to_maturity / 250  # 30 days maturity divided by 250  trading days
    N, M = no_of_time_steps, no_of_simulations
    S0 = data[-1*time_to_maturity, 1]  # current price
    return simulate_random_walk(exp_return, volatility, N, time_to_maturity_annualised, S0, M)


def simulate_BS(stock_price, exercise_price, time_to_maturity, risk_free_rate, current_time):
    S, K, T, r = stock_price[-1*time_to_maturity, 1], exercise_price, time_to_maturity / 250, risk_free_rate,
    t = current_time
    _, volatility = parameter_estimation(stock_price)
    d1 = (np.log(S / K) + (r + 0.5 * np.power(volatility, 2) * (T - t))) / (volatility * np.sqrt(T - t))
    d2 = (np.log(S / K) + (r - 0.5 * np.power(volatility, 2)) * (T - t)) / (volatility * np.sqrt(T - t))
    return S * norm.cdf(d1) - K * np.exp(-1 * r * (T - t)) * norm.cdf(d2)


def expected_option_price(stock_price, exercise_price, risk_free_rate, current_time):
    stock_price_at_expiry, time_to_expiry = stock_price[:, -1], len(stock_price)
    C = np.maximum(stock_price_at_expiry-exercise_price, np.zeros((len(stock_price_at_expiry))))
    return np.exp(-risk_free_rate*(time_to_expiry-current_time))*np.mean(C)


def true_option_price(stock_price, exercise_price, risk_free_rate):
    time_to_expiry, stock_price_at_expiry = len(stock_price), stock_price[-1]
    return np.exp(-risk_free_rate*time_to_expiry)*np.maximum(stock_price_at_expiry-exercise_price,  0)

