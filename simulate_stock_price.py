import numpy as np
# import montecarlo_normal as mc


def calculate_historical_values(data):
    S = data
    logreturn = np.log(S[:-2]/S[1:])
    historical_volatility = np.std(logreturn)
    historical_expected_return = np.mean(logreturn)
    return historical_volatility, historical_expected_return


def simulate_stock_price(num_of_sim, length_of_sim, exp_return, volatility, initial_price):
    S0, mu, sigma = initial_price, exp_return, volatility
    S = np.zeros((num_of_sim, length_of_sim+1))
    S[:, 0] = S0
    for i in range(num_of_sim):
        for j in range(length_of_sim):
            S[i, j+1] = S[i, j]*np.exp((mu-0.5*sigma**2) + sigma*np.random.normal(0, 1, 1))
    return S


def expected_option_price(stock_prices, strike_price, risk_free_rate):
    S, K, r = stock_prices, strike_price, risk_free_rate
    T = len(S)
    mean_S = np.mean(S, axis=1)
    C = np.exp(-r*T)*np.maximum(mean_S-K, np.zeros((len(S))))
    C_expected = np.mean(C)
    return C_expected





