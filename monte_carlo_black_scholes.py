import numpy as np
from sklearn.metrics import mean_squared_error

X = np.loadtxt('testX.txt')
Y = np.loadtxt('testY.txt')
N, M = np.arange(10, 1000, 10), np.arange(10, 1000, 10)

Y_predict_BS  = simulate_BS(stock_price=X[:, 1], strike_price=X[:, 2], time_to_expiry=X[:, 4], risk_free_rate=X[:,3], 0, volatility = X[:, 0])

Y_predict_MC = np.zeros((len(N), len(M)))
for _, n in enumerate(N):
    for _,m in enumerate(M):
        stock_price = simulate_random_walk(exp_return=?, volatility=X[:, 0], current_price=X[:, 1], time_to_expiry=X[:, 4], no_time_steps=,n no_simulations=m)
        Y_predict_MC = expected_option_price(stock_price, exercise_price=X[:, 2], risk_free_rate=X[:, 3], 0):

BS_score =  np.sqrt(mean_squared_error(Y, Y_predict_BS))
MC_score =   np.sqrt(mean_squared_error(Y, Y_predict_MC[-1, -1]))
print('Prediction score with Black-Scholes: ', BS_score)
print('Prediction score with Monte-Carlo: \n', MC_score)
