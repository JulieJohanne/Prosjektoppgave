import numpy as np
import simulate_stock_price as sp
from sklearn.metrics import mean_squared_error


X = np.loadtxt('testX.txt')
print(len(X))
Y = np.loadtxt('testY.txt')
stock_data = np.loadtxt('stock_data_150_5.txt')
N, M = np.array([10]), np.array([10])

Y_predict_BS  = sp.simulate_BS(stock_price=X[:, 1], exercise_price=X[:, 2], time_to_expiry=X[:, 4], risk_free_rate=X[:,3], current_time=0, volatility = X[:, 0])

Y_predict_MC = np.zeros(len(X))
MC_score = np.zeros((len(N), len(M)))
for i,n in enumerate(N):
    for j,m in enumerate(M):
        for k in np.arange(len(X)):
                #print(k, n, m)
                expected_return, _ = sp.parameter_estimation(stock_data[(X[k, 0]).astype(int)/1000, :])
                stock_price = sp.simulate_random_walk(exp_return=expected_return, volatility=X[k, 0], current_price=X[k, 1], time_to_expiry=X[k, 4], no_time_steps=n, no_simulations=m)
                Y_predict_MC[k] = sp.expected_option_price(stock_price=stock_price, exercise_price=X[k, 2], risk_free_rate=X[k, 3], current_time=0)
        MC_score[i, j] = np.sqrt(mean_squared_error(Y, Y_predict_MC))
        print('Monte Carlo Score', (MC_score[i, j]), ' for n = ', n, ', m = ', m, '\n')

BS_score =  np.sqrt(mean_squared_error(Y, Y_predict_BS))
MC_score =   np.sqrt(mean_squared_error(Y, Y_predict_MC))
print('Prediction score with Black-Scholes: ', BS_score)
print('Prediction score with Monte-Carlo: \n', MC_score)
