import numpy as np
import simulate_stock_price as sp
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


data = np.loadtxt('train_data.txt')
index = data[:, 0]
X = data[:, 1:-2]
Y = data[:, -2]/data[:, 3]
moneyness = data[:, -1]
stock_data = np.loadtxt('stock_data.txt')
N, M = np.array([100]), np.array([100])  # 2**5, 2**6, 2**7, 2**8,  2**9])
"""
k = 1
m = 100
n = 200
expected_return, _ = sp.parameter_estimation(stock_data[(index[k]/1000).astype(int), (-62-365):(-62)])#sp.parameter_estimation(stock_data[(X[k, 0]).astype(int)/1000, :])
stock_price = sp.simulate_random_walk(exp_return=expected_return, volatility=X[k, 0], current_price=X[k, 1],
                                      time_to_expiry=X[k, 4], no_time_steps=n, no_simulations=m)
for i in range(m):
    plt.plot(np.linspace(0, 62, n+1), stock_price[i, :])
plt.plot(stock_data[(index[k]/1000).astype(int), -62:], 'b')
plt.show()
"""

Y_predict_BS = sp.simulate_BS(stock_price=X[:, 1], exercise_price=X[:, 2], time_to_expiry=X[:, 4],
                              risk_free_rate=X[:, 3], current_time=0, volatility=X[:, 0])/X[:, 2]


l = len(X)
Y_predict_MC = np.zeros(l)
MC_score = np.zeros((len(N), len(M)))
for i, n in enumerate(N):
    for j, m in enumerate(M):
        for k in np.arange(l):
                print(k)
                expected_return, _ = sp.parameter_estimation(stock_data[(index[k]/1000).astype(int),
                                                         (-X[k, 4].astype(int)-252):(-X[k, 4].astype(int))])
                stock_price = sp.simulate_random_walk(exp_return=expected_return, volatility=X[k, 0], current_price=X[k, 1],
                                                  time_to_expiry=X[k, 4], no_time_steps=n, no_simulations=m)
                Y_predict_MC[k] = sp.expected_option_price(stock_price=stock_price, exercise_price=X[k, 2],
                                                       risk_free_rate=X[k, 3], current_time=0)/X[k, 2]
        MC_score[i, j] = np.sqrt(mean_squared_error(Y[:l], Y_predict_MC))
        print('Monte Carlo Score', (MC_score[i, j]), ' for n = ', n, ', m = ', m, '\n')

BS_score_ME = np.mean(Y_predict_BS - Y)
MC_score_ME = np.mean(Y_predict_MC - Y)
BS_score_MSE = mean_squared_error(Y, Y_predict_BS)
MC_score_MSE = mean_squared_error(Y, Y_predict_MC)
BS_score_RMSE = np.sqrt(BS_score_MSE)
MC_score_RMSE = np.sqrt(MC_score_MSE)
BS_score_MAE = mean_absolute_error(Y, Y_predict_BS)
MC_score_MAE = mean_absolute_error(Y, Y_predict_MC)
BS_score_max = max(np.abs(Y-Y_predict_BS))
MC_score_max = max(np.abs(Y-Y_predict_MC))
print('Prediction score with Black-Scholes: ', BS_score_ME, 'ME')
print('Prediction score with Monte Carlo: ', MC_score_ME, 'ME')
print('Prediction score with Black-Scholes: ', BS_score_RMSE, 'RMSE')
print('Prediction score with Monte Carlo: ', MC_score_RMSE, 'RMSE')
print('Prediction score with Black-Scholes: ', BS_score_MSE, 'MSE')
print('Prediction score with Monte Carlo: ', MC_score_MSE, 'MSE')
print('Prediction score with Black-Scholes: ', BS_score_MAE, 'MAE')
print('Prediction score with Monte Carlo: ', MC_score_MAE, 'MAE')
print('Prediction score with Black-Scholes: ', BS_score_max, 'MAX')
print('Prediction score with Monte Carlo: ', MC_score_max, 'MAX')


# Plot predicted BS vs true data
plt.plot(Y, Y_predict_BS, '*')
plt.plot(np.linspace(0, 0.25, 100), np.linspace(0, 0.25,  100), 'k')
plt.xlabel('true values')
plt.ylabel('predicted values')
plt.title('True values vs predicted values using Black-Scholes')
plt.savefig('truevspred_BS_train.png')
plt.show()

# Plot predicted MC vs true data
plt.plot(Y[:l], Y_predict_MC, '*')
plt.plot(np.linspace(0, 0.25, 100), np.linspace(0, 0.25,  100), 'k')
plt.xlabel('true values')
plt.ylabel('predicted values')
plt.title('true values vs predicted values using Monte Carlo')
plt.savefig('truevspred_MC_train.png')
plt.show()

#plt.loglog(M, MC_score.T.flatten())
#plt.show()aa

"""
plt.plot(moneyness, Y_predict_BS-Y, '*')
plt.show()
"""
