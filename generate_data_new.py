import numpy as np
import simulate_stock_price as sp
import os

# for i in range(1,101):
#     filename = 'comp' + str(i) + '.csv'
#     data[:, i-1] = sp.get_data_alpha_vantage(filename)[:, 1]

directory = 'C:/Users/jju95/OneDrive/Documents/PYTHON/Prosjektoppgave/Prosjektoppgave/Prosjektoppgave/Stock_prices_from_Alpha_Vantage'
data = np.zeros((100, 3))
#index = 0

f = open('company_keys.txt', 'w')

for filename in os.listdir(directory):
    # data[:, index] = sp.get_data_alpha_vantage(directory, filename)[:, 1]
    # index += 1
    f.write(filename[:-10] + '\n')
f.close()

