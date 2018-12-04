from alpha_vantage.timeseries import TimeSeries
import numpy as np
import time
#import pandas as pd


apikey = '82KAKRXYKTSMEUPU'
f = open('company_keys_nn.txt', 'r')
symbol = []
for line in f:
    symbol.append(line[:-1])
f.close()
#symbol.extend(['miii', 'll'])

t = 750
data = np.zeros((len(symbol), t))
for i, company in enumerate(symbol):
    ts = TimeSeries(apikey, output_format='pandas')
    print(company)
    df, metadata = ts.get_daily(company, outputsize='full')
    df = df[['4. close']]
    data[i, :] = df.get_values()[-t:].flatten()
    time.sleep(12)  # Only 5 API requests per minute

np.savetxt('stock_data.txt', data)


