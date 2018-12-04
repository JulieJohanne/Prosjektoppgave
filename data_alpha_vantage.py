from alpha_vantage.timeseries import TimeSeries
import numpy as np
import time
#import pandas as pd


# Collect company symbols from txt file
f = open('company_keys.txt', 'r')
symbol = []
for line in f:
    symbol.append(line[:-1])
f.close()
#symbol.extend([company1, company2])

t = 750
data = np.zeros((len(symbol), t))
apikey = '82KAKRXYKTSMEUPU'
for i, company in enumerate(symbol):
    ts = TimeSeries(apikey, output_format='pandas')
    print(company)
    df, metadata = ts.get_daily(company, outputsize='full')
    df = df[['4. close']]
    data[i, :] = df.get_values()[-t:].flatten()
    time.sleep(12)  # Only 5 API requests per minute

np.savetxt('stock_data.txt', data)


