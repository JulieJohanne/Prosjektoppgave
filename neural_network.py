#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers
from keras import regularizers
from keras.callbacks import EarlyStopping, ModelCheckpoint 
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing
import time


# Set seed
#seed = 7
#np.random.seed(seed)

# Prepare data
data = np.column_stack((np.loadtxt('input_w_moneyness.txt'), np.loadtxt('output_w_moneyness.txt')))
data = np.column_stack((np.arange(len(data)), data))
np.random.shuffle(data)

"""
# Test for nan's
index_delete = []
for j in range(len(data)):
    for k in range(7):
        if np.isnan(data[j, k]):
            index_delete.append(j)
            #print('data has nan, index: ', j, k)
data = np.delete(data, index_delete, 0)
#X = preprocessing.scale(X)
#X = preprocessing.normalize(X) ValueError
"""
"""
moneyness = data[:, -1]
plt.xlabel('S/E')
plt.ylabel('Frequency')
plt.title('Histogram of moneyness')
plt.hist(moneyness)
plt.savefig('histogram_moneyness.png')
plt.show()
"""
otm_data = data[data[:,-1]  <= 0.98, :]
itm_data = data[data[:,-1] >= 1.02, :]
ntm_data = data[(data[:,-1] > 0.98) & (data[:,-1] < 1.02), :]
otmX = np.array([otm_data[:, 1], otm_data[:, 2]/otm_data[:, 3], otm_data[:, 4], otm_data[:, 5]/252]).T
otmY = np.array(otm_data[:, -2]/otm_data[:, 3])
itmX = np.array([itm_data[:, 1], itm_data[:, 2]/itm_data[:, 3], itm_data[:, 4], itm_data[:, 5]/252]).T
itmY = np.array(itm_data[:, -2]/itm_data[:, 3])
ntmX = np.array([ntm_data[:, 1], ntm_data[:, 2]/ntm_data[:, 3], ntm_data[:, 4], ntm_data[:, 5]/252]).T
ntmY =  np.array(ntm_data[:, -2]/ntm_data[:, 3])

np.savetxt('otm_data.txt', otm_data)
np.savetxt('itm_data.txt', itm_data)
np.savetxt('ntm_data.txt', ntm_data)

# Devide into training and test data
index_slice = int(np.round(0.8*len(data)))
train_data = data[1:index_slice, :]
test_data = data[index_slice:, :]
np.savetxt('test_data.txt', test_data)
np.savetxt('train_data.txt', train_data)

trainX = np.array([train_data[:, 1], train_data[:, 2]/train_data[:, 3], train_data[:, 4], train_data[:, 5]/252]).T 
trainY = np.array(train_data[:, -2]/train_data[:, 3])
testX = np.array([test_data[:, 1], test_data[:, 2]/test_data[:, 3], test_data[:, 4], test_data[:, 5]/252]).T 
testY = np.array(test_data[:, -2]/test_data[:, 3])

# Create model
layers = [100, 80, 60, 40, 20] 

model = Sequential()
model.add(Dense(layers[0], input_dim=np.shape(trainX)[1]))
#model.add(Dropout(0.1))
model.add(Dense(layers[1], activation='elu'))#,  kernel_regularizer=regularizers.l2(0.01))) 
#model.add(Dropout(0.1))
model.add(Dense(layers[2], activation='elu'))#,  kernel_regularizer=regularizers.l2(0.01)))  
#model.add(Dropout(0.1))
model.add(Dense(layers[3], activation='elu'))#,  kernel_regularizer=regularizers.l2(0.01)))
#model.add(Dropout(0.1))
model.add(Dense(layers[4], activation='elu'))#,  kernel_regularizer=regularizers.l2(0.01)))
#model.add(Dropout(0.1))
#model.add(Dense(layers[5], activation='softsign'))#,  kernel_regularizer=regularizers.l2(0.01)))
#model.add(Dropout(0.1))
model.add(Dense(1, activation='softplus'))    #activation='softsign')) softplus

model.summary()

# Compile model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae', 'mape'])

#callbacks = [EarlyStopping(monitor='val_loss', patience=2, verbose=2),
#             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

# Fit the model
#history = model.fit(trainX, trainY, epochs=no_epochs, callbacks=callbacks, batch_size=32,  verbose=2,
#        validation_split=0.2, validation_data=None, shuffle=True)
history = model.fit(trainX, trainY, epochs=100, batch_size=128,  verbose=2,
        validation_split=0.2, shuffle=True)
model.evaluate(trainX, trainY, batch_size=128, verbose=1)

#print(history.history.keys()) = ['loss', 'val_mean_absolute_percentage_error', 'mean_absolute_percentage_error', 'val_mean_squared_error', 'val_mean_absolute_error', 'mean_squared_error', 'val_loss', 'mean_absolute_error']
"""
# Plot MAE vs S/E, scatter plot
X = np.array([data[:, 1], data[:, 2]/data[:, 3], data[:, 4], data[:, 5]/252]).T
Y = np.array(data[:, -2]/data[:, 3])
predict = model.predict(X)
plt.plot(moneyness, predict-Y, '*')
plt.savefig('moneynessvserror.png')
plt.show()

# Make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainScore_ME = np.mean(trainPredict-trainY)
testScore_ME = np.mean(testPredict-testY)

trainScore_MSE = mean_squared_error(trainY, trainPredict)
testScore_MSE = mean_squared_error(testY, testPredict)

trainScore_RMSE = np.sqrt(trainScore_MSE)
testScore_RMSE = np.sqrt(testScore_MSE)

trainScore_MAE = mean_absolute_error(trainY, trainPredict)
testScore_MAE = mean_absolute_error(testY, testPredict)

trainScore_max = max(np.abs(trainY, trainPredict))
testScore_max = max(np.abs(testY, testPredict))

print('Train Score: %.6f ME' % (trainScore_ME))
print('Test Score: %.6f ME' % (testScore_ME))

print('Train Score: %.6f RMSE' % (trainScore_RMSE))
print('Test Score: %.6f RMSE' % (testScore_RMSE))

print('Train Score: %.6f MSE' % (trainScore_MSE))
print('Test Score: %.6f MSE' % (testScore_MSE))

print('Train Score: %.6f MAE' % (trainScore_MAE))
print('Test Score: %.6f MAE' % (testScore_MAE))

print('Train Score: %.6f MAX' % (trainScore_max))
print('Test Score: %.6f MAX' % (testScore_max))

# Summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss_mlp.png')
plt.show()

# Plot predictions vs true data, train
plt.plot(trainY, trainPredict, '*')
plt.plot(np.linspace(0, 0.25, 100), np.linspace(0, 0.25, 100), 'k')
plt.xlabel('true values')
plt.ylabel('predicted values')
plt.title('True values vs predicted values for training data')
plt.savefig('truevspred_mlp_train.png')
plt.show()

# Plot predictions vs true data, train
plt.plot(testY, testPredict, '*')
plt.plot(np.linspace(0, 0.25, 100), np.linspace(0, 0.25, 100), 'k')
plt.xlabel('true values')
plt.ylabel('predicted values')
plt.title('True values vs predicted values for test data')
plt.savefig('truevspred_mlp_test.png')
plt.show()

# Calculate OTM, ITM and NTM data
OTM_ME = np.mean(model.predict(otmX) - otmY)
OTM_MSE = mean_squared_error(otmY, model.predict(otmX))
OTM_RMSE = np.sqrt(OTM_MSE)
OTM_MAE = mean_absolute_error(otmY, model.predict(otmX))
OTM_max = max(np.abs(model.predict(otmX) - otmY))

ITM_ME = np.mean(model.predict(itmX) - itmY)
ITM_MSE = mean_squared_error(itmY, model.predict(itmX))
ITM_RMSE = np.sqrt(ITM_MSE)
ITM_MAE = mean_absolute_error(itmY, model.predict(itmX))
ITM_max = max(np.abs(model.predict(itmX) - itmY))

NTM_ME = np.mean(model.predict(ntmX) - ntmY)
NTM_MSE = mean_squared_error(ntmY, model.predict(ntmX))
NTM_RMSE = np.sqrt(NTM_MSE)
NTM_MAE = mean_absolute_error(ntmY, model.predict(ntmX))
NTM_max = max(np.abs(model.predict(ntmX) - ntmY))

print('OTM Score: %.6f ME' % (OTM_ME))
print('OTM Score: %.6f MSE' % (OTM_MSE))
print('OTM Score: %.6f RMSE' % (OTM_RMSE))
print('OTM Score: %.6f MAE' % (OTM_MAE))
print('OTM Score: %.6f MAX' % (OTM_max))

print('ITM Score: %.6f ME' % (ITM_ME))
print('ITM Score: %.6f MSE' % (ITM_MSE))
print('ITM Score: %.6f RMSE' % (ITM_RMSE))
print('ITM Score: %.6f MAE' % (ITM_MAE))
print('ITM Score: %.6f MAX' % (ITM_max))

print('NTM Score: %.6f ME' % (NTM_ME))
print('NTM Score: %.6f MSE' % (NTM_MSE))
print('NTM Score: %.6f RMSE' % (NTM_RMSE))
print('NTM Score: %.6f MAE' % (NTM_MAE))
print('NTM Score: %.6f MAX' % (NTM_max))
"""
print(trainX[:1, :])
y =  trainX[:1, :]
time_start=time.clock()
model.predict(y)
time_elapsed = (time.clock()-time_start)
print('time elapsed', time_elapsed)