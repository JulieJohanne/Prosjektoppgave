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
# from keras.layers import LSTM
# from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing


# Set seed
#seed = 7
#np.random.seed(seed)

# Prepare data
data = np.column_stack((np.loadtxt('input_150.txt'), np.loadtxt('output_150.txt')))
data = np.column_stack((np.arange(len(data)), data))
np.random.shuffle(data)
X = np.array([data[:, 1], data[:, 2]/data[:, 3], data[:, 4], data[:, 5]/252]).T 
Y = np.array(data[:, -1]/data[:, 3])

# Test for nan's
for i in range(len(Y)):
    if np.isnan(Y[i]):
        print('Y has nan, index: ', i)
index_delete = []
for j in range(len(X)):
    for k in range(3):
        if np.isnan(X[j, k]):
            index_delete.append(j)
            #print('X has nan, index: ', j, k)
X = np.delete(X, index_delete, 0)
Y = np.delete(Y, index_delete)
data = np.delete(data, index_delete, 0)
#X = preprocessing.scale(X)
#X = preprocessing.normalize(X) ValueError

index_slice = int(np.round(0.8*len(X)))
trainX = X[:index_slice, :]
trainY = Y[:index_slice]
testX = X[index_slice:, :]
testY = Y[index_slice:]
np.savetxt('testX.txt', np.c_[data[index_slice:, 0], testX])
np.savetxt('testY.txt', testY)

# Create model
#input_size = np.shape(X)[1]
#output_size = 1 #np.shape(Y)[1]c
layers = [60, 60, 60, 60, 60, 60]
#activation_function = 'tanh' #'relu'
#output_function = 'sigmoid' # tanh
model = Sequential()
model.add(Dense(layers[0], input_dim=np.shape(X)[1]))
model.add(Dropout(0.1))
model.add(Dense(layers[1], activation='elu'))#,  kernel_regularizer=regularizers.l2(0.01))) 
model.add(Dropout(0.1))
model.add(Dense(layers[2], activation='elu'))#,  kernel_regularizer=regularizers.l2(0.01)))  
model.add(Dropout(0.1))
model.add(Dense(layers[3], activation='elu'))#,  kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.1))
model.add(Dense(layers[4], activation='elu'))#,  kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.1))
model.add(Dense(layers[5], activation='softsign'))#,  kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.1))
model.add(Dense(1, activation='softsign'))

model.summary()

# Compile model
#sgd = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
#rmsPROP = 
#loss_function = 'mean_squared_error'  # 'mean_absolute_percentage_error'  'binary_crossentropy' 
#optimizing_algorithm = 'adam'  # 'nadam'  # 'sdg'
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae', 'mape'])

# Fit the model
no_epochs = 100
#batch_size = 128  # 32  # None  # 10
#validation_split =  0.2 # None
#validation_data =  None  #(testX,testY) # None
history = model.fit(trainX, trainY, epochs=no_epochs, batch_size=128,  verbose=2,
        validation_split=0.2, validation_data=None, shuffle=True)

model.evaluate(trainX, trainY, batch_size=128, verbose=1)

#print(history.history.keys()) = ['loss', 'val_mean_absolute_percentage_error', 'mean_absolute_percentage_error', 'val_mean_squared_error', 'val_mean_absolute_error', 'mean_squared_error', 'val_loss', 'mean_absolute_error']

# Make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
plt.plot(trainY[-40:], 'r*')
plt.plot(trainPredict[-40:], 'b*')
plt.title('train predict vs true values')
plt.show()
plt.plot(testY[-40:], 'r*')
plt.plot(testPredict[-40:], 'b*')
plt.title('test predict vs true values')
plt.show()
trainScore = np.sqrt(mean_squared_error(trainY, trainPredict))
testScore = np.sqrt(mean_squared_error(testY, testPredict))
print('Train Score: %.6f RMSE' % (trainScore))
print('Test Score: %.6f RMSE' % (testScore))
# round predictions
# rounded = [round(x[0]) for x in predictions]
# print(rounded)

# Summarize history for accuracy
plt.plot(history.history['mean_absolute_percentage_error'])
plt.plot(history.history['val_mean_absolute_percentage_error'])
plt.title('model absolute percentage error')
plt.ylabel('error')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# Summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
