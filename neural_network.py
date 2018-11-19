import numpy as np
import matplotlib as plt
# import pandas
# import math
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
# from keras.layers import LSTM
# from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


# Set seed
seed = 7
np.random.seed(seed)


# Prepare data
X = np.loadtxt('input.txt')
Y = np.array(np.loadtxt('output.txt'))
data = np.column_stack((X, Y))
np.random.shuffle(data)
X = data[:, :-2]
Y = data[:, -1]

trainX = X[:int(np.round(0.8*len(X))), :]
trainY = Y[:int(np.round(0.8*len(X)))]
testX = X[int(np.round(0.8*len(X)))+1:, :]
testY = Y[int(np.round(0.8*len(X)))+1:]


# Create model
input_size = np.shape(X)[1]
output_size = 1 #np.shape(Y)[1]
l1 = 100
l2 = 100
l3 = 100
l4 = 100
activation_function =  'sigmoid'  # 'relu
output_function = 'sigmoid'

model = Sequential()
model.add(Dense(l1, input_dim=input_size, activation=activation_function))
model.add(Dense(l2, activation=activation_function))
model.add(Dense(l3, activation=activation_function))
model.add(Dense(l4, activation=activation_function))
model.add(Dense(output_size, activation=output_function))

# Compile model
loss_function =  'binary_crossentropy'  # 'mean_squared_error' 
optimizing_algorithm = 'adam'
model.compile(loss=loss_function, optimizer=optimizing_algorithm, metrics=['accuracy'])

# Fit the model
no_epochs = np.arange(50,150,5)
batch_size = None  # 10
validation_split = 0.8  # 0.67

trainScore = np.zeros(np.shape(no_epochs))
testScore = np.zeros(np.shape(no_epochs))

for i in no_epochs:
    model.fit(trainX, trainY, epochs=i, batch_size=batch_size,  verbose=2, callbacks=None,
            validation_split=validation_split, validation_data=None, shuffle=True, class_weight=None,
            sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
    model.evaluate(trainX, trainY, batch_size=batch_size, verbose=1, sample_weight=None, steps=None)
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    trainScore[i] = np.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
    testScore[i] = np.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))


plt.plot(no_epochs, testScore, 'r')
plt.plot(no_epochs, trainScore, 'b')
plt.title('Score')
plt.show()
# Make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Train Score: %.2f RMSE' % (trainScore))
print('Test Score: %.2f RMSE' % (testScore))
# round predictions
# rounded = [round(x[0]) for x in predictions]
# print(rounded)
