import numpy as np
import matplotlib.pyplot as plt
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
X = np.loadtxt('input_new.txt')
Y = np.array(np.loadtxt('output_new.txt'))
data = np.column_stack((X, Y))
np.random.shuffle(data)
X = [data[:, 0]. data[:, 1], data[:, 2], data[:, 3], data[:, 4]]
Y = data[:, -1]

trainX = X[:int(np.round(0.8*len(X))), :]
trainY = Y[:int(np.round(0.8*len(X)))]
testX = X[int(np.round(0.8*len(X)))+1:, :]
testY = Y[int(np.round(0.8*len(X)))+1:]


# Create model
input_size = np.shape(X)[1]
output_size = 1 #np.shape(Y)[1]
layer1 = 100
layer2 = 100
layer3 = 100
#layer4 = 100
activation_function = 'relu
output_function = 'sigmoid'

model = Sequential()
model.add(Dense(layer1, input_dim=input_size, activation=activation_function))
model.add(Dense(layer2, activation=activation_function))
model.add(Dense(layer3, activation=activation_function))
#model.add(Dense(layer4, activation=activation_function))
model.add(Dense(output_size, activation=output_function))

model.summary()

# Compile model
loss_function = 'binary_crossentropy'  # 'mean_squared_error' 
optimizing_algorithm = 'nadam'  # 'sdg'  # adam'
model.compile(loss=loss_function, optimizer=optimizing_algorithm, metrics=['accuracy'])

# Fit the model
no_epochs = 50
batch_size = 32  # None  # 10
validation_split = None  # 0.8  # 0.67

history = model.fit(trainX, trainY, epochs=no_epochs, batch_size=batch_size,  verbose=2, callbacks=None,
        validation_split=validation_split, validation_data=(testX,testY), shuffle=True, class_weight=None,
        sample_weight=None, initial_epoch=0, steps_per_epoch=None, validation_steps=None)
model.evaluate(trainX, trainY, batch_size=batch_size, verbose=1, sample_weight=None, steps=None)
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
trainScore = np.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
testScore = np.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))

# Summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
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
