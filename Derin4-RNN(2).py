#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings("ignore")
import math
from keras.models import Sequential
from keras.layers import Dense,LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#%%

data = pd.read_csv("international-airline-passengers.csv",skipfooter=5)

dataset = data.iloc[:,1].values

#%%

plt.plot(dataset)
plt.title("international airline passanger")
plt.xlabel("time")
plt.ylabel("Number of Passenger")
plt.show()

#%%

dataset = dataset.reshape(-1,1)
dataset = dataset.astype("float32")

#%%

scaler = MinMaxScaler()
dataset = scaler.fit_transform(dataset)

#%%

train_size = int(len(dataset)*0.5)
test_size = len(dataset)-train_size

train = dataset[0:train_size,:]
test = dataset[train_size:len(dataset),:]

print("train size: {},test size {}".format(len(train),len(test)))

#%%

time_stemp = 10
dataX = []
dataY = []

for i in range(len(train)-time_stemp-1):
    a = train[i:(i+time_stemp),0]
    dataX.append(a)
    dataY.append(train[i+time_stemp,0])
trainX = np.array(dataX)
trainY = np.array(dataY)

#%%

dataX = []
dataY = []

for i in range(len(train)-time_stemp-1):
    a = test[i:(i+time_stemp),0]
    dataX.append(a)
    dataY.append(test[i+time_stemp,0])
testX = np.array(dataX)
testY = np.array(dataY)


#%%

trainX = np.reshape(trainX,(trainX.shape[0],1,trainX.shape[1]))
testX = np.reshape(testX,(testX.shape[0],1,testX.shape[1]))


#%%

from keras.layers import SimpleRNN,Dropout

model = Sequential()

model.add(LSTM(units=4,activation="tanh",input_shape=(1,time_stemp)))

model.add(Dense(units=1))

model.compile(optimizer="adam",loss="mean_squared_error")
model.fit(trainX,trainY,epochs=50,batch_size=1,verbose=2)
#%%

trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])

testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

trainScore = math.sqrt(mean_squared_error(trainY[0],trainPredict[:,0]))
print("Train Score: %.2f RMSE" % (trainScore))

testScore = math.sqrt(mean_squared_error(testY[0],testPredict[:,0]))
print("Test Score: %.2f RMSE" % (testScore))

#%%

trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[time_stemp:len(trainPredict)+time_stemp,:] = trainPredict

testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(testPredict)+(time_stemp*2)+1:len(dataset)-1,:] = testPredict

plt.plot(scaler.inverse_transform(dataset),color="blue")
plt.plot(trainPredictPlot,color="orange",label="Train Predict")
plt.plot(testPredictPlot,color="green",label="Test Predict")
plt.legend()
plt.show()


#%%

























#%%