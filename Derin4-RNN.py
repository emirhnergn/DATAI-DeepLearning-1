#%%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings("ignore")
import keras
import tensorflow as tf
    

#%%

dataset_train = pd.read_csv("Stock_Price_Train.csv")
dataset_test = pd.read_csv("Stock_Price_test.csv")

train = dataset_train.Open.values.reshape(-1,1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train)

real_stock_price = dataset_test.Open.values.reshape(-1,1)

timesteps = 50
dataset_total = pd.concat((dataset_train.Open,dataset_test.Open),axis=0)
inputs = dataset_total[len(dataset_total)-len(dataset_test)-timesteps:].values.reshape(-1,1)
inputs = scaler.transform(inputs)

#%%
X_train = []
y_train = []
timesteps = 50
for i in range(timesteps,train_scaled.shape[0]):
    X_train.append(train_scaled[i-timesteps:i,0])
    y_train.append(train_scaled[i,0])
X_train,y_train = np.array(X_train),np.array(y_train)
print(X_train.shape)
print(y_train.shape)
X_train = np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
print(X_train.shape)
print(y_train.shape)
X_test = []
for i in range(timesteps,inputs.shape[0]):
    X_test.append(inputs[i-timesteps:i,0])
X_test = np.array(X_test)
X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))

print(train_scaled.shape)


#%%

from keras.models import Sequential
from keras.layers import Dense,SimpleRNN,Dropout,LSTM
from keras.optimizers import RMSprop,Adam

regressor = Sequential()

"""
regressor.add(SimpleRNN(units=50 ,activation="tanh",return_sequences=True,
                        input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))

regressor.add(SimpleRNN(units=50 ,activation="tanh",return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(SimpleRNN(units=50 ,activation="tanh",return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(SimpleRNN(units=50))
regressor.add(Dropout(0.2))
"""


regressor.add(LSTM(units=50 ,activation="tanh",return_sequences=True,
                        input_shape=(X_train.shape[1],1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50 ,activation="tanh",return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50 ,activation="tanh",return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))


regressor.add(Dense(units=1))


#optimizer = RMSprop(lr=0.005,rho=0.9,epsilon=1e-08, decay=0.0)
optimizer = Adam(lr=0.003,beta_1=0.9,beta_2=0.999)

regressor.compile(optimizer=optimizer,loss="mean_squared_error",metrics=["accuracy"])

#%%
#from ann_visualizer.visualize import ann_viz
#ann_viz(regressor,title="LSTM")
"""
import pydot as pyd
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

keras.utils.vis_utils.pydot = pyd
def visualize_model(model):
  return SVG(model_to_dot(model).create(prog='dot', format='svg'))
visualize_model(regressor)"""
keras.utils.plot_model(regressor,show_shapes=True)

#%%

regressor.fit(X_train,y_train,epochs=5,batch_size=20)

#%%

predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = scaler.inverse_transform(predicted_stock_price)

#%%
plt.plot(real_stock_price,color="red",label="Real Google Stock Price")
plt.plot(predicted_stock_price,color="blue",label="Predicted Google Stock Price")
plt.title("Google Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Google Stock Price")
plt.legend()
plt.show()
#%%

from sklearn.metrics import r2_score
print("R2 Score:",r2_score(real_stock_price,predicted_stock_price))


#%%





































#%%