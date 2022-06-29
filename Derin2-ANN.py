#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings("ignore")

#%%

x_l = np.load("X.npy")
y_l = np.load("Y.npy")
img_size = 64
plt.subplot(1,2,1)
plt.imshow(x_l[260].reshape(img_size,img_size))
plt.axis("off")
plt.subplot(1,2,2)
plt.imshow(x_l[900].reshape(img_size,img_size))
plt.axis("off")
plt.show()

#%%

X = np.concatenate((x_l[204:409],x_l[822:1027]),axis=0)

z=np.zeros(205)
o=np.ones(205)

Y = np.concatenate((z,o),axis=0).reshape(X.shape[0],1)

#%%

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.33,random_state=42)

number_of_train = x_train.shape[0]
number_of_test = x_test.shape[0]

#%%

x_train_flatten = x_train.reshape(number_of_train,x_train.shape[1]*x_train.shape[2])
x_test_flatten = x_test.reshape(number_of_test,x_test.shape[1]*x_test.shape[2])

print("X train flatten:",x_train_flatten.shape)
print("X test flatten:",x_test_flatten.shape)

#%%

"""x_train = x_train_flatten.T
x_test = x_test_flatten.T
y_train = y_train.T
y_test = y_test.T"""

x_train = x_train_flatten
x_test = x_test_flatten

#%%

import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
from random import shuffle

def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=8,kernel_initializer="uniform",activation="relu",input_dim=x_train.shape[1]))
    #classifier.add(Dense(units=8,kernel_initializer="uniform",activation="relu"))
    #classifier.add(Dense(units=6,kernel_initializer="uniform",activation="relu"))
    classifier.add(Dense(units=4,kernel_initializer="uniform",activation="relu"))
    classifier.add(Dense(units=1,kernel_initializer="uniform",activation="sigmoid"))
    classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
    return classifier
classifier = KerasClassifier(build_fn=build_classifier,epochs=200)
accuricies = cross_val_score(classifier,X=x_train,y=y_train,scoring="accuracy",cv=3)
mean = accuricies.mean()
variance = accuricies.std()
classifier.fit(x_train,y_train)
dedede = classifier.predict(x_test)
print("Accuracies:",accuricies)
print("Accuracy mean:",mean)
print("Accuracy variance:",variance)


#%%

from sklearn.metrics import accuracy_score,confusion_matrix

print("Confusion Matrix:")
print(confusion_matrix(y_test,dedede))
print("Accuracy Score:",accuracy_score(y_test,dedede))

#%%







































































#%%