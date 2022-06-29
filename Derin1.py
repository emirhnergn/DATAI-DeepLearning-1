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

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.15,random_state=42)

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

from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(x_train,y_train)
print("LogReg Accuracy:",logreg.score(x_test,y_test))

#%%





















































#%%
