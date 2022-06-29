#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from warnings import filterwarnings
filterwarnings("ignore")

#%%

digit_test = pd.read_csv("digit-test.csv")
digit_train = pd.read_csv("digit-train.csv")

#%%

X_train = digit_train.iloc[:,1:].values
Y_train = digit_train.label.values.reshape(-1,1)
X_test = digit_test.values

x_train = X_train/255
x_test =  X_test/255

x_train = x_train.reshape(-1,28,28,1)
X_test = x_test.reshape(-1,28,28,1)

from keras.utils.np_utils import to_categorical

y_train = to_categorical(Y_train,num_classes=10)


#%%

img = x_train[0].reshape((28,28))
plt.imshow(img,cmap="gray")
plt.axis("off")
plt.show()

#%%

from sklearn.model_selection import train_test_split

X_train,X_val,Y_train,Y_val = train_test_split(x_train,y_train,test_size=0.1,random_state=2)

#%%

from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,Dropout,MaxPool2D
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,confusion_matrix
from keras.optimizers import RMSprop,Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

classifier = Sequential()
classifier.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu', input_shape = (28,28,1)))
classifier.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same', 
                 activation ='relu'))
classifier.add(MaxPool2D(pool_size=(2,2)))
classifier.add(Dropout(0.25))


classifier.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
classifier.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same', 
                 activation ='relu'))
classifier.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
classifier.add(Dropout(0.25))

    
classifier.add(Flatten())
    
classifier.add(Dense(units=256,activation="relu",kernel_initializer="uniform"))
classifier.add(Dropout(0.5))

classifier.add(Dense(units=10,activation="softmax",kernel_initializer="uniform"))

#optimizer = Adam(lr=0.001,beta_1=0.9,beta_2=0.999)
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
classifier.compile(optimizer=optimizer,loss="categorical_crossentropy",metrics=["accuracy"])

#%%

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=3, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

epochs = 3
batch_size = 250

#%%

"""datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # dimesion reduction
        rotation_range=0.5,  # randomly rotate images in the range 5 degrees
        zoom_range = 0.5, # Randomly zoom image 5%
        width_shift_range=0.5,  # randomly shift images horizontally 5%
        height_shift_range=0.5,  # randomly shift images vertically 5%
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images"""

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(X_train)

#%%

history = classifier.fit_generator(datagen.flow(X_train,Y_train,batch_size=batch_size),
                              epochs = epochs,validation_data = (X_val,Y_val),
                              verbose=2,
                              steps_per_epoch=X_train.shape[0] // batch_size,
                              callbacks=[learning_rate_reduction])


#%%
plt.plot(history.history['val_loss'], color='b', label="validation loss")
plt.title("Test Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

#%%
import seaborn as sns

pred = classifier.predict(X_val)

pred_classes = np.argmax(pred,axis=1)
pred_true = np.argmax(Y_val,axis=1)

confusion_mtx = confusion_matrix(pred_true,pred_classes)

f,ax = plt.subplots(figsize=(8,8))
sns.heatmap(confusion_mtx,annot=True,linewidths=0.01,cmap="Greens",linecolor="gray",fmt='.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

print("Accuracy:",accuracy_score(pred_true,pred_classes))
#%%

test_predicts = classifier.predict(X_test)
test_trues = np.argmax(test_predicts,axis=1)
print(test_trues)


#%%

































#%%