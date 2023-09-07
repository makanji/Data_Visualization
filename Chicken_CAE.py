#%%
import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split

#%%
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
#%%
#data preprocessing
#scaling and data spitting

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
#%%
#splitting to validation sets
#x_remaining, x_test, y_remaining, y_test = train_test_split(x, y, test_size=ratio_test)
train_ratio = 0.90
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=1 - train_ratio)
print(x_train.shape)
print(y_train.shape)
print(y_val.shape)
print(x_val.shape)
print(x_test.shape)
print(y_test.shape)
print(x_val[0].max())
print(x_test[0].max())
#%%
#CAE definition

input_img = keras.Input(shape=(28, 28, 1))

x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation='relu')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = keras.Model(input_img, decoded, name = 'auto_encoder')
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

#%%
# fittinng model
autoencoder.fit(x_train, x_train,
                epochs= 50, batch_size=256,
                shuffle=True, validation_data=(x_val, x_val))

#%%
# saving whole model
#  reference site https://theailearner.com/2019/01/21/saving-and-loading-models-in-keras/
autoencoder.save('autoencoder_model.h5')
#%%
# loading whole model
from keras.models import load_model
autoencoder= load_model('autoencoder_model.h5')

#%%
encoder = keras.Model(input_img, encoded )
encoder.summary()

#%%
# evaluating saved model
decoded_imgs = autoencoder.predict(x_test)
encoded_imgs = encoder.predict(x_test)
encoded_img_vals = encoder.predict(x_val)
encoded_img_xtrain = encoder.predict(x_train)
decoded_imgs.shape
encoded_imgs.shape
plt.imshow(decoded_imgs[450], cmap='gray')
plt.imshow(encoded_imgs[450,:,:,5], cmap='gray')
plt.imshow(encoded_imgs[0, : , :,1].reshape(2,8), cmap= 'gray')

plt.imshow(encoded_imgs[0, :, :,6], cmap= 'gray')
encoded_imgs[0, :, :,1].shape
encoded_imgs[450,:,:,5].shape
#%%
#viewing through 10 images
n = 10
plt.figure(figsize= (20,4))
for i in range(n):
  #display original images
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(x_test[i].reshape(28,28))
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  #displaying decoder reconstructions
  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(decoded_imgs[i].reshape(28,28))
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()

#%%
# running the classification using encoded images
input_shape = (4,4,8)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
# Creating a Sequential Model and adding the layers
model = Sequential(name = 'class_model')
model.add(Conv2D(28, kernel_size=(3,3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#%%
#training classifier model
model.fit( x = encoded_img_xtrain, y = y_train,
                epochs= 30, batch_size=256,
                shuffle=True,
                validation_data=(encoded_img_vals, y_val))

#%%
#saving classifier weight
model.save('classifier_model.h5', overwrite = True)
#%%
# loading whole model
from keras.models import load_model
model= load_model('classifier_model.h5')
model.summary()
#%%
model.evaluate(encoded_imgs, y_test)
#classification prediction using test_set encoded_imgs
encoded_imgs.shape
image_index = 4444
plt.imshow(encoded_imgs[image_index].reshape(16, 8),cmap='Greys')

plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys') # viewing real image from test
x_test[image_index].shape
encoded_imgs[image_index].shape

#%%
#making prediction
pred = model.predict ( encoded_imgs[image_index])
print(pred.argmax())
encoded_imgs[image_index].ndim
encoded_imgs[image_index].shape

y_test[image_index]
image_index = 5555
plt.imshow(x_test[image_index].reshape(28, 28),cmap='Greys')
x_test[image_index].shape
 .reshape(1, 28, 28, 1))


#%%
#another prediction
image_index = 50
encoded_imgs[image_index].reshape(1,4,4,8) # changing number of dim to 4

pred = model.predict(encoded_imgs[image_index].reshape(1,4,4,8))

print(pred.argmax)
print(y_test[image_index])
image_index

pred = model.predict(encoded_imgs[750].reshape(1,4, 4, 8))
print(pred.argmax)
print(y_test[750])
np.argmax(pred, axis=1)
pred.shape
pred

from numpy import argmax

pred = model.predict(encoded_imgs[1073].reshape(1,4,4,8))
print(pred.argmax)
print(y_test[1073])
np.argmax(pred, axis=0)
pred.shape
pred.argmax()


encoded_imgs.shape