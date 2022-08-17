from tensorflow.keras.datasets import mnist
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow import expand_dims
#from PIL import Image

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
#%matplotlib inline
sns.set(font_scale = 2)
index = np.random.choice(np.arange(len(X_train)), 24, replace = False)
figure, axes = plt.subplots(nrows=4, ncols=6, figsize=(16,9))
for item in zip(axes.ravel(), X_train[index], Y_train[index]):
    axes, pic, target = item
    axes.imshow(pic, cmap=plt.cm.gray_r)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(target)
    plt.tight_layout()
X_train = X_train.reshape((60000, 28, 28, 1))
print(X_train.shape)
X_test = X_test.reshape((10000, 28, 28, 1))
print(X_test.shape)
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255
Y_train = to_categorical(Y_train)
Y_train.shape
print(Y_train[0])
Y_test = to_categorical(Y_test)
print(Y_test.shape)
cnn = Sequential()
cnn.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu'))
cnn.add(MaxPooling2D(pool_size=(2,2)))
cnn.add(Flatten())
cnn.add(Dense(units=128, activation='relu'))
cnn.add(Dense(units=10, activation='softmax'))
print(cnn.summary())
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
cnn.fit(X_train, Y_train, epochs=5, batch_size=64, validation_split=0.1)
loss, accuracy = cnn.evaluate(X_test, Y_test)
print(loss, ' ', accuracy)
przypuszczenia = cnn.predict(X_test)
print(Y_test[0])
for indeks, przypuszczenie in enumerate(przypuszczenia[0]):
    print(f'{indeks}: { przypuszczenie:.10%}')
obrazy = X_test.reshape((10000, 28, 28))
chybione_prognozy = []
for i, (p, e) in enumerate(zip(przypuszczenia, Y_test)):
    prognozowany, spodziewany = np.argmax(p), np.argmax(e)
    if prognozowany != spodziewany:
        chybione_prognozy.append(
        (i, obrazy[i], prognozowany, spodziewany))
print(len(chybione_prognozy))
figure, axes = plt.subplots(nrows=4, ncols=6, figsize=(16,12))
for axes, element in zip(axes.ravel(), chybione_prognozy):
    indeks, obraz, prognozowany, spodziewany = element
    axes.imshow(obraz, cmap=plt.cm.gray_r)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(
        f'index: {indeks}\np: {prognozowany}; s:{spodziewany}')
plt.tight_layout()
plt.show()

images = ['zero', 'one', 'two', 'three', 'four',
          'five', 'six', 'seven', 'eight', 'nine']
for img in images:
    img2 = load_img(img + '.bmp', target_size=(28, 28), color_mode="grayscale", interpolation="bilinear")
    input_arr = img_to_array(img2)
    input_arr = expand_dims(input_arr, 0)
    input_arr /= 255.
    input_arr = 1. - input_arr
    digit = cnn.predict(input_arr)
    plt.imshow(input_arr[0], cmap=plt.cm.gray_r)
    plt.show()
    print(img)
    for indeks, result in enumerate(digit[0]):
        print(f'{indeks}: {result:.10%}')
