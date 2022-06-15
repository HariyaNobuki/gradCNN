
from __future__ import print_function
import keras
from keras.preprocessing.image import load_img,  img_to_array, array_to_img
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

batch_size = 256
num_classes = 10
epochs = 5

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

org_img = x_test[0]  # 適当に0番目の画像を保存 shape=(28,28)

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

#model = Sequential()
#model.add(Conv2D(32, kernel_size=(3, 3),
#                 activation='relu',
#                 input_shape=input_shape, name="conv1"))
#model.add(Conv2D(64, (3, 3), activation='relu', name="conv2"))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
#model.add(Flatten())
#model.add(Dense(128, activation='relu'))
#model.add(Dropout(0.5))
#model.add(Dense(num_classes, activation='softmax', name="output"))
#
#model.compile(loss=keras.losses.categorical_crossentropy,
#              optimizer=keras.optimizers.Adadelta(),
#              metrics=['accuracy'])
#print(model.summary())
#
#model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
#
#score = model.evaluate(x_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

#--- ここから qiita
import numpy as np
def preprocess():
    img = load_img("demodel/x_train.png",
                    grayscale=True, target_size=(28, 28))
    array = img_to_array(img)
    array /= 255
    img_array = np.array([array])
    return img_array
def getCNNpredict():
    model =  keras.models.load_model("demodel/MNIST.h5", compile=False)
    return model
img_array = preprocess()
model = getCNNpredict()

conv_layer = model.get_layer("last_Conv")
conv_layer_weights = conv_layer.get_weights()
print(conv_layer.input.shape)
print(conv_layer.output.shape)

flatten_layer = model.get_layer("flatten")
#flatten_layer = model.get_layer("Flatten")  # 名前を付けていないので自動でついた名前を使用
print(flatten_layer.input.shape)
print(flatten_layer.output.shape)

output_layer = model.get_layer("output")
print(output_layer.input.shape)
print(output_layer.output.shape)

import matplotlib.pyplot as plt
plt.imshow(org_img, cmap='gray')
plt.show()


input_val = x_test[0]  # 入力値 shape=()
print(input_val.shape)
# 予測結果を出す
prediction = model.predict(np.asarray([input_val]), 1)[0]
prediction_idx = np.argmax(prediction)

print(prediction)
print(np.argmax(prediction))

# loss は出力先の結果
loss = model.get_layer("output").output[0][prediction_idx]

# variables は入力層
variables = model.input

# 勾配
grads = K.gradients(loss, variables)[0]
grads_func = K.function([model.input, K.learning_phase()], [grads])

# 結果を取得
values = grads_func([np.asarray([input_val]), 0])
values = values[0]

img = values[0]             # (1,28,28,1) -> (28,28,1)
img = img.reshape((28,28))  # (28,28,1) -> (28,28)
img = np.abs(img)           # 絶対値

# 表示
plt.imshow(img, cmap='gray')
plt.show()


# grad cam
conv_layer_output = model.get_layer("last_Conv").output
input_val = x_test[0]  # 入力値 shape=(28,28,1)

# 予測結果を出す
prediction = model.predict(np.asarray([input_val]), 1)[0]
prediction_idx = np.argmax(prediction)
loss = model.get_layer("output").output[0][prediction_idx]

grads = K.gradients(loss, conv_layer_output)[0]
grads_func = K.function([model.input, K.learning_phase()], [conv_layer_output, grads])

(conv_output, conv_values) = grads_func([np.asarray([input_val]), 0])
conv_output = conv_output[0]  # (24, 24, 64)
conv_values = conv_values[0]  # (24, 24, 64)

weights = np.mean(conv_values, axis=(0, 1))  # 勾配の平均をとる
cam = np.dot(conv_output, weights)           # 出力結果と重さの内積をとる

import cv2
cam = cv2.resize(cam, (28,28), cv2.INTER_LINEAR)
cam = np.maximum(cam, 0)
cam = cam / cam.max()

# モノクロ画像に疑似的に色をつける
cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

# オリジナルイメージもカラー化
org_img = cv2.cvtColor(np.uint8(org_img), cv2.COLOR_GRAY2BGR)  # (w,h) -> (w,h,3)

# 合成
rate = 0.4
cam = cv2.addWeighted(src1=org_img, alpha=(1-rate), src2=cam, beta=rate, gamma=0)
cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)  # 色をRGBに変換

# 表示
plt.imshow(cam)
plt.show()