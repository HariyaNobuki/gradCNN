import tensorflow as tf
import os , sys
from keras import models   
from keras.utils import np_utils
from keras.layers import Conv2D,MaxPooling2D,Dropout,Dense,Flatten,Activation,BatchNormalization
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#tf.keras.backend.set_session(tf.Session(config=config))
#os.environ['CUDA_CACHE_MAXSIZE']='4294967296'

mnist = tf.keras.datasets.mnist
 
(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((60000, 28, 28, 1))
x_test = x_test.reshape((10000, 28, 28, 1))
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

model = models.Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1), activation='relu', name="conv1"))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu',name="last_Conv"))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(10, activation='softmax', name="output"))
model.summary()

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics='accuracy')
model.fit(x_train,y_train,
            batch_size=256,
            epochs=100,
            validation_split=0.2,
            verbose = 1,
            shuffle=True,
            callbacks= [
            EarlyStopping(
                monitor = 'val_loss',
                patience = 10,
                verbose=1,
                restore_best_weights=True,
            ),
            ReduceLROnPlateau(
                monitor = 'val_loss',
                factor=0.1, 
                patience = 5,
                verbose=1,
            ),
            ModelCheckpoint(
                'MNIST.h5',
                monitor='val_loss',
                mode='min',
                verbose=1,
                save_best_only=True,
                save_weights_only=False,
            ),
            #TensorBoard(
            #    log_dir='MNIST',
            #),
            ]
            )

#model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)