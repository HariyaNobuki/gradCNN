# this file have cnn models

# add module
# keras py -3.6 ver 2.4.3
from keras.preprocessing.image import load_img,  img_to_array, array_to_img
from keras import regularizers

from keras.layers import Conv2D,MaxPooling2D,Dropout,Dense,Flatten,Activation,BatchNormalization
from keras import models   
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator            # 色つき処理への懸け橋的な
from keras.preprocessing import image
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau


def Model_GPU(input_shape,num_classes):
    model = models.Sequential()
    # extract image features by convolution and max pooling layers
    model.add(Conv2D(
        64, kernel_size=3, padding="same",
        input_shape=input_shape, activation="relu"
        ))
    model.add(BatchNormalization())
#    model.add(Dropout(0.2))

    model.add(Conv2D(64, kernel_size = 3))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.2))

    model.add(Conv2D(128, kernel_size = 3))
    model.add(BatchNormalization())
    #model.add(Dropout(0.2))
    model.add(Conv2D(128, kernel_size = 3))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.2))

    model.add(Conv2D(128, kernel_size = 3))
    model.add(BatchNormalization())
    #model.add(Dropout(0.2))
    model.add(Conv2D(128, kernel_size = 3))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(Dropout(0.2))

    model.add(Conv2D(256, kernel_size = 3))
    model.add(BatchNormalization())
    #model.add(Dropout(0.2))
    model.add(Conv2D(256, kernel_size = 3))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(512, kernel_size = 3))
    model.add(BatchNormalization())
    #model.add(Dropout(0.2))
    model.add(Conv2D(512, kernel_size = 3))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    #model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(4096 , activation="relu")) 
    model.add(BatchNormalization()) 
    #model.add(Dropout(0.2))
    model.add(Dense(4096 , activation="relu"))
    model.add(BatchNormalization())
    #model.add(Dropout(0.2))
    model.add(Dense(1000, activation="relu"))
    model.add(Dense(1))
    model.add(Activation("linear"))

    #model.summary()

    return model


def Model_CPU(input_shape,num_classes):
    model = models.Sequential()
    # extract image features by convolution and max pooling layers
    model.add(Conv2D(
        3, kernel_size=3, padding="same",
        input_shape=input_shape, activation="relu"
        ))
    model.add(Conv2D(3, kernel_size = 3))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(6, kernel_size = 3))
    model.add(Conv2D(6, kernel_size = 3))
    model.add(MaxPooling2D(pool_size=(2, 2)))


    # Regression the class by fully-connected layers(FC)
    model.add(Flatten())

    model.add(Dense(100 , activation="relu"))
    model.add(Dense(1, activation = "linear"))

    return model

def baumtest(input_shape,num_classes):
    model = models.Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu', name="conv1"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu',name="last_Conv"))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax', name="output"))
    model.summary()
    return model