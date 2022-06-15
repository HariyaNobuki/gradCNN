import tensorflow as tf
import os , sys
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
x_train, x_test = x_train / 255.0, x_test / 255.0
 
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])
 
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
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
            TensorBoard(
                log_dir='MNIST',
            ),
            ]
            )

#model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)