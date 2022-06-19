
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
import os , sys 

os.makedirs("NKTLabLtd",exist_ok=True)
os.makedirs("NKTLabLtd/a",exist_ok=True)
os.makedirs("NKTLabLtd/b",exist_ok=True)

for i in range(45):
    os.makedirs("NKTLabLtd/a/{}".format(i),exist_ok=True)
    os.makedirs("NKTLabLtd/b/{}".format(i),exist_ok=True)

def getCNNpredict():
    model =  keras.models.load_model("demodel/baum.h5", compile=False)
    return model

batch_size = 256
num_classes = 2
epochs = 5
numdata = 45
baumlist = ["a","b"]
model = getCNNpredict()

# input image dimensions
img_rows, img_cols = 224,168



for name in baumlist:
    for i in range(numdata):
        #--- ここから qiita
        import numpy as np
        def preprocess(name,i):
            img = load_img("submit/{}/{}.bmp".format(name,i),
                            grayscale=True, target_size=(224,168))
            array = img_to_array(img)
            array /= 255
            img_array = np.array([array])
            return img_array

        img_array = preprocess(name,i)

        savepath = "NKTLabLtd/" + str(name) + "/" + str(i) + "/"

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
        plt.imshow(img_array.reshape(224,168,1), cmap='gray')
        plt.savefig(savepath + "shrine.png")
        plt.clf()
        plt.close()
        #plt.show()


        input_val = img_array[0]  # 入力値 shape=()
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
        img = img.reshape((img.shape[0],img.shape[1]))  # (28,28,1) -> (28,28)
        img = np.abs(img)           # 絶対値

        # 表示
        plt.imshow(img, cmap='gray')
        plt.savefig(savepath + "output.png")
        plt.clf()
        plt.close()


        # grad cam
        conv_layer_output = model.get_layer("last_Conv").output
        input_val = img_array[0]  # 入力値 shape=(28,28,1)

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
        cam = cv2.resize(cam, (168,224), cv2.INTER_LINEAR)
        cam = np.maximum(cam, 0)
        cam = cam / cam.max()

        # モノクロ画像に疑似的に色をつける
        cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

        # オリジナルイメージもカラー化
        #org_img = cv2.cvtColor(np.uint8(img_array[0]), cv2.COLOR_GRAY2BGR)  # (w,h) -> (w,h,3)
        org_img = cv2.cvtColor(np.uint8(img_array[0]), cv2.COLOR_GRAY2BGR)  # (w,h) -> (w,h,3)

        # 合成
        rate = 0.4
        cam = cv2.addWeighted(src1=org_img, alpha=(1-rate), src2=cam, beta=rate, gamma=0)
        cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)  # 色をRGBに変換

        # 表示
        plt.imshow(cam)
        plt.savefig(savepath + "cam.png")
        plt.clf()
        plt.close()