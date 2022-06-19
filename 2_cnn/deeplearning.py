
# add module 
import os , sys
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import copy
import  cv2
from tqdm import tqdm
import shutil       # file copy module??(I will select program)

# add module my py
import cnnmodel

# option info tensorflow == 2.5.0

# keras py -3.6 ver 2.4.3
from keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img
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
from keras.utils import np_utils        # class categorical ??

class DeepLearnig:
    # コンストラクタ
    # datasetを作成してもいいくらいだけど今回はここで作成してしまいます
    def __init__(self,cnf):
        self.cnf = cnf
        matplotlib.use('Agg')

        self.x_train = []
        self.x_train_storage = []       # for fast preproocess

        self.x_train_Link1_x = []
        self.x_train_Link1_y = []

        self.x_train_Link2_x = []
        self.x_train_Link2_y = []

        self.y_train = []
        self.x_test = []
        self.y_test = []

        self.target_size_Original = self.cnf.target_size_Original
        self.image_shape = (self.targetsize,1)
        self.regression = 2

        self.Link1_x_predict = []
        self.Link1_y_predict = []
        self.Link2_x_predict = []
        self.Link2_y_predict = []

        self.Link1_angle_predict = []
        self.Link2_angle_predict = []

        self.EachAdaption_list = []

    def DLsettings(self,CNN_object,trial):
        self.CNN_object = CNN_object
        self.trial = trial

        self.y_train = np_utils.to_categorical(self.y_train, 2)

        # model setting
        self.ModelSetting()
        self.LRSetting()
        self.model.compile(optimizer=self.optimizer_alpha, loss='categorical_crossentropy', metrics=['accuracy']) 
        self.history = self.model.fit(self.x_train,self.y_train,
                                        batch_size=self.cnf.batch_size,
                                        epochs=self.cnf.maxepochs,
                                        validation_split=0.0,
                                        verbose = self.cnf.verbose,
                                        shuffle=self.cnf.shuffle,
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
                                            '2_1_CNN_RES/{}/trial_{}/{}_{}.h5'.format(CNN_object,trial,CNN_object,trial),
                                            monitor='val_loss',
                                            mode='min',
                                            verbose=1,
                                            save_best_only=True,
                                            save_weights_only=False,
                                        ),
                                        TensorBoard(
                                            log_dir='2_1_CNN_RES/{}/trial_{}/{}_{}'.format(CNN_object,trial,CNN_object,trial),
                                        ),
                                        ]
                                        )

        self.model.save("baum.h5")
        #self.predict_test = self.model.predict(self.x_test,verbose=1)
        #self.predict = []
        #for i in range(self.cnf.test_data_num):
        #    self.predict.append(self.predict_test[i,0])
#
        #self.y_test = self.df_csv_test["{}".format(CNN_object)]
        #if(CNN_object in self.cnf.angle_mode):
        #    self.y_test = np.rad2deg(self.y_test)
        
        #df_test = pd.DataFrame()
        #df_test["real"] = self.y_test
        #df_test["predict"] = self.predict
        #df_test.to_csv("2_1_CNN_RES/{}/trial_{}/DL_predict_{}.csv".format(CNN_object,trial,CNN_object))

        self.history_anarysis(self.history)

    def set_dataset(self):
        self.y_train = np.array([0 for i in range(self.cnf.train_data_num)] + [1 for i in range(self.cnf.test_data_num)])
        self.y_test = np.array([1 for i in range(self.cnf.test_data_num)])

    def targetsize(self,dpi):
        im = cv2.imread('submit/a/0.bmp')  # <class 'numpy.ndarray'>
        #print(im.shape)     # <tuple>
        assert dpi < im.shape[0] , "you choice too big dpi for comparing original pic size (%d->%d)"%(dpi , im.shape[0])
        aspect = im.shape[1] / im.shape[0]
        self.targetsize = int(dpi),int(dpi*aspect),1
        #img = load_img('submit/a/0.bmp', grayscale=False, color_mode='rgb', target_size=(int(dpi),int(dpi*aspect)))
        #i2ary = img_to_array(img)
        #save_img(image_save_dir+ "{}.bmp".format(dpi), i2ary)


    def preprocess(self):
        os.chdir("submit")
        ###train_data###
        os.chdir("a")
        self.x_train = np.array([])
        self.x_train_storage = np.array([])     # storage
        print("===train_data_preprocess===")
        split_count = 0
        for i in tqdm(range(self.cnf.train_data_num)):
            if self.x_train_storage.shape == (0,):
                img = load_img('{}.bmp'.format(i),
                                grayscale=self.cnf.GRAYSCALE, color_mode='grayscale', target_size=self.targetsize)
                array = img_to_array(img)
                array /= 255
                self.x_train_storage = np.array([array])
                split_count += 1
            else:
                img = load_img('{}.bmp'.format(i),
                                grayscale=self.cnf.GRAYSCALE, color_mode='grayscale', target_size=self.targetsize)
                array = img_to_array(img)
                array /= 255
                self.x_train_storage = np.append(self.x_train_storage,np.array([array]),axis=0)
                split_count += 1
            
            if (split_count == self.cnf.split) or (i == self.cnf.train_data_num-1):
                split_count = 0
                if self.x_train.shape == (0,):
                    self.x_train = self.x_train_storage
                else:
                    self.x_train = np.concatenate([self.x_train , self.x_train_storage])
                self.x_train_storage = np.array([])

        os.chdir("..")
        ###test data####
        plt.imshow(self.x_train[1],cmap = "gray")
        plt.savefig("targer_size_checker.png")
        plt.clf()
        plt.close()

        os.chdir("b")
        print("===train_data_preprocess===")
        split_count = 0
        for i in tqdm(range(self.cnf.test_data_num)):
            if self.x_train_storage.shape == (0,):
                img = load_img('{}.bmp'.format(i),
                                grayscale=self.cnf.GRAYSCALE, color_mode='grayscale', target_size=self.targetsize)
                array = img_to_array(img)
                array /= 255
                self.x_train_storage = np.array([array])
                split_count += 1
            else:
                img = load_img('{}.bmp'.format(i),
                                grayscale=self.cnf.GRAYSCALE, color_mode='grayscale', target_size=self.targetsize)
                array = img_to_array(img)
                array /= 255
                self.x_train_storage = np.append(self.x_train_storage,np.array([array]),axis=0)
                split_count += 1
            
            if (split_count == self.cnf.split) or (i == self.cnf.train_data_num-1):
                split_count = 0
                if self.x_train.shape == (0,):
                    self.x_train = self.x_train_storage
                else:
                    self.x_train = np.concatenate([self.x_train , self.x_train_storage])
                self.x_train_storage = np.array([])
        os.chdir("..")
        os.chdir("..")


    
    def DL_main(self):
        print("===This is Deep Learning seccion===")
        image_shape = (self.targetsize,1)

        self.main_path = os.getcwd()
        self.set_dataset()
        self.preprocess()

        # coordinate mode
        for CNN_object in self.cnf.ALL_mode:
            print("###{}###".format(CNN_object))

            self.optimizer = self.cnf.optimizer
            print("===optimizer : {}===".format(self.optimizer))

            self.EachAdaption_list = []
            self.mintrial = 0       # df trial
            for trial in range(self.cnf.CNN_trial):
                self.cnf.reset_seed(trial)      # set random seed
                self.DLsettings(CNN_object,trial)
            self.extract_best_model()
            self.copy_model()
        # angle mode
        #for CNN_object in self.cnf.angle_mode:
        #    print("###{}###".format(CNN_object))
#
        #    self.optimizer = self.cnf.optimizer
        #    print("===optimizer : {}===".format(self.optimizer))
#
        #    self.EachAdaption_list = []
        #    self.mintrial = 0       # df trial
        #    for trial in range(self.cnf.CNN_trial):
        #        self.cnf.reset_seed(trial)      # set random seed
        #        self.DLsettings(CNN_object,trial)
        #    self.extract_best_model()
        #    self.copy_model()

        print("current directory\t:\t",os.getcwd())
    
    # history have all logs
    def history_anarysis(self,history):
        # loss -> mse
        self.history_in_mode = history
        loss = self.history_in_mode.history["loss"]
        #val_loss= self.history_in_mode.history["val_loss"]
        accuracy= self.history_in_mode.history["accuracy"]
        #val_accuracy= self.history_in_mode.history["val_accuracy"]

        #self.EachAdaption_list.append(val_loss[-1])

        epochs = range(1,len(loss) + 1)
        # loss(MSE) learning curb
        fig1 = plt.figure()
        plt.plot(epochs , loss , color = "orange",marker = "o",ms = 2,lw = 0.1, label = "training loss")
        #plt.plot(epochs , val_loss , color = "blue",marker = "o",ms = 2,lw = 0.1, label = "validation loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.legend()
        fig1.savefig("2_1_CNN_RES/{}/trial_{}/loss.png".format(self.CNN_object,self.trial))
        plt.clf()
        plt.close()

        # accuracy learning curb
        fig2 = plt.figure()
        plt.plot(epochs , accuracy , color = "orange",marker = "o",ms = 2,lw = 0.1, label = "training accuracy")
        #plt.plot(epochs , val_accuracy , color = "blue",marker = "o",ms = 2,lw = 0.1, label = "validation accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("accuracy")
        plt.yscale("log")
        plt.legend()
        fig2.savefig("2_1_CNN_RES/{}/trial_{}/accuracy.png".format(self.CNN_object,self.trial))
        plt.clf()
        plt.close()

        # pandas csv save
        df_lrcurb = pd.DataFrame()
        df_lrcurb["epochs"] = epochs
        df_lrcurb["loss"] = loss
        #df_lrcurb["val_loss"] = val_loss
        df_lrcurb["accuracy"] = loss
        #df_lrcurb["val_accuracy"] = val_loss
        df_lrcurb.to_csv("2_1_CNN_RES/{}/trial_{}/learningcurb.csv".format(self.CNN_object,self.trial))


    def extract_best_model(self):
        # self.EachAdaption_list sort and get idx
        index = ["trial_{}".format(i) for i in range(self.cnf.CNN_trial)]
        df_valloss = pd.DataFrame(data = self.EachAdaption_list,index= index)
        df_valloss.to_csv("2_1_CNN_RES/{}/trialtofitness.csv".format(self.CNN_object))

        # want to know best trial
        forbesttrial = copy.deepcopy(self.EachAdaption_list)
        forbesttrial_idx = np.argsort(forbesttrial)
        self.mintrial = forbesttrial_idx[0] # set min trial
        self.saveImportancedata(forbesttrial_idx,forbesttrial)

    def copy_model(self):
        shutil.copy2('2_1_CNN_RES/{}/trial_{}/{}_{}.h5'.format(self.CNN_object,self.mintrial ,self.CNN_object,self.mintrial), '2_1_CNN_RES')

    def ModelSetting(self):
        if(self.cnf.machine == "CPU"):
            self.model = cnnmodel.baumtest(self.targetsize,self.regression)
        elif(self.cnf.machine == "GPU"):
            self.model = cnnmodel.baumtest(self.targetsize,self.regression)


    def LRSetting(self):
        if(self.optimizer == "Adagrad"):
            self.optimizer_alpha = optimizers.Adagrad(learning_rate=1e-4)
        elif(self.optimizer == "Adadelta"):
            self.optimizer_alpha = optimizers.Adam(learning_rate=1.0)
        elif(self.optimizer == "Adam"):
            self.optimizer_alpha = optimizers.Adam(learning_rate=1e-4)
        elif(self.optimizer == "Adamax"):
            self.optimizer_alpha = optimizers.Adamax(learning_rate=2e-4)
        elif(self.optimizer == "Nadam"):
            self.optimizer_alpha = optimizers.Nadam(learning_rate=2e-4)
        elif(self.optimizer == "RMSprop"):
            self.optimizer_alpha = optimizers.RMSprop(learning_rate=1e-3)
        elif(self.optimizer == "SGD"):
            self.optimizer_alpha = optimizers.SGD(learning_rate=1e-3, momentum=0.4,nesterov=True,decay=1e-4,clipnorm=0.4)

    def saveImportancedata(self,idx,data):
        f = open("2_1_CNN_RES/{}/lrdata.txt".format(self.CNN_object),"w")
        f.write("best trial is {}\n".format(idx[0]))
        f.write("best fitness is {}\n".format(data[0]))
        f.close()
