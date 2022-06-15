
# add module 
import os , sys
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import copy
from tqdm import tqdm
import shutil       # file copy module??(I will select program)

# add module my py
import cnnmodel

# option info tensorflow == 2.5.0

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
        self.image_shape = (self.target_size_Original,self.target_size_Original,3)
        self.regression = 1

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
        self.y_train = self.df_csv_train["{}".format(CNN_object)]

        # model setting
        self.ModelSetting()
        self.LRSetting()
        self.model.compile(optimizer=self.optimizer_alpha, loss='mse', metrics=['mae']) 
        self.history = self.model.fit(self.x_train,self.y_train,
                                        batch_size=self.cnf.batch_size,
                                        epochs=self.cnf.maxepochs,
                                        validation_split=0.2,
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


        self.predict_test = self.model.predict(self.x_test,verbose=1)
        self.predict = []
        for i in range(self.cnf.test_data_num):
            self.predict.append(self.predict_test[i,0])

        self.y_test = self.df_csv_test["{}".format(CNN_object)]
        if(CNN_object in self.cnf.angle_mode):
            self.y_test = np.rad2deg(self.y_test)
        
        df_test = pd.DataFrame()
        df_test["real"] = self.y_test
        df_test["predict"] = self.predict
        df_test.to_csv("2_1_CNN_RES/{}/trial_{}/DL_predict_{}.csv".format(CNN_object,trial,CNN_object))

        self.history_anarysis(self.history)

    def set_dataset(self):
        os.chdir("RobotInfo")
        ###train_data###
        self.df_csv_train = pd.read_csv("train_data.csv")
        self.df_csv_test = pd.read_csv("test_data.csv")
        os.chdir("..")


    def preprocess(self):
        os.chdir("RobotSimulation")
        ###train_data###
        os.chdir("train")
        self.x_train = np.array([])
        self.x_train_storage = np.array([])     # storage
        print("===train_data_preprocess===")
        split_count = 0
        for i in tqdm(range(self.cnf.train_data_num)):
            if self.x_train_storage.shape == (0,):
                img = load_img('Arm_{}.jpg'.format(i),
                                grayscale=self.cnf.GRAYSCALE, color_mode="rgb", target_size=(self.cnf.target_size_Original,self.cnf.target_size_Original))
                array = img_to_array(img)
                array /= 255
                self.x_train_storage = np.array([array])
                split_count += 1
            else:
                img = load_img('Arm_{}.jpg'.format(i),
                                grayscale=self.cnf.GRAYSCALE, color_mode="rgb", target_size=(self.cnf.target_size_Original,self.cnf.target_size_Original))
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
        plt.imshow(self.x_train[1])
        plt.savefig("targer_size_checker.png")
        plt.clf()
        plt.close()

        os.chdir("test")
        self.x_test = np.array([])
        print("===test_data_preprocess===")
        for i in tqdm(range(self.cnf.test_data_num)):
            if i==0:
                img = load_img('Arm_{}.jpg'.format(i),
                                 grayscale=self.cnf.GRAYSCALE,color_mode="rgb", target_size=(self.cnf.target_size_Original,self.cnf.target_size_Original))
                array = img_to_array(img)
                array /= 255
                self.x_test = np.array([array])
            else:
                img = load_img('Arm_{}.jpg'.format(i),
                                grayscale=self.cnf.GRAYSCALE,color_mode="rgb", target_size=(self.cnf.target_size_Original,self.cnf.target_size_Original))
                array = img_to_array(img)
                array /= 255
                self.x_test = np.append(self.x_test,np.array([array]),axis=0)
        os.chdir("..")
        os.chdir("..")


    
    def DL_main(self):
        print("===This is Deep Learning seccion===")
        image_shape = (self.target_size_Original,self.target_size_Original,3)
        regression = 1                                                              # 回帰を表現

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
        for CNN_object in self.cnf.angle_mode:
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

        print("current directory\t:\t",os.getcwd())
    
    # history have all logs
    def history_anarysis(self,history):
        # loss -> mse
        self.history_in_mode = history
        loss = self.history_in_mode.history["loss"]
        val_loss= self.history_in_mode.history["val_loss"]
        mae= self.history_in_mode.history["mae"]
        val_mae= self.history_in_mode.history["val_mae"]

        self.EachAdaption_list.append(val_loss[-1])

        epochs = range(1,len(loss) + 1)
        # loss(MSE) learning curb
        fig1 = plt.figure()
        plt.plot(epochs , loss , color = "orange",marker = "o",ms = 2,lw = 0.1, label = "training loss")
        plt.plot(epochs , val_loss , color = "blue",marker = "o",ms = 2,lw = 0.1, label = "validation loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.yscale("log")
        plt.legend()
        fig1.savefig("2_1_CNN_RES/{}/trial_{}/loss.png".format(self.CNN_object,self.trial))
        plt.clf()
        plt.close()

        # mae learning curb
        fig2 = plt.figure()
        plt.plot(epochs , mae , color = "orange",marker = "o",ms = 2,lw = 0.1, label = "training mae")
        plt.plot(epochs , val_mae , color = "blue",marker = "o",ms = 2,lw = 0.1, label = "validation mae")
        plt.xlabel("Epochs")
        plt.ylabel("mae")
        plt.yscale("log")
        plt.legend()
        fig2.savefig("2_1_CNN_RES/{}/trial_{}/mae.png".format(self.CNN_object,self.trial))
        plt.clf()
        plt.close()

        # pandas csv save
        df_lrcurb = pd.DataFrame()
        df_lrcurb["epochs"] = epochs
        df_lrcurb["loss"] = loss
        df_lrcurb["val_loss"] = val_loss
        df_lrcurb["mae"] = loss
        df_lrcurb["val_mae"] = val_loss
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
            self.model = cnnmodel.Model_CPU(self.image_shape,self.regression)
        elif(self.cnf.machine == "GPU"):
            if(self.CNN_object == "Link1_angle"):
                self.model = cnnmodel.Model_GPU(self.image_shape,self.regression)
            elif(self.CNN_object == "Link2_angle"):
                self.model = cnnmodel.Model_GPU(self.image_shape,self.regression)
            else:
                self.model = cnnmodel.Model_GPU(self.image_shape,self.regression)

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
