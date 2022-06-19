from robotsetting import Link
from robotsetting import Circle

# add module
import os , sys
import tensorflow as tf     # py -3.6
import matplotlib
import numpy as np
import random as rd

class Configuration:
    def __init__(self):
        print("###configuration###")
        self.startpath = os.getcwd()
        self.problem = "random"
        self.ALL_mode = ["baum"]       # object for cnn ex) 研究の対象：an object of study
        self.angle_mode = ["Link1_angle","Link2_angle"]
        matplotlib.use("Agg")   # memory flash
    
    def setProblemenvironment(self):
        self.train_data_num = 45
        self.test_data_num = 45
        self.problem = "circle"
        # simulation environment??
        if self.problem == "circle":
            self.test_data_num = 36

    
    def setCompare(self):
        self.comparelist = ["_CGP","_CNN"]
    
    def setSR(self):
        self.Inputs = ["Link1_x","Link1_y","Link2_x","Link2_y"]
        self.Outputs = ["Link1_angle","Link2_angle"]
    
    def setRobotsimulation(self):
        # Simulation Experiment
        self.num_Links = 2
        self.L1 = 12
        self.L2 = 10
        self.margin = 2
        self.Link_info = [[0,self.L1],[0,self.L2]]
        self.arm_height = 0.5 
        ###train###
        self.train_Links = []
        for i in range(self.num_Links):  # Sampling By Genetic Programming
            self.train_Links.append(Link(self.Link_info[i][0],self.Link_info[i][1]))
        ###test###
        self.test_Links = []
        for i in range(self.num_Links):  # Sampling By Genetic Programming
            self.test_Links.append(Link(self.Link_info[i][0],self.Link_info[i][1]))
        
        self.circle = Circle()

        # Simulation Environment
        self.train_data_num = 10
        self.test_data_num = 10

    def setCNN(self):
        self.machine = "CPU"
        self.CNN_trial = 3
        self.target_size_Original = 112
        self.optimizer = "Adam"
        self.maxepochs = 300
        self.batch_size = 32
        self.validation_split = 0.2
        self.verbose = 1
        self.shuffle = True
        self.GRAYSCALE = True
        self.verbose = 1

        self.split = 2      # under train dataset size
    
    def setPreprocess(self):
        self.problem = "random"


    def reset_seed(self,seed=1):    # environment variables
        os.environ['TF_DETERMINISTIC_OPS'] = '{}'.format(seed)
        tf.random.set_seed(seed)
        os.environ['TF_DETERMINISTIC_OPS'] = 'true'
        os.environ['TF_CUDNN_DETERMINISTIC'] = 'true'

        np.random.seed(seed)
        rd.seed(seed)
        
        session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=32, inter_op_parallelism_threads=32)
        tf.compat.v1.set_random_seed(seed)
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)