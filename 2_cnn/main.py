# add module
import numpy as np
import pandas as pd

# add py module
import deeplearning

# add module by myself
import os , sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../tools'))  # 1階層上のfileのimport
import configuration

if __name__ == '__main__':
    print('\033[1m'+'###cnn###'+'\033[0m') # bold custom
    cnf = configuration.Configuration()
    cnf.setRobotsimulation()
    cnf.setCNN()
    cnf.reset_seed()

    os.makedirs("2_1_CNN_RES",exist_ok=True)
    for Mode_name in cnf.ALL_mode:
        os.makedirs("2_1_CNN_RES/{}".format(Mode_name),exist_ok=True)
        for trial in range(cnf.CNN_trial):
            os.makedirs("2_1_CNN_RES/{}/trial_{}".format(Mode_name,trial),exist_ok=True)

    CNN = deeplearning.DeepLearnig(cnf)
    CNN.targetsize(224)     # arity is magic number
    CNN.DL_main()

