from math import pi
import numpy as np
import matplotlib.pyplot as plt 
from scipy.io import loadmat
import os
import re
from glob import glob
import csv
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.utils.data
from numba import jit, njit
from numba.experimental import jitclass

from utils import radian2Degree, degree2Radian

# from nvidia.dali import pipeline_def, Pipeline
# import nvidia.dali.fn as fn
# import nvidia.dali.types as types

def loadHRIR(path):
    names = []
    names += glob(path)
    print(names[0])

    splitnames = [os.path.split(name) for name in names]
    print(len(splitnames))

    p = re.compile('IRC_\d{4,4}')
    print(p)

    subjects = [int(name[4:8]) for base, name in splitnames 
                            if not (p.match(name[-8:]) is None)]
    print(subjects)

    k = 0
    subject = subjects[k]

    for k in range(len(names)):
        subject = subjects[k]
        # filename = os.path.join(names[k], 'IRC_' + str(subject))
        filename = os.path.join(names[k], 'COMPENSATED/MAT/HRIR/IRC_' + str(subject) + '_C_HRIR.mat')
    #     print(filename)

    m = loadmat(filename, struct_as_record=True)
    print(m.keys())
    print(m['l_eq_hrir_S'].dtype)

    l, r = m['l_eq_hrir_S'], m['r_eq_hrir_S']
    hrirSet_l = l['content_m'][0][0]
    hrirSet_r = r['content_m'][0][0]
    elev = l['elev_v'][0][0]
    azim = l['azim_v'][0][0]
    fs_HRIR = m['l_eq_hrir_S']['sampling_hz'][0][0][0][0]

    locLabel = np.hstack((elev, azim)) 
    print("locLabel shape: ", locLabel.shape, " (order: elev, azim)") # locLabel shape: (187, 2)
    # print(locLabel[0:5])

    # 0: left-ear 1: right-ear
    hrirSet = np.vstack((np.reshape(hrirSet_l, (1,) + hrirSet_l.shape),
                            np.reshape(hrirSet_r, (1,) + hrirSet_r.shape)))
    hrirSet = np.transpose(hrirSet, (1,0,2))
    print("hrirSet shape: ", hrirSet.shape)
    
    return hrirSet, locLabel, fs_HRIR

class LocRegion:
    def __init__(self, locLabel):
        self.locLabel = locLabel
        
        self.high_left = []
        self.high_right = []
        self.low_left = []
        self.low_right = []
        for i in range(self.locLabel.shape[0]):
            if self.locLabel[i, 0] > 0 and 0 < self.locLabel[i, 1] < 180:
                self.high_left.append(i)
            elif self.locLabel[i, 0] < 0 and 0 < self.locLabel[i, 1] < 180:
                self.low_left.append(i)
            elif self.locLabel[i, 0] > 0 and 0 < self.locLabel[i, 1] > 180:
                self.high_right.append(i)
            elif self.locLabel[i, 0] < 0 and 0 < self.locLabel[i, 1] > 180:
                self.low_right.append(i)
    
    def getLocRegion(self):
        return (self.high_left, self.low_left, self.high_right, self.low_right)

    def whichRegion(self, locIndex):
        if locIndex in self.high_left:
            return "HL"
        elif locIndex in self.low_left:
            return "LL"
        elif locIndex in self.high_right:
            return "HR"
        elif locIndex in self.low_right:
            return "LR"
        else:
            return "None"



if __name__ == "__main__":
    path = "./HRTF/IRC*"
    hrirSet, locLabel, fs_HRIR = loadHRIR(path)